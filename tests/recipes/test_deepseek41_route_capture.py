# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the debug-serve route capture (routecapture.py): ring contents, predictor law, read-only inputs (eager
and torch.compile), the chunk drain, and the export script's refusal / debug-serve switch."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from ignition.recipes.deepseek41 import routecapture as rc

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tp3-tools" / "ds41-exl3-export-tp2.sh"
PACK = Path(os.environ.get("FLARE_TEST_DS41_EXL3_PACK") or "/nonexistent-exl3-pack")  # the EXL3 pack directory
E, H = 384, 64


@pytest.fixture
def capture(tmp_path, monkeypatch):
    monkeypatch.setenv(rc.ENV, str(tmp_path))
    monkeypatch.setattr(rc, "_dumper", lambda *a, **k: None)   # the drain is tested directly below
    rc._STATE.clear(); rc._LAYOUT.clear()
    yield tmp_path
    rc._STATE.clear(); rc._LAYOUT.clear()


def _router(seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn((E, H), generator=g) * 0.05).bfloat16(), torch.randn((E,), generator=g) * 0.01


def _rows(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((n, H), generator=g).bfloat16(), torch.randint(0, E, (n, 6), generator=g, dtype=torch.int32)


def _numpy_predict(x, w, b, k=rc.PRED):
    logits = (x.float() @ w.float().t()).bfloat16().float().numpy()   # BF16 GEMV output, as predict()
    scores = np.sqrt(np.log1p(np.exp(logits))) + b.numpy()[None]
    return np.argsort(-scores, axis=1, kind="stable")[:, :k]


def test_off_is_a_no_op(monkeypatch):
    monkeypatch.delenv(rc.ENV, raising=False)
    rc._STATE.clear()
    cap = rc.RouteCapture()
    cap.allocate("cpu", "verify", 3)
    cap.step(torch.zeros(6, dtype=torch.int64), "verify")
    assert not cap.enabled and not rc._STATE and not cap.active("decode")


def test_ring_contents_mask_and_predictors(capture):
    cap = rc.RouteCapture()
    own, nxt = _router(1), _router(2)
    hot_map = torch.randperm(E, generator=torch.Generator().manual_seed(3)).int()
    cap.allocate("cpu", "verify", 7, hot_map, 320)
    cap.allocate("cpu", "prefill", 8)                    # prefill / draft graphs are not captured
    cap.allocate("cpu", "draft", 9)
    assert set(rc._LAYOUT) == {7} and rc._LAYOUT[7]["hot_rows"] == 320 and rc._LAYOUT[7]["hot_map"] == hot_map.tolist()
    positions = torch.tensor([10, 11, 12, -1, -1, 13])   # rows 3-4 = graph padding
    moe_in, ids = _rows(6, 4)
    attn_in, _ = _rows(6, 5)
    for step in range(3):
        cap.step(positions, "verify")
        cap.record(ids + step, moe_in, attn_in, 7, "verify", own, nxt, "sqrtsoftplus")
        cap.record(ids, moe_in, attn_in, 39, "verify", own, None, "sqrtsoftplus")   # last layer: no next router
    cap.record(ids, moe_in, attn_in, 7, "draft", own, nxt, "sqrtsoftplus")          # ignored
    s = rc._STATE[None]
    assert int(s["counter"]) == 3
    for step in range(3):
        assert s["meta"][step].tolist() == [step + 1, 6, 2, 4]
        assert s["mask"][step, :6].tolist() == [1, 1, 1, 0, 0, 1] and not s["mask"][step, 6:].any()
        assert torch.equal(s["ids"][step, 7, :6].long(), (ids + step).long())
        assert (s["ids"][step, 7, 6:] == -1).all() and (s["ids"][step, 0] == -1).all()   # layer 0 never recorded
    assert np.array_equal(s["nxt"][0, 7, :6].numpy(), _numpy_predict(moe_in, *nxt))
    assert np.array_equal(s["early"][0, 7, :6].numpy(), _numpy_predict(attn_in, *own))
    assert (s["nxt"][0, 39] == -1).all() and np.array_equal(s["early"][0, 39, :6].numpy(), _numpy_predict(attn_in, *own))


def test_rows_beyond_rmax_and_ring_wrap(capture):
    cap = rc.RouteCapture()
    cap.allocate("cpu", "decode")
    own = _router(1)
    rows = rc.RMAX + 5
    moe_in, ids = _rows(rows, 6)
    for _ in range(rc.RING + 2):
        cap.step(torch.arange(rows), "decode")
        cap.record(ids, moe_in, moe_in, 0, "decode", own, own, "sqrtsoftplus")
    s = rc._STATE[None]
    assert int(s["counter"]) == rc.RING + 2
    assert s["meta"][1].tolist() == [rc.RING + 2, rows, 1, rows]          # slot 1 lapped twice
    assert torch.equal(s["ids"][1, 0].long(), ids[:rc.RMAX].long())


def test_inputs_and_outputs_unchanged_eager_and_compiled(capture):
    """The capture must not change what the surrounding computation produces: run a toy layer tail with and without
    the capture ops, eager and under torch.compile, and compare outputs and inputs bit for bit."""
    cap = rc.RouteCapture()
    own, nxt = _router(7), _router(8)
    cap.allocate("cpu", "verify", 5)
    moe_in, ids = _rows(6, 9)
    attn_in, _ = _rows(6, 10)
    positions = torch.tensor([1, 2, 3, 4, -1, -1])
    w2 = torch.randn((H, H)).bfloat16()

    def tail(moe_in, attn_in, ids, with_capture):
        if with_capture:
            cap.step(positions, "verify")
            cap.record(ids, moe_in, attn_in, 5, "verify", own, nxt, "sqrtsoftplus")
        y = torch.nn.functional.silu(moe_in.float() @ w2.float()) + attn_in.float() * ids[:, :1].float()
        return y.bfloat16()

    snapshot = [t.clone() for t in (moe_in, attn_in, ids, own[0], own[1], nxt[0], nxt[1])]
    want = tail(moe_in, attn_in, ids, False)
    got = tail(moe_in, attn_in, ids, True)
    compiled = torch.compile(tail, backend="inductor", fullgraph=True, dynamic=False)
    want_c = compiled(moe_in, attn_in, ids, False)
    got_c = compiled(moe_in, attn_in, ids, True)
    assert torch.equal(want, got) and torch.equal(want_c, got_c)
    for before, after in zip(snapshot, (moe_in, attn_in, ids, own[0], own[1], nxt[0], nxt[1])):
        assert torch.equal(before, after)
    assert int(rc._STATE[None]["counter"]) == 2                           # eager + compiled capture both ran


def test_drain_writes_finished_slots_and_counts_lost(capture, tmp_path):
    cap = rc.RouteCapture()
    cap.allocate("cpu", "verify")
    own = _router(1)
    moe_in, ids = _rows(6, 11)
    proc = tmp_path / "proc"; proc.mkdir()
    for _ in range(5):
        cap.step(torch.arange(6), "verify")
        cap.record(ids, moe_in, moe_in, 2, "verify", own, own, "sqrtsoftplus")
    s = rc._STATE[None]
    done, chunk, lost = rc.drain(s, str(proc), 0, 0, 0)
    assert (done, chunk, lost) == (4, 1, 0)                               # forward 5 may still be writing
    z = np.load(proc / "chunk-000000.npz")
    assert z["meta"][:, 0].tolist() == [1, 2, 3, 4] and z["ids"].shape == (4, rc.LAYERS, 6, rc.TOPK)
    assert np.array_equal(z["ids"][0, 2], ids.numpy()) and z["mask"].shape == (4, 6)
    for _ in range(rc.RING + 10):
        cap.step(torch.arange(6), "verify")
    done2, chunk2, lost2 = rc.drain(s, str(proc), done, chunk, lost)
    assert done2 == rc.RING + 14 and chunk2 == 2 and lost2 == (rc.RING + 14 - done) - (rc.RING - 1)
    z = np.load(proc / "chunk-000001.npz")
    assert z["meta"].shape[0] == rc.RING - 1 and int(z["lost"]) == lost2


def test_dumper_makes_no_cuda_call_before_go(tmp_path, monkeypatch):
    # seq296: the dumper's synchronize during preparation invalidated a b12x comm.pcie race capture on another thread.
    # Before <dir>/go it may only write layout.json; drain (the only CUDA readout) and stream creation wait for go.
    import threading, time
    calls = []
    monkeypatch.setattr(rc, "drain", lambda state, proc, done, chunk, lost, stream=None:
                        (calls.append(("drain", time.time())), (done, chunk, lost))[1])
    monkeypatch.setattr(torch.cuda, "Stream", lambda *a, **k: calls.append(("stream", 0)))
    rc._LAYOUT.clear(); rc._LAYOUT[3] = {"hot_map": [0], "hot_rows": 1}
    state = {"counter": torch.zeros((1,), dtype=torch.int64)}
    threading.Thread(target=rc._dumper, args=(state, 0, str(tmp_path), 0.02), daemon=True).start()
    time.sleep(0.4)
    proc = tmp_path / f"dev0-pid{os.getpid()}"
    assert (proc / "layout.json").is_file() and calls == []
    rc._LAYOUT[5] = {"hot_map": [1, 0], "hot_rows": 1}          # a later layer registers: the file must follow
    deadline = time.time() + 5
    while len(json.loads((proc / "layout.json").read_text())["hot"]) < 2 and time.time() < deadline:
        time.sleep(0.02)
    assert json.loads((proc / "layout.json").read_text())["hot"]["5"]["hot_rows"] == 1
    (tmp_path / rc.GO).touch()
    deadline = time.time() + 5
    while not calls and time.time() < deadline:
        time.sleep(0.02)
    assert calls and calls[0][0] == "drain"           # CPU counter: no stream; drains start once go exists
    (tmp_path / "mark-start").touch()
    deadline = time.time() + 5
    while not (proc / "marks.jsonl").is_file() and time.time() < deadline:
        time.sleep(0.02)
    assert json.loads((proc / "marks.jsonl").read_text().splitlines()[0])["tag"] == "start"
    rc._LAYOUT.clear()


def test_a_mark_also_starts_the_readout(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rc, "drain", lambda state, proc, done, chunk, lost, stream=None:
                        (calls.append(1), (done, chunk, lost))[1])
    assert not rc._serving(str(tmp_path))
    (tmp_path / "mark-x").touch()
    assert rc._serving(str(tmp_path))


class _ToyForward:
    """A prepared-forward stand-in holding BOTH debug-serve recorders, as PreparedLayer / PreparedDecoder do."""

    def __init__(self):
        from ignition.recipes.deepseek41.routecount import RouteCounter
        self.counter, self.capture = RouteCounter(), rc.RouteCapture()
        self.counter.allocate("cpu", E)
        self.capture.allocate("cpu", "verify", 3)
        self.own, self.nxt = _router(21), _router(22)
        self.w2 = torch.randn((H, H)).bfloat16()

    def __call__(self, moe_in, attn_in, ids, positions):
        self.capture.step(positions, "verify")
        self.counter.record(ids, positions, 3, "verify", E)
        self.capture.record(ids, moe_in, attn_in, 3, "verify", self.own, self.nxt, "sqrtsoftplus")
        return (torch.nn.functional.silu(moe_in.float() @ self.w2.float()) + attn_in.float()).bfloat16()


def test_counter_and_capture_trace_clean_in_a_compile_worker(capture, monkeypatch):
    """seq297: forwards are traced in compile-worker processes from the PICKLED forward, where the modules' _STATE is
    empty; a recorder that looked its state up there allocated and started a thread inside the trace (dynamo:
    _thread.daemon_threads_allowed unsupported). Both recorders must carry their state on the object and trace with
    tensor ops only: pickle the forward, clear both _STATE dicts, forbid threads, compile fullgraph with inductor."""
    import pickle
    import types
    from ignition.recipes.deepseek41 import routecount as rcount
    monkeypatch.setenv(rcount.ENV, str(capture / "counts"))
    monkeypatch.setattr(rcount, "_dumper", lambda *a, **k: None)
    rcount._STATE.clear()
    worker = pickle.loads(pickle.dumps(_ToyForward()))
    rcount._STATE.clear(); rc._STATE.clear()

    def no_thread(*a, **k):
        raise AssertionError("a recorder started a thread during tracing")
    for module in (rcount, rc):   # inductor itself uses threads: forbid them in the recorder modules only
        monkeypatch.setattr(module, "threading", types.SimpleNamespace(Thread=no_thread))
    moe_in, ids = _rows(6, 31)
    attn_in, _ = _rows(6, 32)
    positions = torch.tensor([5, 6, 7, 8, -1, -1])
    compiled = torch.compile(worker, backend="inductor", fullgraph=True, dynamic=False)
    want = (torch.nn.functional.silu(moe_in.float() @ worker.w2.float()) + attn_in.float()).bfloat16()
    assert torch.equal(compiled(moe_in, attn_in, ids, positions), want)
    assert rcount._STATE == {} and rc._STATE == {}                         # nothing allocated during tracing
    assert int(worker.counter.table[rcount.LAYERS + 3].sum()) == 4 * 6     # 4 live rows x top-6, step group
    assert int(worker.capture.state["counter"]) == 1
    assert np.array_equal(worker.capture.state["ids"][0, 3, :6].numpy(), ids.numpy().astype(np.int16))
    rcount._STATE.clear()


def test_recorders_refuse_to_record_unallocated(capture):
    from ignition.recipes.deepseek41.routecount import RouteCounter
    moe_in, ids = _rows(2, 1)
    with pytest.raises(RuntimeError, match="before allocate"):
        rc.RouteCapture().record(ids, moe_in, moe_in, 1, "verify", _router(1), None, "sqrtsoftplus")
    with pytest.raises(RuntimeError, match="before allocate"):
        RouteCounter(enabled=True).record(ids, torch.arange(2), 1, "decode", E)


def _script(args, **env):
    full = {k: v for k, v in os.environ.items()
            if not k.startswith(("FLARE_DS41_", "DEBUG_SERVE", "HELIOPAUSE_PATH", "FLARE_DEBUG_SERVE"))}
    full.update(env, CKPT=str(PACK), CUDA_VISIBLE_DEVICES="")
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=full, timeout=300)


@pytest.mark.skipif(not (PACK / "config.json").is_file(), reason="EXL3 pack not present")
@pytest.mark.parametrize("var", ["FLARE_DS41_EXL3_ROUTE_CAPTURE", "FLARE_DS41_ROUTE_COUNTS", "FLARE_DEBUG_SERVE_SERVER_JSON"])
def test_export_refuses_debug_switches(var):
    out = _script(["--dry-run"], **{var: "/tmp/x"})
    assert out.returncode == 2 and "never export with it" in out.stderr


@pytest.mark.skipif(not (PACK / "config.json").is_file(), reason="EXL3 pack not present")
def test_debug_serve_argv_matches_export_argv():
    export = _script(["--dry-run"], PARK="2-4,6-39", HOT="320")
    serve = _script(["--dry-run"], PARK="2-4,6-39", HOT="320", DEBUG_SERVE="1", PORT="15015",
                    FLARE_DS41_EXL3_ROUTE_CAPTURE="/tmp/x", HELIOPAUSE_PATH="/nonexistent-qual",
                    FLARE_DEBUG_SERVE_SERVER_JSON=json.dumps({"mtp_adaptive_verification_alpha": 0.8}))
    assert export.returncode == 0 and serve.returncode == 0, (export.stderr, serve.stderr)
    a = next(line for line in export.stdout.splitlines() if line.startswith("export argv:")).split()
    b = next(line for line in serve.stdout.splitlines() if line.startswith("export argv:")).split()
    assert a[3] == "export" and b[3] == "debug-serve"
    assert b[:3] + b[4:-4] == a[:3] + a[4:] and b[-4:] == ["--host", "127.0.0.1", "--port", "15015"]
    pick = lambda out: [line for line in out.stdout.splitlines() if line.startswith(("FLARE_DS41_", "FLARE_TUNE_", "B12X_"))]
    assert [line for line in pick(serve) if not line.startswith("FLARE_DS41_EXL3_ROUTE_CAPTURE=")] == pick(export)


def test_decoder_wiring_present():
    """The decoder imports the capture and calls it on the step and per layer (source-level check; the GPU stage proves
    the graphs)."""
    src = (ROOT / "ignition/recipes/deepseek41/decoder.py").read_text()
    assert "self.route_capture.step(metadata.positions, self.mode)" in src
    assert "self.route_capture.record(route.ids, normalized, attention_normalized" in src
    assert "layer.route_capture_next = (following.router, following.correction)" in src
    from ignition.recipes.deepseek41 import decoder  # noqa: F401  (imports cleanly)
