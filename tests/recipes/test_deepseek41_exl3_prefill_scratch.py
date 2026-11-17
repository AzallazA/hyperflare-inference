# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the prefill scratch levers (exp/ds41-exl3-final-scratch): decode-once passes over ranges of the sorted
routes (FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS), the prefill group count (FLARE_DS41_EXL3_PREFILL_GROUPS) and the cold
prefill launch on a side stream (FLARE_DS41_EXL3_PREFILL_COLD_SIDE).

Equivalence: the NumPy mirror of the decode-once path gives bit-identical float64 output with and without passes, and
the production op, run on a fake extension whose kernels compute a toy expert from exactly the operands they are given
(job table built from the passed counts, sliced sorted arrays, scratch views) on exactly representable inputs, gives
bit-identical output for every switch combination (each route applied once, by its expert, in some launch)."""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest

from ignition.recipes.deepseek41.exl3 import do_reference as dor

torch = pytest.importorskip("torch")
from ignition.recipes.deepseek41.exl3 import experts as ex  # noqa: E402
from ignition.recipes.deepseek41.exl3 import xmoe  # noqa: E402

PER_ROW = 2 * 5120 + 2 * 1152


def test_switches(monkeypatch):
    for name in ("FLARE_DS41_EXL3_PREFILL_GROUPS", "FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS",
                 "FLARE_DS41_EXL3_PREFILL_COLD_SIDE"):
        monkeypatch.delenv(name, raising=False)
    assert ex.prefill_groups() == 4 and ex.prefill_pass_routes() == 0 and not ex.prefill_cold_side()
    monkeypatch.setenv("FLARE_DS41_EXL3_PREFILL_GROUPS", "2")
    monkeypatch.setenv("FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS", "8192")
    monkeypatch.setenv("FLARE_DS41_EXL3_PREFILL_COLD_SIDE", "1")
    assert ex.prefill_groups() == 2 and ex.prefill_pass_routes() == 8192 and ex.prefill_cold_side()
    for name, bad in (("FLARE_DS41_EXL3_PREFILL_GROUPS", "0"), ("FLARE_DS41_EXL3_PREFILL_GROUPS", "5"),
                      ("FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS", "1000"), ("FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS", "-128"),
                      ("FLARE_DS41_EXL3_PREFILL_COLD_SIDE", "yes")):
        monkeypatch.setenv(name, bad)
        with pytest.raises(ValueError, match=name):
            {"FLARE_DS41_EXL3_PREFILL_GROUPS": ex.prefill_groups,
             "FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS": ex.prefill_pass_routes,
             "FLARE_DS41_EXL3_PREFILL_COLD_SIDE": ex.prefill_cold_side}[name]()
        monkeypatch.delenv(name)


def test_scratch_sizes_at_tp2():
    full = ex.scratch_halfs(4096, 1152, 4, decode_once=True)
    assert full == 24576 * PER_ROW                                       # today: 588 MiB
    assert 2 * full // 2**20 == 588
    shrunk = ex.scratch_halfs(4096, 1152, 2, decode_once=True, pass_routes=8192)
    assert shrunk == 2 * 4096 * PER_ROW == 8192 * PER_ROW                 # 196 MiB: max(2 groups, one 8192-route pass)
    side = ex.scratch_halfs(4096, 1152, 2, decode_once=True, pass_routes=4096, cold_side=True)
    assert side == 2 * 4096 * PER_ROW + 4096 * PER_ROW                    # 294 MiB: both regions at once
    assert ex.scratch_halfs(4096, 1152, 4, decode_once=True, pass_routes=32768) == full   # pass above the chunk
    assert ex.scratch_halfs(4096, 1152, 2) == 2 * 4096 * PER_ROW           # group mode: groups only
    report = ex.scratch_report(4096, 1152, 40, groups=2, pass_routes=8192, cold_side=False, mode="do")
    assert report["default_bytes"] == 2 * full and report["bytes"] == 2 * shrunk
    assert report["saved_bytes"] == 2 * (full - shrunk)


@pytest.mark.parametrize("routed,step", [(24576, 0), (24576, 8192), (24576, 4096), (24576, 10240), (768, 4096),
                                         (0, 128)])
def test_passes_tile_the_routes(routed, step):
    passes = ex.do_passes(routed, step)
    assert passes[0][0] == 0 and passes[-1][1] == routed
    assert all(a[1] == b[0] for a, b in zip(passes, passes[1:]))
    assert all(stop - start <= (step or routed) for start, stop in passes)


@pytest.mark.parametrize("seed", range(5))
def test_pass_counts_match_brute_force_and_jobs_cover_hot_routes_once(seed):
    rng = np.random.default_rng(seed)
    experts = 384
    rows = 4096
    counts = np.append(rng.multinomial(rows * 6, rng.dirichlet(np.full(experts, 0.3))), 0)
    hot = int(rng.choice([experts, 320, 0]))
    order_ids = np.repeat(np.arange(experts + 1), counts)            # expert of each sorted route
    step = int(rng.choice([128, 4096, 8192, 10240]))
    covered = np.zeros(order_ids.size, np.int64)
    capacity = ex.do_job_capacity(rows * 6, experts)
    for start, stop in ex.do_passes(order_ids.size, step):
        got = ex.pass_counts(torch.from_numpy(counts), start, stop).numpy()
        want = np.bincount(order_ids[start:stop], minlength=experts + 1)
        assert np.array_equal(got, want) and np.array_equal(dor.pass_counts(counts, start, stop), want)
        row0, n, e, njobs = dor.build_jobs(got, hot, ex.do_job_capacity(stop - start, experts))
        assert njobs <= capacity
        for j in range(njobs):
            assert np.all(order_ids[start + row0[j]:start + row0[j] + n[j]] == e[j])
            covered[start + row0[j]:start + row0[j] + n[j]] += 1
    assert np.array_equal(covered, (order_ids < hot).astype(np.int64))


def _mats(rng, experts, hidden=256, inter=128):
    mats = {}
    for e in range(experts):
        mats[e] = {name: (rng.standard_normal((i, o)) / np.sqrt(i), rng.choice([-1.0, 1.0], i) * rng.uniform(0.5, 1.5, i),
                          rng.uniform(0.5, 1.5, o))
                   for name, (i, o) in (("w1", (hidden, inter)), ("w3", (hidden, inter)), ("w2", (inter, hidden)))}
    return mats


@pytest.mark.parametrize("hot_rows,pass_routes", [(12, 256), (9, 384), (9, 128), (0, 512), (12, 1 << 20)])
def test_mirror_passes_are_bit_identical(hot_rows, pass_routes):
    rng = np.random.default_rng(hot_rows + pass_routes)
    experts, tokens, top_k = 12, 300, 6
    mats = _mats(rng, experts)
    x = rng.standard_normal((tokens, 256))
    p = np.linspace(2, 1, experts) / np.linspace(2, 1, experts).sum()
    ids = np.stack([rng.choice(experts, top_k, replace=False, p=p) for _ in range(tokens)])
    w = rng.dirichlet(np.ones(top_k), tokens)
    one = dor.do_moe_reference(x, ids, w, mats, hot_rows=hot_rows)
    split = dor.do_moe_reference(x, ids, w, mats, hot_rows=hot_rows, pass_routes=pass_routes)
    assert np.array_equal(one, split)      # contributions land in the same (expert-sorted) order


# ---- the production op on a computing fake extension ----

class Stream:
    def __init__(self, name):
        self.name, self.waits = name, []

    def wait_stream(self, other):
        self.waits.append(other.name)


class ComputeExt:
    """Fake xmoe: each kernel computes out[token] += weight * (x[token] * (expert + 1) + expert) in FP32, route by
    route in sorted order, from exactly the operands passed (counts, sorted slices, row_exp, job table), and records
    the scratch views and the active stream so the test can check placement."""

    def __init__(self, streams):
        self.streams, self.calls = streams, []
        self.force = self.width = self.budget = self.slot = self.bulk = 0

    def xmoe_set_flat_bulk(self, v): self.bulk = v
    def xmoe_set_force_group(self, v): self.force = v
    def xmoe_set_group_width(self, v): self.width = v
    def xmoe_set_sm_budget(self, v): self.budget = v
    def xmoe_set_lock_slot(self, v): self.slot = v

    @staticmethod
    def _apply(x, out, tok, wts, experts):
        for t, w, e in zip(tok.tolist(), wts.float().tolist(), experts):
            out[t] += np.float32(w) * (x[t].float() * (e + 1) + e)

    def xmoe_do_moe(self, x, out, tok, wts, row_exp, counts, hot_rows, job_i32, job_ptr, overflow, *rest):
        xg, xu, cg, cu = rest[9:13]
        assert job_i32.shape[1] >= ex.do_job_capacity(tok.shape[0], counts.shape[0] - 1)
        assert xg.shape[0] >= tok.shape[0] and cu.shape[0] >= tok.shape[0]
        row0, n, e, njobs = dor.build_jobs(counts.numpy(), int(hot_rows), job_i32.shape[1])
        rows = []
        for j in range(njobs):
            assert torch.all(row_exp[row0[j]:row0[j] + n[j]] == int(e[j]))
            rows += [(r, int(e[j])) for r in range(row0[j], row0[j] + n[j])]
        assert sorted(r for r, _ in rows) == torch.nonzero(row_exp >= 0).flatten().tolist()
        idx = [r for r, _ in rows]
        self._apply(x, out, tok[idx], wts[idx], [ee for _, ee in rows])
        self.calls.append(dict(kind="do", stream=self.streams["now"], routes=tok.shape[0], tok_ptr=tok.data_ptr(),
                               spans=[(v.data_ptr(), v.data_ptr() + v.numel() * 2) for v in (xg, xu, cg, cu)]))

    def exl3_moe(self, hidden, out, counts, token_sorted, weight_sorted, sg, su, ig, iu, *rest):
        start, idx, experts = 0, [], []
        for e, c in enumerate(counts.tolist()):
            if c > 0 and c <= sg.shape[1]:
                idx += range(start, start + c)
                experts += [e] * c
            start += abs(c)
        self._apply(hidden, out, token_sorted[idx], weight_sorted[idx], experts)
        self.calls.append(dict(kind="group", stream=self.streams["now"], force=self.force, slot=self.slot, bulk=self.bulk,
                               width=self.width, groups=sg.shape[0], per=sg.shape[1],
                               spans=[(v.data_ptr(), v.data_ptr() + v.numel() * 2) for v in (sg, su, ig, iu)]))


def _bank(experts, hot_rows):
    return SimpleNamespace(num_experts=experts, local_intermediate=1152, parked=hot_rows < experts, hot_rows=hot_rows,
                           codebook="mul1", bits=3, pointers={(m, p): torch.zeros(experts, dtype=torch.int64)
                                                              for m in ("w1", "w2", "w3") for p in ("trellis", "suh", "svh")})


def _run(monkeypatch, *, groups, pass_routes, cold_side, hot_rows, rows=256, experts=16, seed=7):
    streams = {"now": "main"}
    main, side = Stream("main"), Stream("side")

    @contextlib.contextmanager
    def on(stream):
        saved, streams["now"] = streams["now"], stream.name
        try:
            yield
        finally:
            streams["now"] = saved

    fake = ComputeExt(streams)
    monkeypatch.setattr(xmoe, "extension", lambda: fake)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: main)
    monkeypatch.setattr(torch.cuda, "stream", on)
    monkeypatch.setattr(ex, "PREFILL_MODE", "do")
    monkeypatch.setattr(ex, "PREFILL_GROUPS", groups)
    monkeypatch.setattr(ex, "PREFILL_PASS_ROUTES", pass_routes)
    monkeypatch.setattr(ex, "PREFILL_COLD_SIDE", cold_side)
    ex._SCRATCH.clear(); ex._DO_JOBS.clear(); ex._SIDE.clear(); ex._PREFILL_SIDE.clear()
    bank = _bank(experts, hot_rows)
    if cold_side and bank.parked:
        ex._PREFILL_SIDE[None] = side          # CPU tensors: device.index is None
    ex._BANKS[:] = [(bank, 23)]
    g = torch.Generator().manual_seed(seed)
    p = torch.linspace(3, 1, experts)
    ids = torch.stack([torch.multinomial(p, 6, generator=g) for _ in range(rows)])
    # exactly representable inputs (small integers, weights in eighths): every sum is exact in FP32, so the output does
    # not depend on the order of the adds (on the GPU the atomics are unordered) and any lost, doubled or mis-assigned
    # route changes it
    w = torch.randint(1, 9, (rows, 6), generator=g).float() / 8
    x = torch.randint(-8, 9, (rows, 5120), generator=g).to(torch.bfloat16)
    out = torch.empty(rows, 5120, dtype=torch.bfloat16)
    ex.ds41_exl3_moe(0, x, ids, w, out, 10.0)
    return out, fake.calls, main, side


@pytest.mark.parametrize("hot_rows", [16, 10, 0])
@pytest.mark.parametrize("groups,pass_routes,cold_side", [(2, 512, False), (2, 384, True), (4, 0, True), (1, 128, True),
                                                          (2, 1 << 16, False)])
def test_op_output_is_bit_identical_across_switches(monkeypatch, hot_rows, groups, pass_routes, cold_side):
    rows = 256
    want, base_calls, _, _ = _run(monkeypatch, groups=4, pass_routes=0, cold_side=False, hot_rows=hot_rows, rows=rows)
    got, calls, main, side = _run(monkeypatch, groups=groups, pass_routes=pass_routes, cold_side=cold_side,
                                  hot_rows=hot_rows, rows=rows)
    assert torch.equal(want.view(torch.int16), got.view(torch.int16))
    do = [c for c in calls if c["kind"] == "do"]
    group = [c for c in calls if c["kind"] == "group"]
    passes = ex.do_passes(rows * 6, pass_routes)
    assert [c["routes"] for c in do] == [b - a for a, b in passes]
    parked = hot_rows < 16
    assert len(group) == (1 if parked else 0)
    if not parked:
        return
    cold = group[0]
    assert cold["force"] == 1 and cold["groups"] == groups and cold["per"] == rows and cold["width"] == 32
    if cold_side:
        assert cold["stream"] == "side" and cold["slot"] == 1 and calls[0] is cold    # launched first, beside the passes
        assert side.waits == ["main"] and main.waits == ["side"]
        assert all(c["stream"] == "main" for c in do)
        lo = min(s for s, _ in cold["spans"]); hi = max(e for _, e in cold["spans"])
        for c in do:                                                     # disjoint scratch regions
            assert all(e <= lo or s >= hi for s, e in c["spans"])
    else:
        assert cold["stream"] == "main" and cold["slot"] == 0 and calls[-1] is cold   # after the passes, as before
        assert side.waits == [] and main.waits == []


def test_default_switches_keep_the_old_launches(monkeypatch):
    out, calls, _, _ = _run(monkeypatch, groups=4, pass_routes=0, cold_side=False, hot_rows=10)
    assert [c["kind"] for c in calls] == ["do", "group"]
    assert calls[0]["routes"] == 256 * 6 and calls[1]["groups"] == 4
    assert calls[1]["bulk"] == 0        # prefill launches never take the xmoe_bulk loop
    # the decode-once views start at the scratch base, as before (one overlapping region)
    assert calls[0]["spans"][0][0] == calls[1]["spans"][0][0]


def test_cold_side_needs_decode_once(monkeypatch):
    monkeypatch.setattr(ex, "PREFILL_COLD_SIDE", True)
    monkeypatch.setattr(ex, "PREFILL_MODE", "group")
    with pytest.raises(ValueError, match="PREFILL=do"):
        ex._ensure_prefill_side(torch.device("cpu"), _bank(16, 10), 4096)
    monkeypatch.setattr(ex, "PREFILL_MODE", "do")
    ex._PREFILL_SIDE.clear()
    ex._ensure_prefill_side(torch.device("cpu"), _bank(16, 16), 4096)   # resident: no stream
    ex._ensure_prefill_side(torch.device("cpu"), _bank(16, 10), 64)     # decode rows: no stream
    assert not ex._PREFILL_SIDE
