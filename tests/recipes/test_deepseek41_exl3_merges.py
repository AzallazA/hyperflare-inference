# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the DS41 EXL3 tail merges (exl3/merges.py, exp/ds41-exl3-final-merges): each merged kernel's algorithm
equals the op sequence it replaces bit for bit (NumPy mirrors of exl3_merges.cu; the Triton inverse-RoPE quantizer in
Triton's CPU interpreter), the switches parse strictly, and the decoder/op wiring feeds the MoE kernels identical
inputs with every switch on as with every switch off."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ignition.recipes.deepseek41.exl3 import experts as ex  # noqa: E402
from ignition.recipes.deepseek41.exl3 import merges, xmoe  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


# ---- c-i: mapped prep ------------------------------------------------------------------------------------------------

def mapped_prep_mirror(ids, hot_map, weights, hidden, num_experts, hot_rows):
    """exl3_merges.cu xmoe_decode_prep_mapped step by step (block 0: e = map[id], histogram, exclusive scan, rank by
    earlier routes to the same expert; other blocks: fp16 activations, zeroed accumulator)."""
    raw = ids.reshape(-1).astype(np.int64)
    if hot_map is not None:
        assert ((raw >= 0) & (raw < hot_map.size)).all()
        flat = hot_map[raw].astype(np.int64)
    else:
        flat = raw
    routes, top_k, bins = flat.size, ids.shape[1], num_experts + 1
    assert routes <= 512 and bins <= 512 and ((flat >= 0) & (flat < bins)).all()
    hist = np.bincount(flat, minlength=bins)
    off = np.concatenate([[0], np.cumsum(hist)[:-1]])
    token_sorted = np.empty(routes, np.int64)
    weight_sorted = np.empty(routes, np.float16)
    w = weights.reshape(-1)
    for t in range(routes):
        e = flat[t]
        pos = off[e] + int((flat[:t] == e).sum())
        token_sorted[pos] = t // top_k
        weight_sorted[pos] = np.float16(np.float32(w[t]))
    slot = np.arange(bins)
    hot = np.where(slot < min(hot_rows, bins), hist, -hist)
    return token_sorted, weight_sorted, hot, -hot, hidden.astype(np.float32).astype(np.float16), np.zeros(hidden.shape, np.float32)


def _same(a, b, name):
    a = a.numpy() if isinstance(a, torch.Tensor) else a
    assert a.dtype == b.dtype and a.shape == b.shape, name
    if a.dtype in (np.float16, np.float32):
        bits = np.uint16 if a.dtype == np.float16 else np.uint32
        assert np.array_equal(a.view(bits), b.view(bits)), name
    else:
        assert np.array_equal(a, b), name


def _routes(rows, experts, seed, top_k=6):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(experts, generator=g)[:top_k] for _ in range(rows)]).to(torch.int32)
    return ids, torch.rand(rows, top_k, generator=g), torch.randn(rows, 5120, generator=g).to(torch.bfloat16)


@pytest.mark.parametrize("rows,experts,hot_rows,mapped", [(1, 384, 308, True), (6, 384, 308, True), (24, 384, 320, True),
                                                          (48, 384, 308, True), (64, 384, 300, True), (6, 384, 385, False),
                                                          (24, 16, 10, True), (14, 128, 129, False)])
def test_mapped_prep_equals_index_select_then_prep(rows, experts, hot_rows, mapped):
    ids, w, x = _routes(rows, experts, rows * 7 + experts)
    hot_map = torch.randperm(experts, generator=torch.Generator().manual_seed(rows)).to(torch.int32) if mapped else None
    want = merges.torch_mapped_prep(ids, hot_map, w, x, experts, hot_rows)
    got = mapped_prep_mirror(ids.numpy(), None if hot_map is None else hot_map.numpy(), w.numpy(), x.float().numpy(),
                             experts, hot_rows)
    for name, a, b in zip(("token_sorted", "weight_sorted", "hot", "cold", "x16", "acc"), want, got):
        _same(a, b, name)


def test_mapped_prep_duplicate_heavy_rows():
    # every row picks the same experts: rank by route index = stable argsort, also after the map
    ids = torch.tensor([[3, 1, 2, 0, 5, 4]] * 10, dtype=torch.int32)
    hot_map = torch.tensor([7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.int32)
    w, x = torch.rand(10, 6), torch.randn(10, 5120).to(torch.bfloat16)
    want = merges.torch_mapped_prep(ids, hot_map, w, x, 8, 4)
    got = mapped_prep_mirror(ids.numpy(), hot_map.numpy(), w.numpy(), x.float().numpy(), 8, 4)
    _same(want[0], got[0], "token_sorted")
    _same(want[1], got[1], "weight_sorted")


# ---- c-iii: epilogue -------------------------------------------------------------------------------------------------

def _bf16_rn(values):
    """fp32 -> bf16 round to nearest even (NaN-free inputs), as __float2bfloat16_rn, returned as fp32."""
    bits = np.asarray(values, np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def epilogue_mirror(acc, add):
    """exl3_merges.cu xmoe_epilogue_add: r = bf16_rn(acc); out = bf16_rn(fp32(r) + fp32(add)) (no FTZ)."""
    return _bf16_rn(_bf16_rn(acc) + add.astype(np.float32))


def test_epilogue_equals_copy_then_add():
    g = torch.Generator().manual_seed(5)
    acc = torch.randn(48, 5120, generator=g) * 3
    acc[0, :16] = torch.tensor([1e-40, -1e-40, 3e-39, 0.0, -0.0, 65504.0, 1e30, -1e30, 2 ** -126, 2 ** -133,
                                1.00390625, 1.01171875, -1.00390625, 3.0e38, 1e-45, -1e-45])
    add = (torch.randn(48, 5120, generator=g) * 2).to(torch.bfloat16)
    add[0, :16] = torch.tensor([-1e-40, 1e-40, -3e-39, 0.0, 0.0, 0.5, -1e30, 1e30, 2 ** -127, 0.0, 0.0, 0.0, 0.0,
                                3.0e38, 0.0, 0.0]).to(torch.bfloat16)
    want = merges.torch_epilogue(acc, add).float().numpy()
    with np.errstate(over="ignore"):
        got = epilogue_mirror(acc.numpy(), add.float().numpy())
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))


# ---- c-ii: inverse RoPE + WO-A quantization (Triton interpreter on CPU) ------------------------------------------------

def test_inv_rope_quantizer_equals_rotate_then_quantize():
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH")))))
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("exl3_merges_interpret.py"))],
                            capture_output=True, text=True, env=env, timeout=600, cwd=ROOT)
    assert result.returncode == 0 and "ROPEQ INTERPRET OK 4" in result.stdout, result.stdout + result.stderr[-3000:]


def test_state_view_runs_b12x_sequence_with_merged_quantizer():
    calls = []

    class State:
        def __init__(self):
            self.quantizers = SimpleNamespace(quantize_a=lambda s, o: calls.append(("stock_a", s)),
                                              quantize_b=lambda s, o: calls.append(("b", s)), a_chunks=16)
            self.ordinary_a = SimpleNamespace(run=lambda *a, **k: calls.append(("gemm_a",)))

        def run(self, binding):
            self.quantizers.quantize_a(binding, None)
            self.ordinary_a.run()
            self.quantizers.quantize_b(binding, None)

    state = State()
    q = merges._RopeQuantizers(state.quantizers, None, None, 512, 448)
    assert q.a_chunks == 16 and q.quantize_b is state.quantizers.quantize_b
    q.quantize_a = lambda s, o: calls.append(("merged_a", s))       # stands in for the Triton launch
    type(state).run(merges._StateView(state, q), "B")
    assert calls == [("merged_a", "B"), ("gemm_a",), ("b", "B")]


# ---- switches --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,reader", [("FLARE_DS41_EXL3_MERGE_TOPK", merges.merge_topk),
                                         ("FLARE_DS41_EXL3_MERGE_ROPEQ", merges.merge_ropeq),
                                         ("FLARE_DS41_EXL3_MERGE_EPILOGUE", merges.merge_epilogue)])
def test_switches(monkeypatch, name, reader):
    monkeypatch.delenv(name, raising=False)
    assert reader() is False
    monkeypatch.setenv(name, "0")
    assert reader() is False
    monkeypatch.setenv(name, "1")
    assert reader() is True
    monkeypatch.setenv(name, "on")
    with pytest.raises(ValueError, match=name):
        reader()


def test_defaults_off():
    for name in ("FLARE_DS41_EXL3_MERGE_TOPK", "FLARE_DS41_EXL3_MERGE_ROPEQ", "FLARE_DS41_EXL3_MERGE_EPILOGUE"):
        if os.environ.get(name, "") in ("", "0"):
            assert getattr(merges, name.removeprefix("FLARE_DS41_EXL3_")) is False


def test_merge_prep_rows(monkeypatch):
    monkeypatch.setattr(merges, "MERGE_TOPK", True)
    monkeypatch.setattr(ex, "FUSED_PREP", True)
    monkeypatch.setattr(ex, "FUSED_PREP_MAX_ROWS", 24)
    assert [ex.merge_prep_rows(r) for r in (1, 6, 24, 48, 64, 256)] == [True, True, True, False, False, False]
    monkeypatch.setattr(ex, "FUSED_PREP", False)
    assert not ex.merge_prep_rows(6)
    monkeypatch.setattr(ex, "FUSED_PREP", True)
    monkeypatch.setattr(merges, "MERGE_TOPK", False)
    assert not ex.merge_prep_rows(6)


# ---- op wiring: merged ops feed the MoE launches exactly what the old path does ------------------------------------

class MergeExt:
    """Fake xmoe + merges extension: preps by the mirrors, records exl3_moe launches, epilogue by its mirror."""

    def __init__(self):
        self.launches, self.preps, self.mapped, self.epilogues = [], 0, 0, 0
        self.bulk = 0

    def xmoe_set_flat_bulk(self, v): self.bulk = v
    def xmoe_set_force_group(self, v): self.force = v
    def xmoe_set_group_width(self, v): self.width = v
    def xmoe_set_sm_budget(self, v): self.budget = v
    def xmoe_set_lock_slot(self, v): self.slot = v

    def _write(self, outs, dst):
        for d, s in zip(dst, outs):
            d.copy_(torch.from_numpy(np.ascontiguousarray(s)))

    def xmoe_decode_prep(self, ids, weights, hidden, hot_rows, *dst):
        self.preps += 1
        self._write(mapped_prep_mirror(ids.numpy(), None, weights.float().numpy(), hidden.float().numpy(),
                                       dst[2].numel() - 1, hot_rows), dst)

    def xmoe_decode_prep_mapped(self, ids, table, weights, hidden, hot_rows, *dst):
        self.mapped += 1
        self._write(mapped_prep_mirror(ids.numpy(), table.numpy() if table.numel() else None, weights.float().numpy(),
                                       hidden.float().numpy(), dst[2].numel() - 1, hot_rows), dst)

    def xmoe_epilogue_add(self, acc, add, out):
        self.epilogues += 1
        out.copy_(torch.from_numpy(epilogue_mirror(acc.numpy(), add.float().numpy())).to(torch.bfloat16))

    def exl3_moe(self, hidden, out, counts, token_sorted, weight_sorted, sg, su, ig, iu, act, kg, ku, kd, *rest):
        self.launches.append(dict(hidden=hidden.clone(), counts=counts.clone(), tokens=token_sorted.clone(),
                                  weights=weight_sorted.clone(), force=self.force, width=self.width, groups=sg.shape[0],
                                  bulk=self.bulk))
        out.add_(1.0 + len(self.launches))      # a deterministic stand-in for the routed partial sum


def _bank(parked):
    return SimpleNamespace(num_experts=16, local_intermediate=1152, parked=parked, hot_rows=10 if parked else 16,
                           codebook="mul1", bits=3,
                           pointers={(m, p): torch.zeros(16, dtype=torch.int64) for m in ("w1", "w2", "w3")
                                     for p in ("trellis", "suh", "svh")})


@pytest.mark.parametrize("rows,parked", [(1, True), (6, True), (24, True), (6, False)])
def test_merged_ops_equal_old_sequence(monkeypatch, rows, parked):
    """Old: index_select(hot_map) -> ds41_exl3_moe_after (fused prep inside) -> routed + shared. Merged:
    ds41_exl3_moe_prep (raw ids + hot_map) -> ds41_exl3_moe_prepped(add_after) -> routed is the sum."""
    monkeypatch.setattr(ex, "FUSED_PREP", True)
    monkeypatch.setattr(ex, "FUSED_PREP_MAX_ROWS", 24)
    monkeypatch.setattr(ex, "COLD_WIDTH", 0)
    monkeypatch.setattr(ex, "PREFILL_MODE", "group")
    monkeypatch.setattr(merges, "MERGE_TOPK", True)
    g = torch.Generator().manual_seed(rows)
    raw = torch.stack([torch.randperm(16, generator=g)[:6] for _ in range(rows)]).to(torch.int32)
    hot_map = torch.randperm(16, generator=g).to(torch.int32) if parked else None
    w = torch.rand(rows, 6, generator=g)
    x = torch.randn(rows, 5120, generator=g).to(torch.bfloat16)
    shared = (torch.randn(rows, 5120, generator=g) * 4).to(torch.bfloat16)
    results = {}
    for merged in (False, True):
        fake = MergeExt()
        monkeypatch.setattr(xmoe, "extension", lambda: fake)
        monkeypatch.setattr(merges, "extension", lambda: fake)
        ex._SCRATCH.clear(); ex._SIDE.clear()
        ex._BANKS[:] = []
        layer = ex.PreparedExl3Experts.__new__(ex.PreparedExl3Experts)
        layer.experts, layer.limit, layer.handle, layer.top_k = _bank(parked), 10.0, 0, 6
        layer.output_layout = ((rows, 5120), torch.bfloat16)
        ex._BANKS.append((layer.experts, 23))
        if merged:
            prepped = layer.prep(x, raw, w, hot_map)
            out = layer(x, raw, w, after=shared, prepped=prepped, add_after=True)
            partial = out
        else:
            ids = raw if hot_map is None else torch.index_select(hot_map, 0, raw.reshape(-1)).view(raw.shape)
            routed = layer(x, ids, w, after=shared)
            partial = routed + shared
        results[merged] = (fake, partial)
    (old, p_old), (new, p_new) = results[False], results[True]
    assert old.preps == 1 and old.mapped == 0 and old.epilogues == 0
    assert new.preps == 0 and new.mapped == 1 and new.epilogues == 1
    assert len(old.launches) == len(new.launches) == (2 if parked else 1)
    for a, b in zip(old.launches, new.launches):
        for key in ("force", "width", "groups"):
            assert a[key] == b[key], key
        for key in ("counts", "tokens", "weights", "hidden"):
            assert torch.equal(a[key], b[key]), key
    assert torch.equal(p_old.view(torch.int16), p_new.view(torch.int16))


def test_after_sum_op_on_torch_prep_rows(monkeypatch):
    """Rows above the fused-prep limit: ds41_exl3_moe_after_sum = ds41_exl3_moe_after + add, torch prep."""
    monkeypatch.setattr(ex, "FUSED_PREP", True)
    monkeypatch.setattr(ex, "FUSED_PREP_MAX_ROWS", 24)
    monkeypatch.setattr(ex, "COLD_WIDTH", 0)
    monkeypatch.setattr(ex, "PREFILL_MODE", "group")
    rows = 48
    ids, w, x = _routes(rows, 16, 3)
    shared = torch.randn(rows, 5120).to(torch.bfloat16)
    outs = {}
    for summed in (False, True):
        fake = MergeExt()
        monkeypatch.setattr(xmoe, "extension", lambda: fake)
        monkeypatch.setattr(merges, "extension", lambda: fake)
        ex._SCRATCH.clear(); ex._SIDE.clear()
        ex._BANKS[:] = [(_bank(True), 23)]
        out = torch.empty(rows, 5120, dtype=torch.bfloat16)
        if summed:
            ex.ds41_exl3_moe_after_sum(0, x, ids.long(), w, shared, out, 10.0)
            outs[summed] = out
        else:
            ex.ds41_exl3_moe_after(0, x, ids.long(), w, shared, out, 10.0)
            outs[summed] = out + shared
        assert fake.preps == 0 and fake.epilogues == int(summed)
    assert torch.equal(outs[False].view(torch.int16), outs[True].view(torch.int16))


def test_add_after_needs_shared():
    layer = ex.PreparedExl3Experts.__new__(ex.PreparedExl3Experts)
    layer.output_layout = ((1, 5120), torch.bfloat16)
    with pytest.raises(ValueError, match="shared"):
        layer(torch.zeros(1, 5120), None, None, add_after=True)


def test_export_script_documents_switches():
    text = (ROOT / "tp3-tools/ds41-exl3-export-tp2.sh").read_text()
    for name in ("FLARE_DS41_EXL3_MERGE_TOPK", "FLARE_DS41_EXL3_MERGE_ROPEQ", "FLARE_DS41_EXL3_MERGE_EPILOGUE"):
        assert name in text, name


# ---- decoder wiring: PreparedLayer._run with the merges on vs off ---------------------------------------------------

def _stub_layer(merge_prep, merge_epilogue, rows, log):
    from ignition.recipes.deepseek41 import decoder
    from ignition.recipes.deepseek41.probe import Probe

    layer = decoder.PreparedLayer.__new__(decoder.PreparedLayer)
    hidden = torch.full((rows, 8), 2.0)
    raw = torch.arange(rows * 6, dtype=torch.int32).view(rows, 6) % 16
    layer.probe, layer.probe_index, layer.probe_mode = Probe(), 3, "decode"
    layer.engram = None
    layer.attention_hc = lambda *incoming, **kwargs: (incoming[0], None, None, hidden)
    layer.attention = SimpleNamespace(_run=lambda normalized, metadata, selection, **kwargs:
                                      (normalized, None, selection, {}))
    layer.attention_reduce = lambda source: source
    layer.ffn_hc = lambda output, residual, post, comb, **kwargs: (residual, post, comb, output)
    layer.router_projection = lambda source: source
    layer.router = lambda logits, **kwargs: SimpleNamespace(ids=raw, weights=torch.ones(rows, 6), selected_logits=None)
    layer.hot_map = torch.arange(15, -1, -1, dtype=torch.int32)
    layer.has_image_mask = False
    shared_out = torch.full((rows, 8), 0.5)

    class Shared:
        def start(self, normalized):
            log.append(("start",)); return "pending"

        def finish(self, pending, routed):
            log.append(("finish", routed)); return shared_out

    class Experts:
        def prep(self, normalized, ids, weights, hot_map):
            log.append(("prep", ids, hot_map))
            return SimpleNamespace(token_sorted=torch.zeros(rows * 6, dtype=torch.int64))

        def __call__(self, normalized, ids, weights, **kwargs):
            log.append(("experts", ids, kwargs))
            return torch.full((rows, 8), 3.0) + (shared_out if kwargs.get("add_after") else 0)

    layer.concurrent_shared, layer.shared_joins_before_experts = Shared(), True
    layer.exl3_merge_prep, layer.exl3_merge_epilogue = merge_prep, merge_epilogue
    layer.experts = Experts()
    layer.mlp_reduce = lambda source: log.append(("reduce", source)) or source
    layer.post = None
    return layer, raw


@pytest.mark.parametrize("merge_prep,merge_epilogue", [(False, False), (True, False), (False, True), (True, True)])
def test_decoder_run_wiring(merge_prep, merge_epilogue):
    rows, log = 6, []
    layer, raw = _stub_layer(merge_prep, merge_epilogue, rows, log)
    metadata = SimpleNamespace(token_ids=torch.zeros(rows, dtype=torch.int64))
    layer._run((torch.zeros((rows, 4, 8)),), torch.zeros((rows, 4)), metadata, None, (None, None, None))
    names = [entry[0] for entry in log]
    if merge_prep:
        assert names == ["start", "prep", "finish", "experts", "reduce"]
        _, ids, hot_map = log[1]
        assert ids is raw and hot_map is layer.hot_map                     # raw router ids, map folded into the prep
        assert log[2][1].dtype == torch.int64 and log[2][1].shape == (rows * 6,)   # the join waits for the prep
        assert log[3][2]["prepped"] is not None
    else:
        assert names == ["start", "finish", "experts", "reduce"]
        mapped = torch.index_select(layer.hot_map, 0, raw.reshape(-1)).view(raw.shape)
        assert torch.equal(log[1][1], mapped) and torch.equal(log[2][1], mapped)
    kwargs = log[-2][2]
    assert kwargs.get("add_after", False) is merge_epilogue
    assert torch.equal(log[-1][1], torch.full((rows, 8), 3.5))     # partial = routed + shared either way
