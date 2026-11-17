# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of FLARE_DS41_EXL3_FUSED_PREP (exp/ds41-exl3-next): the fused kernel's algorithm (NumPy mirror of
quant/exl3_prep.cu) equals the op's torch prep exactly, and the op feeds the same launches from either prep."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ignition.recipes.deepseek41.exl3 import experts as ex  # noqa: E402
from ignition.recipes.deepseek41.exl3 import xmoe  # noqa: E402


def kernel_mirror(ids, weights, hidden, num_experts, hot_rows):
    """quant/exl3_prep.cu step by step: shared histogram, exclusive scan, rank = earlier routes to the same expert."""
    flat = ids.reshape(-1).astype(np.int64)
    routes, top_k, bins = flat.size, ids.shape[1], num_experts + 1
    assert routes <= 512 and bins <= 512
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


class PrepExt:
    """FakeExt + xmoe_decode_prep implemented by the mirror (writes into the given outputs like the kernel)."""

    def __init__(self):
        self.launches, self.preps, self.bulk = [], 0, 0

    def xmoe_set_flat_bulk(self, v): self.bulk = v
    def xmoe_set_force_group(self, v): self.force = v
    def xmoe_set_group_width(self, v): self.width = v
    def xmoe_set_sm_budget(self, v): self.budget = v
    def xmoe_set_lock_slot(self, v): self.slot = v

    def xmoe_decode_prep(self, ids, weights, hidden, hot_rows, token_sorted, weight_sorted, hot, cold, x16, acc):
        self.preps += 1
        outs = kernel_mirror(ids.numpy(), weights.float().numpy(), hidden.float().numpy(), hot.numel() - 1, hot_rows)
        for dst, src in zip((token_sorted, weight_sorted, hot, cold, x16, acc), outs):
            dst.copy_(torch.from_numpy(np.ascontiguousarray(src)))

    def exl3_moe(self, hidden, out, counts, token_sorted, weight_sorted, sg, su, ig, iu, act, kg, ku, kd, *rest):
        self.launches.append(dict(hidden=hidden.clone(), counts=counts.clone(), tokens=token_sorted.clone(),
                                  weights=weight_sorted.clone(), force=self.force, width=self.width, budget=self.budget,
                                  slot=self.slot, groups=sg.shape[0], acc=out.clone(), bulk=self.bulk))


def _routes(rows, experts, seed, top_k=6):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(experts, generator=g)[:top_k] for _ in range(rows)])
    return ids, torch.rand(rows, top_k, generator=g), torch.randn(rows, 5120, generator=g).to(torch.bfloat16)


@pytest.mark.parametrize("rows,experts,hot_rows", [(1, 384, 320), (6, 384, 320), (14, 384, 308), (64, 384, 320),
                                                   (6, 16, 10), (24, 16, 17), (6, 128, 129)])
def test_kernel_algorithm_equals_torch_prep(rows, experts, hot_rows):
    ids, w, x = _routes(rows, experts, rows * 1000 + experts)
    want = ex.torch_prep(ids, w, x, experts, hot_rows)
    got = kernel_mirror(ids.numpy(), w.numpy(), x.float().numpy(), experts, hot_rows)
    for name, a, b in zip(("token_sorted", "weight_sorted", "hot", "cold", "x16", "acc"), want, got):
        assert a.numpy().dtype == b.dtype, name
        assert np.array_equal(a.numpy(), b), name      # bit-exact (fp16 compared as values; no NaN inputs)
        if a.dtype in (torch.float16, torch.float32):
            assert np.array_equal(a.numpy().view(np.uint16 if a.dtype == torch.float16 else np.uint32),
                                  b.view(np.uint16 if b.dtype == np.float16 else np.uint32)), name


def test_duplicate_heavy_routing_keeps_stable_order():
    # many routes to few experts (several rows choosing the same experts): rank by route index = stable argsort
    ids = torch.tensor([[3, 1, 2, 0, 5, 4]] * 10)
    w = torch.rand(10, 6)
    x = torch.randn(10, 5120).to(torch.bfloat16)
    want = ex.torch_prep(ids, w, x, 8, 4)
    got = kernel_mirror(ids.numpy(), w.numpy(), x.float().numpy(), 8, 4)
    assert np.array_equal(want[0].numpy(), got[0]) and np.array_equal(want[1].numpy(), got[1])


def test_switch():
    import os
    old = os.environ.pop("FLARE_DS41_EXL3_FUSED_PREP", None)
    try:
        assert not ex.fused_prep()
        os.environ["FLARE_DS41_EXL3_FUSED_PREP"] = "1"
        assert ex.fused_prep()
        os.environ["FLARE_DS41_EXL3_FUSED_PREP"] = "yes"
        with pytest.raises(ValueError, match="FUSED_PREP"):
            ex.fused_prep()
    finally:
        os.environ.pop("FLARE_DS41_EXL3_FUSED_PREP", None)
        if old is not None:
            os.environ["FLARE_DS41_EXL3_FUSED_PREP"] = old


@pytest.mark.parametrize("rows,parked,side", [(6, False, False), (6, True, False), (24, True, False), (64, True, False),
                                              (256, True, False)])
def test_op_feeds_identical_launches(monkeypatch, rows, parked, side):
    runs = {}
    for fused in (False, True):
        fake = PrepExt()
        monkeypatch.setattr(xmoe, "extension", lambda: fake)
        monkeypatch.setattr(ex, "FUSED_PREP", fused)
        monkeypatch.setattr(ex, "COLD_WIDTH", 16)
        monkeypatch.setattr(ex, "PREFILL_MODE", "group")
        ex._SCRATCH.clear(); ex._SIDE.clear()
        bank = SimpleNamespace(num_experts=16, local_intermediate=1152, parked=parked, hot_rows=10 if parked else 16,
                               codebook="mul1", bits=3, pointers={(m, p): torch.zeros(16, dtype=torch.int64)
                                                                  for m in ("w1", "w2", "w3") for p in ("trellis", "suh", "svh")})
        ex._BANKS[:] = [(bank, 23)]
        ids, w, x = _routes(rows, 16, rows)
        out = torch.empty(rows, 5120, dtype=torch.bfloat16)
        ex.ds41_exl3_moe(0, x, ids, w, out, 10.0)
        runs[fused] = fake
    plain, fused = runs[False], runs[True]
    assert fused.preps == (1 if rows <= 64 else 0) and plain.preps == 0
    assert len(plain.launches) == len(fused.launches) == (2 if parked else 1)
    for a, b in zip(plain.launches, fused.launches):
        for key in ("force", "width", "budget", "slot", "groups"):
            assert a[key] == b[key], key
        if rows > 64:
            continue
        # the unparked torch path passes raw counts (padding bin positive); the fused path passes hot == counts there too
        assert torch.equal(a["counts"][:-1], b["counts"][:-1])
        for key in ("tokens", "weights", "hidden", "acc"):
            assert torch.equal(a[key], b[key]), key


def test_max_rows_switch(monkeypatch):
    monkeypatch.delenv("FLARE_DS41_EXL3_FUSED_PREP_MAX_ROWS", raising=False)
    assert ex.fused_prep_max_rows() == 64
    monkeypatch.setenv("FLARE_DS41_EXL3_FUSED_PREP_MAX_ROWS", "12")
    assert ex.fused_prep_max_rows() == 12
    for bad in ("0", "65", "x"):
        monkeypatch.setenv("FLARE_DS41_EXL3_FUSED_PREP_MAX_ROWS", bad)
        with pytest.raises(ValueError):
            ex.fused_prep_max_rows()


def test_mixed_graphs_by_row_limit(monkeypatch):
    """With the row limit, graphs up to N rows take the fused prep and larger decode graphs the torch prep; both feed
    the same launches."""
    monkeypatch.setattr(ex, "FUSED_PREP", True)
    monkeypatch.setattr(ex, "FUSED_PREP_MAX_ROWS", 12)
    monkeypatch.setattr(ex, "COLD_WIDTH", 0)
    monkeypatch.setattr(ex, "PREFILL_MODE", "group")
    for rows, want_fused in ((6, True), (12, True), (14, False), (24, False), (64, False)):
        fake = PrepExt()
        monkeypatch.setattr(xmoe, "extension", lambda: fake)
        ex._SCRATCH.clear(); ex._SIDE.clear()
        bank = SimpleNamespace(num_experts=16, local_intermediate=1152, parked=True, hot_rows=10, codebook="mul1", bits=3,
                               pointers={(m, p): torch.zeros(16, dtype=torch.int64)
                                         for m in ("w1", "w2", "w3") for p in ("trellis", "suh", "svh")})
        ex._BANKS[:] = [(bank, 23)]
        ids, w, x = _routes(rows, 16, rows + 7)
        ex.ds41_exl3_moe(0, x, ids, w, torch.empty(rows, 5120, dtype=torch.bfloat16), 10.0)
        assert fake.preps == int(want_fused), rows
        want = ex.torch_prep(ids, w, x, 16, 10)
        hot, cold = fake.launches
        assert torch.equal(hot["tokens"], want[0]) and torch.equal(hot["weights"], want[1])
        assert torch.equal(hot["counts"], want[2]) and torch.equal(cold["counts"], want[3])
