# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of FLARE_DS41_EXL3_BULK_MIN_ROWS (exp/ds41-exl3-final-bulk): diffbot's xmoe_bulk hot loop is requested
only for the hot (flat) launch of decode/verify graphs with >= N rows, never for cold or prefill launches, the flag is
cleared after every launch (captured graphs keep what they were captured with), and the ported kernel sources keep
their MIT headers and are built into the extension."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from ignition.recipes.deepseek41.exl3 import experts as ex  # noqa: E402
from ignition.recipes.deepseek41.exl3 import xmoe  # noqa: E402



class PrepExt:
    """Records each exl3_moe launch with the host flags set at that moment; xmoe_decode_prep = the torch prep."""

    def __init__(self):
        self.launches, self.preps, self.bulk = [], 0, 0

    def xmoe_set_flat_bulk(self, v): self.bulk = v
    def xmoe_set_force_group(self, v): self.force = v
    def xmoe_set_group_width(self, v): self.width = v
    def xmoe_set_sm_budget(self, v): self.budget = v
    def xmoe_set_lock_slot(self, v): self.slot = v

    def xmoe_decode_prep(self, ids, weights, hidden, hot_rows, token_sorted, weight_sorted, hot, cold, x16, acc):
        self.preps += 1
        for dst, src in zip((token_sorted, weight_sorted, hot, cold, x16, acc),
                            ex.torch_prep(ids, weights, hidden, hot.shape[0] - 1, hot_rows)):
            dst.copy_(src)

    def exl3_moe(self, hidden, out, counts, *rest):
        self.launches.append(dict(force=self.force, width=self.width, budget=self.budget, slot=self.slot, bulk=self.bulk))


def _routes(rows, experts, seed, top_k=6):
    g = torch.Generator().manual_seed(seed)
    ids = torch.stack([torch.randperm(experts, generator=g)[:top_k] for _ in range(rows)])
    return ids, torch.rand(rows, top_k, generator=g), torch.randn(rows, 5120, generator=g).to(torch.bfloat16)


def _bank(parked):
    return SimpleNamespace(num_experts=16, local_intermediate=1152, parked=parked, hot_rows=10 if parked else 16,
                           codebook="mul1", bits=3, pointers={(m, p): torch.zeros(16, dtype=torch.int64)
                                                              for m in ("w1", "w2", "w3") for p in ("trellis", "suh", "svh")})


def test_switch(monkeypatch):
    monkeypatch.delenv("FLARE_DS41_EXL3_BULK_MIN_ROWS", raising=False)
    assert ex.bulk_min_rows() == 0
    monkeypatch.setenv("FLARE_DS41_EXL3_BULK_MIN_ROWS", "14")
    assert ex.bulk_min_rows() == 14
    for bad in ("-1", "65", "x"):
        monkeypatch.setenv("FLARE_DS41_EXL3_BULK_MIN_ROWS", bad)
        with pytest.raises(ValueError, match="BULK_MIN_ROWS"):
            ex.bulk_min_rows()


def test_use_bulk(monkeypatch):
    monkeypatch.setattr(ex, "BULK_MIN_ROWS", 0)
    assert not any(ex.use_bulk(r) for r in (1, 14, 64, 256))
    monkeypatch.setattr(ex, "BULK_MIN_ROWS", 14)
    assert [ex.use_bulk(r) for r in (1, 6, 13, 14, 24, 64, 65, 256)] == [False] * 3 + [True] * 3 + [False] * 2


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("parked,side", [(False, False), (True, False), (True, True)])
def test_hot_launch_only_from_min_rows(monkeypatch, fused, parked, side):
    monkeypatch.setattr(ex, "BULK_MIN_ROWS", 14)
    monkeypatch.setattr(ex, "FUSED_PREP", fused)
    monkeypatch.setattr(ex, "FUSED_PREP_MAX_ROWS", 64)
    monkeypatch.setattr(ex, "COLD_WIDTH", 16)
    monkeypatch.setattr(ex, "PREFILL_MODE", "group")
    if side:   # torch.cuda.Stream is not available on CPU: a stand-in stream object and a no-op stream context
        class _S:
            def wait_stream(self, other): pass
        monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _S())
        monkeypatch.setattr(torch.cuda, "stream", lambda s: __import__("contextlib").nullcontext())
    for rows in (6, 13, 14, 24, 64, 256):
        fake = PrepExt()
        monkeypatch.setattr(xmoe, "extension", lambda: fake)
        ex._SCRATCH.clear(); ex._SIDE.clear()
        if side and rows <= 64:
            ex._SIDE[None] = (_S(), 172, 16)
        ex._BANKS[:] = [(_bank(parked), 23)]
        ids, w, x = _routes(rows, 16, rows)
        ex.ds41_exl3_moe(0, x, ids, w, torch.empty(rows, 5120, dtype=torch.bfloat16), 10.0)
        want = 14 <= rows <= 64
        hot = [l for l in fake.launches if not l["force"]]
        cold = [l for l in fake.launches if l["force"]]
        assert len(hot) == 1 and len(cold) == int(parked), rows
        assert hot[0]["bulk"] == int(want), rows
        assert all(l["bulk"] == 0 for l in cold), rows
        assert fake.bulk == 0, rows          # cleared after the launch


def test_off_by_default_launches_unchanged(monkeypatch):
    monkeypatch.setattr(ex, "BULK_MIN_ROWS", 0)
    monkeypatch.setattr(ex, "FUSED_PREP", True)
    monkeypatch.setattr(ex, "PREFILL_MODE", "group")
    for rows in (6, 24, 64):
        fake = PrepExt()
        monkeypatch.setattr(xmoe, "extension", lambda: fake)
        ex._SCRATCH.clear(); ex._SIDE.clear()
        ex._BANKS[:] = [(_bank(True), 23)]
        ids, w, x = _routes(rows, 16, rows)
        ex.ds41_exl3_moe(0, x, ids, w, torch.empty(rows, 5120, dtype=torch.bfloat16), 10.0)
        assert [l["bulk"] for l in fake.launches] == [0, 0]


def test_sources_and_headers():
    quant = xmoe.KERNELS / "quant"
    assert quant / "exl3_moe_transfer.cu" in xmoe.sources()
    for name in ("exl3_moe_transfer.cu", "exl3_moe_transfer_kernel.cuh", "exl3_gemm_inner_transfer.cuh",
                 "exl3_flat128_helpers.cuh", "exl3_flat128_jobs.cuh"):
        head = (quant / name).read_text().splitlines()[:3]
        assert head[0] == "// SPDX-License-Identifier: MIT" and "LICENSE.diffbot" in "".join(head), name
    for lic in ("LICENSE.diffbot", "LICENSE.exllamav3"):
        assert (xmoe.KERNELS / lic).is_file()
    dispatch = (quant / "exl3_moe.cu").read_text()
    # host flag consumed at launch time, regular flat kernel kept when the bulk kernel does not fit
    assert "void xmoe_set_flat_bulk(int v)" in dispatch and "xmoe_bulk_ok[device] > 0" in dispatch
    assert "xmoe_set_flat_bulk" in (xmoe.KERNELS / "bindings.cpp").read_text()
