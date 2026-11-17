# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of EXL3 expert loading, parking layout and the routed-expert call's routing preparation (no GPU)."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ignition.recipes.deepseek41.exl3 import experts as ex  # noqa: E402
from ignition.recipes.deepseek41.exl3 import xmoe  # noqa: E402


class FakeLoader:
    def __init__(self, count, codebook="mul1"):
        self.device = torch.device("cpu")
        g = torch.Generator().manual_seed(5)
        self.tensors = {}
        for e in range(count):
            for m in ("w1", "w3", "w2"):
                rows, cols = (5120, 2304) if m != "w2" else (2304, 5120)
                p = f"layers.0.ffn.experts.{e}.{m}"
                self.tensors[p + ".trellis"] = torch.randint(-32768, 32767, (rows // 16, cols // 16, 48),
                                                             dtype=torch.int16, generator=g)
                self.tensors[p + ".suh"] = torch.randn(rows, generator=g).half()
                self.tensors[p + ".svh"] = torch.randn(cols, generator=g).half()
                self.tensors[p + "." + codebook] = torch.tensor(0, dtype=torch.int32)
        self._sources = self.tensors

    def _copy(self, destination, source):
        destination.copy_(source)


def _prefixes(order):
    return [f"layers.0.ffn.experts.{e}" for e in order]


@pytest.mark.parametrize("rank", [0, 1])
def test_banks_hold_rank_slices(rank):
    loader = FakeLoader(3)
    bank = ex.load_exl3_experts(loader, _prefixes([2, 0, 1]), rank=rank, world=2)
    assert bank.num_experts == 3 and bank.local_intermediate == 1152 and not bank.parked
    assert tuple(bank.trellis["w1"].shape) == (3, 320, 72, 48) and tuple(bank.trellis["w2"].shape) == (3, 72, 320, 48)
    for slot, e in enumerate([2, 0, 1]):
        for m in ("w1", "w3", "w2"):
            p = f"layers.0.ffn.experts.{e}.{m}"
            want = xmoe.shard_matrix(m, {k: loader.tensors[f"{p}.{k}"] for k in ("trellis", "suh", "svh")}, rank, 2)
            assert torch.equal(bank.trellis[m][slot], want["trellis"])
            assert torch.equal(bank.suh[m][slot], want["suh"]) and torch.equal(bank.svh[m][slot], want["svh"])
            step = bank.trellis[m].stride(0) * 2
            assert int(bank.pointers[m, "trellis"][slot]) == bank.trellis[m].data_ptr() + slot * step


def test_parked_bank_uses_mixed_rows_and_checks_codebook():
    loader = FakeLoader(4)
    calls = []

    def mixed(name, shape, dtype, *, device_rows):
        calls.append((name, shape, device_rows))
        return SimpleNamespace(device_view=torch.empty(shape, dtype=dtype))

    bank = ex.load_exl3_experts(loader, _prefixes([3, 1, 0, 2]), rank=0, world=2, hot_rows=2, mixed_allocate=mixed)
    assert bank.parked and bank.hot_rows == 2 and len(calls) == 3 and all(c[2] == 2 for c in calls)
    with pytest.raises(ValueError, match="codebook"):
        ex.load_exl3_experts(FakeLoader(2, codebook="mcg"), _prefixes([0, 1]), rank=0, world=2)
    with pytest.raises(ValueError, match="mixed allocator"):
        ex.load_exl3_experts(loader, _prefixes([0, 1, 2, 3]), rank=0, world=2, hot_rows=1)


class FakeExt:
    def __init__(self):
        self.launches = []

    def xmoe_set_force_group(self, v):
        self.force = v

    def xmoe_set_group_width(self, v):
        self.width = v

    def xmoe_set_sm_budget(self, v):
        self.budget = v

    def xmoe_set_lock_slot(self, v):
        self.slot = v

    def exl3_moe(self, hidden, out, counts, token_sorted, weight_sorted, sg, su, ig, iu, act, kg, ku, kd, *rest):
        self.launches.append(dict(counts=counts.clone(), tokens=token_sorted.clone(), weights=weight_sorted.clone(),
                                  force=self.force, width=self.width, groups=sg.shape[0], per=sg.shape[1],
                                  budget=self.budget, slot=self.slot,
                                  mcg=rest[9], mul1=rest[10], limit=rest[15], active=rest[16]))


@pytest.mark.parametrize("rows,parked", [(6, False), (24, True), (256, False)])
def test_routing_preparation(monkeypatch, rows, parked):
    fake = FakeExt()
    monkeypatch.setattr(xmoe, "extension", lambda: fake)
    ex._SCRATCH.clear()
    count = 16
    bank = SimpleNamespace(num_experts=count, local_intermediate=1152, parked=parked, hot_rows=10 if parked else count,
                           codebook="mul1", bits=3, pointers={(m, p): torch.zeros(count, dtype=torch.int64)
                                                              for m in ("w1", "w2", "w3") for p in ("trellis", "suh", "svh")})
    ex._BANKS[:] = [(bank, 23)]
    g = torch.Generator().manual_seed(rows)
    ids = torch.stack([torch.randperm(count, generator=g)[:6] for _ in range(rows)])
    w = torch.rand(rows, 6, generator=g)
    out = torch.empty(rows, 5120, dtype=torch.bfloat16)
    ex.ds41_exl3_moe(0, torch.randn(rows, 5120, dtype=torch.bfloat16), ids, w, out, 10.0)
    flat = ids.reshape(-1)
    counts = torch.bincount(flat, minlength=count + 1)
    first = fake.launches[0]
    order = torch.argsort(flat, stable=True)
    assert torch.equal(first["tokens"], order // 6)
    assert torch.equal(first["weights"], w.reshape(-1)[order].half())
    assert first["mul1"] and not first["mcg"] and first["limit"] == 10.0 and first["active"] == -1
    prefill = rows > 64
    assert first["groups"] == (ex.PREFILL_GROUPS if prefill else 23) and first["per"] == rows
    assert first["width"] == (ex.PREFILL_GROUP_WIDTH if prefill else 0)
    if not parked:
        assert len(fake.launches) == 1 and torch.equal(first["counts"], counts) and first["force"] == 0
    else:
        hot, cold = fake.launches
        slots = torch.arange(count + 1)
        assert torch.equal(hot["counts"], torch.where(slots < 10, counts, -counts)) and hot["force"] == 0
        assert torch.equal(cold["counts"], torch.where(slots < 10, -counts, counts)) and cold["force"] == 1
        assert torch.equal(hot["counts"].abs(), counts) and torch.equal(cold["counts"].abs(), counts)
        assert hot["budget"] == cold["budget"] == 0 and hot["slot"] == cold["slot"] == 0   # side stream off: sequential
    assert np.isfinite(out.float().numpy()).all()


def test_model_config_accepts_exl3_only_in_its_exact_form():
    import copy
    import json
    from pathlib import Path

    from ignition.recipes.deepseek41.config import ModelConfig

    pack = Path(os.environ.get("FLARE_TEST_DS41_EXL3_PACK") or "/nonexistent-exl3-pack") / "config.json"
    if not pack.is_file():
        pytest.skip("EXL3 pack not present")
    base = json.loads(pack.read_text())
    assert ModelConfig.from_dict(copy.deepcopy(base)).exl3
    for change in ({"bits": 4}, {"codebook": "mcg"}, {"mtp_experts": "fp4"}, {"engram_embed_format": "fp8"}):
        bad = copy.deepcopy(base)
        bad["quantization_config"].update(change)
        with pytest.raises(ValueError, match="EXL3"):
            ModelConfig.from_dict(bad)
    stock = copy.deepcopy(base)
    stock["quantization_config"] = {"quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0",
                                    "expert_dtype": "fp4", "activation_scheme": "dynamic"}
    assert not ModelConfig.from_dict(stock).exl3
    stock["quantization_config"]["expert_dtype"] = "exl3"
    with pytest.raises(ValueError, match="native MXFP4"):
        ModelConfig.from_dict(stock)


def test_parked_rows_are_staged_never_direct(monkeypatch):
    """b12x's batch executor writes only into CUDA device allocations: direct copies must target VRAM rows or the
    staging chunk, never a parked bank's host rows; the host rows still end up with the right bytes."""
    loader = FakeLoader(5)
    banks = {}

    def mixed(name, shape, dtype, *, device_rows):
        view = torch.zeros(shape, dtype=dtype)
        banks[name] = (view, device_rows)
        return SimpleNamespace(device_view=view)

    direct = []
    copy = loader._copy

    def guarded(destination, source):
        for view, rows in banks.values():
            if destination.untyped_storage().data_ptr() == view.untyped_storage().data_ptr():
                row = view[0].numel() * view.element_size()
                start = destination.data_ptr() - view.data_ptr()
                assert start + destination.numel() * destination.element_size() <= rows * row, "host row written directly"
        direct.append(destination.shape)
        copy(destination, source)

    loader._copy = guarded
    monkeypatch.setattr(ex, "STAGING_EXPERTS", 2)
    order = [4, 2, 0, 1, 3]
    bank = ex.load_exl3_experts(loader, _prefixes(order), rank=1, world=2, hot_rows=2, mixed_allocate=mixed)
    for slot, e in enumerate(order):
        for m in ("w1", "w3", "w2"):
            p = f"layers.0.ffn.experts.{e}.{m}"
            want = xmoe.shard_matrix(m, {k: loader.tensors[f"{p}.{k}"] for k in ("trellis", "suh", "svh")}, 1, 2)
            assert torch.equal(bank.trellis[m][slot], want["trellis"]), (slot, m)


def test_declared_hot_rows_map_fewer_vram_rows_and_declare_the_rest(monkeypatch):
    """FLARE_DS41_EXL3_EXPORT_HOT: the export maps min(n, hot_rows) VRAM rows (direct copies only there, the rest
    staged), the kernels keep hot_rows, and each mixed region declares hot_rows rows in whole granules."""
    from ignition.preparation.runtime_core import mixed_device_bytes

    loader = FakeLoader(6)
    banks = {}

    def mixed(name, shape, dtype, *, device_rows):
        view = torch.zeros(shape, dtype=dtype)
        row = view[0].numel() * view.element_size()
        granule = 1 << 20
        up = lambda n: -(-n // granule) * granule
        region = SimpleNamespace(device_view=view, name=name, granularity=granule, extent=up(shape[0] * row),
                                 device_bytes=min(up(device_rows * row), up(shape[0] * row)))
        banks[name] = (view, device_rows, region)
        return region

    copy = loader._copy

    def guarded(destination, source):
        for view, rows, _ in banks.values():
            if destination.untyped_storage().data_ptr() == view.untyped_storage().data_ptr():
                row = view[0].numel() * view.element_size()
                start = destination.data_ptr() - view.data_ptr()
                assert start + destination.numel() * destination.element_size() <= rows * row, "unmapped row written directly"
        copy(destination, source)

    loader._copy = guarded
    monkeypatch.setattr(ex, "STAGING_EXPERTS", 2)
    monkeypatch.setenv("FLARE_DS41_EXL3_EXPORT_HOT", "2")
    order = [5, 4, 2, 0, 1, 3]
    bank = ex.load_exl3_experts(loader, _prefixes(order), rank=0, world=2, hot_rows=4, mixed_allocate=mixed)
    assert bank.hot_rows == 4 and bank.mapped_rows == 2 and bank.parked
    for view, rows, region in banks.values():
        row = view[0].numel() * view.element_size()
        assert rows == 2
        assert region.declared_device_rows == 4
        assert mixed_device_bytes(region) == min(-(-4 * row // region.granularity) * region.granularity, region.extent)
        assert mixed_device_bytes(region) >= region.device_bytes
    for slot, e in enumerate(order):
        p = f"layers.0.ffn.experts.{e}.w2"
        want = xmoe.shard_matrix("w2", {k: loader.tensors[f"{p}.{k}"] for k in ("trellis", "suh", "svh")}, 0, 2)
        assert torch.equal(bank.trellis["w2"][slot], want["trellis"]), slot
    # unset or >= hot_rows: unchanged layout, nothing declared
    monkeypatch.setenv("FLARE_DS41_EXL3_EXPORT_HOT", "9")
    banks.clear()
    bank = ex.load_exl3_experts(loader, _prefixes(order), rank=0, world=2, hot_rows=4, mixed_allocate=mixed)
    assert bank.mapped_rows is None and all(r == 4 and not hasattr(g, "declared_device_bytes") for _, r, g in banks.values())
    monkeypatch.setenv("FLARE_DS41_EXL3_EXPORT_HOT", "-1")
    with pytest.raises(ValueError, match="EXPORT_HOT"):
        ex.export_hot()


def test_mixed_device_bytes_rejects_bad_declarations():
    from ignition.preparation.runtime_core import mixed_device_bytes

    region = SimpleNamespace(name="r", device_bytes=4 << 20, extent=16 << 20, granularity=2 << 20)
    assert mixed_device_bytes(region) == 4 << 20
    region.declared_device_bytes = 8 << 20
    assert mixed_device_bytes(region) == 8 << 20
    for bad in (2 << 20, 18 << 20, (8 << 20) + 4096):
        region.declared_device_bytes = bad
        with pytest.raises(ValueError, match="declared VRAM prefix"):
            mixed_device_bytes(region)


def test_per_layer_hot_counts_parse(monkeypatch, tmp_path):
    monkeypatch.delenv("FLARE_DS41_EXL3_HOT_LAYERS", raising=False)
    assert ex.hot_layers() == {} and ex.layer_hot_count(7, 320) == 320
    monkeypatch.setenv("FLARE_DS41_EXL3_HOT_LAYERS", "7:300, 25:340")
    assert ex.hot_layers() == {7: 300, 25: 340}
    assert ex.layer_hot_count(7, 320) == 300 and ex.layer_hot_count(8, 320) == 320
    f = tmp_path / "h.json"
    f.write_text('{"2": 338, "39": 321}')
    monkeypatch.setenv("FLARE_DS41_EXL3_HOT_LAYERS", f"@{f}")
    assert ex.hot_layers() == {2: 338, 39: 321}
    for bad in ("7", "7:400", "7:300,7:301", "-1:3"):
        monkeypatch.setenv("FLARE_DS41_EXL3_HOT_LAYERS", bad)
        with pytest.raises(ValueError, match="HOT_LAYERS"):
            ex.hot_layers()


def test_cold_side_switch_and_budgets(monkeypatch):
    monkeypatch.delenv("FLARE_DS41_EXL3_COLD_SIDE", raising=False)
    assert ex.cold_side_sms() == 0 and ex.side_budgets(188, 0) == (188, 0)
    monkeypatch.setenv("FLARE_DS41_EXL3_COLD_SIDE", "16")
    assert ex.cold_side_sms() == 16
    hot, cold = ex.side_budgets(188, 16)
    assert (hot, cold) == (172, 16) and hot + cold <= 188     # every block of both kernels co-resident
    for bad in ("12", "-8"):
        monkeypatch.setenv("FLARE_DS41_EXL3_COLD_SIDE", bad)
        with pytest.raises(ValueError, match="COLD_SIDE"):
            ex.cold_side_sms()
    with pytest.raises(ValueError, match="3/4"):
        ex.side_budgets(188, 56)


def test_launch_resets_budget_and_lock_slot():
    calls = []

    class Ext:
        def __getattr__(self, name):
            return lambda *a: calls.append((name, a))

    bank = SimpleNamespace(codebook="mul1", bits=3, pointers={(m, p): None for m in ("w1", "w3", "w2")
                                                               for p in ("trellis", "suh", "svh")})
    ex._launch(Ext(), bank, None, None, None, None, None, (None,) * 4, 10.0, force_group=True, width=0,
               sm_budget=16, lock_slot=1)
    names = [c[0] for c in calls]
    assert names[:4] == ["xmoe_set_force_group", "xmoe_set_group_width", "xmoe_set_sm_budget", "xmoe_set_lock_slot"]
    assert calls[2][1] == (16,) and calls[3][1] == (1,) and names[4] == "exl3_moe"
    assert calls[-2:] == [("xmoe_set_sm_budget", (0,)), ("xmoe_set_lock_slot", (0,))]


def test_shared_side_switch_and_ordering_op(monkeypatch):
    monkeypatch.delenv("FLARE_DS41_EXL3_SHARED_SIDE", raising=False)
    assert not ex.shared_side()
    monkeypatch.setenv("FLARE_DS41_EXL3_SHARED_SIDE", "1")
    assert ex.shared_side()
    monkeypatch.setenv("FLARE_DS41_EXL3_SHARED_SIDE", "yes")
    with pytest.raises(ValueError, match="SHARED_SIDE"):
        ex.shared_side()
    # the ordering op runs exactly the plain op's launches; its extra operand is only read for ordering
    runs = {}
    for name in ("plain", "after"):
        fake = FakeExt()
        monkeypatch.setattr(xmoe, "extension", lambda: fake)
        ex._SCRATCH.clear()
        bank = SimpleNamespace(num_experts=16, local_intermediate=1152, parked=True, hot_rows=10, codebook="mul1", bits=3,
                               pointers={(m, p): torch.zeros(16, dtype=torch.int64)
                                         for m in ("w1", "w2", "w3") for p in ("trellis", "suh", "svh")})
        ex._BANKS[:] = [(bank, 23)]
        g = torch.Generator().manual_seed(3)
        ids = torch.stack([torch.randperm(16, generator=g)[:6] for _ in range(6)])
        w = torch.rand(6, 6, generator=g)
        x = torch.randn(6, 5120, dtype=torch.bfloat16, generator=g)
        out = torch.empty(6, 5120, dtype=torch.bfloat16)
        if name == "plain":
            ex.ds41_exl3_moe(0, x, ids, w, out, 10.0)
        else:
            ex.ds41_exl3_moe_after(0, x, ids, w, torch.zeros(6, 5120, dtype=torch.bfloat16), out, 10.0)
        runs[name] = fake.launches
    assert len(runs["plain"]) == len(runs["after"]) == 2
    for a, b in zip(runs["plain"], runs["after"]):
        assert torch.equal(a["counts"], b["counts"]) and a["force"] == b["force"]


def test_cold_width_switch_applies_to_decode_cold_launch_only(monkeypatch):
    monkeypatch.delenv("FLARE_DS41_EXL3_COLD_WIDTH", raising=False)
    assert ex.cold_width() == 0
    for bad in ("12", "64", "-8"):
        monkeypatch.setenv("FLARE_DS41_EXL3_COLD_WIDTH", bad)
        with pytest.raises(ValueError, match="COLD_WIDTH"):
            ex.cold_width()
    monkeypatch.setattr(ex, "COLD_WIDTH", 24)
    monkeypatch.setattr(ex, "PREFILL_MODE", "group")
    for rows, want_cold in ((6, 24), (256, ex.PREFILL_GROUP_WIDTH)):
        fake = FakeExt()
        monkeypatch.setattr(xmoe, "extension", lambda: fake)
        ex._SCRATCH.clear(); ex._SIDE.clear()
        bank = SimpleNamespace(num_experts=16, local_intermediate=1152, parked=True, hot_rows=10, codebook="mul1", bits=3,
                               pointers={(m, p): torch.zeros(16, dtype=torch.int64)
                                         for m in ("w1", "w2", "w3") for p in ("trellis", "suh", "svh")})
        ex._BANKS[:] = [(bank, 23)]
        g = torch.Generator().manual_seed(rows)
        ids = torch.stack([torch.randperm(16, generator=g)[:6] for _ in range(rows)])
        ex.ds41_exl3_moe(0, torch.randn(rows, 5120, dtype=torch.bfloat16), ids, torch.rand(rows, 6, generator=g),
                         torch.empty(rows, 5120, dtype=torch.bfloat16), 10.0)
        hot, cold = fake.launches
        assert not hot["force"] and cold["force"]
        assert hot["width"] == (ex.PREFILL_GROUP_WIDTH if rows > 64 else 0) and cold["width"] == want_cold
