# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""FLARE_DS41_EXL3_WOA_SINGLE: the TP2 WO-A tile-major copy is kept when off and never persistently allocated when on
(CPU, meta tensors; b12x's CUDA-tensor checks stubbed as in tp3-tools/ds41_exl3_plan_check.py)."""

import contextlib
from types import SimpleNamespace

import pytest
import torch

import ignition.components.block_fp8_weights as bfw
from ignition.recipes.deepseek41 import weights as dw

TEXT = dict(o_lora_rank=1024, num_attention_heads=64, head_dim=512, o_groups=8, hidden_size=5120)


def _loader(rank_groups):
    loader = bfw.BlockFP8WeightLoader.__new__(bfw.BlockFP8WeightLoader)
    loader.config = SimpleNamespace(text=TEXT, quantization={"weight_block_size": [32, 32]})
    loader._session = SimpleNamespace(flush=lambda: None)
    rows = (rank_groups[1] - rank_groups[0])

    def dense_blocks(prefix, *, rows=None, columns=None, pad=False, temporary=True):
        shape = (rows[1] - rows[0], 4096) if prefix.endswith("wo_a") else (5120, columns[1] - columns[0])
        return (torch.empty(shape, dtype=torch.float8_e4m3fn, device="meta"),
                torch.empty((shape[0] // 32, shape[1] // 32), dtype=torch.uint8, device="meta"))

    loader._dense_blocks = dense_blocks
    assert rows % 1024 == 0
    return loader


@pytest.fixture
def persistent(monkeypatch):
    import b12x.gemm._shared.wo_mxfp8 as wo_mxfp8

    monkeypatch.setattr(wo_mxfp8, "_check_gpu_tensor", lambda *args, **kwargs: None)
    shapes = []
    real = bfw.persistent_empty

    def record(shape, dtype, device, **kwargs):
        shapes.append(tuple(shape))
        return real(shape, dtype, device, **kwargs)

    monkeypatch.setattr(bfw, "persistent_empty", record)
    return shapes


def test_tp2_wo_a_tiled_copy_kept_by_default(persistent):
    packed = _loader((0, 4096)).load_attention_output("layers.3.attn", (0, 4096), pad=True)
    assert tuple(packed.wo_a.values.shape) == (1024, 4096, 4)
    assert tuple(packed.wo_a.values_tiled.shape) == (4, 16, 32, 64, 128)
    assert (4, 16, 32, 64, 128) in persistent
    assert packed.wo_b.values_tiled is None          # (5120, 4096) WO-B: b12x packs no copy at DS41 TP2


def test_tp2_wo_a_single_drops_tiled_copy(persistent):
    packed = _loader((0, 4096)).load_attention_output("layers.3.attn", (0, 4096), pad=True, tiled=False)
    assert packed.wo_a.values_tiled is None and packed.wo_b.values_tiled is None
    assert all(len(shape) != 5 for shape in persistent), persistent      # the copy never took persistent storage
    # the operands the WO plan reads are unchanged
    reference = _loader((0, 4096)).load_attention_output("layers.3.attn", (0, 4096), pad=True)
    for matrix in ("wo_a", "wo_b"):
        for plane in ("values", "scale_rows", "scale_mma"):
            got, want = getattr(getattr(packed, matrix), plane), getattr(getattr(reference, matrix), plane)
            assert (tuple(got.shape), got.stride(), got.dtype) == (tuple(want.shape), want.stride(), want.dtype)
    assert (packed.groups, packed.group_width, packed.rank, packed.hidden) == (4, 4096, 1024, 5120)


def test_tp3_shapes_have_no_tiled_copy(persistent):
    # TP3 pads 8 groups to 3 per rank: (3, 1024, 4096) is not a b12x tile-major shape, the switch is a no-op there.
    packed = _loader((0, 3072)).load_attention_output("layers.3.attn", (0, 3072), pad=True)
    assert packed.wo_a.values_tiled is None


def test_untiled_allocate_only_diverts_tiled_shapes(persistent):
    t = bfw._untiled_allocate((4, 16, 32, 64, 128), dtype=torch.float8_e4m3fn, device="meta")
    assert tuple(t.shape) == (4, 16, 32, 64, 128) and persistent == []
    bfw._untiled_allocate((4096, 4096), dtype=torch.float8_e4m3fn, device="meta")
    bfw._untiled_allocate((4, 16, 32, 64, 128), dtype=torch.uint8, device="meta")
    assert persistent == [(4096, 4096), (4, 16, 32, 64, 128)]


@pytest.mark.parametrize("value,expected", [(None, False), ("", False), ("0", False), ("1", True)])
def test_woa_single_switch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("FLARE_DS41_EXL3_WOA_SINGLE", raising=False)
    else:
        monkeypatch.setenv("FLARE_DS41_EXL3_WOA_SINGLE", value)
    assert dw.woa_single() is expected


def test_woa_single_switch_rejects_other_values(monkeypatch):
    monkeypatch.setenv("FLARE_DS41_EXL3_WOA_SINGLE", "yes")
    with pytest.raises(ValueError, match="WOA_SINGLE"):
        dw.woa_single()


def test_load_layer_passes_the_switch(monkeypatch):
    seen = []

    class Loader(dw.DeepSeekWeightLoader):
        def __init__(self):
            pass

    loader = Loader()
    loader.config = SimpleNamespace(text=dict(kv_source_layer_ids=(), index_source_layer_ids=(), engram_layer_ids=(),
                                              compress_ratios={}))
    loader.shard = SimpleNamespace(padded_partition=lambda n: (0, 4))
    for name in ("load_linear_group", "load_tensor", "load_padded_tensor"):
        setattr(loader, name, lambda *args, **kwargs: object())
    loader.load_attention_output = lambda *args, **kwargs: seen.append(kwargs["tiled"]) or object()

    for value, tiled in (("0", True), ("1", False)):
        monkeypatch.setenv("FLARE_DS41_EXL3_WOA_SINGLE", value)
        with contextlib.suppress(Exception):    # the stub loader stops past the attention dict; only "output" matters
            loader.load_layer(3)
        assert seen[-1] is tiled
