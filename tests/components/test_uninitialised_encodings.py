# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Invocation-local activation encodings start uninitialised only where their writer covers them.

The allocator is poisoned (every ``torch.empty`` returns 0x7F bytes, the pattern recycled graph-pool
memory can hold) and each emulated writer stores exactly the elements its kernel stores. The
encodings it leaves must equal those of zero-initialised bases bit for bit.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import expert_overlap, linear

POISON = 0x7F


def _poisoning(empty):
    def poisoned_empty(*args, **kwargs):
        tensor = empty(*args, **kwargs)
        if tensor.numel():
            tensor.view(torch.uint8).fill_(POISON)
        return tensor
    return poisoned_empty


@pytest.fixture
def poisoned(monkeypatch):
    monkeypatch.setattr(torch, "empty", _poisoning(torch.empty))


def _layout(rows, columns):
    # mxfp8.dense_rows_storage_shapes, without importing the CUDA-gated quantizer.
    return (((rows, columns), torch.float8_e4m3fn), ((1, rows, columns // 32), torch.uint8),
            ((1, (rows + 127) // 128, columns // 128, 32, 4, 4), torch.uint8))


def _quantize_like_cute(source, values, scale_rows, scale_mma):
    """Store what b12x's CuTe MXFP8 row quantizer stores (``_lib/quant/mxfp8_rows.py``).

    Every row below M writes all K values and all K/32 compact scales; ``_store_scale`` writes the
    swizzled MMA byte of (row, group) only for rows below M.
    """
    rows, columns = source.shape
    groups = columns // 32
    blocks = source.float().view(rows, groups, 32)
    amax = blocks.abs().amax(-1)
    exponent = torch.where(amax > 0, torch.ceil(torch.log2(amax / 448)), torch.zeros_like(amax))
    scale_bytes = (exponent + 127).clamp(1, 254).to(torch.uint8)
    values.copy_((blocks / torch.exp2(exponent)[..., None]).view(rows, columns).to(torch.float8_e4m3fn))
    scale_rows.view(rows, groups).copy_(scale_bytes)
    flat = scale_mma.view(-1)
    k_tiles = (groups + 3) // 4
    for row in range(rows):
        for group in range(groups):
            offset = ((row % 32) * 16 + ((row // 32) % 4) * 4 + (row // 128) * k_tiles * 512
                      + group % 4 + (group // 4) * 512)
            flat[offset] = scale_bytes[row, group]


@pytest.mark.parametrize("rows", [1, 5, 32, 127, 128, 130, 256])
def test_mxfp8_dense_bases_match_zeroed_bases_after_the_quantizer(monkeypatch, rows):
    columns = 256
    source = torch.randn(rows, columns, dtype=torch.bfloat16)
    expected = [torch.zeros(shape, dtype=dtype) for shape, dtype in _layout(rows, columns)]
    _quantize_like_cute(source, *expected)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", _poisoning(torch.empty))
        bases = linear.mxfp8_dense_bases(_layout(rows, columns), torch.device("cpu"))
    # Values and compact scales are left to the quantizer; MMA scales are zero unless M fills its tiles.
    assert bases[0].view(torch.uint8).eq(POISON).all() and bases[1].eq(POISON).all()
    assert bases[2].eq(POISON if rows % 128 == 0 else 0).all()
    _quantize_like_cute(source, *bases)
    for actual, reference in zip(bases, expected, strict=True):
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


def _block_fp8_quantize(source, values, scales):
    """``per_token_group_quant_8bit`` with caller output: every value and every 128-column scale."""
    rows, columns = source.shape
    blocks = source.float().view(rows, columns // 128, 128)
    scale = blocks.abs().amax(-1).clamp_min(1e-10) / 448
    values.copy_((blocks / scale[..., None]).view(rows, columns).to(torch.float8_e4m3fn))
    scales.copy_(scale)


def test_shared_expert_encodings_poison_only_fully_written_segments(poisoned):
    rows = 5
    layouts = (((rows, 512), torch.float8_e4m3fn), ((rows, 4), torch.float32),   # block-FP8 gate/up
               *_layout(rows, 256))                                               # MXFP8 down
    owner = SimpleNamespace(quantizers=[0, 11], encoding_offsets=[0, 2, 5], encoded_layouts=list(layouts))
    encoded = expert_overlap.ConcurrentSharedExpert._encoded_bases(owner, torch.device("cpu"))
    assert [(tuple(t.shape), t.dtype) for t in encoded] == [(shape, dtype) for shape, dtype in layouts]
    assert all(t.view(torch.uint8).eq(POISON).all() for t in encoded[:4])
    assert encoded[4].eq(0).all()

    gate_up, down = torch.randn(rows, 512, dtype=torch.bfloat16), torch.randn(rows, 256, dtype=torch.bfloat16)
    expected = [torch.zeros(shape, dtype=dtype) for shape, dtype in layouts]
    for bases in (encoded, expected):
        _block_fp8_quantize(gate_up, *bases[:2])
        _quantize_like_cute(down, *bases[2:])
    for actual, reference in zip(encoded, expected, strict=True):
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))


def test_shared_expert_encodings_zero_unclaimed_layouts(poisoned):
    owner = SimpleNamespace(quantizers=[], encoding_offsets=[], encoded_layouts=[((2, 128), torch.float32)] * 2)
    encoded = expert_overlap.ConcurrentSharedExpert._encoded_bases(owner, torch.device("cpu"))
    assert len(encoded) == 2 and all(t.eq(0).all() for t in encoded)


def test_functional_quantizer_projection_allocates_mxfp8_dense_bases(monkeypatch):
    from b12x import gemm
    from b12x.quantization import mxfp8

    rows, columns = 3, 128
    projection = object.__new__(linear.QuantizedProjection)
    projection.output = None
    projection.output_layout = ((rows, 4), (4, 1), torch.float32)
    projection.workspace = None
    projection.workspace_nbytes = 16
    projection.encoded = None
    projection.encoded_layout = _layout(rows, columns)
    projection.quantizer_plan = object()
    projection.plan = object()
    projection.parts = (torch.ones(4, columns), None, None)
    projection.bias = None
    seen = []

    def quantize(source, values, scale_rows, scale_mma, *, plan):
        seen.append((values.view(torch.uint8).clone(), scale_rows.clone(), scale_mma.clone()))
        _quantize_like_cute(source, values, scale_rows, scale_mma)

    def mm(values, scale_mma, weight, weight_scales, output, workspace, *, plan):
        output.copy_(values.float() @ weight.T)

    monkeypatch.setattr(mxfp8, "quantize_dense_rows", quantize)
    monkeypatch.setattr(gemm, "mxfp8_mm_bases", mm)
    monkeypatch.setattr(torch, "empty", _poisoning(torch.empty))
    source = torch.randn(rows, columns, dtype=torch.bfloat16)
    projection(source)
    values, scale_rows, scale_mma = seen[0]
    assert values.eq(POISON).all() and scale_rows.eq(POISON).all() and scale_mma.eq(0).all()
