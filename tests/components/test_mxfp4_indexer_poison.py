# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The functional MXFP4 indexer leaves its query encodings to the quantizer and keeps its sentinels."""

import torch

from ignition.components import mxfp4_indexer

POISON = 0x7F


def _quantizer_stores(rows):
    """Byte offsets b12x's dsa_indexer ``_Quantize`` kernel stores (non-paged), per its index formulas."""
    data, scales = set(), set()
    for group in range(rows * 4):
        row = group // 4
        for lane in range(0, 32, 2):
            data.add(row * 64 + (group % 4) * 16 + lane // 2)
        scales.add(row * 4 + group % 4)
    return data, scales


def test_quantizer_covers_every_query_byte_and_scale():
    for rows in (1, 6, 64 * 3):
        data, scales = _quantizer_stores(rows)
        assert data == set(range(rows * 64)) and scales == set(range(rows * 4))


def test_functional_call_fills_only_selection_sentinels(monkeypatch):
    rows, heads = 3, 2
    owner = object.__new__(mxfp4_indexer.PreparedMXFP4Indexer)
    owner.arguments = None
    owner.layouts = {"q_mxfp4": ((rows, heads, 64), (heads * 64, 64, 1), torch.uint8),
                     "q_scales": ((rows, heads, 4), (heads * 4, 4, 1), torch.uint8),
                     "output_indices": ((rows, 8), (8, 1), torch.int32),
                     "candidate_output": ((rows, 16), (16, 1), torch.int32),
                     "candidate_output_lengths": ((rows,), (1,), torch.int32)}
    owner.scratch_layout = (((32,), torch.uint8),)
    owner.cache, owner.plan = None, type("Plan", (), {"handle": 3})()
    empty_strided = torch.empty_strided

    def poisoned(*args, **kwargs):
        tensor = empty_strided(*args, **kwargs)
        tensor.view(torch.uint8).fill_(POISON)
        return tensor

    seen = {}

    def quantize(query, packed, scales, handle):
        seen["packed"], seen["scales"] = packed.clone(), scales.clone()

    def select(packed, scales, weights, cache, pages, lengths, active, output, scratch, candidates, counts,
               candidates_out, counts_out, handle):
        seen["output"], seen["candidates"], seen["counts"] = output.clone(), candidates_out.clone(), counts_out.clone()

    monkeypatch.setattr(torch, "empty_strided", poisoned)
    monkeypatch.setattr(mxfp4_indexer, "_quantize", quantize)
    monkeypatch.setattr(mxfp4_indexer, "_select", select)
    owner(torch.zeros(rows, heads, 128, dtype=torch.bfloat16))
    assert seen["packed"].eq(POISON).all() and seen["scales"].eq(POISON).all()
    assert seen["output"].eq(-1).all() and seen["candidates"].eq(-1).all() and seen["counts"].eq(0).all()
