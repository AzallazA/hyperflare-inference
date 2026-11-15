# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Which GLM per-forward scratch buffers are left unfilled, and which stay zeroed.

`MLAScratch`, `IndexScratch` and `KDAScratch` are zeroed for preparation and allocated once per call in the
compiled forward (`forward=True`), where every zero fill feeding an opaque op is a launch. A buffer may be left
unfilled only if the forward writes every element a reader can see before the read (the allocators' docstrings
name the writer of each). These tests pin that set: `torch.empty` is poisoned with 0x7F bytes, so a buffer that
reads as all zeros was zero filled. Recycled memory is not zero; the CUDA dirty-allocator runs check the
writers themselves.
"""

from dataclasses import fields

import pytest
import torch

from ignition.recipes.glm53.shards import Shards

POISON = 0x7F


@pytest.fixture
def poisoned_empty(monkeypatch):
    original = torch.empty

    def empty(*shape, **kwargs):
        tensor = original(*shape, **kwargs)
        if tensor.numel():
            tensor.view(torch.uint8).fill_(POISON) if tensor.dtype != torch.bool else tensor.fill_(True)
        return tensor

    monkeypatch.setattr(torch, "empty", empty)


def state(tensor):
    if tensor.numel() == 0:
        return "none"
    if tensor.dtype == torch.bool:
        return "zero" if not tensor.any() else "unfilled"
    raw = tensor.contiguous().view(torch.uint8)
    if not raw.any():
        return "zero"
    return "unfilled" if bool((raw == POISON).all()) else "other"


def layout(scratch):
    return {field.name: state(getattr(scratch, field.name)) for field in fields(scratch)
            if isinstance(getattr(scratch, field.name), torch.Tensor)}


@pytest.mark.parametrize("world", [1, 3])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_index_scratch_leaves_only_whole_written_selection_buffers_unfilled(poisoned_empty, world, mode):
    from ignition.recipes.glm53.indexer import IndexScratch

    tokens, blocks = 6, 4
    assert set(layout(IndexScratch.allocate(tokens, blocks, mode, "cpu", context_ranks=world)).values()) <= {
        "zero", "none", "other"}
    observed = layout(IndexScratch.allocate(tokens, blocks, mode, "cpu", context_ranks=world, forward=True))
    single = {"physical": "zero", "selected_lengths": "zero"}
    context = ({"scores": "unfilled", "candidates": "unfilled", "gathered": "unfilled", "global_pools": "unfilled"}
               if world > 1 else {"scores": "none", "candidates": "none", "gathered": "none", "global_pools": "none"})
    expected = {"query": "unfilled", "query_fp8": "unfilled", "query_scales": "unfilled",
                "normalized_key": "unfilled", "gate": "unfilled", "weights": "unfilled", "pool_lengths": "unfilled",
                "pools": "unfilled", "page_table": "unfilled",
                **(single if world == 1 else {"physical": "unfilled", "selected_lengths": "unfilled"}), **context}
    assert observed == expected
