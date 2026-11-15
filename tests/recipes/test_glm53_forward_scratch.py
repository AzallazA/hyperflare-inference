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


@pytest.mark.parametrize("world", [1, 3])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_mla_scratch_keeps_exchange_and_record_windows_zeroed(poisoned_empty, world, mode):
    from ignition.recipes.glm53.mla import MLAScratch

    tokens, blocks, heads = 6, 2, Shards.of(world).heads
    prepared = MLAScratch.allocate(tokens, heads, blocks, mode, "cpu", context_ranks=world, remote=world > 1)
    assert all(v in ("zero", "none") for k, v in layout(prepared).items() if k not in ("index", "slab")), \
        layout(prepared)
    scratch = MLAScratch.allocate(tokens, heads, blocks, mode, "cpu", context_ranks=world, remote=world > 1,
                                  forward=True)
    observed = {k: v for k, v in layout(scratch).items() if k != "index"}
    cp_decode, cp_prefill = world > 1 and mode == "decode", world > 1 and mode == "prefill"
    for name in ("projected_input", "query_input", "query_residual", "latent_input", "latent", "query",
                 "absorbed_query", "attended", "expanded", "output", "visible"):
        assert observed.pop(name) == "unfilled", name
    for name in ("gathered", "partial", "lse"):
        assert observed.pop(name) == ("unfilled" if cp_decode else "none"), name
    assert observed.pop("slab") == ("unfilled" if cp_prefill else "none")
    # Cross-host staging and record windows keep their zeros.
    assert set(observed.values()) <= {"zero", "none"}, observed
    assert any(v == "zero" for v in observed.values()) == (world > 1)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("tokens", [1, 3])
def test_kda_scratch_leaves_only_the_recurrence_output_unfilled(poisoned_empty, split, tokens):
    from ignition.recipes.glm53.kda import KDAScratch

    heads = Shards.of(3).heads
    assert set(layout(KDAScratch.allocate(tokens, heads, "cpu", split=split)).values()) == {"zero"}
    observed = layout(KDAScratch.allocate(tokens, heads, "cpu", split=split, forward=True))
    assert observed.pop("recurrence") == "unfilled"
    assert set(observed.values()) == {"zero"}, observed
