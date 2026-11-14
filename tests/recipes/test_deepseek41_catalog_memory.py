# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 catalog memory (G1 2026-11-14): shared vision declaration scratch and the parked-copy load pool."""

from contextlib import contextmanager, nullcontext
from dataclasses import fields, is_dataclass

import torch

from ignition.components.deepseek_vision import VisionScratch
from ignition.preparation.persistent import _ACTIVE_STORAGE
from ignition.recipes.deepseek41.contract import PATCH_CAPACITIES
from ignition.recipes.deepseek41.weights import _parking_pool


def _leaves(value):
    if isinstance(value, torch.Tensor):
        return [value]
    return [leaf for item in fields(value) for leaf in _leaves(getattr(value, item.name))]


def test_vision_capacities_are_row_prefixes_of_the_largest_scratch():
    largest = VisionScratch.allocate(max(PATCH_CAPACITIES), "meta", hidden=5120)
    for capacity in PATCH_CAPACITIES:
        shared = largest.rows(capacity)
        expected = VisionScratch.allocate(capacity, "meta", hidden=5120)
        assert is_dataclass(shared) and type(shared) is type(expected)
        for view, own, base in zip(_leaves(shared), _leaves(expected), _leaves(largest), strict=True):
            # Same geometry the per-capacity allocation had, carved from the largest scratch.
            assert (view.shape, view.dtype, view.stride()) == (own.shape, own.dtype, own.stride())
            assert view.is_contiguous() and (view is base or view._base is base)


def test_parked_copies_load_in_a_pool_of_their_own():
    entered = []

    class Storage:
        @contextmanager
        def transient(self):
            entered.append(True)
            yield

    assert isinstance(_parking_pool(), nullcontext)
    token = _ACTIVE_STORAGE.set(Storage())
    try:
        with _parking_pool():
            pass
    finally:
        _ACTIVE_STORAGE.reset(token)
    assert entered == [True]
