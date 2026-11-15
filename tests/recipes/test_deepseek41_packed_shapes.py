# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 packed verification shapes above 8 slots (tp3.packed_keep_batches / packed_wide_graphs / packed_wide_tokens), CPU.

Default = upstream's enumeration (one 64-graph budget spread over every batch capacity: a 16-slot export gets coarser
4- and 8-request shapes). With packed_keep_batches=8 a 16-slot export keeps the 8-slot export's shapes for 1-8 requests
exactly and adds the 16-request shapes on top.
"""

import collections

import pytest

from ignition.components.capacity import packed_speculative_shapes
from ignition.recipes.deepseek41.bootstrap import PreparationConfig, packed_shapes
from ignition.recipes.deepseek41.tp3_config import TP3Config


def config(slots, budget=64, **tp3):
    return PreparationConfig(devices=(0, 1, 2), max_active=slots, dspark=True, mtp_only=True,
                             vocabulary_parallel_sampling=True, speculative_widths=tuple(range(1, 7)),
                             speculative_graph_budget=budget, tp3=TP3Config(**tp3))


def by_batch(shapes):
    grouped = collections.defaultdict(list)
    for batch, tokens in shapes:
        grouped[batch].append(tokens)
    return dict(grouped)


EIGHT = {1: [2, 3, 4, 5, 6], 2: list(range(3, 13)), 4: list(range(5, 25)),
         8: [*range(9, 25), 26, 27, 29, 30, 32, 33, 35, 37, 39, 41, 43, 45, 48]}


def test_default_enumeration_is_unchanged():
    assert by_batch(config(8).packed_speculative_shapes) == EIGHT  # the 151744 catalog
    sixteen = by_batch(config(16).packed_speculative_shapes)
    assert len(config(16).packed_speculative_shapes) == 64
    assert sixteen[4] == [*range(5, 15), 16, 18, 20, 22, 24]
    assert sixteen[8] == [9, 10, 11, 12, 14, 15, 17, 19, 21, 23, 26, 29, 32, 35, 39, 43, 48]
    assert sixteen[16] == [17, 19, 21, 23, 26, 29, 32, 36, 40, 45, 50, 56, 62, 69, 77, 86, 96]
    # the batches argument defaults to every batch and restricts without changing the split rule
    assert packed_speculative_shapes(16, 6, 128, 64) == packed_speculative_shapes(16, 6, 128, 64, batches=(1, 2, 4, 8, 16))
    with pytest.raises(ValueError, match="batch capacities"):
        packed_speculative_shapes(8, 6, 128, 64, batches=(16,))


def test_keep_eight_slot_shapes_and_add_wide_ones():
    keep = config(16, packed_keep_batches=8, packed_wide_graphs=12, packed_wide_tokens=48)
    shapes = by_batch(keep.packed_speculative_shapes)
    assert {batch: tokens for batch, tokens in shapes.items() if batch <= 8} == EIGHT
    assert shapes[16] == [17, 19, 20, 22, 25, 27, 29, 33, 37, 42, 45, 48]
    assert len(keep.packed_speculative_shapes) == 76
    assert keep.decode_tokens == 48  # the widest packed graph
    endpoints = by_batch(config(16, packed_keep_batches=8).packed_speculative_shapes)
    assert endpoints[16] == [17, 96] and endpoints[8] == EIGHT[8]


def test_same_budget_split():
    """Option (b): 52 graphs for 1-8 requests + 12 for 16 = the default 64."""
    split = config(16, budget=52, packed_keep_batches=8, packed_wide_graphs=12, packed_wide_tokens=48)
    shapes = by_batch(split.packed_speculative_shapes)
    assert len(split.packed_speculative_shapes) == 64
    assert shapes[4] == [*range(5, 19), 20, 22, 24]
    assert [t for t in shapes[8] if t <= 24] == [*range(9, 18), 19, 21, 23, 24]


def test_keep_is_a_no_op_at_or_below_its_batch():
    for slots in (4, 8):
        assert (config(slots, packed_keep_batches=8).packed_speculative_shapes
                == config(slots).packed_speculative_shapes)


def test_knob_validation_and_environment():
    with pytest.raises(ValueError, match="power of two"):
        TP3Config(packed_keep_batches=6)
    with pytest.raises(ValueError, match="need packed_keep_batches"):
        TP3Config(packed_wide_graphs=4)
    with pytest.raises(ValueError, match="exceed the widest batch"):
        packed_shapes(16, 6, 128, 64, TP3Config(packed_keep_batches=8, packed_wide_tokens=16))
    env = TP3Config.from_environment({"FLARE_DS41_PACKED_KEEP_BATCHES": "8", "FLARE_DS41_PACKED_WIDE_GRAPHS": "12",
                                      "FLARE_DS41_PACKED_WIDE_TOKENS": "48"})
    assert (env.packed_keep_batches, env.packed_wide_graphs, env.packed_wide_tokens) == (8, 12, 48)
    record = env.to_metadata()
    assert (record["packed_keep_batches"], record["packed_wide_graphs"], record["packed_wide_tokens"]) == (8, 12, 48)
    base = TP3Config.from_environment({})
    assert (base.packed_keep_batches, base.packed_wide_graphs, base.packed_wide_tokens) == (0, 0, 0)
