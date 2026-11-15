# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The functional sparse indexer skips its -1 / 0 fills only on b12x's fused decode route.

That route stores every top-k slot of every row (see the comment in ``PreparedSparseIndexer.__call__`` and the
CUDA poison test); the tiled and packed-contiguous routes keep the fills.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import indexer

ROWS, TOPK = 3, 8


def _owner(monkeypatch, route):
    owner = object.__new__(indexer.PreparedSparseIndexer)
    owner.plan = SimpleNamespace(handle=4, scratch_specs=lambda: (SimpleNamespace(shape=(32,), dtype=torch.uint8),))
    owner.inputs = {"index_k_cache": None}
    owner.output = owner.scores = owner.binding = owner._finalize = None
    owner.output_layout = (ROWS, TOPK), (TOPK, 1), torch.int32
    owner.score_layout = (ROWS, TOPK), (TOPK, 1), torch.float32
    state = SimpleNamespace(layout=SimpleNamespace(route=route))
    monkeypatch.setattr(indexer, "require_prepared", lambda plan, component, device=None: state)
    owner._finalize_functional()
    return owner


@pytest.mark.parametrize("route,filled", [("paged_fused", False), ("paged_tiled", True),
                                          ("packed_contiguous", True)])
def test_fills_follow_the_route(monkeypatch, route, filled):
    owner = _owner(monkeypatch, route)
    empty_strided, seen = torch.empty_strided, {}

    def poisoned(*args, **kwargs):
        tensor = empty_strided(*args, **kwargs)
        tensor.view(torch.uint8).fill_(0x7F)
        return tensor

    def select(query, weights, cache, pages, lengths, active, output, scores, scratch, handle):
        seen["output"], seen["scores"] = output.clone(), scores.clone()

    monkeypatch.setattr(torch, "empty_strided", poisoned)
    monkeypatch.setattr(indexer, "_select", select)
    output, scores = owner(torch.zeros(ROWS, 2, 128), None, None, None, None)
    assert output.shape == (ROWS, TOPK) and scores.shape == (ROWS, TOPK)
    if filled:
        assert (seen["output"] == -1).all() and (seen["scores"] == 0).all()
    else:
        assert (seen["output"].view(torch.uint8) == 0x7F).all() and (seen["scores"].view(torch.uint8) == 0x7F).all()


def test_states_without_a_route_keep_the_fills(monkeypatch):
    owner = _owner(monkeypatch, None)
    assert owner.writes_every_slot is False
