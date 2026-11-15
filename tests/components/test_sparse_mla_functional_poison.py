# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Functional sparse MLA output and LSE are wholly copied from b12x's result, so they need no zero fill."""

import pytest
import torch

from ignition.components import sparse_mla

ROWS, HEADS, DIM = 3, 4, 8


def _poisoned(empty_strided):
    def allocate(*args, **kwargs):
        tensor = empty_strided(*args, **kwargs)
        tensor.view(torch.uint8).fill_(0xFF)
        return tensor
    return allocate


@pytest.mark.parametrize("return_lse", [False, True])
def test_functional_output_and_lse_are_every_element_copied(monkeypatch, return_lse):
    result = torch.randn(ROWS, HEADS, DIM).bfloat16()
    result[-1] = 0  # b12x writes idle rows as exact zero in its own scratch result.
    result_lse = torch.randn(ROWS, HEADS)
    result_lse[-1] = float("-inf")
    owner = object.__new__(sparse_mla.PreparedSparseMLA)
    owner.output = None
    owner.output_layout = (ROWS, HEADS, DIM), (HEADS * DIM, DIM, 1), torch.bfloat16
    owner.return_lse = return_lse
    owner.lse_layout = ((ROWS, HEADS), (HEADS, 1)) if return_lse else None
    owner.scratch_layout = (((64,), torch.uint8),)
    owner.inputs = {"kv_cache": torch.zeros(1, 1, 1), "attention_sink": None}
    owner.plan = type("Plan", (), {"handle": 7})()
    seen = {}

    def run(binding):
        seen["output"] = binding["output_seen"]
        return (result, result_lse) if return_lse else result

    def bind(plan, **kwargs):
        return {"output_seen": True}

    monkeypatch.setattr(sparse_mla, "plan_from_handle", lambda handle: handle)
    monkeypatch.setattr(sparse_mla.sparse_mla, "bind", bind)
    monkeypatch.setattr(sparse_mla.sparse_mla, "run", run)
    monkeypatch.setattr(torch, "empty_strided", _poisoned(torch.empty_strided))
    returned = owner(torch.zeros(ROWS, HEADS, DIM), torch.zeros(ROWS, 1, dtype=torch.int32),
                     torch.zeros(ROWS, dtype=torch.int32), torch.zeros(ROWS, dtype=torch.int32))
    output, lse = returned if return_lse else (returned, None)
    assert seen["output"]
    assert torch.equal(output.view(torch.int16), result.view(torch.int16))
    assert not output[-1].count_nonzero()
    if return_lse:
        assert torch.equal(lse.view(torch.int32), result_lse.view(torch.int32))


def test_functional_call_hands_unfilled_storage_to_the_operator(monkeypatch):
    owner = object.__new__(sparse_mla.PreparedSparseMLA)
    owner.output = None
    owner.output_layout = (ROWS, HEADS, DIM), (HEADS * DIM, DIM, 1), torch.bfloat16
    owner.return_lse, owner.lse_layout = True, ((ROWS, HEADS), (HEADS, 1))
    owner.scratch_layout = (((64,), torch.uint8),)
    owner.inputs = {"kv_cache": None, "attention_sink": None}
    owner.plan = type("Plan", (), {"handle": 7})()
    seen = {}

    def attend(*args):
        seen["output"], seen["lse"] = args[6].view(torch.uint8).clone(), args[7].view(torch.uint8).clone()

    monkeypatch.setattr(sparse_mla, "_attend", attend)
    monkeypatch.setattr(torch, "empty_strided", _poisoned(torch.empty_strided))
    owner(torch.zeros(ROWS, HEADS, DIM), None, None, None)
    assert (seen["output"] == 0xFF).all() and (seen["lse"] == 0xFF).all()
