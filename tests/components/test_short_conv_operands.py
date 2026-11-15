# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""run_short_conv hands b12x private operands of its own, never dtype views at offsets into a byte scratch.

Inductor (PyTorch 2.14) clones a byte scratch passed as offset dtype views to a mutating custom op and bases
the clone on the offset view, so the clone reads past the scratch's end. These tests pin the operands and
check the compiled forward has no such clone (CPU Inductor).
"""

from types import SimpleNamespace

import pytest
import torch
from torch._inductor.utils import run_and_get_code

from ignition.components import recurrent

SEQS, CHANNELS, STATE, TOKENS = 1, 9216, 3, 2


@torch.library.custom_op("flare_test::short_conv", mutates_args=("conv_state", "out", "gathered_state", "request_ids"))
def _fake_pipeline(x: torch.Tensor, conv_weight: torch.Tensor, query_start_loc: torch.Tensor,
                   state_slot_ids: torch.Tensor, state_is_fresh: torch.Tensor, num_accepted_tokens: torch.Tensor,
                   request_is_prefill: torch.Tensor, num_seqs: torch.Tensor, num_tokens: torch.Tensor,
                   conv_state: torch.Tensor, out: torch.Tensor, gathered_state: torch.Tensor,
                   request_ids: torch.Tensor, conv_bias: torch.Tensor | None, plan_handle: int) -> None:
    request_ids.copy_(torch.arange(request_ids.numel(), dtype=torch.int32))
    gathered_state.copy_(conv_state[:1].to(gathered_state.dtype))
    out.copy_(x * 2 + request_ids[:, None].to(x.dtype) + gathered_state.float().sum())


@_fake_pipeline.register_fake
def _(*args):
    pass


@pytest.fixture
def prepared(monkeypatch):
    from b12x.preparation import types
    from b12x.sequence.short_conv import api

    query = SimpleNamespace(max_seqs=SEQS, channels=CHANNELS, state_length=STATE, max_tokens=TOKENS)
    state = SimpleNamespace(query=query)
    monkeypatch.setattr(types, "require_prepared", lambda plan, component, device: state)
    monkeypatch.setattr(api, "_pipeline", torch.ops.flare_test.short_conv)
    return SimpleNamespace(handle=5)


def _call(plan, x, conv_state):
    scratch = torch.empty(55304, dtype=torch.uint8)
    out = torch.empty_like(x)
    meta = torch.zeros(2, dtype=torch.int32)
    recurrent.run_short_conv(plan, scratch=scratch, x=x, conv_weight=torch.ones(CHANNELS, 4),
        query_start_loc=meta, state_slot_ids=torch.zeros(1, dtype=torch.int64),
        state_is_fresh=torch.zeros(1, dtype=torch.bool), num_accepted_tokens=meta[:1], num_seqs=meta[:1],
        num_tokens=meta[:1], conv_state=conv_state, out=out)
    return out


def test_private_operands_do_not_alias_the_scratch(prepared, monkeypatch):
    seen = {}
    from b12x.sequence.short_conv import api
    monkeypatch.setattr(api, "_pipeline", lambda *args: seen.update(gathered=args[11], ids=args[12]))
    _call(prepared, torch.randn(TOKENS, CHANNELS), torch.randn(4, CHANNELS, STATE).bfloat16())
    gathered, ids = seen["gathered"], seen["ids"]
    assert gathered.shape == (SEQS, CHANNELS, STATE) and gathered.dtype == torch.bfloat16
    assert ids.shape == (TOKENS,) and ids.dtype == torch.int32
    for tensor in (gathered, ids):
        assert tensor._base is None and tensor.storage_offset() == 0 and tensor.is_contiguous()


def test_compiled_short_conv_has_no_scratch_clone(prepared):
    x, conv_state = torch.randn(TOKENS, CHANNELS), torch.randn(4, CHANNELS, STATE).bfloat16()
    expected = _call(prepared, x, conv_state)
    torch._dynamo.reset()
    actual, (code, *_) = run_and_get_code(
        torch.compile(lambda: _call(prepared, x, conv_state) * 1, backend="inductor", fullgraph=True))
    assert torch.equal(actual, expected)
    call = code[code.index("def call("):]
    assert "clone" not in call and "view.dtype" not in call, call
