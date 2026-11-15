# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The functional compressed MLA hands b12x an unfilled output; b12x writes every element (see the CUDA test)."""

import torch

from ignition.components import compressed_mla


def test_functional_output_reaches_the_operator_unfilled(monkeypatch):
    rows, heads = 3, 24
    owner = object.__new__(compressed_mla.PreparedCompressedMLA)
    owner.inputs = None
    owner.output_layout = (rows, heads, 512), (heads * 512, 512, 1), torch.bfloat16
    owner.scratch_layout = (((64,), torch.uint8),)
    owner.window, owner.indexed, owner.sink, owner.scale = None, None, None, 1.0
    owner.plan = type("Plan", (), {"handle": 9})()
    empty_strided, seen = torch.empty_strided, {}

    def poisoned(*args, **kwargs):
        tensor = empty_strided(*args, **kwargs)
        tensor.view(torch.uint8).fill_(0xFF)
        return tensor

    def attend(*args):
        seen["before"] = args[9].view(torch.uint8).clone()
        args[9].fill_(1.5)

    monkeypatch.setattr(torch, "empty_strided", poisoned)
    monkeypatch.setattr(compressed_mla, "_attend", attend)
    output = owner(torch.zeros(rows, heads, 512, dtype=torch.bfloat16), None, None)
    assert (seen["before"] == 0xFF).all()
    assert output.shape == (rows, heads, 512) and (output == 1.5).all()
