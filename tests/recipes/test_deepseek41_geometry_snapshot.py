# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek V4.1 attention geometry is realised once per forward, not once per reading layer.

Every attention layer reads the clamped positions, window lengths and compression lengths through opaque
mutating operators. Left as pointwise expressions, Inductor recomputes them into fresh buffers beside each
reader; the geometry snapshots them so the compiled forward writes each vector once (CPU Inductor).
"""

import re

import torch
from torch._inductor.lowering import register_lowering
from torch._inductor.utils import run_and_get_code

from ignition.components.paged_batch import Metadata
from ignition.preparation import compilation
from ignition.recipes.deepseek41.attention import AttentionIndices

TOKENS, READERS = 5, 6


@torch.library.custom_op("flare_test::geometry_reader", mutates_args=("out",))
def _reader(x: torch.Tensor, position: torch.Tensor, slots: torch.Tensor, lengths: torch.Tensor,
            compressed: torch.Tensor, index_lengths: torch.Tensor, out: torch.Tensor) -> None:
    out.copy_(x + position.sum() + slots.sum() + lengths.sum() + compressed.sum() + index_lengths.sum())


@_reader.register_fake
def _(*args):
    pass


def _metadata():
    m = Metadata.allocate(TOKENS, 2, 4, torch.device("cpu"))
    m.positions.copy_(torch.tensor([3, 200, -1, 700, 0]))
    m.requests.copy_(torch.tensor([0, 1, 0, 1, 0]))
    return m


def _forward(indices, m, x):
    geometry = indices(m)
    compressed = geometry.compressed(2)
    for _ in range(READERS):
        out = torch.empty_like(x)
        _reader(x, geometry.position, geometry.window_slots, geometry.window_lengths, compressed.lengths,
                compressed.index_lengths, out)
        x = out * 1
    return x


def _written_buffers(code):
    call = code[code.index("def call("):]
    return sum(len(re.findall(r"\bbuf\d+\b", line)) for line in call.splitlines()
               if re.search(r"cpp_fused\w*clamp_min\w*\(", line))


def test_geometry_vectors_are_written_once_per_forward():
    register_lowering(torch.ops.flare.snapshot.default)(compilation._snapshot_lowering)
    m = _metadata()
    indices = AttentionIndices(m, mode="decode", tokens=TOKENS, page_size=128, width=128, ratios=(2,))
    x = torch.randn(TOKENS)
    expected = _forward(indices, m, x)
    torch._dynamo.reset()
    actual, (code, *_) = run_and_get_code(
        torch.compile(lambda: _forward(indices, m, x), backend="inductor", fullgraph=True))
    assert torch.equal(actual, expected)
    # Five vectors read by six layers: written once each, not once per reader.
    assert 0 < _written_buffers(code) < 2 * 5, code[code.index("def call("):]
