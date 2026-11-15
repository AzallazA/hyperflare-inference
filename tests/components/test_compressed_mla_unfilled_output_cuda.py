# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compressed MLA writes every output element itself, so its functional output needs no zero fill.

Each replay starts from a poisoned (all-0xFF, NaN) output and must match a replay that started from zeros bit
for bit, for decode (split merge) and the prefill-like modes, with an idle row, a short row and both arithmetic
choices.
"""

import os

import pytest
import torch
from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference

from ignition.components.compressed_mla import PreparedCompressedMLA
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                              reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("compute_mode", ["auto", "bf16"])
@pytest.mark.parametrize("mode,rows", [("decode", 4), ("extend", 17), ("verify", 7), ("draft_extend", 7)])
@torch.no_grad()
def test_poisoned_output_replays_like_zeroed_output(mode, rows, compute_mode):
    device = torch.device("cuda:0")
    generator = torch.Generator(device=device).manual_seed(4119)
    source = torch.randn((128, 512), device=device, generator=generator).bfloat16()
    cache = pack_deepseek_v41_cache_reference(source, page_size=128, cache_kind="swa")
    query = torch.randn((rows, 24, 512), device=device, generator=generator).bfloat16()
    indices = torch.arange(128, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
    lengths = torch.full((rows,), 128, dtype=torch.int32, device=device)
    lengths[0] = 3
    lengths[-1] = 0
    sink = torch.randn(24, device=device, generator=generator)
    output = torch.empty_like(query)
    arena = ScratchArena(device)
    graph = torch.cuda.CUDAGraph()
    with RecordedPreparationSession(device=device, checkpoint="compressed-mla-unfilled-output",
            workload_schema="heads24", numerical_policy="checkpoint", compile_workers=2) as session:
        session.register_scratch(arena)
        operation = PreparedCompressedMLA(query, cache, indices, lengths, sink, output,
            mode=mode, session=session, name="attention", workspace=arena, compute_mode=compute_mode)
        session.tune(show_progress=False)
        operation()
        torch.cuda.synchronize()
        with session.capture(), torch.cuda.graph(graph):
            operation()
        try:
            output.zero_()
            graph.replay()
            torch.cuda.synchronize()
            zeroed = output.clone()
            assert torch.isfinite(zeroed).all() and not zeroed[-1].count_nonzero()
            for _ in range(2):
                output.view(torch.uint8).fill_(0xFF)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(output.view(torch.int16), zeroed.view(torch.int16))
        finally:
            graph.reset()
