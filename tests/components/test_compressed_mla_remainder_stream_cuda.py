# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Twenty-four-head BF16 decode: the concurrent remainder grid is bit-identical and (optionally) timed.

Uses ignition/components/mla_remainder_stream.py (no b12x overlay). One prepared
DS41-shaped decode (24 heads, 128-token window + mapped indexed pages, BF16 arithmetic) is captured twice in one
session: B12X_MLA_DECODE_REMAINDER_STREAM unset (today's serial 16 + 8 head grids) and =1 (8-head grid on a side
stream). Both graphs are replayed on the same inputs and must write the same bytes; both must match the reference.
FLARE_ATTN1_TIMING=1 also replays each graph FLARE_ATTN1_REPS times (default 500) and prints the median per replay.
"""

import os
import statistics

import pytest
import torch
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference, pack_deepseek_v41_cache_reference,
)

from ignition.components import mla_remainder_stream
from ignition.components.compressed_mla import PreparedCompressedMLA
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession


ENV = "B12X_MLA_DECODE_REMAINDER_STREAM"
pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


@pytest.fixture(autouse=True)
def hook():
    """Install the wrapper for this test; it reads the switch per call, so off = b12x's own serial launches."""
    from b12x.attention._shared.mla import kernel

    installed_before = isinstance(kernel._sparse_mla_decode_grid_flat_launch, mla_remainder_stream._RemainderStreamLaunch)
    mla_remainder_stream.install()
    yield
    if not installed_before:
        mla_remainder_stream.uninstall()


def _replay_us(graph, reps):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples)


@pytest.mark.parametrize("rows", [1, 6, 12, 21, 43])
@pytest.mark.parametrize("width", [512, 1024])
@torch.no_grad()
def test_remainder_stream_is_bit_identical(rows, width, monkeypatch):
    device = torch.device("cuda:0")
    gen = torch.Generator(device=device).manual_seed(41115)
    heads, window_page, page_size = 24, 136, 64
    logical_pages = width // page_size
    num_pages = logical_pages + 1
    window_values = (torch.randn(3 * window_page, 512, device=device, generator=gen) * .2).bfloat16()
    indexed_values = (torch.randn(num_pages * page_size, 512, device=device, generator=gen) * .2).bfloat16()
    window = pack_deepseek_v41_cache_reference(window_values, page_size=window_page, cache_kind="swa")
    indexed = pack_deepseek_v41_cache_reference(indexed_values, page_size=page_size, cache_kind="indexed")
    query = (torch.randn(rows, heads, 512, device=device, generator=gen) * .2).bfloat16()
    window_indices = (torch.arange(128, dtype=torch.int32, device=device) + window_page).expand(rows, -1).contiguous()
    indexed_indices = torch.arange(width, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
    window_lengths = torch.full((rows,), 128, dtype=torch.int32, device=device)
    indexed_lengths = torch.full((rows,), width, dtype=torch.int32, device=device)
    if rows > 1:
        window_lengths[-1], indexed_lengths[-1] = 17, width // 3
    table = torch.arange(1, num_pages, dtype=torch.int32, device=device).expand(rows, -1).contiguous()
    sink = torch.randn(heads, device=device, generator=gen)
    output = torch.empty(rows, heads, 512, device=device, dtype=torch.bfloat16)
    arena = ScratchArena(device)
    serial, concurrent = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()

    monkeypatch.delenv(ENV, raising=False)
    with RecordedPreparationSession(device=device, checkpoint="compressed-mla-remainder-stream",
            workload_schema=f"heads24-bf16-rows{rows}-w{width}", numerical_policy="checkpoint",
            compile_workers=2) as session:
        session.register_scratch(arena)
        attention = PreparedCompressedMLA(query, window, window_indices, window_lengths, sink, output,
            indexed=indexed, indexed_indices=indexed_indices, indexed_lengths=indexed_lengths,
            page_table=table, window_page_size=window_page, indexed_page_size=page_size,
            mode="decode", decode_row_capacity=rows, session=session, name="attention",
            workspace=arena, compute_mode="bf16")
        session.tune(show_progress=False)
        launch = attention.plan.prepared.state.launch
        assert not launch.native_dsv41_fp8 and launch.h_blocks_full == 1 and launch.rem_heads == 8
        attention()
        torch.cuda.synchronize()
        with session.capture():
            with torch.cuda.graph(serial):
                attention()
            monkeypatch.setenv(ENV, "1")
            with torch.cuda.graph(concurrent):
                attention()
            monkeypatch.delenv(ENV)
    assert 0 in mla_remainder_stream._STREAMS  # the concurrent capture really forked onto the side stream
    try:
        output.fill_(torch.nan)
        serial.replay()
        torch.cuda.synchronize()
        expected = output.clone()
        output.fill_(torch.nan)
        concurrent.replay()
        torch.cuda.synchronize()
        assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
        physical = (table.long().gather(1, (indexed_indices.long() // page_size).clamp(0, logical_pages - 1))
                    * page_size + indexed_indices.long() % page_size).int()
        reference = compressed_sparse_mla_reference(query, window, window_indices, window_lengths,
            sm_scale=512 ** -.5, attn_sink=sink, extra_k_cache=indexed, extra_indices=physical,
            extra_topk_lengths=indexed_lengths, swa_page_size=window_page, extra_page_size=page_size,
            cache_format="deepseek_v41")
        torch.testing.assert_close(output, reference, atol=.035, rtol=.035)
        for _ in range(3):  # repeat: concurrency must not make the result run-dependent
            concurrent.replay()
            torch.cuda.synchronize()
            assert torch.equal(output.view(torch.int16), expected.view(torch.int16))
        if os.environ.get("FLARE_ATTN1_TIMING") == "1":
            reps = int(os.environ.get("FLARE_ATTN1_REPS", "500"))
            for graph in (serial, concurrent):
                _replay_us(graph, 50)
            a, b = _replay_us(serial, reps), _replay_us(concurrent, reps)
            print(f"\nATTN1 rows={rows} width={width} splits={launch.num_splits} serial_us={a:.2f} "
                  f"concurrent_us={b:.2f} saved_us={a - b:.2f}")
    finally:
        serial.reset()
        concurrent.reset()
