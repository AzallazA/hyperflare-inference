# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Twenty-four-head one-launch BF16 decode (FLARE_DS41_MLA_H24): bit-identical to b12x's 16 + 8 grids.

One DS41-shaped decode (24 heads, 128-token window + mapped indexed pages, BF16 arithmetic, attention sink, the last
row with short window/indexed lengths so some splits are empty) is prepared and captured twice, in two sessions:
b12x's 16-head + 8-head grids (remainder on the side stream, the production setting) and the H24 hook selecting
these rows. Both at the same forced split count (FLARE_DS41_MLA_DECODE_SPLITS hook; None = b12x's 10). The H24
preparation must have lowered to our program plus the inert remainder launcher, and both graphs must write the same
bytes (and match the reference). FLARE_H24_TIMING=1 also prints the median replay time of both graphs.

Review fixes (REVIEW-FABLE-H24.md): the hooks are selected through their environment variables and installed with
install_from_environment(), so the session's spawned compile workers plan the same programs (B1); tuning is off (one
split geometry, B2); each arm keeps its own scratch arena alive and every replay starts from poisoned split partials
(mid_out NaN, mid_lse 0.0), so a missing H24 write cannot be hidden by the other arm's partials (B3).
"""

import os
import statistics
from contextlib import contextmanager

import pytest
import torch
from b12x.attention._shared.mla.compressed_reference import (
    compressed_sparse_mla_reference, pack_deepseek_v41_cache_reference,
)

from ignition.components import mla_decode_splits, mla_h24_decode, mla_remainder_stream
from ignition.components.compressed_mla import PreparedCompressedMLA
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession


pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device")


@pytest.fixture(autouse=True)
def hooks(monkeypatch):
    """Remainder stream installed and on (production), no H24/split hook outside the arm that installs it."""
    from b12x.attention._shared.mla import kernel

    mla_h24_decode.uninstall()
    mla_decode_splits.uninstall()
    installed_before = isinstance(kernel._sparse_mla_decode_grid_flat_launch,
                                  mla_remainder_stream._RemainderStreamLaunch)
    mla_remainder_stream.install()
    monkeypatch.setenv(mla_remainder_stream.ENV, "1")
    yield
    mla_h24_decode.uninstall()
    mla_decode_splits.uninstall()
    if not installed_before:
        mla_remainder_stream.uninstall()


def _inputs(rows, width, device):
    gen = torch.Generator(device=device).manual_seed(41118 + rows)
    heads, window_page, page_size = 24, 136, 64
    logical_pages = width // page_size
    num_pages = logical_pages + 1
    window_values = (torch.randn(3 * window_page, 512, device=device, generator=gen) * .2).bfloat16()
    indexed_values = (torch.randn(num_pages * page_size, 512, device=device, generator=gen) * .2).bfloat16()
    data = dict(
        window=pack_deepseek_v41_cache_reference(window_values, page_size=window_page, cache_kind="swa"),
        indexed=pack_deepseek_v41_cache_reference(indexed_values, page_size=page_size, cache_kind="indexed"),
        query=(torch.randn(rows, heads, 512, device=device, generator=gen) * .2).bfloat16(),
        window_indices=(torch.arange(128, dtype=torch.int32, device=device) + window_page).expand(rows, -1).contiguous(),
        indexed_indices=torch.arange(width, dtype=torch.int32, device=device).expand(rows, -1).contiguous(),
        window_lengths=torch.full((rows,), 128, dtype=torch.int32, device=device),
        indexed_lengths=torch.full((rows,), width, dtype=torch.int32, device=device),
        table=torch.arange(1, num_pages, dtype=torch.int32, device=device).expand(rows, -1).contiguous(),
        sink=torch.randn(heads, device=device, generator=gen),
        window_page=window_page, page_size=page_size, logical_pages=logical_pages)
    if rows > 1:
        data["window_lengths"][-1], data["indexed_lengths"][-1] = 17, width // 3
    return data


@contextmanager
def _environment(values):
    """Set/delete variables for one arm (spawned compile workers inherit them) and restore them afterwards."""
    saved = {name: os.environ.get(name) for name in values}
    try:
        for name, value in values.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@contextmanager
def _record_partials(sink):
    """Outermost passive wrapper on b12x's grid launcher: remembers the split partial views (mid_out, mid_lse)."""
    from b12x.attention._shared.mla import kernel

    inner = kernel._sparse_mla_decode_grid_flat_launch

    def recording(*args, **kwargs):
        if hasattr(args[3], "is_cuda") and args[3].is_cuda:
            sink.append((args[3], args[4]))
        return inner(*args, **kwargs)

    kernel._sparse_mla_decode_grid_flat_launch = recording
    try:
        yield
    finally:
        kernel._sparse_mla_decode_grid_flat_launch = inner


def _capture(data, rows, width, splits, arm, output):
    device = output.device
    values = {mla_decode_splits.ENV: None if splits is None else f"{rows}:{splits}",
              mla_h24_decode.ENV: f"{rows}:1" if arm == "h24" else None}
    graph = torch.cuda.CUDAGraph()
    arena = ScratchArena(device)
    partials = []
    with _environment(values):
        mla_decode_splits.install_from_environment()
        hook = mla_h24_decode.install_from_environment()
    assert (hook is not None) == (arm == "h24")
    try:
        with _environment(values), RecordedPreparationSession(device=device, checkpoint="mla-h24-decode",
                workload_schema=f"heads24-bf16-rows{rows}-w{width}-s{splits}-{arm}", numerical_policy="checkpoint",
                compile_workers=2, autotune_effort="off") as session:
            session.register_scratch(arena)
            attention = PreparedCompressedMLA(data["query"], data["window"], data["window_indices"],
                data["window_lengths"], data["sink"], output, indexed=data["indexed"],
                indexed_indices=data["indexed_indices"], indexed_lengths=data["indexed_lengths"],
                page_table=data["table"], window_page_size=data["window_page"], indexed_page_size=data["page_size"],
                mode="decode", decode_row_capacity=rows, session=session, name="attention", workspace=arena,
                compute_mode="bf16")
            session.tune(show_progress=False)
            launch = attention.plan.prepared.state.launch
            assert not launch.native_dsv41_fp8 and launch.h_blocks_full == 1 and launch.rem_heads == 8
            if splits is not None:
                assert launch.num_splits == splits
            if hook is not None:
                assert mla_h24_decode._is_h24_program(launch.grid_programs[0]), launch.grid_programs
                assert launch.grid_programs[1] is mla_h24_decode.REMAINDER_DONE
                assert hook.applied and all(entry == (rows, launch.num_splits) for entry in hook.applied), hook.applied
            else:
                assert not mla_h24_decode._is_h24_program(launch.grid_programs[0])
            attention()
            torch.cuda.synchronize()
            with session.capture(), _record_partials(partials):
                with torch.cuda.graph(graph):
                    attention()
    finally:
        mla_h24_decode.uninstall()
        mla_decode_splits.uninstall()
    assert partials, "capture recorded no split partial views"
    if hook is not None:
        assert hook.replays >= 1, "the captured graph did not run the prepared H24 program"
    # Keep the arena and attention alive with the graph: each arm owns its scratch memory.
    return graph, launch, (arena, attention, partials)


def _poison(owners):
    for mid_out, mid_lse in owners[2]:
        mid_out.fill_(torch.nan)
        mid_lse.fill_(0.0)


def _replay(graph, output, owners):
    output.fill_(torch.nan)
    _poison(owners)
    graph.replay()
    torch.cuda.synchronize()
    return output.clone()


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


CASES = [(rows, splits) for rows in (13, 43) for splits in (None, 5, 3, 2, 1)] + \
        [(rows, splits) for rows in (1, 6, 23) for splits in (None, 5)]


@pytest.mark.parametrize("rows,splits", CASES)
@torch.no_grad()
def test_h24_is_bit_identical_to_16_plus_8(rows, splits):
    device = torch.device("cuda:0")
    width = 512
    data = _inputs(rows, width, device)
    output = torch.empty(rows, 24, 512, device=device, dtype=torch.bfloat16)
    base, base_launch, base_owners = _capture(data, rows, width, splits, "base", output)
    fused, fused_launch, fused_owners = _capture(data, rows, width, splits, "h24", output)
    try:
        assert base_launch.num_splits == fused_launch.num_splits
        assert base_owners[0].tensor.data_ptr() != fused_owners[0].tensor.data_ptr()
        expected = _replay(base, output, base_owners)
        assert not expected.isnan().any(), "base 16 + 8 output read a poisoned partial"
        got = _replay(fused, output, fused_owners)
        assert torch.equal(got.view(torch.int16), expected.view(torch.int16)), (
            f"max |diff| {(got.float() - expected.float()).abs().max().item():.3e}")
        for _ in range(3):  # run-to-run stable
            assert torch.equal(_replay(fused, output, fused_owners).view(torch.int16), expected.view(torch.int16))
        physical = (data["table"].long().gather(1, (data["indexed_indices"].long() // data["page_size"]).clamp(
            0, data["logical_pages"] - 1)) * data["page_size"] + data["indexed_indices"].long() % data["page_size"]).int()
        reference = compressed_sparse_mla_reference(data["query"], data["window"], data["window_indices"],
            data["window_lengths"], sm_scale=512 ** -.5, attn_sink=data["sink"], extra_k_cache=data["indexed"],
            extra_indices=physical, extra_topk_lengths=data["indexed_lengths"], swa_page_size=data["window_page"],
            extra_page_size=data["page_size"], cache_format="deepseek_v41")
        torch.testing.assert_close(got, reference, atol=.035, rtol=.035)
        if os.environ.get("FLARE_H24_TIMING") == "1":
            reps = int(os.environ.get("FLARE_H24_REPS", "500"))
            for graph in (base, fused):
                _replay_us(graph, 50)
            a, b = _replay_us(base, reps), _replay_us(fused, reps)
            print(f"\nH24 rows={rows} splits={base_launch.num_splits} base_us={a:.2f} h24_us={b:.2f} "
                  f"saved_us={a - b:.2f}")
    finally:
        base.reset()
        fused.reset()
