# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GPU microbench for DS41 TP3 decode lever 4 (attention 16 + 8 head grids): split counts per row count.

One GPU, nothing else running on it. Times b12x's prepared DSV41 BF16 compressed MLA decode exactly as a TP3 rank runs
it (24 heads, SWA 128 + indexed 512 candidates, page table, attention sink, provided output, remainder grid on a side
stream = B12X_MLA_DECODE_REMAINDER_STREAM=1) inside a CUDA graph of back-to-back calls, for:

  * splits S in {10 (production), 5, 4, 3, 2, 1}, forced through ignition/components/mla_decode_splits.py (the
    FLARE_DS41_MLA_DECODE_SPLITS hook; B12X_MLA_SM120_NUM_SPLITS does not reach DSV41 decode);
  * 16 heads (one grid: the lower bound for a one-launch 24-head CTA) at S = 10 and 5;
  * FP8 arithmetic with 8-head blocks (b12x's existing one-launch 3 x 8 path; numerics differ, timing reference only);
  * remainder stream off (serial grids) at S = 10, to price the stream itself.

Rows: the graph capture sizes seen in the 181718 traces (5, 9-10, 12-14, 18-19, 21-29, 32, 37-40, 80) plus 1.
KV gathers hit a 1.2 GB indexed pool through random pages (L2 misses as in serving), not the tuner's one-page fixture.
Each case also reports max |out - out(S=10)| (expect BF16 rounding only, <= ~2e-3) as a sanity check.

Usage (GPUs exclusive: stop the server first):
  CUDA_VISIBLE_DEVICES=0 B12X_OVERLAY=... PYTHONPATH=<worktree> B12X_COMPILE_CACHE_DIR=<scratch>/b12x \\
      TRITON_CACHE_DIR=<scratch>/triton python tp3-tools/attn_split_bench.py --out <dir>/attn-splits.tsv
  --quick: rows 5,13,23,32,40,80 and S 10,5,2 only (~4 min after compiles).
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace

import torch

ROWS_FULL = (1, 5, 9, 10, 13, 18, 19, 23, 26, 28, 29, 32, 37, 40, 80)
ROWS_QUICK = (5, 13, 23, 32, 40, 80)
SPLITS_FULL = (10, 5, 4, 3, 2, 1)
SPLITS_QUICK = (10, 5, 2)
SWA_WIDTH, INDEXED_WIDTH, SWA_PAGE, INDEXED_PAGE = 128, 512, 128, 64
POOL_PAGES, TABLE_WIDTH = 65536, 2048  # 64-token pages: 4.2M-token pool, 131K-token contexts per row
SM_SCALE = 512 ** -0.5


def build_inputs(rows, heads, device, generator):
    from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference

    values = (torch.randn((SWA_PAGE, 512), device=device, generator=generator) * 0.2).bfloat16()
    swa_page = pack_deepseek_v41_cache_reference(values, page_size=SWA_PAGE, cache_kind="swa")
    swa_pages = 2 * rows + 2
    swa = swa_page.repeat(swa_pages, 1).contiguous()
    values = (torch.randn((INDEXED_PAGE, 512), device=device, generator=generator) * 0.2).bfloat16()
    indexed_page = pack_deepseek_v41_cache_reference(values, page_size=INDEXED_PAGE, cache_kind="indexed")
    indexed = torch.empty((POOL_PAGES, indexed_page.shape[1]), dtype=torch.uint8, device=device)
    indexed.copy_(indexed_page.expand(POOL_PAGES, -1))
    q = (torch.randn((rows, heads, 512), device=device, generator=generator) * 0.2).bfloat16()
    # Each row's window: 128 consecutive slots in its own two pages (ring offset varies per row).
    base = (torch.arange(rows, device=device, dtype=torch.int32) * 2 + 1) * SWA_PAGE
    start = torch.randint(0, SWA_PAGE, (rows,), device=device, generator=generator, dtype=torch.int32)
    swa_indices = (base + start)[:, None] + torch.arange(SWA_WIDTH, device=device, dtype=torch.int32)[None]
    swa_lengths = torch.full((rows,), SWA_WIDTH, dtype=torch.int32, device=device)
    table = torch.randint(1, POOL_PAGES, (rows, TABLE_WIDTH), device=device, generator=generator, dtype=torch.int32)
    logical = torch.stack([torch.randperm(TABLE_WIDTH * INDEXED_PAGE, device=device, generator=generator)[:INDEXED_WIDTH]
                           for _ in range(rows)]).to(torch.int32)
    indexed_lengths = torch.full((rows,), INDEXED_WIDTH, dtype=torch.int32, device=device)
    sink = torch.zeros((heads,), dtype=torch.float32, device=device)
    return dict(q=q, swa=swa, indexed=indexed, swa_indices=swa_indices, swa_lengths=swa_lengths,
                indexed_indices=logical, indexed_lengths=indexed_lengths, table=table, sink=sink)


def prepare(session, x, *, arithmetic, override_fp8_h8):
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention.compressed_sparse_mla._tuning import TUNING
    from b12x.preparation import PreparedCall

    rows, heads = x["q"].shape[:2]
    output = torch.empty_like(x["q"])
    caps = mla.Caps(device=x["q"].device, num_q_heads=heads, max_q_rows=rows, max_width=SWA_WIDTH + INDEXED_WIDTH,
                    swa_width=SWA_WIDTH, indexed_width=INDEXED_WIDTH, max_page_table_width=TABLE_WIDTH,
                    swa_page_size=SWA_PAGE, indexed_page_size=INDEXED_PAGE, cache_format="deepseek_v41",
                    mode="decode", decode_row_capacity=rows, use_cuda_graph=True, compute_mode=arithmetic)
    invocation = mla.invocation_from_tensors(q=x["q"], swa_k_cache=x["swa"], indexed_k_cache=x["indexed"],
                                             attn_sink=x["sink"], out=output)
    plan = mla.plan(caps, invocation=invocation)
    if override_fp8_h8:
        config = replace(TUNING.configure(plan.query, device=session.device.identity).default,
                         v41_compute_mode="fp8", v41_heads_per_block=8)
        plan = mla.plan(caps, invocation=invocation, override=config)
    bind_args = dict(q=x["q"], swa_indices=x["swa_indices"], swa_lengths=x["swa_lengths"],
                     indexed_indices=x["indexed_indices"], indexed_lengths=x["indexed_lengths"],
                     indexed_page_table=x["table"])
    run_args = dict(swa_k_cache=x["swa"], indexed_k_cache=x["indexed"], attn_sink=x["sink"], sm_scale=SM_SCALE)

    def prepare_call(state):
        (spec,) = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=output.device)
        trial = torch.empty_like(output)
        binding = state.bind_for_preparation(scratch=scratch, **bind_args)
        return PreparedCall(run=lambda: state.run(binding, **run_args, out=trial), output=trial,
                            owners=(scratch, binding))

    session.prepare((plan.request(name="attn-split-bench", prepare_call=prepare_call,
                                  benchmark_call=prepare_call),))
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=output.device)
    binding = mla.bind(plan, scratch=scratch, **bind_args)

    def call():
        return mla.run(binding=binding, swa_k_cache=x["swa"], swa_page_size=SWA_PAGE, indexed_k_cache=x["indexed"],
                       indexed_page_size=INDEXED_PAGE, attn_sink=x["sink"], sm_scale=SM_SCALE, out=output)

    return plan, call, output, (scratch, binding)


def time_graph(call, *, calls=20, replays=60, warmup=10):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(calls):
                call()
    torch.cuda.synchronize()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / calls)
    samples.sort()
    return samples[len(samples) // 2], samples[len(samples) // 10]


def run_case(device, rows, heads, splits, variant, cache_dir):
    """variant: bf16 (production), fp8h8 (b12x one-launch 3 x 8 heads, FP8 arithmetic), serial (stream off)."""
    from b12x.preparation import PreparationSession
    from b12x.attention._shared.mla import kernel as mla_kernel
    from ignition.components import mla_decode_splits, mla_remainder_stream

    os.environ[mla_remainder_stream.ENV] = "0" if variant == "serial" else "1"
    mla_remainder_stream.install()
    hook = mla_decode_splits.install(((rows, splits),)) if splits is not None else None
    generator = torch.Generator(device=device).manual_seed(4118 + rows)
    x = build_inputs(rows, heads, device, generator)
    try:
        with PreparationSession(device=device, autotune=False, compile_workers=0, cache_dir=cache_dir) as session:
            plan, call, output, owners = prepare(session, x, arithmetic="auto" if variant == "fp8h8" else "bf16",
                                                 override_fp8_h8=variant == "fp8h8")
            plan_info = dict(mla_kernel.LAST_DECODE_PLAN)
            median, p10 = time_graph(call)
            call()
            torch.cuda.synchronize()
            result = output.float().clone()
    finally:
        mla_decode_splits.uninstall()
    return dict(rows=rows, heads=heads, forced_splits=splits, variant=variant, us=round(median, 3),
                us_p10=round(p10, 3), num_splits=plan_info.get("num_splits"),
                chunks_per_split=plan_info.get("chunks_per_split"), applied=hook.applied if hook else []), result


def cases(rows_list, splits_list):
    for rows in rows_list:
        for splits in splits_list:
            yield rows, 24, splits, "bf16"
        yield rows, 16, 10, "bf16"
        yield rows, 16, 5, "bf16"
        yield rows, 24, None, "fp8h8"
        yield rows, 24, 10, "serial"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cache-dir", default=None, help="b12x preparation cache (default: <out dir>/prep-cache)")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        sys.exit("run with exactly one visible GPU (CUDA_VISIBLE_DEVICES=N)")
    device = torch.device("cuda", 0)
    cache_dir = args.cache_dir or os.path.join(os.path.dirname(os.path.abspath(args.out)), "prep-cache")
    os.makedirs(cache_dir, exist_ok=True)
    rows_list, splits_list = (ROWS_QUICK, SPLITS_QUICK) if args.quick else (ROWS_FULL, SPLITS_FULL)
    reference = {}
    started = time.time()
    with open(args.out, "w") as handle:
        handle.write("rows\theads\tvariant\tforced_splits\tnum_splits\tchunks_per_split\tus_median\tus_p10\t"
                     "max_abs_diff_vs_S10\tapplied\n")
        for rows, heads, splits, variant in cases(rows_list, splits_list):
            info, out = run_case(device, rows, heads, splits, variant, cache_dir)
            key = (rows, heads)
            if variant == "bf16" and splits == 10:
                reference[key] = out
            diff = (out - reference[key]).abs().max().item() if key in reference else float("nan")
            handle.write(f"{rows}\t{heads}\t{variant}\t{splits}\t{info['num_splits']}\t{info['chunks_per_split']}\t"
                         f"{info['us']}\t{info['us_p10']}\t{diff:.3e}\t{json.dumps(info['applied'])}\n")
            handle.flush()
            print(json.dumps(dict(info, diff=diff, elapsed_s=round(time.time() - started))), flush=True)


if __name__ == "__main__":
    main()
