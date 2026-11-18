# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GPU microbench + bit-identity check for DS41 TP3 decode lever 5 (Markov proposal score tiles, all rows per tile).

One GPU, nothing else running on it. DS41 TP3 rank shapes: 43,136-column vocabulary shard (169 tiles of 256), rank-256
row-scaled FP8 Markov head, BF16 embedding 129,280 x 256 (VRAM; with --host the table is in pinned host memory as
tp3.markov_embed_host), staged predecessor rows (tp3.markov_stage_rows, export 4 default) and direct reads, penalties on.
For batches 1/4/8/16 and draft steps 0-4 it times _score_tiles (production) and _score_tiles_rows for ROWS in
{1, 2, 4, 8, 16} (<= batch) inside a CUDA graph (5 steps = one draft round, stage + score only) and requires
probabilities and partial records to be bit-identical to production with the real Gumbel noise.

Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<worktree> TRITON_CACHE_DIR=<scratch>/triton \\
           python tp3-tools/markov_tile_bench.py --out <dir>/markov-tiles.tsv [--host]
About 2 minutes including Triton compiles.
"""

import argparse
import sys

import torch
import triton

LOCAL, VOCAB, RANK, BLOCK, WIDTH, DRAFTS = 43136, 129280, 256, 256, 6, 5


def build(b, device, host, generator):
    from ignition.components.sampling import RUNNING

    capacity = 32
    g = generator
    x = dict(
        offset=torch.tensor([LOCAL], dtype=torch.int32, device=device),  # rank 1's shard
        logits=(torch.randn((b, WIDTH, LOCAL), device=device, generator=g) * 2).to(torch.bfloat16),
        head=(torch.randn((LOCAL, RANK), device=device, generator=g) * 4).to(torch.float8_e4m3fn),
        scale=torch.rand((LOCAL,), device=device, generator=g) * 0.1 + 0.01,
        pending=torch.randint(0, VOCAB, (b,), device=device, generator=g, dtype=torch.int64),
        candidates=torch.randint(0, VOCAB, (b, DRAFTS), device=device, generator=g, dtype=torch.int64),
        histogram=torch.randint(0, 2, (capacity, VOCAB), device=device, generator=g, dtype=torch.int32),
        slots=torch.randperm(capacity, device=device, generator=g)[:b].to(torch.int32),
        generations=torch.full((b,), 7, dtype=torch.int64, device=device),
        live=torch.tensor([b], dtype=torch.int32, device=device),
        generation=torch.full((capacity,), 7, dtype=torch.int64, device=device),
        status=torch.full((capacity,), RUNNING, dtype=torch.int32, device=device),
        params=torch.tensor([[1.0, 1, 0, 1.05, 0.0, 0.0]] * capacity, dtype=torch.float32, device=device).flatten(),
        top_k=torch.zeros((capacity,), dtype=torch.int32, device=device),
        grammar=torch.zeros((capacity, triton.cdiv(VOCAB, 32)), dtype=torch.int32, device=device),
        grammar_enabled=torch.zeros((capacity,), dtype=torch.int32, device=device),
        seeds=torch.arange(b, dtype=torch.int64, device=device) + 11,
        offsets=torch.arange(b, dtype=torch.int64, device=device) * 4,
        running=RUNNING, capacity=capacity,
    )
    table = (torch.randn((VOCAB, RANK), device=device, generator=g) * 0.5).to(torch.bfloat16)
    x["table"] = table.cpu().pin_memory() if host else table
    x["staged"] = torch.empty((b, RANK), dtype=torch.bfloat16, device=device)
    return x


def launches(x, b, step, staged, rows, probabilities, partials):
    from ignition.components import vocabulary_sampling as v

    blocks = triton.cdiv(LOCAL, BLOCK)
    out = []
    if staged:
        out.append(lambda: v._stage_markov_rows[(b,)](x["table"], x["pending"], x["candidates"], x["staged"],
                   DRAFTS=DRAFTS, VOCAB=VOCAB, RANK=RANK, RANK_PAD=RANK, STEP=step, num_warps=1))
    embedding = x["staged"] if staged else x["table"]
    args = (x["offset"], x["logits"], probabilities, partials, embedding, x["head"], x["scale"], x["pending"],
            x["candidates"], x["histogram"], x["slots"], x["generations"], x["live"], x["generation"], x["status"],
            x["params"], x["top_k"], x["grammar"], x["grammar_enabled"], x["seeds"], x["offsets"],
            x["logits"].stride(0), x["logits"].stride(1), x["histogram"].stride(0), x["grammar"].stride(0))
    common = dict(LOCAL=LOCAL, VOCAB=VOCAB, WIDTH=WIDTH, DRAFTS=DRAFTS, PREFIX=step, BLOCKS=blocks,
                  CAPACITY=x["capacity"], BATCH=b, RUNNING=x["running"], RANK=RANK, FP8=True, STEP=step, BLOCK=BLOCK,
                  STAGED=staged, num_warps=4, enable_fp_fusion=False)
    if rows is None:
        out.append(lambda: v._score_tiles[(blocks, b)](*args, **common, MARKOV=True, SAMPLE=True))
    else:
        out.append(lambda: v._score_tiles_rows[(blocks, triton.cdiv(b, rows))](*args, **common, ROWS=rows))
    return out


def measure(x, b, staged, rows, replays=200):
    probabilities = torch.full((b, WIDTH, LOCAL), -7.0, device="cuda")
    partials = torch.full((b, 5, triton.cdiv(LOCAL, BLOCK)), -9.0, device="cuda")
    ops = [op for step in range(5) for op in launches(x, b, step, staged, rows, probabilities, partials)]
    for op in ops:
        op()  # compile + produce results
    torch.cuda.synchronize()
    result = (probabilities.clone(), partials.clone())
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        for op in ops:
            op()
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(replays):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2], result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--host", action="store_true", help="Markov table in pinned host memory (markov_embed_host)")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1:
        sys.exit("run with exactly one visible GPU (CUDA_VISIBLE_DEVICES=N)")
    generator = torch.Generator(device="cuda").manual_seed(1118)
    failures = 0
    with open(args.out, "w") as handle:
        handle.write("batch\tstaged\trows_per_program\tus_per_round_5_steps\tus_vs_production\tbit_identical\n")
        for b in (1, 4, 8, 16):
            x = build(b, "cuda", args.host, generator)
            for staged in (True, False):
                base_us, base = measure(x, b, staged, None)
                handle.write(f"{b}\t{int(staged)}\tproduction\t{base_us:.2f}\t0\t1\n")
                for rows in (r for r in (1, 2, 4, 8, 16) if r <= b):
                    us, result = measure(x, b, staged, rows)
                    same = all(torch.equal(a.view(torch.int32), c.view(torch.int32)) for a, c in zip(base, result))
                    failures += not same
                    handle.write(f"{b}\t{int(staged)}\t{rows}\t{us:.2f}\t{us - base_us:+.2f}\t{int(same)}\n")
                    handle.flush()
                    print(b, staged, rows, f"{base_us:.1f} -> {us:.1f} us", "identical" if same else "DIFFERS",
                          flush=True)
    if failures:
        sys.exit(f"{failures} configurations differ from production")


if __name__ == "__main__":
    main()
