# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 fork (DS41 tp3.markov_tile_rows): all-rows-per-tile Markov proposal scores are bit-identical.

``_score_tiles_rows`` scores ROWS request rows per program with one read of each head tile; ``_score_tiles`` scores one
(tile, row) per program. Runs both in Triton's CPU interpreter for batches 1/4/8/16, draft steps 0-4, direct and staged
embedding reads, BF16 and row-scaled FP8 heads, several ROWS (including ones that do not divide the batch), with
penalties, a grammar row and predecessors outside the vocabulary, and requires every stored score and every partial
record to be bit-identical, and every other byte of the probability buffer untouched.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import triton
import triton.language as tl


def test_tile_rows_score_bit_identically_in_the_cpu_interpreter():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH")))))
    completed = subprocess.run([sys.executable, "-m", "tests.components.test_markov_tile_rows_interpret"],
                               cwd=root, env=env, capture_output=True, text=True, timeout=3600)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "markov tile rows ok 80" in completed.stdout, completed.stdout + completed.stderr
    assert "markov tile rows seven-column ok 8" in completed.stdout, completed.stdout + completed.stderr


def test_tile_rows_is_off_by_default_and_recorded_only_when_set():
    from ignition.recipes.deepseek41.tp3_config import TP3Config

    assert TP3Config().markov_tile_rows is None
    assert "markov_tile_rows" not in TP3Config().to_metadata()
    on = TP3Config.from_environment({"FLARE_DS41_MARKOV_TILE_ROWS": "4"})
    assert on.markov_tile_rows == 4 and on.to_metadata()["markov_tile_rows"] == 4
    with pytest.raises(ValueError, match="markov_tile_rows"):
        TP3Config(markov_tile_rows=0)


ROWS_FOR_BATCH = {1: (1, 4), 4: (1, 3, 4), 8: (2, 3, 8), 16: (4, 5, 16)}


def _inputs(b, fp8, generator):
    from ignition.components.sampling import RUNNING

    width, drafts, vocab, local, rank, capacity = 6, 5, 700, 600, 256, b + 2  # rank 256 as DS41 (4 chunks)
    offset = torch.tensor([64], dtype=torch.int32)  # this shard holds global ids 64..663
    logits = torch.randn((b, width, local), generator=generator).to(torch.bfloat16)
    table = torch.randn((vocab, rank), generator=generator).to(torch.bfloat16)
    if fp8:
        head = (torch.randn((local, rank), generator=generator) * 4).to(torch.float8_e4m3fn)
        scale = torch.rand((local,), generator=generator) * 0.1 + 0.01
    else:
        head = torch.randn((local, rank), generator=generator).to(torch.bfloat16)
        scale = torch.ones((1,))
    pending = torch.randint(-3, vocab + 40, (b,), generator=generator, dtype=torch.int64)
    candidates = torch.randint(-2, vocab + 2, (b, drafts), generator=generator, dtype=torch.int64)
    histogram = torch.randint(0, 3, (capacity, vocab), generator=generator, dtype=torch.int32)
    slots = torch.randperm(capacity, generator=generator)[:b].to(torch.int32)
    generations = torch.full((b,), 7, dtype=torch.int64)
    live = torch.tensor([b], dtype=torch.int32)
    generation = torch.full((capacity,), 7, dtype=torch.int64)
    status = torch.full((capacity,), RUNNING, dtype=torch.int32)
    params = torch.tensor([[0.7, 1, 0, 1.1, 0.2, 0.05]] * capacity, dtype=torch.float32)
    params[0, 3:] = torch.tensor([1.0, 0.0, 0.0])  # one slot without penalties
    params[1, 0] = 0.0  # one greedy slot (temperature 0 -> divide by 1)
    top_k = torch.zeros((capacity,), dtype=torch.int32)
    grammar = torch.randint(-2**31, 2**31 - 1, (capacity, triton.cdiv(vocab, 32)), generator=generator,
                            dtype=torch.int64).to(torch.int32)
    grammar_enabled = torch.zeros((capacity,), dtype=torch.int32)
    grammar_enabled[int(slots[0])] = 1  # first row grammar-masked
    seeds = torch.arange(b, dtype=torch.int64) + 11
    offsets = torch.arange(b, dtype=torch.int64) * 3
    return dict(width=width, drafts=drafts, vocab=vocab, local=local, rank=rank, capacity=capacity, offset=offset,
                logits=logits, table=table, head=head, scale=scale, pending=pending, candidates=candidates,
                histogram=histogram, slots=slots, generations=generations, live=live, generation=generation,
                status=status, params=params.flatten(), top_k=top_k, grammar=grammar,
                grammar_enabled=grammar_enabled, seeds=seeds, offsets=offsets, running=RUNNING)


def _run(x, b, step, fp8, staged, rows, parameters=6):
    from ignition.components import vocabulary_sampling as v

    block = 256
    blocks = triton.cdiv(x["local"], block)
    probabilities = torch.full((b, x["width"], x["local"]), -7.0)
    partials = torch.full((b, 5, blocks), -9.0)
    embedding = x["table"]
    if staged:
        embedding = torch.full((b, x["rank"]), float("nan"), dtype=torch.bfloat16)
        v._stage_markov_rows[(b,)](x["table"], x["pending"], x["candidates"], embedding, DRAFTS=x["drafts"],
                                   VOCAB=x["vocab"], RANK=x["rank"], RANK_PAD=triton.next_power_of_2(x["rank"]),
                                   STEP=step)
    args = (x["offset"], x["logits"], probabilities, partials, embedding, x["head"], x["scale"], x["pending"],
            x["candidates"], x["histogram"], x["slots"], x["generations"], x["live"], x["generation"], x["status"],
            x["params"], x["top_k"], x["grammar"], x["grammar_enabled"], x["seeds"], x["offsets"],
            x["logits"].stride(0), x["logits"].stride(1), x["histogram"].stride(0), x["grammar"].stride(0))
    common = dict(LOCAL=x["local"], VOCAB=x["vocab"], WIDTH=x["width"], DRAFTS=x["drafts"], PREFIX=step,
                  BLOCKS=blocks, CAPACITY=x["capacity"], BATCH=b, RUNNING=x["running"], RANK=x["rank"], FP8=fp8,
                  STEP=step, BLOCK=block, STAGED=staged, PARAMETERS=parameters)
    if rows is None:
        v._score_tiles[(blocks, b)](*args, **common, MARKOV=True, SAMPLE=True)
    else:
        v._score_tiles_rows[(blocks, triton.cdiv(b, rows))](*args, **common, ROWS=rows)
    return probabilities, partials


def _case(b, step, fp8, staged, x):
    base = _run(x, b, step, fp8, staged, None)
    for rows in ROWS_FOR_BATCH[b]:
        new = _run(x, b, step, fp8, staged, rows)
        for name, a, c in zip(("probabilities", "partials"), base, new):
            assert torch.equal(a.view(torch.int32), c.view(torch.int32)), (
                f"b {b} step {step} fp8 {fp8} staged {staged} rows {rows}: {name} differ")
    # Not vacuous: the step's scores were stored (other steps keep the poison) and the records were written.
    assert not torch.equal(base[0][:, step], torch.full_like(base[0][:, step], -7.0))
    assert torch.equal(base[0][:, :step], torch.full_like(base[0][:, :step], -7.0))
    assert not (base[1] == -9.0).any()
    # Not vacuous: the Markov bias moves the scores (a zero table gives others).
    zero = dict(x, table=torch.zeros_like(x["table"]))
    assert not torch.equal(_run(zero, b, step, fp8, staged, ROWS_FOR_BATCH[b][-1])[0][:, step], base[0][:, step])


def _seven_columns(x):
    """Steering cores (PR #404 port) carry a seventh request parameter column; the sampling values are unchanged."""
    params = x["params"].view(x["capacity"], 6)
    scale = torch.linspace(0.0, 1.0, x["capacity"]).unsqueeze(1)
    return dict(x, params=torch.cat((params, scale), 1).contiguous().flatten())


def _case_seven(b, step, staged, x):
    """Both Markov score kernels read slot parameters with the request-state stride: a 7-column state scores exactly
    as its 6-column twin (before the port, _score_tiles_rows read a fixed stride of 6)."""
    seven = _seven_columns(x)
    for rows in (None, *ROWS_FOR_BATCH[b]):
        a = _run(x, b, step, False, staged, rows)
        c = _run(seven, b, step, False, staged, rows, parameters=7)
        for name, u, w in zip(("probabilities", "partials"), a, c):
            assert torch.equal(u.view(torch.int32), w.view(torch.int32)), (
                f"seven columns b {b} step {step} staged {staged} rows {rows}: {name} differ")


@triton.jit
def _noise(Seed, Offset, token):
    return ((token * 7 + Seed * 3 + Offset) % 13).to(tl.float32) * 0.01


def _interpretable_noise():
    """libdevice.log1p (Gumbel noise) has no CPU-interpreter lowering; both kernels draw the same stand-in noise."""
    from ignition.components import vocabulary_sampling as v

    v._token_noise = _noise


def main():
    _interpretable_noise()
    generator = torch.Generator().manual_seed(1118)
    count = 0
    for b in (1, 4, 8, 16):
        for fp8 in (False, True):
            x = _inputs(b, fp8, generator)
            for step in range(5):
                for staged in (False, True):
                    _case(b, step, fp8, staged, x)
                    count += 1
        print(f"batch {b} ok", flush=True)
    print(f"markov tile rows ok {count}")
    seven = 0
    for b in (4, 16):
        x = _inputs(b, False, generator)
        for step in (0, 2):
            for staged in (False, True):
                _case_seven(b, step, staged, x)
                seven += 1
    print(f"markov tile rows seven-column ok {seven}")


if __name__ == "__main__":
    main()
