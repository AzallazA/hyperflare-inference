# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 fork (DS41 tp3.markov_stage_rows): staged Markov predecessor rows give bit-identical proposal scores.

``_stage_markov_rows`` copies each request's predecessor row of the Markov embedding into a [rows, rank] buffer once
per draft step; ``_score_tiles(STAGED=True)`` then reads that copy instead of the table. Runs both kernels in Triton's
CPU interpreter on the real ``_score_tiles`` (every argument, sampling step 0..2, BF16 and row-scaled FP8 heads,
predecessors outside the vocabulary, penalties on) and requires the stored scores and every partial record to be
bit-identical to the direct read, with the staging buffer poisoned before each step.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


def test_staged_markov_rows_score_bit_identically_in_the_cpu_interpreter():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH")))))
    completed = subprocess.run([sys.executable, "-m", "tests.components.test_markov_stage_rows_interpret"],
                               cwd=root, env=env, capture_output=True, text=True, timeout=900)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "markov stage ok 12" in completed.stdout, completed.stdout + completed.stderr


def test_stage_rows_is_off_by_default_and_needs_the_parked_table():
    from ignition.recipes.deepseek41.tp3_config import TP3Config

    assert TP3Config().markov_stage_rows is False
    assert "markov_stage_rows" not in TP3Config().to_metadata()  # recorded only when on: default records unchanged
    on = TP3Config.from_environment({"FLARE_DS41_MARKOV_EMBED_HOST": "1", "FLARE_DS41_MARKOV_STAGE_ROWS": "1"})
    assert on.markov_stage_rows is True and on.to_metadata()["markov_stage_rows"] is True
    import pytest

    with pytest.raises(ValueError, match="needs markov_embed_host"):
        TP3Config.from_environment({"FLARE_DS41_MARKOV_STAGE_ROWS": "1"})
    with pytest.raises(TypeError):
        TP3Config(markov_embed_host=True, markov_stage_rows=1)


def _case(step, fp8, generator):
    import triton

    from ignition.components import vocabulary_sampling as v
    from ignition.components.sampling import RUNNING

    b, width, drafts, vocab, local, rank, capacity, block = 3, 4, 4, 700, 600, 128, 4, 256
    blocks = triton.cdiv(local, block)
    offset = torch.tensor([64], dtype=torch.int32)  # this shard holds global ids 64..663
    logits = torch.randn((b, width, local), generator=generator).to(torch.bfloat16)
    table = torch.randn((vocab, rank), generator=generator).to(torch.bfloat16)
    if fp8:
        head = (torch.randn((local, rank), generator=generator) * 4).to(torch.float8_e4m3fn)
        scale = torch.rand((local,), generator=generator) * 0.1 + 0.01
    else:
        head = torch.randn((local, rank), generator=generator).to(torch.bfloat16)
        scale = torch.ones((1,))
    pending = torch.tensor([5, -3, vocab + 40], dtype=torch.int64)  # clamped to 0 and vocab - 1, as the direct read
    candidates = torch.randint(-2, vocab + 2, (b, drafts), generator=generator, dtype=torch.int64)
    histogram = torch.randint(0, 3, (capacity, vocab), generator=generator, dtype=torch.int32)
    slots = torch.tensor([2, 0, 3], dtype=torch.int32)
    generations = torch.full((b,), 7, dtype=torch.int64)
    live = torch.tensor([b], dtype=torch.int32)
    generation = torch.full((capacity,), 7, dtype=torch.int64)
    status = torch.full((capacity,), RUNNING, dtype=torch.int32)
    params = torch.tensor([[0.7, 1, 0, 1.1, 0.2, 0.05]] * capacity, dtype=torch.float32).flatten()
    top_k = torch.zeros((capacity,), dtype=torch.int32)
    grammar = torch.zeros((capacity, triton.cdiv(vocab, 32)), dtype=torch.int32)
    grammar_enabled = torch.zeros((capacity,), dtype=torch.int32)
    seeds = torch.arange(b, dtype=torch.int64) + 11
    offsets = torch.zeros((b,), dtype=torch.int64)
    out = {}
    for staged_mode in (False, True, "zero"):
        probabilities = torch.full((b, width, local), -7.0)
        partials = torch.full((b, 5, blocks), -9.0)
        embedding = torch.zeros_like(table) if staged_mode == "zero" else table
        if staged_mode is True:
            staged = torch.full((b, rank), float("nan"), dtype=torch.bfloat16)  # poisoned: every byte must be written
            v._stage_markov_rows[(b,)](table, pending, candidates, staged, DRAFTS=drafts, VOCAB=vocab, RANK=rank,
                                       RANK_PAD=triton.next_power_of_2(rank), STEP=step)
            predecessor = (pending if step == 0 else candidates[:, step - 1]).clamp(0, vocab - 1)
            assert torch.equal(staged.view(torch.int16), table[predecessor].view(torch.int16)), "staged rows differ"
            embedding = staged
        v._score_tiles[(blocks, b)](offset, logits, probabilities, partials, embedding, head, scale, pending,
            candidates, histogram, slots, generations, live, generation, status, params, top_k, grammar,
            grammar_enabled, seeds, offsets, logits.stride(0), logits.stride(1), histogram.stride(0), grammar.stride(0),
            LOCAL=local, VOCAB=vocab, WIDTH=width, DRAFTS=drafts, PREFIX=step, BLOCKS=blocks, CAPACITY=capacity,
            BATCH=b, RUNNING=RUNNING, MARKOV=True, RANK=rank, FP8=fp8, SAMPLE=True, STEP=step, BLOCK=block,
            STAGED=staged_mode is True)
        out[staged_mode] = (probabilities, partials)
    direct, staged = out[False], out[True]
    for a, c in zip(direct, staged):
        assert torch.equal(a.view(torch.int32), c.view(torch.int32)), f"step {step} fp8 {fp8}: scores differ"
    # Not vacuous: the scores were stored, and the Markov bias changed them (a zero table gives other scores).
    assert not torch.equal(direct[0][:, step], torch.full_like(direct[0][:, step], -7.0))
    assert not torch.equal(direct[0][:, step], out["zero"][0][:, step])


@triton.jit
def _noise(Seed, Offset, token):
    return ((token * 7 + Seed * 3 + Offset) % 13).to(tl.float32) * 0.01


def _interpretable_noise():
    """libdevice.log1p (Gumbel noise) has no CPU-interpreter lowering. Both arms draw the same noise and the staged
    change touches only the Markov term, so a deterministic stand-in keeps the comparison exact."""
    from ignition.components import vocabulary_sampling as v

    v._token_noise = _noise


def main():
    _interpretable_noise()
    generator = torch.Generator().manual_seed(1118)
    count = 0
    for fp8 in (False, True):
        for step in (0, 1, 2):
            for _ in range(2):
                _case(step, fp8, generator)
                count += 1
    print(f"markov stage ok {count}")


if __name__ == "__main__":
    main()
