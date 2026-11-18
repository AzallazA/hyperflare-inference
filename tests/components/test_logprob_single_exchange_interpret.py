# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 fork (DS41 tp3.logprob_single_exchange): the one-exchange vocabulary-parallel logprob reduction writes the
same completion logprobs as the two-exchange reduction.

Runs both reductions of ``PreparedVocabularyParallelLogprobs`` in Triton's CPU interpreter (a subprocess,
TRITON_INTERPRET=1) for three emulated vocabulary ranks: each rank owns its instance (offset rank x local), the
kernels of ``_preparation`` are launched in order on every rank and each exchange is emulated by concatenating the
ranks' summaries (the PCIe all-gather's layout: rank 0's columns, then rank 1's, ...). Rows cover top-20 requests,
explicit token-id requests that span ranks, a row without logprobs, idle columns, padding rows, three tiles per
shard, a padded final shard, -inf logits and exact ties. The written ``completion_logprobs`` must be bit-identical.

The adversarial case puts 40 distinct raw logits a few ULPs apart on one rank at the top of a flat distribution, so
that subtracting the normalizer collapses them to the same FP32 values at the top-20 boundary: there the reported
values must still be bit-identical and only token ids among exactly equal values may differ (the documented limit).
"""

import os
import subprocess
import sys
from pathlib import Path

import torch

WORLD, LOCAL, VOCAB = 3, 2400, 7000  # three 1024-token tiles per shard; the last shard is padded
BATCH, WIDTH, SLOTS, TOKENS = 4, 4, 5, 6


def test_single_exchange_logprobs_match_the_two_exchange_reduction_in_the_cpu_interpreter():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH")))))
    completed = subprocess.run([sys.executable, "-m", "tests.components.test_logprob_single_exchange_interpret"],
                               cwd=root, env=env, capture_output=True, text=True, timeout=1800)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "logprob single exchange ok 4" in completed.stdout, completed.stdout + completed.stderr
    assert "collapse: values identical" in completed.stdout, completed.stdout


def test_switches_are_off_by_default_parsed_recorded_and_exclusive():
    import pytest

    from ignition.recipes.deepseek41.tp3_config import TP3Config

    base = TP3Config()
    assert base.logprob_single_exchange is False and base.no_generation_logprobs is False
    assert "logprob_single_exchange" not in base.to_metadata() and "no_generation_logprobs" not in base.to_metadata()
    on = TP3Config.from_environment({"FLARE_DS41_LOGPROB_SINGLE_EXCHANGE": "1"})
    assert on.logprob_single_exchange is True and on.to_metadata()["logprob_single_exchange"] is True
    drop = TP3Config.from_environment({"FLARE_DS41_NO_GENERATION_LOGPROBS": "1"})
    assert drop.no_generation_logprobs is True and drop.to_metadata()["no_generation_logprobs"] is True
    with pytest.raises(ValueError, match="set one of them"):
        TP3Config(logprob_single_exchange=True, no_generation_logprobs=True)
    with pytest.raises(TypeError):
        TP3Config(no_generation_logprobs=1)


def test_completion_without_generation_logprobs_declares_none_and_reductions_are_inert():
    """tp3.no_generation_logprobs: the manifest gets no generation-logprob buffers (Helios max_top_logprobs() == 0,
    requests answered with HTTP 400) and both reductions launch nothing."""
    from ignition.components.completion import CompletionOutputs, validate_completion_buffers
    from ignition.components.logprobs import PreparedLogprobs
    from ignition.components.vocabulary_logprobs import PreparedVocabularyParallelLogprobs

    full = CompletionOutputs.allocate((2, SLOTS, 16 + TOKENS), device=torch.device("cpu"))
    assert {"completion_logprobs", "logprob_requested", "logprob_token_ids",
            "logprob_token_id_counts"} <= set(full.buffers())
    none = CompletionOutputs.allocate((2, SLOTS, 16 + TOKENS), device=torch.device("cpu"), generation_logprobs=False)
    assert set(none.buffers()) == {"completion_records", "completion_speculative_survival"}
    assert validate_completion_buffers(none.buffers(), SLOTS) == TOKENS
    assert none.logprobs is None and none.private().logprobs is None
    logits = torch.zeros((BATCH, WIDTH, LOCAL))
    rows = torch.zeros(BATCH, dtype=torch.int64), torch.zeros(BATCH, dtype=torch.int64), torch.ones(1, dtype=torch.int32)
    frame = torch.zeros(1, dtype=torch.int32)
    plain = PreparedLogprobs(logits, none, *rows, frame)
    plain.prepare()
    plain.functional(dynamic_inputs=True)
    assert plain.run(logits) is None and plain.statistics is None
    parallel = PreparedVocabularyParallelLogprobs(logits, none, *rows, frame, vocabulary=VOCAB, offset=0,
        runtime=None, session=None, coordinator=None, name="inert")
    parallel.prepare()
    assert parallel.functional(session=None) is parallel and parallel.operations == ()
    assert parallel() is None and parallel.run(logits) is None
    import pytest

    with pytest.raises(TypeError):
        CompletionOutputs.allocate((2, SLOTS, 16 + TOKENS), device=torch.device("cpu"), generation_logprobs=1)


# --- interpreter body -----------------------------------------------------------------------------------------

def _completion(generator):
    from ignition.components.completion import (CompletionOutputs, EMITTED_COUNT_FIELD, VECTOR_TOKENS_FIELD)

    completion = CompletionOutputs.allocate((2, SLOTS, VECTOR_TOKENS_FIELD + TOKENS), device=torch.device("cpu"))
    frame = torch.tensor([1], dtype=torch.int32)
    slots = torch.tensor([2, 0, 1, 4], dtype=torch.int64)       # row 3 is a padding row (beyond live)
    epochs = torch.tensor([7, 9, 11, 13], dtype=torch.int64)
    live = torch.tensor([3], dtype=torch.int32)
    counts = {2: WIDTH, 0: 2, 1: 3, 4: 4}
    for row, slot in enumerate(slots.tolist()):
        record = completion.records[1, slot]
        record[0] = epochs[row]
        record[EMITTED_COUNT_FIELD] = counts[slot]
        record[VECTOR_TOKENS_FIELD:VECTOR_TOKENS_FIELD + WIDTH] = torch.randint(0, VOCAB, (WIDTH,), generator=generator)
    completion.records[1, 2, VECTOR_TOKENS_FIELD] = VOCAB - 1           # chosen token on the padded last shard
    completion.requested.copy_(torch.tensor([1, 0, 1, 0, 1], dtype=torch.int32))   # slot 1 (row 2) not requested
    completion.token_id_counts[0] = 5                                    # slot 0 (row 1) asks for explicit ids
    completion.token_ids[0, :5] = torch.tensor([0, LOCAL - 1, LOCAL, 2 * LOCAL + 5, VOCAB - 1], dtype=torch.int32)
    return completion, slots, epochs, live, frame


def _run(logits, single, completion_factory):
    """Both ranks' reductions in lockstep; returns rank 0's and every rank's completion logprobs."""
    from ignition.components import vocabulary_logprobs as module

    class FakeGather:  # the exchange is emulated below, between kernel launches
        def __init__(self, *args, **kwargs):
            pass

    class FakeRuntime:
        world_size = WORLD

    saved = module.PreparedAllGather
    module.PreparedAllGather = FakeGather
    try:
        ranks = []
        for rank in range(WORLD):
            completion, slots, epochs, live, frame = completion_factory()
            shard = torch.full((BATCH, WIDTH, LOCAL), -float("inf"))
            width = min(LOCAL, VOCAB - rank * LOCAL)
            shard[..., :width] = logits[..., rank * LOCAL:rank * LOCAL + width]
            shard = shard.contiguous()
            ranks.append((module.PreparedVocabularyParallelLogprobs(shard, completion, slots, epochs, live, frame,
                vocabulary=VOCAB, offset=rank * LOCAL, runtime=FakeRuntime(), session=None, coordinator=None,
                name=f"rank{rank}", single_exchange=single), completion))
    finally:
        module.PreparedAllGather = saved
    reduction = ranks[0][0]
    exchanges = reduction._exchange_before
    for index in range(len(reduction._preparation)):
        if index in exchanges:
            gathered = torch.stack([r.summary for r, _ in ranks], 1).flatten(1)   # [rows, world * summary words]
            for r, _ in ranks:
                r.gathered.copy_(gathered)
        for r, _ in ranks:
            kernel, args, grid, warps = r._preparation[index]
            kernel[grid](*args, num_warps=warps)
    return [completion.logprobs.clone() for _, completion in ranks]


def _patch_interpreter_unsigned_invert():
    """Test-only: Triton 3.8's CPU interpreter builds the all-ones operand of ``~x`` as ``np.full(1, -1, dtype)``,
    which NumPy 2 rejects for unsigned dtypes (OverflowError), so ``_sort_key``'s ``~bits`` cannot run there. Use the
    dtype's all-ones value (the GPU lowering's) for unsigned types; signed and boolean types are unchanged."""
    import numpy as np
    from triton.runtime import interpreter

    original = interpreter.InterpreterBuilder.get_all_ones_value

    def get_all_ones_value(self, type):
        np_type = interpreter._get_np_dtype(type)
        if np.issubdtype(np_type, np.unsignedinteger):
            return interpreter.TensorHandle(np.full(1, np.iinfo(np_type).max, dtype=np_type), type.scalar)
        return original(self, type)

    interpreter.InterpreterBuilder.get_all_ones_value = get_all_ones_value


def _main():
    from ignition.components.completion import LOGPROB_PAIR_FIELDS, MAX_TOP_LOGPROBS

    _patch_interpreter_unsigned_invert()

    generator = torch.Generator().manual_seed(11)
    state = torch.Generator().manual_seed(5)

    def completion_factory():
        state.manual_seed(5)
        return _completion(state)

    cases = []
    logits = torch.randn((BATCH, WIDTH, VOCAB), generator=generator) * 3
    cases.append(("random", logits))
    ties = logits.clone()
    ties[..., 100:160] = 9.0                       # 60 exactly equal top logits spread over a tile and a shard edge
    ties[..., LOCAL - 10:LOCAL + 10] = 9.0
    cases.append(("ties", ties))
    sparse = logits.clone()
    sparse[..., ::3] = -float("inf")
    cases.append(("neg-inf", sparse))
    peaked = logits.clone()
    peaked[..., 4321] = 60.0                       # one dominant token: x - lse ~ 0 for it
    cases.append(("peaked", peaked))
    checked = 0
    for name, case in cases:
        two = _run(case, False, completion_factory)
        one = _run(case, True, completion_factory)
        for rank in range(WORLD):
            assert torch.equal(two[rank], one[rank]), (name, rank, (two[rank] != one[rank]).nonzero()[:8])
        assert two[0][1].abs().sum() > 0          # something was written into the live frame
        checked += 1
    # Adversarial collapse: 40 distinct raw logits one ULP apart at the top of rank 1, flat elsewhere.
    flat = torch.randn((BATCH, WIDTH, VOCAB), generator=generator) * 0.05
    top = torch.tensor(0.5, dtype=torch.float32)
    run = []
    for _ in range(40):
        run.append(top.item())
        top = torch.nextafter(top, torch.tensor(1.0))
    flat[..., LOCAL + 100:LOCAL + 140] = torch.tensor(run)
    two = _run(flat, False, completion_factory)
    one = _run(flat, True, completion_factory)
    values = slice(1, 2 * (1 + MAX_TOP_LOGPROBS), LOGPROB_PAIR_FIELDS)  # chosen and top-20 value words
    for rank in range(WORLD):
        assert torch.equal(two[rank][..., values], one[rank][..., values]), rank
        assert torch.equal(two[rank][..., 0], one[rank][..., 0])
    differing = int((two[0][..., 2:2 * (1 + MAX_TOP_LOGPROBS):2] != one[0][..., 2:2 * (1 + MAX_TOP_LOGPROBS):2]).sum())
    print(f"collapse: values identical, ids differing among equal values: {differing}")
    print(f"logprob single exchange ok {checked}")


if __name__ == "__main__":
    _main()
