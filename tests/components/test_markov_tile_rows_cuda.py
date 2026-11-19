# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 fork (DS41 tp3.markov_tile_rows): prepared proposals with all rows per tile equal production bit for bit on GPU.

GPU twin of test_markov_tile_rows_interpret.py through PreparedVocabularySoftmax (prepare, real Gumbel noise, the full
5-kernel step chain): probabilities, validity and sampled tokens must be identical for every ROWS.
"""

import os

import pytest
import torch

from tests.components.test_vocabulary_sampling_cuda import fixture

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


@pytest.mark.parametrize("batch,rows", [(1, 1), (4, 1), (4, 3), (4, 4), (8, 2), (8, 8), (16, 4), (16, 5), (16, 16)])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("stage_rows", [False, True])  # True = the production setting (tp3.markov_stage_rows)
def test_tile_rows_proposals_are_bit_identical(batch, rows, fp8, stage_rows):
    results = []
    for options in ({"stage_rows": stage_rows}, {"stage_rows": stage_rows, "tile_rows": rows}):
        torch.manual_seed(1118)
        op = fixture(batch, 6, 129280, rank=256, fp8=fp8, **options)
        op.pending.copy_(torch.randint(0, 129280, (batch,), device="cuda"))
        if rows > 1 and "tile_rows" in options:
            assert op.steps[0][0].kernel.fn.__name__ == "_score_tiles_rows"
        op()
        torch.cuda.synchronize()
        results.append((op.probabilities.clone(), op.valid.clone(), op.tokens.clone()))
    for a, b in zip(*results):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    assert results[0][1].all()
