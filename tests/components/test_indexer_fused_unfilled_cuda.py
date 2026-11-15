# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""b12x's fused decode indexer stores every top-k slot of every row, so the functional call skips its fills.

GLM-shaped decode selection (32 heads, top-512) must resolve to the fused route, and a replay that starts from
poisoned indices (-42) and scores (NaN) must match one that starts from -1 / 0 (per row, up to merge order),
with an idle row, a short row (fewer records than top-k) and long rows that take the multi-CTA merge.
"""

import os

import pytest
import torch
from b12x.attention import dsa_indexer
from b12x.attention.dsa_indexer.reference import pack_index_k_cache_reference
from b12x.preparation.types import require_prepared

from ignition.components.indexer import PreparedSparseIndexer
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="explicit CUDA assignment required")


@pytest.mark.parametrize("rows", [2, 16])
@torch.no_grad()
def test_fused_decode_writes_every_slot(rows):
    device = torch.device("cuda:0")
    torch.manual_seed(5120 + rows)
    pages, heads, topk = 512, 32, 512
    query = torch.randn((rows, heads, 128), device=device).to(torch.float8_e4m3fn)
    weights = torch.randn((rows, heads), device=device)
    cache = pack_index_k_cache_reference(torch.randn((pages * 64, 128), device=device))
    table = torch.randperm(pages, device=device, dtype=torch.int32).unsqueeze(0).expand(rows, -1).contiguous()
    lengths = torch.full((rows,), pages * 64, dtype=torch.int32, device=device)
    lengths[0] = 0
    lengths[1] = 127
    width = torch.tensor([pages * 64], dtype=torch.int32, device=device)
    output = torch.empty((rows, topk), dtype=torch.int32, device=device)
    scores = torch.empty((rows, topk), device=device)
    arena = ScratchArena(device)
    graph = torch.cuda.CUDAGraph()
    with RecordedPreparationSession(device=device, checkpoint="sparse-indexer-fused-unfilled",
            workload_schema="glm-indexer-v1", numerical_policy="default") as session:
        session.register_scratch(arena)
        caps = dsa_indexer.Caps(device=device, num_q_heads=heads, max_q_rows=rows,
            max_page_table_width=pages, topk=topk, mode="decode")
        operation = PreparedSparseIndexer(query, weights, cache, table, lengths, width, output,
            caps=caps, session=session, name="indexer", workspace=arena, scores=scores)
        session.tune()
        assert require_prepared(operation.plan, "attention.dsa_indexer").layout.route == "paged_fused"
        compiled = compile_forward(lambda: operation())
        compiled()
        torch.cuda.synchronize()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        try:
            for long in (pages * 64, 4096, 700):
                lengths[2:] = long
                output.fill_(-1)
                scores.zero_()
                graph.replay()
                torch.cuda.synchronize()
                expected, expected_scores = output.clone(), scores.clone()
                assert (expected[0] == -1).all() and (expected[1] >= 0).sum() == 127
                for _ in range(2):
                    output.fill_(-42)
                    scores.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    # Multi-CTA merges place winners and cutoff ties in atomic arrival order, so compare per row
                    # as multisets: identical sorted scores, the same -1 padding, and only live record ids.
                    assert torch.equal(scores.sort(1).values.view(torch.int32),
                                       expected_scores.sort(1).values.view(torch.int32))
                    assert torch.equal((output == -1).sum(1), (expected == -1).sum(1))
                    assert ((output == -1) | ((output >= 0) & (output < lengths[:, None]))).all()
        finally:
            graph.reset()
