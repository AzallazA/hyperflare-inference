# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""exp/ds41-exl3-speed: FLARE_DS41_EXL3_SHARED_SIDE on one GPU with the EXL3 pack's real layer-3 weights (TP2 rank 0).

The shared expert forks after the FFN norm, overlaps the router projection + top-k and joins before the xmoe call
(flare::ds41_exl3_moe_after reads the joined output). Compiled with Flare's compile_forward and captured as a CUDA graph,
like the decoder; compared with the serial forward (router, xmoe, then the shared MLP):
- the shared expert's output must be bit-identical (same kernels and plans, disjoint buffers);
- the routed output must stay within the spread of repeated serial replays (xmoe accumulates in FP32 atomics);
- every replay rewrites the inputs first and scribbles the arena afterwards, so a join placed after the xmoe call or a
  missing join reads stale data and fails the comparison.
Requires FLARE_RUN_CUDA_TESTS=1 and FLARE_TEST_DS41_EXL3_PACK (the pack directory).
"""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
                                or not os.environ.get("FLARE_TEST_DS41_EXL3_PACK"),
                                reason="requires an assigned GPU and the EXL3 pack")

ROWS = tuple(int(r) for r in os.environ.get("FLARE_SHARED_SIDE_ROWS", "6,24").split(","))


@pytest.fixture(scope="module")
def layer_weights():
    os.environ.setdefault("FLARE_DS41_EXL3", "1")
    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

    with DeepSeekWeightLoader(inspect_indexed_checkpoint(Path(os.environ["FLARE_TEST_DS41_EXL3_PACK"])), "cuda:0",
                              shard=TensorParallelShard(0, 2), prefixes=("layers.3.",)) as loader:
        weights = loader.load_layer(3).experts
        config = loader.config.text
    return weights, config


@pytest.mark.parametrize("rows", ROWS)
@torch.no_grad()
def test_shared_side_is_identical_and_joins_before_xmoe(layer_weights, rows):
    from ignition.components.expert_overlap import ConcurrentSharedExpert
    from ignition.components.linear import UnquantizedProjection
    from ignition.components.mlp import PreparedMLP
    from ignition.components.moe import PreparedRouter
    from ignition.preparation.compilation import compile_forward
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.deepseek41.decoder import Scratch
    from ignition.recipes.deepseek41.exl3.experts import Exl3Experts, PreparedExl3Experts

    weights, config = layer_weights
    assert isinstance(weights.experts, Exl3Experts)
    device = torch.device("cuda:0")
    arena = ScratchArena(device, overlap=True)
    source = torch.randn((rows, 5120), device=device, dtype=torch.bfloat16)
    storage = Scratch.allocate(rows, rows, 2, device)
    live, logits = storage.ffn_hc.normalized, storage.router_logits
    route, shared_buffers, routed_buffer = storage.router, storage.shared, storage.routed
    output, shared_output, routed_output = (torch.empty_like(source) for _ in range(3))
    with RecordedPreparationSession(device=device, checkpoint="ds41-exl3-shared-side",
            workload_schema="tp4-shard", numerical_policy="checkpoint", compile_workers=16) as session:
        session.register_scratch(arena)
        projection = UnquantizedProjection(live, weights.router, logits, session=session, name="router")
        router = PreparedRouter(logits, route, session=session, name="route",
            score_func=config["scoring_func"], renormalize=config["norm_topk_prob"],
            routed_scaling_factor=config["routed_scaling_factor"], correction_bias=weights.correction)
        experts = PreparedExl3Experts(live, weights.experts, route.ids, route.weights, routed_buffer,
            session=session, name="experts", workspace=arena, router_plan=router.plan)
        shared = PreparedMLP(live, weights.shared, shared_buffers.gate_up, shared_buffers.intermediate,
            shared_buffers.output, session=session, name="shared", workspace=arena,
            swiglu_limit=config["swiglu_limit"], round_silu=False, activation_mode="auto")
        concurrent = ConcurrentSharedExpert(shared, parallel_plans=(projection.plan, router.plan),
            session=session, workspace=arena, defer_launch=rows > 4)
        projection.functional(session=session)
        router.functional(session=session)
        experts.functional(session=session)
        concurrent.functional(session=session)
        session.tune(show_progress=False)
        arena.release_preparation()

        def side_forward():
            live.copy_(source)
            pending = concurrent.start(live)
            selected = router(projection(live))
            projected = concurrent.finish(pending, selected.ids)          # join before xmoe (decoder.py)
            routed = experts(live, selected.ids, selected.weights, after=projected)
            output.copy_(routed + projected)
            shared_output.copy_(projected)
            routed_output.copy_(routed)
            arena.tensor.fill_(255)
            live.zero_()

        def serial_forward():
            live.copy_(source)
            selected = router(projection(live))
            routed = experts(live, selected.ids, selected.weights)
            projected = shared(live)
            output.copy_(routed + projected)
            shared_output.copy_(projected)
            routed_output.copy_(routed)
            arena.tensor.fill_(255)
            live.zero_()

        graphs = {}
        for name, forward in (("serial", serial_forward), ("side", side_forward)):
            compiled = compile_forward(forward)
            compiled()
            torch.cuda.synchronize()
            graphs[name] = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graphs[name]):
                compiled()
        for _ in range(8):
            source.normal_()
            graphs["serial"].replay()
            torch.cuda.synchronize()
            want_shared, want_routed = shared_output.clone(), routed_output.clone()
            spread = 0.0
            for _ in range(3):
                graphs["serial"].replay()
                torch.cuda.synchronize()
                spread = max(spread, (routed_output.float() - want_routed.float()).abs().max().item())
            for _ in range(3):
                graphs["side"].replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(shared_output, want_shared, atol=0, rtol=0)
                diff = (routed_output.float() - want_routed.float()).abs().max().item()
                scale = want_routed.float().abs().max().item()
                assert diff <= max(2 * spread, 1e-2 * scale), (diff, spread, scale)
