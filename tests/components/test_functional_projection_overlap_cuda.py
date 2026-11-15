# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Projection-chain temporaries remain live through auxiliary-stream joins."""

import os

import pytest
import torch

from ignition.components.linear import UnquantizedProjection
from ignition.components.projection_overlap import ConcurrentProjectionChain
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an assigned CUDA device")


@pytest.mark.parametrize("rows", [3, 129])
@torch.no_grad()
def test_functional_projection_chain_replay_preserves_local_input(rows):
    device = torch.device("cuda:0")
    torch.manual_seed(321)
    seed = torch.randn(rows, 128, dtype=torch.bfloat16, device=device)
    template = torch.empty_like(seed)
    weights = [torch.randn(128, 128, dtype=torch.bfloat16, device=device) * .1 for _ in range(3)]
    expected_side, expected_main = torch.empty_like(seed), torch.empty_like(seed)
    side, main_output = torch.empty_like(seed), torch.empty_like(seed)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="projection-chain",
            workload_schema="projection-chain", numerical_policy="bf16") as session:
        session.register_scratch(arena)
        first = UnquantizedProjection(template, weights[0], torch.empty_like(seed), session=session, name="first")
        second = UnquantizedProjection(first.output, weights[1], torch.empty_like(seed), session=session, name="second")
        main = UnquantizedProjection(template, weights[2], torch.empty_like(seed), session=session, name="main")
        chain = ConcurrentProjectionChain((first, second), parallel_plans=(main.plan,), session=session, workspace=arena)
        chain.functional(session=session)
        main.functional(session=session)
        session.tune()
        arena.release_preparation()
        assert chain.projections is chain.workspace is chain.arena is None

        def forward():
            source = seed.clone()
            pending = chain.start(source)
            dependency = main(source)
            result = chain.finish(pending, dependency)
            side.copy_(result)
            main_output.copy_(dependency)
            source.fill_(float("nan"))

        compiled = compile_forward(forward)
        compiled()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        try:
            for gain in (1., -.5, 2.):
                seed.mul_(gain)
                expected_side.copy_(second(first(seed)))
                expected_main.copy_(main(seed))
                side.fill_(float("nan"))
                main_output.fill_(float("nan"))
                torch.cuda.synchronize()
                before = torch.cuda.memory_stats()
                graph.replay()
                torch.cuda.synchronize()
                after = torch.cuda.memory_stats()
                for field in ("allocation.all.allocated", "allocation.all.freed", "segment.all.allocated", "segment.all.freed"):
                    assert after[field] == before[field]
                torch.testing.assert_close(side, expected_side, rtol=0, atol=0)
                torch.testing.assert_close(main_output, expected_main, rtol=0, atol=0)
        finally:
            graph.reset()


def _poison(empty, byte):
    def poisoned(*args, **kwargs):
        tensor = empty(*args, **kwargs)
        if tensor.numel():
            tensor.view(torch.uint8).fill_(byte)
        return tensor
    return poisoned


@pytest.mark.parametrize("rows", [1, 5, 16])
@pytest.mark.parametrize("kind", ["bf16", "mxfp8-atomic"])
@torch.no_grad()
def test_functional_chain_outputs_need_no_zero_fill(monkeypatch, rows, kind):
    """Chain outputs start uninitialised; recycled 0x7F/0xFF memory must not reach the result."""
    from b12x.gemm import blockscaled

    from ignition.components.linear import QuantizedProjection

    device = torch.device("cuda:0")
    torch.manual_seed(rows)
    width = 4096
    seed = torch.randn(rows, width, dtype=torch.bfloat16, device=device)
    arena = ScratchArena(device)

    def projection(source, features, name, session):
        output = torch.empty(rows, features, dtype=torch.bfloat16, device=device)
        if kind == "bf16":
            weight = torch.randn(features, source.shape[1], dtype=torch.bfloat16, device=device) * .02
            return UnquantizedProjection(source, weight, output, session=session, name=name)
        values = (torch.randn(features, source.shape[1], device=device) * 4).to(torch.float8_e4m3fn)
        scales = torch.randint(118, 124, (features, source.shape[1] // 32), dtype=torch.uint8, device=device)
        weight = blockscaled.pack_weight(values, scales, recipe="mxfp8")
        return QuantizedProjection(source, weight, output, session=session, name=name, workspace=arena,
                                   activation_mode="quantized", allow_bf16_atomic_reduction=True)

    with RecordedPreparationSession(device=device, checkpoint="projection-chain-poison",
            workload_schema=f"chain-poison-{kind}-{rows}",
            numerical_policy="bf16" if kind == "bf16" else "default") as session:
        session.register_scratch(arena)
        first = projection(seed, 1024, "first", session)
        second = projection(first.output, 128, "second", session)
        main = projection(seed, 1024, "main", session)
        chain = ConcurrentProjectionChain((first, second), parallel_plans=(main.plan,), session=session,
                                          workspace=arena)
        chain.functional(session=session)
        main.functional(session=session)
        session.tune(show_progress=False)
        arena.release_preparation()
    for plan in (p.plan for p in chain.plan_owners):
        print(f"[plan] {kind} rows {rows}: {getattr(plan, 'query', plan)!r}"[:400], flush=True)

    def run():
        pending = chain.start(seed)
        return chain.finish(pending, main(seed)).clone()

    expected = run()
    for byte in (0x7F, 0xFF):
        with monkeypatch.context() as patch:
            patch.setattr(torch, "empty_strided", _poison(torch.empty_strided, byte))
            patch.setattr(torch, "empty", _poison(torch.empty, byte))
            actual = run()
        torch.cuda.synchronize()
        assert torch.equal(actual, expected), f"poison 0x{byte:02X}"
    assert expected.isfinite().all()
