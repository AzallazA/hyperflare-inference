# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Accepted KDA prefixes, replayed from verification records, match plain decode across widths."""

import contextlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.components.recurrent import DeltaMetadata
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.glm53.kda import KDARecords, KDAScratch, PreparedGlmKDA
from ignition.recipes.glm53.model import create_model_tree
from ignition.recipes.glm53.pool import ContinuationPool
from ignition.recipes.glm53.speculative import PreparedKDACommit, PreparedRound, convolution_regions
from ignition.recipes.glm53.weights import GlmWeightLoader, TEXT_PREFIX

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                               reason="requires an explicitly assigned CUDA device")


def _learned_kda_round(session_stack):
    """Layer-0 TP4 GLM KDA: plain decode (width 1) and a captured verification + commit round (width 6)."""
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    manifest = inspect_indexed_checkpoint(snapshot)
    module = create_model_tree(snapshot).model.language_model.layers[0].self_attn
    with GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 4),
                         prefixes=(f"{TEXT_PREFIX}.layers.0.self_attn.",)) as loader:
        weights = loader.load_attention(0)
    pool = ContinuationPool.allocate(("linear_attention",), 4, 10, 1, "cuda:0", mtp=True)
    state = pool.decoder.layers[0]
    arena = ScratchArena("cuda:0")
    metas = {width: DeltaMetadata.allocate(1, "cuda:0", verification_columns=8) for width in (1, 6)}
    sources = {width: torch.zeros((width, 4096), dtype=torch.bfloat16, device="cuda:0") for width in metas}
    slots = torch.tensor([0], dtype=torch.int32, device="cuda:0")
    counts = torch.ones(1, dtype=torch.int32, device="cuda:0")
    op = PreparedRound.__new__(PreparedRound)
    op.owner = SimpleNamespace(pool=pool, requests=SimpleNamespace(capacity=1),
                              inputs=SimpleNamespace(row_slot_ids=slots,
                                  request_state_rows=torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], device="cuda:0")))
    op.width, op.batch, op.rows = 6, 1, torch.arange(1, device="cuda:0")
    op.convolution_regions = convolution_regions(pool)
    records = KDARecords.allocate({0: weights}, state.recurrent.shape[0], state.recurrent.shape[1], "cuda:0")
    hidden = torch.zeros((6, 4096), dtype=torch.bfloat16, device="cuda:0")
    verified_output = torch.empty_like(hidden)
    operators = {}
    session = session_stack.enter_context(RecordedPreparationSession(device=torch.device("cuda:0"),
            checkpoint="glm53-mtp-state", workload_schema="glm-kda-prefix-v1", numerical_policy="default"))
    session.register_scratch(arena)
    for width in metas:
        operators[width] = PreparedGlmKDA(module, weights, sources[width],
            KDAScratch.allocate(width, 16, "cuda:0"), state, metas[width],
            mode="decode", session=session, name=f"decode{width}", workspace=arena,
            records=records.layer(0) if width > 1 else None)
    op.kda_commit = PreparedKDACommit(pool, records, operators[6].recurrent_plan,
        metas[6].state_indices[:, 0], metas[6].num_seqs, session=session, workspace=arena)
    for operator in operators.values():
        operator.functional(session=session)
    session.tune()
    arena.release_preparation()
    operators[1](sources[1], metas[1])
    for width, meta in metas.items():
        meta.query_start_loc.copy_(torch.tensor([0, width], dtype=torch.int32))
        meta.convolution_slots.fill_(1)
        # Verification reads the live row in column zero; plain decode writes it back there.
        meta.state_indices.copy_(torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]))
        meta.num_seqs.fill_(1)
        meta.num_tokens.fill_(width)
    generator = torch.Generator(device="cuda:0").manual_seed(531)
    chain = torch.randn((7, 4096), generator=generator, device="cuda:0", dtype=torch.bfloat16) * .1
    state.recurrent.normal_(0, .01, generator=generator)
    state.convolution.normal_(0, .1, generator=generator)
    initial_r, initial_c = state.recurrent.clone(), state.convolution.clone()
    sources[6].copy_(chain[:6])
    verified_output.copy_(operators[6](sources[6], metas[6], records=records.layer(0)))
    op.commit_state(slots.long(), counts, records, metas[6], hidden)
    graph = torch.cuda.CUDAGraph()
    with session.capture(), torch.cuda.graph(graph):
        verified_output.copy_(operators[6](sources[6], metas[6], records=records.layer(0)))
        op.commit_state(slots.long(), counts, records, metas[6], hidden)
    session.freeze()
    return SimpleNamespace(state=state, operators=operators, metas=metas, sources=sources, slots=slots,
                           counts=counts, records=records, op=op, hidden=hidden, graph=graph,
                           verified_output=verified_output, chain=chain, initial_r=initial_r, initial_c=initial_c)


@torch.no_grad()
def test_learned_kda_verification_commits_each_prefix_before_plain_decode():
    with contextlib.ExitStack() as stack:
        r = _learned_kda_round(stack)
        state, operators, metas, sources, counts, graph = r.state, r.operators, r.metas, r.sources, r.counts, r.graph
        verified_output, chain, initial_r, initial_c = r.verified_output, r.chain, r.initial_r, r.initial_c
        # Each verified token's output matches decoding the tokens one at a time. Plain decode and verification
        # read different columns of an arbitrary convolution history, so this check starts from an empty one.
        plain = []
        state.recurrent.copy_(initial_r)
        state.convolution.zero_()
        for index in range(6):
            sources[1].copy_(chain[index:index + 1])
            plain.append(operators[1](sources[1], metas[1]).clone())
        state.recurrent.copy_(initial_r)
        state.convolution.zero_()
        counts.fill_(1)
        graph.replay()
        torch.cuda.synchronize()
        verified = verified_output.float()
        expected = torch.cat(plain).float()
        relative = ((verified - expected).norm() / expected.norm()).item()
        # Recording rounds the recurrence to BF16 before the gated norm, as b12x's recording does.
        assert relative < 1e-2, relative
        for accepted in range(1, 7):
            state.recurrent.copy_(initial_r)
            state.convolution.copy_(initial_c)
            for index in range(accepted):
                sources[1].copy_(chain[index:index + 1])
                operators[1](sources[1], metas[1])
            expected_r, expected_c = state.recurrent[1].clone(), state.convolution[1, :, :3].clone()
            sources[1].copy_(chain[6:])
            expected_output = operators[1](sources[1], metas[1]).clone()
            state.recurrent.copy_(initial_r)
            state.convolution.copy_(initial_c)
            counts.fill_(accepted)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(state.recurrent[1], expected_r, rtol=.025, atol=.002)
            torch.testing.assert_close(state.convolution[1, :, :3], expected_c, rtol=.025, atol=.002)
            actual_output = operators[1](sources[1], metas[1])
            torch.testing.assert_close(actual_output, expected_output, rtol=.025, atol=.005)


@pytest.mark.parametrize("records_fill", [0x7F, 0xFF])
@torch.no_grad()
def test_poisoned_kda_records_commit_like_zeroed_records(records_fill):
    """SpeculativePreparation.records hands out uninitialised record planes: whatever they hold before a round, the
    round's verified outputs and committed states must equal those of a round started on zeroed records."""
    with contextlib.ExitStack() as stack:
        r = _learned_kda_round(stack)
        planes = (r.records.corrections, r.records.keys_and_gates)
        for accepted in range(1, 7):
            results = []
            for fill in (None, records_fill):
                r.state.recurrent.copy_(r.initial_r)
                r.state.convolution.copy_(r.initial_c)
                r.counts.fill_(accepted)
                for plane in planes:
                    plane.view(torch.uint8).fill_(0 if fill is None else fill)
                r.verified_output.fill_(float("nan"))
                r.graph.replay()
                torch.cuda.synchronize()
                results.append((r.state.recurrent.clone(), r.state.convolution.clone(), r.verified_output.clone()))
            for zeroed, poisoned in zip(*results, strict=True):
                assert torch.equal(zeroed, poisoned), f"accepted {accepted}: poison 0x{records_fill:02X} reached the round"
            assert results[0][2].isfinite().all()


def _round_state(pool, width, rows, kda_commit=None):
    """A round's state commit over `rows`: requests in slots 0, 1, ... own their rows' column zero."""
    op = PreparedRound.__new__(PreparedRound)
    batch = rows.shape[0]
    op.owner = SimpleNamespace(pool=pool, requests=SimpleNamespace(capacity=batch),
        inputs=SimpleNamespace(row_slot_ids=torch.arange(batch, dtype=torch.int32, device="cuda:0"),
                               request_state_rows=rows))
    op.width, op.batch, op.rows = width, batch, torch.arange(batch, device="cuda:0")
    op.convolution_regions = convolution_regions(pool)
    op.kda_commit = kda_commit
    return op


@torch.no_grad()
def test_width_one_advances_live_rows_in_place_between_recorded_rounds():
    """Fork #217 on the functional round: width one decodes in place, recorded rounds resume from its results."""
    snapshot = Path(os.environ.get("FLARE_GLM53_SNAPSHOT",
        "/data/checkpoint-releases/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"))
    manifest = inspect_indexed_checkpoint(snapshot)
    module = create_model_tree(snapshot).model.language_model.layers[0].self_attn
    with GlmWeightLoader(manifest, "cuda:0", shard=TensorParallelShard(0, 4),
                         prefixes=(f"{TEXT_PREFIX}.layers.0.self_attn.",)) as loader:
        weights = loader.load_attention(0)
    pool = ContinuationPool.allocate(("linear_attention",), 4, 10, 1, "cuda:0", mtp=True)
    state = pool.decoder.layers[0]
    arena = ScratchArena("cuda:0")
    # Request zero owns row 1. Request one's row 2 stays inactive, as a rejected or idle request does.
    rows = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]], device="cuda:0")
    # Width one decodes in place through one state column. Its recorded form, with the same projection
    # capacity, is the reference: recorded rounds verify against eight columns and commit accepted records.
    shapes = {"in_place": (1, 1), "recorded": (1, 8), "verification": (6, 8)}
    metas = {key: DeltaMetadata.allocate(2, "cuda:0", verification_columns=columns)
             for key, (_, columns) in shapes.items()}
    sources = {key: torch.zeros((2 * width, 4096), dtype=torch.bfloat16, device="cuda:0")
               for key, (width, _) in shapes.items()}
    records = KDARecords.allocate({0: weights}, state.recurrent.shape[0], state.recurrent.shape[1], "cuda:0")
    with RecordedPreparationSession(device=torch.device("cuda:0"), checkpoint="glm53-mtp-state",
            workload_schema="glm-kda-width-one-v1", numerical_policy="default") as session:
        session.register_scratch(arena)
        operators = {key: PreparedGlmKDA(module, weights, sources[key],
            KDAScratch.allocate(2 * width, 16, "cuda:0"), state, metas[key],
            mode="decode", session=session, name=key, workspace=arena,
            records=None if key == "in_place" else records.layer(0)) for key, (width, _) in shapes.items()}
        assert not operators["in_place"].recording and operators["in_place"].columns == 1
        rounds = {key: _round_state(pool, width, rows, None if key == "in_place" else
            PreparedKDACommit(pool, records, operators[key].recurrent_plan, metas[key].state_indices[:, 0],
                              metas[key].num_seqs, session=session, workspace=arena))
            for key, (width, _) in shapes.items()}
        for operator in operators.values():
            operator.functional(session=session)
        session.tune()
        arena.release_preparation()
        counts = torch.zeros(2, dtype=torch.int32, device="cuda:0")

        def run(key, tokens, accepted):
            # Metadata as PreparedRound.target_metadata packs it with request one inactive.
            meta, length = metas[key], tokens.shape[0]
            meta.query_start_loc.copy_(torch.tensor([0, length, length], dtype=torch.int32))
            meta.convolution_slots.copy_(torch.tensor([1, -1]))
            meta.state_indices.zero_()
            meta.state_indices[0, 0] = 1
            meta.num_seqs.fill_(2)
            meta.num_tokens.fill_(length)
            sources[key].zero_()
            sources[key][:length].copy_(tokens)
            layer_records = None if key == "in_place" else records.layer(0)
            output = operators[key](sources[key], meta, records=layer_records)[:length].clone()
            counts.copy_(torch.tensor([accepted, 0], dtype=torch.int32))
            hidden = torch.zeros((2 * shapes[key][0], 4096), dtype=torch.bfloat16, device="cuda:0")
            rounds[key].commit_state(rounds[key].owner.inputs.row_slot_ids.long(), counts, records, meta, hidden)
            return output

        generator = torch.Generator(device="cuda:0").manual_seed(977)
        chain = torch.randn((7, 4096), generator=generator, device="cuda:0", dtype=torch.bfloat16) * .1
        state.recurrent.normal_(0, .01, generator=generator)
        state.convolution.normal_(0, .1, generator=generator)
        initial_r, initial_c = state.recurrent.clone(), state.convolution.clone()
        # Each sequence runs width one, a recorded round that accepts three of six candidates, and width one.
        results = {}
        for key in ("recorded", "in_place"):
            state.recurrent.copy_(initial_r)
            state.convolution.copy_(initial_c)
            outputs = (run(key, chain[:1], 1), run("verification", chain[1:7], 3)[:3], run(key, chain[4:5], 1))
            results[key] = outputs, state.recurrent.clone(), state.convolution.clone()
            # The inactive request's row and the null row are unchanged by every round.
            for row in (0, 2):
                torch.testing.assert_close(state.recurrent[row], initial_r[row], rtol=0, atol=0)
                torch.testing.assert_close(state.convolution[row], initial_c[row], rtol=0, atol=0)
        (expected, expected_r, expected_c), (actual, actual_r, actual_c) = results["recorded"], results["in_place"]
        # Convolution does not depend on recording; recording rounds the recurrence to BF16 before the gated
        # norm, as b12x's recording does, and replays the committed state from its records.
        torch.testing.assert_close(actual_c, expected_c, rtol=0, atol=0)
        torch.testing.assert_close(actual_r[1], expected_r[1], rtol=.025, atol=.002)
        for value, reference in zip(actual, expected, strict=True):
            relative = ((value.float() - reference.float()).norm() / reference.float().norm()).item()
            assert relative < 1e-2, relative
