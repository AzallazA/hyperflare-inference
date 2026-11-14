# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 bootstrap wiring on CPU: the shared world-3 all-reduce runtime (TP3Config.ar_rows), the top_p pin under
vocabulary-parallel sampling, and the copied-history draft inputs (flare run --history-drafts)."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.dspark import SpeculativePreparation
from ignition.preparation.execution import GraphSignature
from ignition.recipes.deepseek41 import bootstrap
from ignition.recipes.deepseek41.tp3_config import TP3Config


def _runtimes(monkeypatch, config):
    calls = []

    def runtime(kind):
        def create(*args, **kwargs):
            result = SimpleNamespace(kind=kind, **kwargs)
            calls.append(result)
            return result
        return create

    for name in ("create_all_reduce_runtime", "create_dma_all_reduce_runtime", "create_all_gather_runtime",
                 "create_result_broadcast_runtime"):
        monkeypatch.setattr(bootstrap, name, runtime(name))
    result = bootstrap._collective_runtimes(config, None, None, coordinator=SimpleNamespace(spans_hosts=False),
        core_storage=SimpleNamespace(peer_storage=object()), verification_samples=8)
    return result, calls


@pytest.mark.parametrize("ar_rows,expected", [(0, 4096), (1024, 1024), (8192, 4096)])
def test_world_three_shares_one_all_reduce_runtime_sized_by_ar_rows(monkeypatch, ar_rows, expected):
    config = bootstrap.PreparationConfig(devices=(0, 1, 2), max_active=8, tp3=TP3Config(ar_rows=ar_rows))
    (collective, prefill, *_), calls = _runtimes(monkeypatch, config)
    reductions = [call for call in calls if "all_reduce" in call.kind]
    assert [call.kind for call in reductions] == ["create_all_reduce_runtime"]  # 5120 % 3: no DMA reduce-scatter
    assert prefill is collective
    assert collective.max_tokens == max(config.all_reduce_tokens, expected)
    assert config.all_reduce_tokens < 1024  # the decode capacity alone would not cover a prefill chunk


def test_world_four_keeps_upstream_decode_and_dma_prefill_runtimes(monkeypatch):
    config = bootstrap.PreparationConfig(devices=(0, 1, 2, 3), max_active=8, tp3=TP3Config(ar_rows=1024))
    (collective, prefill, *_), _ = _runtimes(monkeypatch, config)
    assert collective.kind == "create_all_reduce_runtime" and collective.max_tokens == config.all_reduce_tokens
    assert prefill.kind == "create_dma_all_reduce_runtime" and prefill.max_tokens == config.prefill_chunk_size


@pytest.mark.parametrize("parallel", [False, True])
def test_vocabulary_parallel_sampling_pins_top_p(parallel):
    config = bootstrap.PreparationConfig(devices=(0, 1, 2), max_active=8, vocabulary_parallel_sampling=parallel)
    defaults = bootstrap._sampling_defaults(config)
    upstream = bootstrap.RECIPE.sampling_defaults.as_dict()
    if not parallel:
        assert defaults == upstream
        return
    assert defaults["top_p"] == 1. and defaults["ranges"]["top_p"] == {"min": 1., "max": 1.}
    assert {key: value for key, value in defaults.items() if key not in ("top_p", "ranges")} == \
        {key: value for key, value in upstream.items() if key not in ("top_p", "ranges")}
    assert defaults["ranges"]["repetition_penalty"] == upstream["ranges"]["repetition_penalty"]


@pytest.mark.parametrize("parallel", [False, True])
def test_history_draft_inputs_only_under_vocabulary_parallel_sampling(monkeypatch, parallel):
    from ignition.components import packed_speculation, parallel_draft

    class DraftRound:
        def __init__(self, *args, **kwargs):
            pass

        def prepare(self):
            pass

    monkeypatch.setattr(parallel_draft, "PreparedParallelDraftRound", DraftRound)
    monkeypatch.setattr(packed_speculation, "PreparedPackedSpeculativeRound", lambda *args, **kwargs: object())
    config = bootstrap.PreparationConfig(devices=(0, 1, 2), max_active=8, vocabulary_parallel_sampling=parallel)
    copies = {name: torch.zeros(1) for name in SpeculativePreparation.COPY_INPUTS}
    speculation = SimpleNamespace(COPY_INPUTS=SpeculativePreparation.COPY_INPUTS, copy_buffers=lambda: copies)
    records, buffers = [], {}

    def host_allocate(name, shape, dtype):
        value = torch.zeros(shape, dtype=dtype)
        return SimpleNamespace(device_view=value, host_view=value)

    def add(name, forward, kind, rows, tokens, fields, **kwargs):
        records.append((name, tuple(fields)))

    bootstrap._prepare_speculation_catalog(config, speculation, add=add, buffers=buffers,
        forward_inputs=("row_slot_ids", "query_lengths", "request_state_rows"), host_allocate=host_allocate,
        device="cpu")
    packed = [fields for name, fields in records if name.startswith("packed-speculative")]
    assert packed and all((set(SpeculativePreparation.COPY_INPUTS) <= set(fields)) == parallel for fields in packed)
    assert all(not set(SpeculativePreparation.COPY_INPUTS) & set(fields)
               for name, fields in records if not name.startswith("packed-speculative"))
    assert (set(copies) <= set(buffers)) == parallel


def test_expert_selection_summary_names_micro_graphs():
    tuning = {"selections": [
        {"name": "decode1.layer3.experts", "config": {"implementation": "micro"}},
        {"name": "decode1.layer4.experts", "config": {"implementation": "micro"}},
        {"name": "prefill4096.layer3.experts", "config": {"implementation": "dynamic"}},
        {"name": "verification8x6-tokens48.layer3.experts", "config": {"tile": 64, "mode": "static_micro"}},
        {"name": "decode1.layer3.router", "config": {"implementation": "micro"}},
    ]}
    line = bootstrap._expert_selection_summary(tuning)
    assert line == ("Routed-expert selections: micro x3, dynamic x1; "
                    "micro/tiny_decode in: decode1 x2, verification8x6-tokens48 x1")
    assert bootstrap._expert_selection_summary({}).endswith("none; micro/tiny_decode in: none")
