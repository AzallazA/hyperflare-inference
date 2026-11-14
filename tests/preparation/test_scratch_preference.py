# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Autotuning prefers a lower-scratch candidate when it is near-equal in speed."""

from types import SimpleNamespace

import pytest

from ignition.preparation.collectives import _tuning_results
from ignition.preparation.scratch_preference import (
    MIN_SAVING_ENV, TOLERANCE_ENV, ScratchPreference, from_environment, select_shard_winner,
)

MiB = 1 << 20
PREFERENCE = ScratchPreference(0.02, 16 * MiB)


@pytest.mark.parametrize(("entries", "reference", "expected"), [
    # 1% slower with 300 MiB less scratch: the lean candidate wins.
    (((100.0, 0, 438 * MiB), (101.0, 1, 138 * MiB), (150.0, 2, 0)), None, 1),
    # Lean but 3% slower: outside the tolerance.
    (((100.0, 0, 438 * MiB), (103.0, 1, 138 * MiB)), None, 0),
    # Within tolerance, but the saving is below the floor (decode-sized workspaces).
    (((100.0, 0, 8 * MiB), (100.5, 1, 0)), None, 0),
    # Equal scratch keeps the fastest.
    (((100.5, 1, 64 * MiB), (100.0, 0, 64 * MiB)), None, 1),
    # An earlier, faster measurement bounds the tolerance: no drift beyond 2% of it.
    (((101.0, 0, 438 * MiB), (102.5, 1, 138 * MiB)), 100.0, 0),
    # Nothing within the bound: the fastest of this set.
    (((110.0, 3, 438 * MiB), (111.0, 4, 0)), 100.0, 0),
    # Ineligible (infinite) scratch never wins on scratch.
    (((100.0, 0, float("inf")), (101.0, 1, float("inf"))), None, 0),
])
def test_choose_prefers_lean_candidates_only_within_tolerance(entries, reference, expected):
    assert PREFERENCE.choose(entries, reference) == expected


def test_environment_defaults_disable_and_validation(monkeypatch):
    monkeypatch.delenv(TOLERANCE_ENV, raising=False)
    monkeypatch.delenv(MIN_SAVING_ENV, raising=False)
    assert from_environment() == ScratchPreference(0.02, 16 * MiB)
    monkeypatch.setenv(TOLERANCE_ENV, "0")
    assert from_environment() is None
    monkeypatch.setenv(TOLERANCE_ENV, "0.05")
    monkeypatch.setenv(MIN_SAVING_ENV, "0")
    assert from_environment() == ScratchPreference(0.05, 1)
    for bad in ("2", "-0.1", "nan", "two"):
        monkeypatch.setenv(TOLERANCE_ENV, bad)
        with pytest.raises(ValueError):
            from_environment()


def _shard(index, latency, scratch, fastest):
    from b12x.preparation.types import TuningRequirement

    return TuningRequirement("k" * 8, (0, 1, 2), {"width": index}, latency, index,
                             scratch_nbytes=scratch, fastest_latency_us=fastest)


def test_cross_shard_winner_respects_each_shards_fastest_measurement(monkeypatch):
    monkeypatch.delenv(TOLERANCE_ENV, raising=False)
    monkeypatch.delenv(MIN_SAVING_ENV, raising=False)
    # Shard 2 picked its lean candidate 2 at 2056.6 us over its fastest (2022 us).
    shards = [_shard(3, 2129.6, 300 * MiB, 2129.6), _shard(4, 2135.9, 296 * MiB, 2135.9),
              _shard(2, 2056.6, 296 * MiB, 2022.0)]
    assert select_shard_winner(shards, PREFERENCE).candidate_index == 2
    (winner,) = _tuning_results(shards)
    assert winner.candidate_index == 2
    # A lean shard winner beyond 2% of another shard's fastest candidate loses.
    shards[2] = _shard(2, 2090.0, 296 * MiB, 2022.0)
    shards[0] = _shard(3, 2040.0, 438 * MiB, 2040.0)
    assert select_shard_winner(shards, PREFERENCE).candidate_index == 3
    monkeypatch.setenv(TOLERANCE_ENV, "0")
    assert _tuning_results(shards)[0].candidate_index == 3


def test_contributions_without_scratch_keep_the_fastest():
    items = [SimpleNamespace(latency_us=100.0, candidate_index=0, scratch_nbytes=None),
             SimpleNamespace(latency_us=100.5, candidate_index=1, scratch_nbytes=0)]
    assert select_shard_winner(items, PREFERENCE).candidate_index == 0


def _race(tmp_path, monkeypatch, preference, latencies, scratch):
    """One single-rank b12x race over fake candidates; returns the selection and counters."""
    from b12x.preparation import (
        DetectedDevice, Knob, MemoryRequirements, Plan, PreparationSession, PreparedCall,
        TuningContract, _measurement,
    )
    from b12x.preparation._cache import SelectionCache
    from b12x._lib.scratch import ScratchBufferSpec
    import torch

    def prepare(calls, **kwargs):
        yield
        return SimpleNamespace(calls=calls, close=lambda: None, completed_rounds=0,
            planned_rounds=0, active_count=len(calls), latest_round_us=())

    def measure(race, **kwargs):
        yield
        return _measurement.RaceMeasurements(tuple(latencies[call.output] for call in race.calls), 0)

    monkeypatch.setattr(_measurement, "prepare_race_steps", prepare)
    monkeypatch.setattr(_measurement, "measure_race_steps", measure)
    contract = TuningContract(
        component_id="test.scratch_preference", query_schema_version=1, config_schema_version=1,
        query_fields=frozenset({"rows"}), config_fields=frozenset({"width"}),
        encode_query=lambda query: {"rows": query}, encode_config=lambda config: {"width": config},
        decode_config=lambda payload: payload["width"], validate_query=lambda query, device: None,
        validate_config=lambda query, config, device: None, default_config=lambda query, device: 1,
        knobs=(Knob(name="width", values=tuple(latencies)),))

    def memory(config, device):
        size = scratch[config]
        return MemoryRequirements(scratch=(ScratchBufferSpec(name="workspace", shape=(size,),
            dtype=torch.uint8, device=torch.device("cpu")),) if size else ())

    plan = Plan(contract=contract, query=4096, _compile_jobs=lambda config, device: (),
                _memory_requirements=memory, _materialize=lambda selection, device: selection.config)
    request = plan.request(name="prefill.experts", prepare_call=lambda state: PreparedCall(run=lambda: state),
        benchmark_call=lambda state: PreparedCall(run=lambda: state, produce=lambda: None))
    with PreparationSession(device=DetectedDevice(None, None), cache_dir=tmp_path, compile_workers=0) as engine:
        engine._cache = SelectionCache(tmp_path, {"schema_version": 23, "tuning_cache_version": 1})
        engine.configure_scratch_preference(preference)
        result = engine.prepare((request,))
        return result.selections["prefill.experts"].config, result


@pytest.mark.parametrize("preference", [None, PREFERENCE])
def test_b12x_race_selects_lean_candidate_and_reuses_recorded_measurements(tmp_path, monkeypatch, preference):
    latencies = {1: 2022.0, 2: 2056.6, 3: 2129.6, 4: 2400.0}
    scratch = {1: 438 * MiB, 2: 296 * MiB, 3: 296 * MiB, 4: 0}
    config, result = _race(tmp_path, monkeypatch, preference, latencies, scratch)
    assert config == (1 if preference is None else 2)
    assert result.benchmarked_candidates == len(latencies)
    # Toggling the policy re-selects from the recorded race without measuring again.
    other = PREFERENCE if preference is None else None
    config, result = _race(tmp_path, monkeypatch, other, latencies, scratch)
    assert config == (2 if preference is None else 1)
    assert result.benchmarked_candidates == 0
    assert result.restored_candidates == len(latencies)


def test_configured_preference_cannot_change_after_preparation_begins(tmp_path):
    from b12x.preparation import DetectedDevice, PreparationSession

    with PreparationSession(device=DetectedDevice(None, None), cache_dir=tmp_path, compile_workers=0) as engine:
        with pytest.raises(TypeError):
            engine.configure_scratch_preference(object())
        engine.configure_scratch_preference(PREFERENCE)
        engine._scratch_limit_locked = True
        with pytest.raises(RuntimeError):
            engine.configure_scratch_preference(None)
