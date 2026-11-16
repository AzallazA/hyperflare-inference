# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 VRAM reclaim switches (2026-11-15 memory map items 2, 5, 8), CPU only.

- share_engram_token_map: b12x Engram hash plans read one persistent token map instead of their own copy.
- trim_vocab_gather / skip_draft_vocab_gather: vocabulary all-gather channel sizes with vocabulary-parallel sampling.
- share_vocab_columns: vocabulary-parallel rounds share one copy of the rank's global vocabulary ids.

Every switch defaults to the base behaviour; each is read from its own FLARE_DS41_* variable, and
FLARE_DS41_VRAM_RECLAIM=1 turns all of them on unless an item's own variable is set.
"""

import inspect
from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.deepseek41.tp3_config import TP3Config, VRAM_RECLAIM

NAMES = tuple(VRAM_RECLAIM)


# --- switches and record -------------------------------------------------------------------------------------------

def test_switches_default_off_and_are_recorded():
    base = TP3Config()
    assert all(getattr(base, name) is False for name in NAMES)
    assert all(getattr(TP3Config.from_environment({}), name) is False for name in NAMES)
    record = base.to_metadata()
    assert all(record[name] is False for name in NAMES)
    assert base.describe() == "TP3 config: defaults"
    assert set(NAMES) <= {item.name for item in fields(TP3Config)}


@pytest.mark.parametrize("name", NAMES)
def test_each_switch_has_its_own_variable(name):
    config = TP3Config.from_environment({"FLARE_DS41_" + VRAM_RECLAIM[name]: "1"})
    assert {other: getattr(config, other) for other in NAMES} == {other: other == name for other in NAMES}
    assert config.to_metadata()[name] is True and f"{name}=True" in config.describe()
    with pytest.raises(TypeError, match=f"{name} must be a boolean"):
        TP3Config(**{name: 1})


def test_umbrella_turns_all_on_and_items_override_it():
    on = TP3Config.from_environment({"FLARE_DS41_VRAM_RECLAIM": "1"})
    assert all(getattr(on, name) for name in NAMES)
    one_off = TP3Config.from_environment({"FLARE_DS41_VRAM_RECLAIM": "1", "FLARE_DS41_TRIM_VOCAB_GATHER": "0"})
    assert {name: getattr(one_off, name) for name in NAMES} == {name: name != "trim_vocab_gather" for name in NAMES}
    # an empty item variable counts as unset (follows the umbrella), as everywhere in TP3Config
    empty = TP3Config.from_environment({"FLARE_DS41_VRAM_RECLAIM": "1", "FLARE_DS41_SHARE_VOCAB_COLUMNS": ""})
    assert empty.share_vocab_columns is True
    assert not any(getattr(TP3Config.from_environment({"FLARE_DS41_VRAM_RECLAIM": "0"}), name) for name in NAMES)


# --- item 5: vocabulary channels -----------------------------------------------------------------------------------

def _config(**tp3):
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    # The served shape: TP3, 8 requests, DSpark 5 drafts (widths 1-6), vocabulary-parallel sampling, MTP only.
    return PreparationConfig(devices=(0, 1, 2), max_active=8, dspark=True, mtp_only=True,
                             vocabulary_parallel_sampling=True, speculative_widths=tuple(range(1, 7)),
                             tp3=TP3Config(**tp3))


def test_vocabulary_channels_base_and_trimmed():
    from ignition.recipes.deepseek41.bootstrap import vocabulary_channels, vram_reclaim_report

    base = _config()
    samples = base.request_capacity * max(base.speculative_widths)
    assert samples == 48
    assert vocabulary_channels(base, samples) == (48, True)
    assert vocabulary_channels(_config(trim_vocab_gather=True), samples) == (max(base.batch_capacities), True)
    assert max(base.batch_capacities) == 8
    assert vocabulary_channels(_config(skip_draft_vocab_gather=True), samples) == (48, False)
    both = _config(trim_vocab_gather=True, skip_draft_vocab_gather=True, share_engram_token_map=True,
                   share_vocab_columns=True)
    assert vocabulary_channels(both, samples) == (8, False)
    assert vram_reclaim_report(both, samples) == {
        "share_engram_token_map": True, "vocabulary_gather_rows": 8, "vocabulary_gather_rows_base": 48,
        "draft_vocabulary_gather": False, "share_vocab_columns": True}
    assert vram_reclaim_report(base, samples)["vocabulary_gather_rows"] == 48


def test_vocabulary_channels_unchanged_without_vocabulary_parallel_sampling():
    """The draft-vocabulary channel is the DSpark head's gather channel when heads gather: keep it reachable."""
    from ignition.recipes.deepseek41.bootstrap import vocabulary_channels, vram_reclaim_report

    config = replace(_config(trim_vocab_gather=True, skip_draft_vocab_gather=True, share_vocab_columns=True),
                     vocabulary_parallel_sampling=False)
    assert vocabulary_channels(config, 48) == (48, True)
    assert vram_reclaim_report(config, 48)["share_vocab_columns"] is False
    on = TP3Config(trim_vocab_gather=True, skip_draft_vocab_gather=True)
    no_dspark = SimpleNamespace(tp3=on, dspark=False, vocabulary_parallel_sampling=True, batch_capacities=(1, 2))
    assert vocabulary_channels(no_dspark, 48) == (2, False)  # no DSpark: no draft channel either way


def test_vocabulary_channel_rows_cover_every_gather_left_on_it():
    """With vocabulary-parallel sampling the speculative target and draft heads are local-only, so what still gathers
    through the vocabulary channel is prefill heads (1 row), resume heads (``rows=capacity``) and, without
    --mtp-only, non-speculative decode heads (``rows=capacity``): never more than max(batch_capacities)."""
    from ignition.components import parallel_draft
    from ignition.recipes.deepseek41 import bootstrap, decoder, dspark

    # speculative rounds pass vocabulary_parallel=True to both the target decoder and the draft forward ...
    source = inspect.getsource(parallel_draft.PreparedParallelDraftRound.__init__)
    assert source.count('**({"vocabulary_parallel": True} if self.vocabulary_parallel else {})') == 2
    # ... and those heads are local-only then
    assert "gather=not vocabulary_parallel" in inspect.getsource(decoder.PreparedDecoder._prepare_head)
    assert "gather=not vocabulary_parallel" in inspect.getsource(dspark.PreparedDraft.__init__)
    # the remaining users are built per batch capacity (resume, decode) or with one row (prefill)
    catalog = inspect.getsource(bootstrap.prepare_rank)
    assert "rows = 1 if mode == \"prefill\" else capacity" in catalog
    assert "rows=capacity, runtime=collective, vocabulary_runtime=vocabulary_collective" in catalog


# --- item 2: shared Engram token map -------------------------------------------------------------------------------

def _hash_plan(vocab=8):
    from b12x.sequence import engram

    geometry = engram.build_geometry(layer_ids=(2,), base_table_size=1000, compressed_vocab_size=vocab)
    token_map = tuple(range(vocab))
    # Caps want a CUDA device (no CUDA call is made); the CPU test then re-homes the plan to the CPU so CPU tensors
    # pass share_hash_token_map's same-device check.
    plan = engram.plan(engram.Caps(device=torch.device("cuda", 0), max_tokens=4, max_seqs=4, max_requests=4,
                                   vocab_size=vocab, layer_id=2, tp_size=1, tp_rank=0),
                       token_map=token_map, geometry=geometry, invocation={"operation": "hash"})
    object.__setattr__(plan, "_device", torch.device("cpu"))
    return plan, token_map


@pytest.fixture
def hash_plan():
    try:
        return _hash_plan()
    except Exception as error:  # noqa: BLE001 - the b12x plan constructor may need a CUDA device on some builds
        pytest.skip(f"b12x Engram hash plan not constructible on CPU: {error}")


def _state_like_b12x(plan, token_map):
    """The _State b12x's materialize returns, minus the compile (programs) - built from the same constructor."""
    from b12x.sequence.engram._impl import _State

    def tensor(values):
        return torch.tensor(values, dtype=torch.int64)

    return _State(None, None, tensor(token_map), tensor([3]), tensor([5]), tensor([7]), 2, 10, 0, 10, 10, "hash",
                  False, (), ("programs",))


def test_shared_token_map_replaces_only_the_token_map(hash_plan):
    from ignition.components.engram import share_hash_token_map

    plan, token_map = hash_plan
    own = _state_like_b12x(plan, token_map)
    object.__setattr__(plan, "_materialize", lambda selection, device: own)
    handle, query = plan.handle, plan.query
    shared = torch.tensor(token_map, dtype=torch.int64)
    assert share_hash_token_map(plan, shared) is plan
    assert plan.handle == handle and plan.query is query  # identity and declaration unchanged
    state = plan._materialize(None, None)
    assert state.token_map is shared
    for name in ("multipliers", "primes", "offsets", "programs", "pad_id", "operation"):
        assert getattr(state, name) is getattr(own, name)
    # two plans, one tensor
    other, _ = _hash_plan()
    object.__setattr__(other, "_materialize", lambda selection, device: _state_like_b12x(other, token_map))
    share_hash_token_map(other, shared)
    assert other._materialize(None, None).token_map is shared


def test_shared_token_map_must_equal_the_plans_own(hash_plan):
    from ignition.components.engram import share_hash_token_map

    plan, token_map = hash_plan
    object.__setattr__(plan, "_materialize", lambda selection, device: _state_like_b12x(plan, token_map))
    wrong = torch.tensor(token_map, dtype=torch.int64).flip(0).contiguous()
    share_hash_token_map(plan, wrong)
    with pytest.raises(ValueError, match="differs from the plan's own"):
        plan._materialize(None, None)
    with pytest.raises(ValueError, match="contiguous int64 vector"):
        share_hash_token_map(_hash_plan()[0], torch.tensor(token_map, dtype=torch.int32))
    with pytest.raises(ValueError, match="contiguous int64 vector"):
        share_hash_token_map(_hash_plan()[0], torch.arange(len(token_map) + 1))


def test_b12x_token_map_is_a_read_only_int64_copy_of_the_tuple():
    """Guards for the bit-identity argument: if b12x changes how it builds or uses the map, these fail on CPU."""
    from b12x.sequence.engram import _impl, _kernels, _preparation

    from ignition.recipes.deepseek41.engram import Tables

    assert "torch.tensor(token_map, dtype=torch.int64, device=caps.device)" in inspect.getsource(_preparation)
    assert "token_map = tuple(token_map)" in inspect.getsource(_preparation.make_plan)
    assert ("torch.tensor(self.token_map, dtype=torch.int64, device=device)"
            in inspect.getsource(Tables.device_token_map.func))
    # the compress kernel only loads from the map; the hash op never receives it as a mutable argument
    compress = inspect.getsource(_kernels._compress.fn)
    assert "tl.load(token_map + raw" in compress and "tl.store(token_map" not in compress
    assert "token_map" not in str(_kernels.hash_op._opoverload._schema).split("->")[0].replace("plan_handle", "")
    # b12x's own binding contract lists it read-only
    assert '("token_map", p.token_map)' in inspect.getsource(_impl)


def test_table_requests_pass_the_shared_map_only_when_switched_on(monkeypatch):
    from ignition.recipes.deepseek41 import engram as engram_module

    seen = []

    class Requests:
        def __init__(self, *args, **options):
            seen.append(options.get("shared_token_map", "missing"))

        def functional(self):
            return self

    monkeypatch.setattr(engram_module, "PreparedEngramRequests", Requests)
    device_map = torch.arange(4)

    def tables(share):
        inputs = {2: SimpleNamespace(), 3: SimpleNamespace()}
        return SimpleNamespace(device_token_map=device_map, token_map=(0, 1, 2, 3), geometry=None, sequence=None,
                               inputs=inputs, mailboxes={2: SimpleNamespace(device_view=None),
                                                         3: SimpleNamespace(device_view=None)},
                               row_mapping={2: None, 3: None}, share_token_map=share)

    metadata = SimpleNamespace(starts=torch.zeros(2, dtype=torch.int64), token_ids=None, query_start_loc=None,
                               live_counts=None)
    engram_module.PreparedTableRequests(tables(False), metadata, None, session=None, name="g")
    assert seen == [None, None]
    seen.clear()
    engram_module.PreparedTableRequests(tables(True), metadata, None, session=None, name="g")
    assert seen == [device_map, device_map]


# --- item 8: shared vocabulary columns -----------------------------------------------------------------------------

def test_vocabulary_columns_shared_only_when_switched_on():
    from ignition.components.parallel_draft import vocabulary_columns

    base = SimpleNamespace()
    a, b = vocabulary_columns(base, 16, 32, "cpu"), vocabulary_columns(base, 16, 32, "cpu")
    assert a is not b and torch.equal(a, b) and torch.equal(a, torch.arange(16) + 32) and a.dtype == torch.int64
    shared = SimpleNamespace(share_vocabulary_columns=True, vocabulary_columns={})
    c, d = vocabulary_columns(shared, 16, 32, "cpu"), vocabulary_columns(shared, 16, 32, "cpu")
    assert c is d and torch.equal(c, a)
    e = vocabulary_columns(shared, 16, 48, "cpu")  # another rank's start: its own tensor
    assert e is not c and torch.equal(e, torch.arange(16) + 48)


def test_vocabulary_columns_are_only_compared():
    """The one reader of copy_columns compares it; no round writes it (sharing cannot change a graph's values)."""
    import re

    from ignition.components import parallel_draft

    source = inspect.getsource(parallel_draft)
    uses = re.findall(r".*copy_columns.*", source)
    assert len(uses) == 2, uses
    assert any("self.copy_columns = vocabulary_columns(" in use for use in uses)
    assert any("(self.copy_columns[None, None, :] == ids[:, :, None])" in use for use in uses)


def test_speculative_preparation_threads_the_switch():
    from ignition.components.dspark import SpeculativePreparation
    from ignition.recipes.deepseek41 import bootstrap

    assert inspect.signature(SpeculativePreparation).parameters["share_vocabulary_columns"].default is False
    assert "share_vocabulary_columns=tp3.share_vocab_columns" in inspect.getsource(bootstrap.prepare_rank)
    assert "share_token_map=tp3.share_engram_token_map" in inspect.getsource(bootstrap.prepare_rank)
