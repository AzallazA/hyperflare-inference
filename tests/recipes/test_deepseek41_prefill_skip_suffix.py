# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 encoder-only prefill twins (FLARE_DS41_PREFILL_SKIP_SUFFIX): switch, catalog and the state contract.

A skipped (encoder-only) chunk must still write everything later chunks read: every row's state of layers 0-19
and layer 20's global KV (compressed KV, index keys, latent pages: the sources 2/8/14/20 serve every later layer).
It may leave stale exactly what the next chunk (always longer than the 128-row window, Helios guarantees it)
rewrites or never reads: the SWA windows of layers 20-39, the DSpark draft ring and ``last_hidden``.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.window_indices import PreparedWindowIndices
from ignition.recipes.deepseek41.config import ModelConfig
from ignition.recipes.deepseek41.decoder_prefill import (
    PreparedDecoderSuffix, PreparedEncoderGeneration, encoder_capacities,
)
from ignition.recipes.deepseek41.tp3_config import SKIP_SUFFIX_DEFAULT, TP3Config

LADDER = (128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)


# --- switch and catalog -------------------------------------------------------------------------------------------

def test_skip_suffix_switch_is_off_unrecorded_by_default_and_parses_lists():
    for value in (None, "", "0"):
        env = {} if value is None else {"FLARE_DS41_PREFILL_SKIP_SUFFIX": value}
        config = TP3Config.from_environment(env)
        assert config.prefill_skip_suffix is None
        assert "prefill_skip_suffix" not in config.to_metadata()  # a default export's record is unchanged
    on = TP3Config.from_environment({"FLARE_DS41_PREFILL_SKIP_SUFFIX": "1"})
    assert on.prefill_skip_suffix == SKIP_SUFFIX_DEFAULT == (3072, 3584, 4096)
    assert on.to_metadata()["prefill_skip_suffix"] == (3072, 3584, 4096)
    assert "prefill_skip_suffix" in on.describe()
    listed = TP3Config.from_environment({"FLARE_DS41_PREFILL_SKIP_SUFFIX": "2048,4096"})
    assert listed.prefill_skip_suffix == (2048, 4096)
    for bad in ("128,4096", "4096,3072", "3000", "4096,4096", "x"):
        with pytest.raises(ValueError, match="PREFILL_SKIP_SUFFIX|prefill_skip_suffix"):
            TP3Config.from_environment({"FLARE_DS41_PREFILL_SKIP_SUFFIX": bad})


def test_encoder_twins_must_be_captured_prefill_capacities_above_the_window():
    assert encoder_capacities(None, LADDER, chunk=4096) == ()
    assert encoder_capacities(SKIP_SUFFIX_DEFAULT, LADDER, chunk=4096) == (4096, 3584, 3072)
    with pytest.raises(ValueError, match="3584 not captured"):
        encoder_capacities(SKIP_SUFFIX_DEFAULT, (128, 256, 512, 1024, 2048, 3072, 4096), chunk=4096)
    with pytest.raises(ValueError, match="4096 not captured"):  # a smaller chunk has no 4096 graph
        encoder_capacities(SKIP_SUFFIX_DEFAULT, (128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584), chunk=3584)


def test_encoder_prefill_contract_binds_exactly_the_prefill_uploads():
    from ignition.preparation import runtime_core_pb2 as p
    from ignition.preparation.execution import GraphSignature
    from ignition.recipes.deepseek41.contract import prepare
    from ignition.recipes.deepseek41.pool import ContinuationPool, PAGE_TOKENS

    fields = ("request_state_rows", "row_slot_ids", "row_generations", "live_rows", "page_tables",
              "sampling_grammar_mask", "sample_enabled", "query_lengths", "token_ids", "prefill_positions")
    signatures = {"prefill4096": GraphSignature("prefill", 1, 4096, fields, speculation_enabled=True),
                  "encoder_prefill4096": GraphSignature("encoder_prefill", 1, 4096, fields, speculation_enabled=True)}
    model = ModelConfig({"num_hidden_layers": 40, "hidden_size": 5120, "sliding_window": 128,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20, "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36]}, {})
    contract = prepare(signatures, {}, ContinuationPool.allocate(model, 2, 3, "cpu"),
                       page_tokens=PAGE_TOKENS, draft_lag=None)
    graphs = {graph["graph"]: graph for graph in contract["graphs"]}
    base, twin = graphs["prefill4096"], graphs["encoder_prefill4096"]
    assert base["kind"] == p.PREFILL and twin["kind"] == p.ENCODER_PREFILL == 14
    assert twin["inputs"] == base["inputs"] and twin["state"] == base["state"]
    assert twin["embedding_tables"] == base["embedding_tables"]
    assert twin["pending_token"] == base["pending_token"]
    for capacity in (128, 100):
        with pytest.raises(ValueError, match="encoder-only prefill"):
            GraphSignature("encoder_prefill", 1, capacity, fields)


# --- what the encoder-only forward runs -----------------------------------------------------------------------------

def _fake_decoder(*, encoder_only, layers, decoder_start=20, rows=4096, log):
    """A PreparedDecoder shell whose components record calls; __call__ is the real method."""
    from ignition.recipes.deepseek41 import decoder as module
    from ignition.recipes.deepseek41.probe import Probe

    class Boundary:
        def __init__(self, index):
            self.index = index

        def global_only(self, incoming, pre_mix, metadata, indices, indexer):
            log.append(("global_only", self.index, metadata.positions.numel(), indexer))

        def __call__(self, incoming, pre_mix, metadata, indices, indexer):
            log.append(("boundary_gather", self.index, metadata.positions.numel(), indexer))
            return PreparedDecoderSuffix(metadata).begin(metadata), incoming, pre_mix

    def layer_stub(index):
        def call(incoming, pre_mix, metadata, indices, selection, **kwargs):
            log.append(("layer", index, metadata.positions.numel()))
            return incoming, pre_mix, None, selection
        return SimpleNamespace(boundary=Boundary(index) if index == decoder_start else None,
                               attention=SimpleNamespace(selectors=[f"selector{index}"]), call=call)

    shell = object.__new__(module.PreparedDecoder)
    stubs = [layer_stub(index) for index in range(layers)]

    class Layer:
        def __init__(self, stub):
            self.stub = stub
            self.boundary, self.engram_side, self.attention = stub.boundary, None, stub.attention

        def __call__(self, *args, **kwargs):
            return self.stub.call(*args, **kwargs)

    table_requests = type("Tables", (), {
        "__call__": lambda self, metadata: log.append(("tables",)) or "history",
        "commit": lambda self, lengths, history, metadata: log.append(("commit", history, int(lengths[0]))),
    })()
    shell.__dict__.update(
        mode="prefill", probe=Probe(False), route_capture=SimpleNamespace(enabled=False),
        table_requests=table_requests, vision=None, host_rows=None, staging_shape=None,
        weights=SimpleNamespace(embedding=torch.zeros(8, 1)),
        embedding=lambda token_ids, num_rows: torch.zeros(token_ids.numel(), 4),
        window_metadata=lambda metadata: "full-indices", aux_layers=(), auxiliary_shape=(128, 0),
        layers=[Layer(stub) for stub in stubs], encoder_only=encoder_only,
        suffix_window_metadata=lambda metadata, replay_start=None: "suffix-indices")
    return shell


def test_encoder_only_forward_runs_layers_0_19_and_layer_20_global_kv_on_every_row_then_commits():
    log = []
    shell = _fake_decoder(encoder_only=True, layers=21, log=log)
    metadata = Metadata.allocate(4096, 1, 40, "cpu")
    metadata.lengths.fill_(4096)
    assert shell(metadata) is None
    assert log[0] == ("tables",)
    assert log[1:21] == [("layer", index, 4096) for index in range(20)]
    # Layer 20: mHC + compressor + index keys + latent pages on all 4,096 rows, through layer 20's own selector.
    assert log[21] == ("global_only", 20, 4096, "selector20")
    assert log[22:] == [("commit", "history", 4096)]


def test_encoder_only_flag_does_not_change_the_ordinary_prefill_forward_order():
    # Control: the same shell without the flag gathers the suffix at layer 20 (the forward continues past it).
    log = []
    shell = _fake_decoder(encoder_only=False, layers=21, log=log)
    metadata = Metadata.allocate(4096, 1, 40, "cpu")
    metadata.lengths.fill_(4096)
    metadata.query_start_loc[1] = 4096
    metadata.positions.copy_(torch.arange(4096))
    with pytest.raises(AttributeError):  # the shell has no head: the ordinary forward went past the boundary
        shell(metadata)
    assert ("boundary_gather", 20, 4096, "selector20") in log
    assert ("layer", 20, 128) in log  # layer 20's body on the 128-row suffix
    assert not any(entry[0] == "global_only" for entry in log)


@dataclass(frozen=True)
class _LayerWeights:
    engram: object = None
    attention: dict = field(default_factory=dict)


def test_encoder_only_decoder_builds_through_the_boundary_layer_without_head(monkeypatch):
    from ignition.recipes.deepseek41 import decoder as module

    built, boundaries = [], []
    monkeypatch.setattr(module, "PreparedTableRequests", lambda *args, **kwargs: None)
    monkeypatch.setattr(module.Scratch, "allocate", staticmethod(lambda *args, **kwargs: SimpleNamespace()))
    monkeypatch.setattr(module, "PreparedEmbedding",
                        lambda *args, **kwargs: SimpleNamespace(functional=lambda **kw: "embedding"))
    monkeypatch.setattr(module, "Rotary", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "AttentionIndices", lambda *args, **kwargs: SimpleNamespace(functional=lambda: None))
    monkeypatch.setattr(module, "PreparedLayer", lambda index, layer, *args, **kwargs: built.append(
        (index, kwargs["boundary"] is not None)) or SimpleNamespace(boundary=kwargs["boundary"]))
    monkeypatch.setattr(module, "PreparedDecoderBoundary", lambda index, *args, **kwargs: boundaries.append(index) or index)
    monkeypatch.setattr(module, "configure_engram_side", lambda *args: None)
    monkeypatch.setattr(module, "register_finalizer", lambda *args: None)
    monkeypatch.setattr(module.PreparedDecoder, "_prepare_head",
                        lambda self, **kwargs: pytest.fail("an encoder-only twin has no vocabulary head"))
    weights = SimpleNamespace(config=SimpleNamespace(decoder_start=20, sources=tuple(
        SimpleNamespace(layer=layer) for layer in (2, 8, 14, 20)), draft=SimpleNamespace(target_layers=(23, 31, 39)),
        text={"compress_ratios": [0, 0] + [2] * 18 + [1] * 20}),
        draft=object(), layers=tuple(_LayerWeights() for _ in range(40)),
        shard=SimpleNamespace(world_size=3), embedding=None)
    scratch = SimpleNamespace(embedding=torch.zeros(1), hidden=torch.zeros((4096, 8)),
        local_logits=torch.zeros((1, 1)), logits=torch.zeros((1, 1)))
    decoder = module.PreparedDecoder(weights, None, scratch, Metadata.allocate(4096, 1, 40, "cpu"), None,
        mode="prefill", runtime=None, vocabulary_runtime=None, coordinator=None, session=None,
        name="encoder_prefill4096", workspace=None, encoder_only=True)
    assert [index for index, _ in built] == list(range(21))
    assert [index for index, has_boundary in built if has_boundary] == [20] and boundaries == [20]
    assert decoder.head is None and decoder.auxiliary is None and decoder.encoder_only
    # A chunk that fits the window has no suffix to skip.
    with pytest.raises(ValueError, match="larger than the decoder suffix"):
        module.PreparedDecoder(weights, None, scratch, Metadata.allocate(128, 1, 40, "cpu"), None,
            mode="prefill", runtime=None, vocabulary_runtime=None, coordinator=None, session=None,
            name="encoder_prefill128", workspace=None, encoder_only=True)


def test_encoder_generation_consumes_and_records_without_sampling_context_or_hidden():
    log = []
    forward = object.__new__(PreparedEncoderGeneration)
    metadata = SimpleNamespace(**{name: name for name in
                                  ("live_counts", "starts", "lengths", "query_start_loc", "token_ids", "positions")})
    forward.batch = SimpleNamespace(validate=lambda: log.append("validate") or ("v", "present"),
        begin=lambda value, present: log.append(("begin", value, present)) or metadata,
        finish=lambda m: log.append(("finish", m is metadata)))
    forward.validation = lambda value: log.append("all-reduce") or "summed"
    forward.decoder = lambda m: log.append(("decoder", m is metadata))
    forward.completion = SimpleNamespace(record_completion=lambda: log.append("record"),
        run=lambda *a, **k: pytest.fail("an encoder-only chunk never samples"))
    observed = forward.observe()
    assert log == ["validate", "all-reduce", ("begin", "summed", "present"), ("decoder", True),
                   ("finish", True), "record"]
    assert set(observed) == {"target." + name for name in vars(metadata)}
    with pytest.raises(ValueError, match="encoder-only decoder"):
        PreparedEncoderGeneration(SimpleNamespace(encoder_only=False), None, None, None, None,
                                  runtime=None, coordinator=None, session=None, name="x")


# --- state contract: written state and the stale state the next chunk replaces --------------------------------------

def test_layers_after_the_boundary_read_only_layer_20_global_kv():
    model = ModelConfig({"num_hidden_layers": 40, "hidden_size": 5120, "sliding_window": 128,
        "compress_ratios": [0, 0] + [2] * 18 + [1] * 20, "kv_source_layer_ids": [2, 8, 14, 20],
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36]}, {})
    assert model.decoder_start == 20
    assert [source.layer for source in model.sources] == [2, 8, 14, 20]
    # Every compressed layer reads a source at or before the boundary, all of which the encoder twin writes.
    assert {model.source(layer).layer for layer in range(40) if model.source(layer)} == {2, 8, 14, 20}
    assert {model.source(layer).layer for layer in range(20, 40)} == {20}
    # Index layers 24/28/32/36 select from layer 20's index pages (written by the twin's global KV step).
    assert all(model.source(layer).layer == 20 for layer in (24, 28, 32, 36))


def _set_chunk(metadata, start, count):
    metadata.starts.fill_(start)
    metadata.lengths.fill_(count)
    metadata.state_rows.fill_(3)
    metadata.query_start_loc.copy_(torch.tensor([0, count], dtype=torch.int32))
    metadata.live_counts.copy_(torch.tensor([count, 1], dtype=torch.int32))
    metadata.sample_indices.fill_(max(count - 1, 0))
    metadata.positions.fill_(-1)
    metadata.positions[:count] = torch.arange(start, start + count)
    metadata.token_ids[:count] = torch.arange(count) + 7


@pytest.mark.parametrize("start", [3072, 4096, 4224])
@pytest.mark.parametrize("count", [129, 512, 3072, 4096])
def test_a_successor_longer_than_the_window_never_reads_the_stale_layer_20_39_windows(start, count):
    capacity = 4096
    source = Metadata.allocate(capacity, 1, 40, "cpu")
    packing = PreparedDecoderSuffix(source)
    indices = PreparedWindowIndices(packing.metadata, mode="prefill", tokens=128, page_size=256, width=128,
                                    replay_start=packing.replay_start)
    _set_chunk(source, start, count)
    packed = packing.begin(source)
    window = indices(packed.metadata, replay_start=packed.replay_start)
    live = packed.metadata.positions >= 0
    assert int(live.sum()) == 128 and packed.replay_start.item() == start + count - 128
    # Window slots 0..127 of the staged page hold the persistent window gathered from the pool (the state an
    # encoder-only predecessor left stale); 128.. hold this chunk's rows. A > 128-row successor reads only the latter.
    read = window.window_indices[live]
    assert read[read >= 0].min().item() >= 128
    # Its commit rewrites all 128 ring slots of each layer >= 20 (attention._commit_window: tokens
    # max(0, length - 128) .. length - 1 of the chunk land at (start + token) % 128).
    committed = {(start + token) % 128 for token in range(max(0, count - 128), count)}
    assert committed == set(range(128))


@pytest.mark.parametrize("count", [1, 64, 128])
def test_a_window_sized_successor_would_read_the_stale_window(count):
    # Why Helios never schedules a <= 128-row chunk right after an encoder-only one: the ordinary (unpacked) path
    # reads the persistent window for its first rows, and the decoder suffix of a <= 128 chunk replays no boundary.
    start = 4096
    metadata = Metadata.allocate(128, 1, 40, "cpu")
    indices = PreparedWindowIndices(metadata, mode="prefill", tokens=128, page_size=256, width=128)
    _set_chunk(metadata, start, count)
    window = indices(metadata)
    read = window.window_indices[metadata.positions >= 0]
    assert read[read >= 0].min().item() < 128
    committed = {(start + token) % 128 for token in range(max(0, count - 128), count)}
    assert len(committed) == count  # a short chunk leaves 128 - count stale ring slots behind


def _context(compact):
    from ignition.components.dspark import PreparedContext

    context = object.__new__(PreparedContext)
    written = []
    context.compact = compact
    context.project = context.norm = context.context_projection = lambda value: value
    context.rotary = SimpleNamespace(apply=lambda *args, **kwargs: None)
    context.stages = [(lambda value: value, lambda value, slots: written.append(slots.clone()))]
    return context, written


@pytest.mark.parametrize("count", [129, 512, 4096])
def test_the_next_chunk_rewrites_every_draft_ring_slot(count):
    # The DSpark context of a > 128-row prefill chunk writes 128 consecutive positions: every ring slot.
    start, capacity = 4096, 4096
    source = Metadata.allocate(capacity, 1, 40, "cpu")
    _set_chunk(source, start, count)
    context, written = _context(compact=True)
    context(torch.zeros((128, 512 * 1)), source.lengths, source)
    slots = written[0]
    assert sorted(slots[slots >= 0].tolist()) == [3 * 128 + slot for slot in range(128)]


def test_last_hidden_is_rewritten_by_the_next_chunk():
    from ignition.components.state import save_hidden

    last_hidden = torch.full((4, 6), -7.0)  # row 3 holds the stale value the skipped chunk did not refresh
    hidden = torch.arange(6, dtype=torch.float32).view(1, 6)
    save_hidden(hidden, last_hidden, torch.tensor([3]), torch.tensor([True]))
    assert torch.equal(last_hidden[3], hidden[0])
    assert torch.equal(last_hidden[:3], torch.full((3, 6), -7.0))
