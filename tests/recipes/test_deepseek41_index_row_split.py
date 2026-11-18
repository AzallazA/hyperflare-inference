# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 indexer row split (``TP3Config.index_row_split``, index_row_split.py), CPU only.

- geometry: contiguous uneven block ownership (4096 rows: 11 / 11 / 10 blocks), channel rows on the FP32 vocabulary
  channel (86,272 BF16 lanes per rank at TP3; 4096 rows need 17);
- pack / emulated column gather / unpack is a bit copy (every int32 pattern, BF16 NaN halves, 0x8000 halves);
- reference-model equivalence: a CPU top-512 selector over three emulated ranks, split + exchange == the replicated
  selection on every rank; with rank-dependent inputs, every rank holds the owners' selections (the identity
  precondition in PLAN.md);
- ``PreparedIndexExchange`` call path through a stand-in gather; channel and transport refusals;
- switch parsing, validation, record, vocabulary channel sizing and which graphs split.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.deepseek41 import index_row_split as split_module
from ignition.recipes.deepseek41.index_row_split import (ROW_SPLIT_DEFAULT, ExchangeLayout, IndexRowSplitContext,
                                                         RowSplit, channel_lanes, emulate_gather, exchange_rows, pack,
                                                         prefill_split_context, split_capacities, unpack)
from ignition.recipes.deepseek41.tp3_config import TP3Config

WORLD = 3
LANES = 86272  # TP3 FP32 vocabulary channel: padded 43,136-column shard x 2 BF16 lanes


# --- geometry ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("capacity,blocks", [(4096, (11, 11, 10)), (3584, (10, 9, 9)), (3072, (8, 8, 8)),
                                             (512, (2, 1, 1)), (384, (1, 1, 1))])
def test_blocks_are_contiguous_uneven_and_cover_the_chunk(capacity, blocks):
    split = RowSplit(capacity, WORLD)
    assert tuple(len(split.owned_blocks(rank)) for rank in range(WORLD)) == blocks
    owned = [ordinal for rank in range(WORLD) for ordinal in split.owned_blocks(rank)]
    assert owned == list(range(split.blocks))
    rows = [split.rows(rank) for rank in range(WORLD)]
    assert rows[0][0] == 0 and rows[-1][1] == capacity
    assert all(rows[i][1] == rows[i + 1][0] for i in range(WORLD - 1))
    assert split.max_rows == max(blocks) * 128


def test_short_last_block_is_clipped():
    split = RowSplit(448, WORLD)  # 4 blocks, the last one 64 rows
    assert split.row_counts() == (256, 128, 64)
    assert split.rows(2) == (384, 448)


def test_geometry_refusals():
    with pytest.raises(ValueError):
        RowSplit(256, WORLD)  # two blocks for three ranks
    with pytest.raises(ValueError):
        RowSplit(4096, 1)
    with pytest.raises(ValueError):
        RowSplit(4096, WORLD).owned_blocks(3)
    with pytest.raises(ValueError):
        ExchangeLayout(RowSplit(4096, WORLD), 101)


def test_tp3_vocabulary_channel_rows():
    assert channel_lanes(WORLD) == LANES
    layout = ExchangeLayout(RowSplit(4096, WORLD), LANES)
    assert layout.channel_rows == 17  # 1408 x 512 x 4 B = 2.75 MiB over 172,544-byte channel rows
    assert layout.payload_bytes == 17 * LANES * 2
    assert exchange_rows(ROW_SPLIT_DEFAULT, WORLD) == 17
    assert exchange_rows((3072,), WORLD) == 13
    assert exchange_rows((), WORLD) == 0


# --- bit-exact transport -------------------------------------------------------------------------------------------

def _patterns(rows, seed):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randint(-2 ** 31, 2 ** 31 - 1, (rows, 512), generator=generator, dtype=torch.int64)
    values = values.to(torch.int32)
    special = torch.tensor([-1, 0, 0x8000, 0x80000000 - 2 ** 32, 0x7FC00000, 0x7F810000, 0x00007FC1, 0xFFFF7F81 - 2 ** 32,
                            0x80008000 - 2 ** 32, 0x7F807F80, 524287, 1048575], dtype=torch.int64).to(torch.int32)
    values[0, :special.numel()] = special
    values[-1, -special.numel():] = special
    return values


@pytest.mark.parametrize("capacity", [4096, 3584, 3072, 448])
def test_pack_gather_unpack_is_a_bit_copy(capacity):
    layout = ExchangeLayout(RowSplit(capacity, WORLD), LANES)
    full = _patterns(capacity, capacity)
    partitions = [pack(full[slice(*layout.split.rows(rank))], layout, rank) for rank in range(WORLD)]
    assert all(p.dtype == torch.int32 and tuple(p.shape) == (layout.channel_rows, LANES // 2) for p in partitions)
    gathered = emulate_gather(partitions)
    result = unpack(gathered, layout)
    assert result.dtype == torch.int32 and torch.equal(result, full)


def test_pack_refuses_the_wrong_share():
    layout = ExchangeLayout(RowSplit(4096, WORLD), LANES)
    with pytest.raises(ValueError):
        pack(torch.zeros((1280, 512), dtype=torch.int32), layout, 0)  # rank 0 owns 1408 rows
    with pytest.raises(ValueError):
        pack(torch.zeros((1408, 512), dtype=torch.int64), layout, 0)
    with pytest.raises(ValueError):
        unpack(torch.zeros((16, 3 * LANES // 2), dtype=torch.int32), layout)
    with pytest.raises(ValueError):
        unpack(torch.zeros((17, 3 * LANES), dtype=torch.bfloat16), layout)


def test_compiled_pack_and_unpack_stay_bit_exact():
    """Inductor (CPU) traces the int32 transforms and keeps every bit, including -1 and BF16-NaN-shaped halves."""
    layout = ExchangeLayout(RowSplit(4096, WORLD), LANES)
    full = _patterns(4096, 5)
    owns = [full[slice(*layout.split.rows(rank))].clone() for rank in range(WORLD)]

    def step(a, b, c):
        return unpack(emulate_gather([pack(a, layout, 0), pack(b, layout, 1), pack(c, layout, 2)]), layout)

    assert torch.equal(torch.compile(step, fullgraph=True)(*owns), full)


# --- reference-model equivalence -----------------------------------------------------------------------------------

def _reference_select(query, weights, keys, lengths, topk=512):
    """CPU stand-in for one DSA selector call: score_k = sum_h relu(q_h . k) * w_h over the row's visible keys,
    top-``topk`` key indices in ascending order, -1 padding (the b12x indexer's output contract)."""
    scores = (torch.einsum("rhd,kd->rhk", query.float(), keys.float()).relu() * weights.float()[..., None]).sum(1)
    visible = torch.arange(keys.shape[0])[None] < lengths[:, None]
    scores = scores.masked_fill(~visible, -torch.inf)
    count = min(topk, keys.shape[0])
    values, chosen = scores.topk(count, dim=1, sorted=True)
    chosen = torch.where(torch.isneginf(values), torch.iinfo(torch.int32).max, chosen)
    ordered = chosen.sort(dim=1).values
    ordered = torch.where(ordered == torch.iinfo(torch.int32).max, -1, ordered).to(torch.int32)
    out = torch.full((query.shape[0], topk), -1, dtype=torch.int32)
    out[:, :count] = ordered
    return out


def _replicated(query, weights, keys, lengths, block=128):
    """Today's prefill path: one selector per 128-row block over every row, concatenated."""
    return torch.cat([_reference_select(query[first:first + block], weights[first:first + block], keys,
                                        lengths[first:first + block])
                      for first in range(0, query.shape[0], block)])


def _split(inputs, layout, keys):
    """The split path on three emulated ranks: each rank's owned blocks, pack, column gather, unpack on every rank.
    ``inputs[rank]`` = that rank's (query, weights, lengths) for the whole chunk."""
    split = layout.split
    partitions = []
    for rank in range(split.world):
        query, weights, lengths = inputs[rank]
        outputs = []
        for ordinal in split.owned_blocks(rank):
            first, last = ordinal * split.block_rows, min(split.capacity, (ordinal + 1) * split.block_rows)
            outputs.append(_reference_select(query[first:last], weights[first:last], keys, lengths[first:last]))
        partitions.append(pack(torch.cat(outputs), layout, rank))
    gathered = emulate_gather(partitions)
    return [unpack(gathered, layout) for _ in range(split.world)]  # every rank unpacks the same gathered bytes


def _chunk(capacity, keys_count, context, seed):
    generator = torch.Generator().manual_seed(seed)
    # 8 index heads instead of 32 keep the CPU reference fast; the selection contract does not depend on the count.
    query = (torch.randn((capacity, 8, 128), generator=generator) * .2).bfloat16()
    weights = (torch.randn((capacity, 8), generator=generator) / 8).bfloat16()
    keys = (torch.randn((keys_count, 128), generator=generator) * .2).bfloat16()
    # Causal prefill chunk at the end of `context` tokens, ratio-2 compressed keys: row i sees (start + i + 1) // 2.
    start = context - capacity
    lengths = ((start + torch.arange(capacity) + 1) // 2).clamp(max=keys_count).to(torch.int32)
    lengths[-3:] = 0  # padded rows past the live prompt select nothing
    return query, weights, keys, lengths


@pytest.mark.parametrize("capacity,context", [(3072, 4096), (3584, 6144), (448, 2048)])
def test_split_exchange_equals_the_replicated_selection_on_every_rank(capacity, context):
    query, weights, keys, lengths = _chunk(capacity, context // 2, context, seed=capacity)
    full = _replicated(query, weights, keys, lengths)
    assert (full[:, 0] >= 0).sum() > 0 and (full == -1).any()  # some rows short of 512 visible keys
    layout = ExchangeLayout(RowSplit(capacity, WORLD), LANES)
    for selected in _split([(query, weights, lengths)] * WORLD, layout, keys):
        assert torch.equal(selected, full)


def test_rank_dependent_inputs_give_every_rank_the_owners_selection():
    """When a rank's indexer input differs (one-shot all-reduce tail rows), the split hands every rank the owner's
    selection: identical across ranks, equal to the owner's replicated result, different from a non-owner's."""
    capacity, context = 3072, 4096
    query, weights, keys, lengths = _chunk(capacity, context // 2, context, seed=7)
    layout = ExchangeLayout(RowSplit(capacity, WORLD), LANES)
    inputs, replicated = [], []
    for rank in range(WORLD):
        q = query.clone()
        if rank == 1:
            q[1100:1200] = (q[1100:1200].float() * 1.5).bfloat16()  # rows owned by rank 1 (1024..2047)
            q[2100:2200] = -q[2100:2200]  # rows owned by rank 2: rank 1's own replicated view differs there
        inputs.append((q, weights, lengths))
        replicated.append(_replicated(q, weights, keys, lengths))
    results = _split(inputs, layout, keys)
    owners = torch.cat([replicated[rank][slice(*layout.split.rows(rank))] for rank in range(WORLD)])
    for selected in results:
        assert torch.equal(selected, owners)
    assert not torch.equal(replicated[1], owners)  # rank 1's replicated view of rank 2's rows is not what it gets
    assert not torch.equal(replicated[0], owners)  # rank 0 now reads rank 1's selection of rank 1's rows


# --- the prepared exchange's call path -----------------------------------------------------------------------------

class _ChannelStub:
    world_size, heads_per_rank, query_head_dim = WORLD, LANES // 64, 64

    def __init__(self, rows):
        self.max_batch_size = rows


def _stub_channel(monkeypatch, rows, partitions_by_rank, ranks):
    """PreparedAllGather planning replaced by a recorder; the column transport by a copy of the emulated gather of
    every rank's partition (``ranks[0]`` = the calling rank, the stand-in publishes its source there)."""
    from ignition.components import collectives

    class Gather:
        handle = 4101

        def __init__(self, source, output, *, runtime, session, coordinator, name):
            assert source.dtype == output.dtype == torch.bfloat16
            assert source.shape == (rows, LANES) and output.shape == (rows, WORLD * LANES)

        def functional(self, *, session):
            return self

    def transport(source, output, handle):
        assert handle == Gather.handle and source.dtype == output.dtype == torch.bfloat16
        partitions_by_rank[ranks[0]] = source.view(torch.int32).clone()
        output.view(torch.int32).copy_(emulate_gather(partitions_by_rank))

    monkeypatch.setattr(collectives, "PreparedAllGather", Gather)
    monkeypatch.setattr(collectives, "_gather_columns", transport)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: False)


def test_prepared_exchange_packs_gathers_and_unpacks(monkeypatch):
    capacity = 4096
    full = _patterns(capacity, 11)
    split = RowSplit(capacity, WORLD)
    layout = ExchangeLayout(split, LANES)
    partitions = [pack(full[slice(*split.rows(rank))], layout, rank) for rank in range(WORLD)]
    calling = [0]
    _stub_channel(monkeypatch, layout.channel_rows, partitions, calling)
    for rank in range(WORLD):
        calling[0] = rank
        context = IndexRowSplitContext(split, rank, _ChannelStub(layout.channel_rows), coordinator=None, examples={})
        exchange = split_module.PreparedIndexExchange(context, "cpu", session=None, name="layer2.index_exchange")
        exchange.functional(session=None)
        assert torch.equal(exchange(full[slice(*split.rows(rank))]), full)
    # Traced as a serving forward traces it: the custom op's output mutation and the int32 transforms compile.
    compiled = torch.compile(lambda own: exchange(own), fullgraph=True)
    assert torch.equal(compiled(full[slice(*split.rows(WORLD - 1))].clone()), full)


def test_prepared_exchange_refuses_a_short_channel_and_roce(monkeypatch):
    from ignition.components import collectives

    split = RowSplit(4096, WORLD)
    _stub_channel(monkeypatch, 17, [None] * WORLD, [0])
    with pytest.raises(ValueError, match="17 vocabulary channel rows"):
        split_module.PreparedIndexExchange(IndexRowSplitContext(split, 0, _ChannelStub(16), None), "cpu",
                                           session=None, name="x")
    with pytest.raises(ValueError, match="PCIe"):
        split_module.PreparedIndexExchange(IndexRowSplitContext(split, 0, None, None), "cpu", session=None, name="x")
    monkeypatch.setattr(collectives, "_roce", lambda runtime: True)
    with pytest.raises(ValueError, match="PCIe"):
        split_module.PreparedIndexExchange(IndexRowSplitContext(split, 0, _ChannelStub(17), None), "cpu",
                                           session=None, name="x")


# --- switch, record, channel sizing, graph selection -----------------------------------------------------------------

def test_switch_parsing_validation_and_record():
    assert TP3Config().index_row_split is None
    assert "index_row_split" not in TP3Config().to_metadata()  # a default export's record is unchanged
    assert TP3Config.from_environment({}).index_row_split is None
    assert TP3Config.from_environment({"FLARE_DS41_INDEX_ROW_SPLIT": "0"}).index_row_split is None
    on = TP3Config.from_environment({"FLARE_DS41_INDEX_ROW_SPLIT": "1"})
    assert on.index_row_split == ROW_SPLIT_DEFAULT == (3072, 3584, 4096)
    assert on.to_metadata()["index_row_split"] == (3072, 3584, 4096)
    assert "index_row_split=" in on.describe()
    assert TP3Config.from_environment({"FLARE_DS41_INDEX_ROW_SPLIT": "4096"}).index_row_split == (4096,)
    for bad in ((256,), (4000,), (4096, 3072), (3072, 3072)):
        with pytest.raises(ValueError, match="index_row_split"):
            TP3Config(index_row_split=bad)
    with pytest.raises(ValueError):
        TP3Config.from_environment({"FLARE_DS41_INDEX_ROW_SPLIT": "big"})


def _config(**tp3):
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    return PreparationConfig(devices=(0, 1, 2), max_active=8, dspark=True, mtp_only=True,
                             vocabulary_parallel_sampling=True, speculative_widths=tuple(range(1, 7)),
                             tp3=TP3Config(**tp3))


def test_vocabulary_channel_grows_only_for_the_split():
    from ignition.recipes.deepseek41.bootstrap import vocabulary_channels

    base = _config()
    samples = base.request_capacity * max(base.speculative_widths)
    assert vocabulary_channels(_config(trim_vocab_gather=True), samples) == (8, True)
    assert vocabulary_channels(_config(trim_vocab_gather=True, index_row_split=ROW_SPLIT_DEFAULT), samples) == (17, True)
    assert vocabulary_channels(_config(trim_vocab_gather=True, index_row_split=(3072,)), samples) == (13, True)
    assert vocabulary_channels(_config(index_row_split=ROW_SPLIT_DEFAULT), samples) == (48, True)  # already wider
    assert vocabulary_channels(base, samples) == (48, True)


def test_split_capacities_must_be_captured_prefill_graphs():
    assert split_capacities(None, (128, 256, 4096), chunk=4096) == ()
    assert split_capacities((3072, 4096), (128, 3072, 3584, 4096), chunk=4096) == (3072, 4096)
    with pytest.raises(ValueError, match="3584"):
        split_capacities((3584, 4096), (128, 3072, 4096), chunk=4096)
    with pytest.raises(ValueError, match="INDEX_ROW_SPLIT"):
        _config(index_row_split=(4224,))


def test_only_listed_prefill_graphs_without_scoring_split():
    tp3 = TP3Config(index_row_split=(3072, 4096))
    shard = SimpleNamespace(rank=1, world_size=WORLD)
    runtime, coordinator = object(), object()

    def context(**overrides):
        arguments = dict(mode="prefill", capacity=4096, scoring=None, tp3=tp3, shard=shard, runtime=runtime,
                         coordinator=coordinator)
        return prefill_split_context(**{**arguments, **overrides})

    split = context()
    assert split.split == RowSplit(4096, WORLD) and split.rank == 1 and split.runtime is runtime
    assert context(capacity=3584) is None
    assert context(mode="decode") is None and context(mode="verify") is None
    assert context(scoring=object()) is None
    assert context(tp3=TP3Config()) is None
    assert context(shard=SimpleNamespace(rank=0, world_size=1)) is None


def test_attention_splits_only_index_layers_before_the_candidate_source():
    """Wiring guard (source level): the split is limited to prefill index layers before the candidate source, runs
    the owned selector ordinals only, and exchanges right after the concatenation."""
    import inspect

    from ignition.recipes.deepseek41 import attention, decoder

    init = inspect.getsource(attention.PreparedAttention.__init__)
    assert 'index_split is not None and mode == "prefill" and layer < candidate_source' in init
    run = inspect.getsource(attention.PreparedAttention._run)
    assert "ordinals = range(len(self.selectors)) if self.index_owned is None else self.index_owned" in run
    assert run.index("selected = torch.cat(outputs, dim=0)") < run.index("selected = self.index_exchange(selected)")
    assert attention.INDEXER_QUERY_ROWS == split_module.BLOCK_ROWS
    layer = inspect.getsource(decoder.PreparedDecoder.__init__)
    assert "index_split=index_split if layer_metadata is metadata else None" in layer  # never the 128-row suffix
