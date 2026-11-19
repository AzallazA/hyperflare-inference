# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 indexer row split (``TP3Config.index_row_split``, index_row_split.py), CPU only.

- geometry: contiguous uneven block ownership (4096 rows: 11 / 11 / 10 blocks), equal per-rank windows (rank-symmetric
  trace: 11 selector launches on every rank), channel rows on the FP32 vocabulary channel (86,272 BF16 lanes per rank
  at TP3; 4096 rows need 17);
- rank symmetry under the export's compiled-forward binding (``aot_forward`` + ``guard_check``): the earlier pattern
  (Python ownership ordinals) fails the guards on the other ranks, the window pattern binds on all three;
- pack / emulated column gather / unpack is a bit copy (every int32 pattern, BF16 NaN halves, 0x8000 halves);
- reference-model equivalence: a CPU top-512 selector over three emulated ranks, split + exchange == the replicated
  selection on every rank; with rank-dependent inputs, every rank holds the owners' selections (the identity
  precondition in PLAN.md);
- ``PreparedIndexExchange`` call path through a stand-in gather; channel and transport refusals;
- switch parsing, validation, record, vocabulary channel sizing and which graphs split;
- check mode (debug-serve): checksums in the padding (channel rows unchanged), transport mismatches found per source
  rank, structural row check against the lengths, the counter op through the prepared call path (eager and compiled).
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
                                             (512, (2, 1, 1)), (384, (1, 1, 1)), (640, (2, 2, 1))])
def test_blocks_are_contiguous_uneven_and_cover_the_chunk(capacity, blocks):
    split = RowSplit(capacity, WORLD)
    assert tuple(len(split.owned_blocks(rank)) for rank in range(WORLD)) == blocks
    owned = [ordinal for rank in range(WORLD) for ordinal in split.owned_blocks(rank)]
    assert owned == list(range(split.blocks))
    rows = [split.rows(rank) for rank in range(WORLD)]
    assert rows[0][0] == 0 and rows[-1][1] == capacity
    assert all(rows[i][1] == rows[i + 1][0] for i in range(WORLD - 1))
    assert split.max_rows == max(blocks) * 128


@pytest.mark.parametrize("capacity,starts,offsets", [(4096, (0, 11, 21), (0, 0, 128)), (3584, (0, 10, 18), (0, 0, 128)),
                                                      (3072, (0, 8, 16), (0, 0, 0)), (640, (0, 2, 3), (0, 0, 128))])
def test_windows_have_one_size_and_contain_the_owned_blocks(capacity, starts, offsets):
    split = RowSplit(capacity, WORLD)
    assert tuple(split.window_start(rank) for rank in range(WORLD)) == starts
    assert split.window_offsets() == offsets
    for rank in range(WORLD):
        rows = split.window_rows(rank)
        assert rows.dtype == torch.int64 and rows.numel() == split.window_rows_count == split.window_blocks * 128
        first, last = split.rows(rank)
        assert int(rows[0]) <= first and last <= int(rows[-1]) + 1 <= capacity
        assert int(rows[0]) + split.window_offsets()[rank] == first


def test_short_last_block_is_refused():
    with pytest.raises(ValueError, match="whole"):
        RowSplit(448, WORLD)


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
    assert exchange_rows((3584,), WORLD) == 16
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


def _windows(full, split):
    return [full.index_select(0, split.window_rows(rank)) for rank in range(split.world)]


@pytest.mark.parametrize("capacity", [4096, 3584, 3072, 640])
def test_pack_gather_unpack_is_a_bit_copy(capacity):
    layout = ExchangeLayout(RowSplit(capacity, WORLD), LANES)
    full = _patterns(capacity, capacity)
    partitions = [pack(window, layout) for window in _windows(full, layout.split)]
    assert all(p.dtype == torch.int32 and tuple(p.shape) == (layout.channel_rows, LANES // 2) for p in partitions)
    gathered = emulate_gather(partitions)
    result = unpack(gathered, layout)
    assert result.dtype == torch.int32 and torch.equal(result, full)


def test_pack_refuses_the_wrong_share():
    layout = ExchangeLayout(RowSplit(4096, WORLD), LANES)
    with pytest.raises(ValueError):
        pack(torch.zeros((1280, 512), dtype=torch.int32), layout)  # every window is 1408 rows
    with pytest.raises(ValueError):
        pack(torch.zeros((1408, 512), dtype=torch.int64), layout)
    with pytest.raises(ValueError):
        unpack(torch.zeros((16, 3 * LANES // 2), dtype=torch.int32), layout)
    with pytest.raises(ValueError):
        unpack(torch.zeros((17, 3 * LANES), dtype=torch.bfloat16), layout)


def test_compiled_pack_and_unpack_stay_bit_exact():
    """Inductor (CPU) traces the int32 transforms and keeps every bit, including -1 and BF16-NaN-shaped halves."""
    layout = ExchangeLayout(RowSplit(4096, WORLD), LANES)
    full = _patterns(4096, 5)
    owns = _windows(full, layout.split)

    def step(a, b, c):
        return unpack(emulate_gather([pack(a, layout), pack(b, layout), pack(c, layout)]), layout)

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
    """The split path on three emulated ranks, as attention runs it: gather the window rows through the rank's row-id
    tensor, the same number of selector blocks on every rank, pack, column gather, unpack on every rank.
    ``inputs[rank]`` = that rank's (query, weights, lengths) for the whole chunk."""
    split = layout.split
    partitions = []
    for rank in range(split.world):
        rows = split.window_rows(rank)
        query, weights, lengths = (value.index_select(0, rows) for value in inputs[rank])
        outputs = [_reference_select(query[first:first + split.block_rows], weights[first:first + split.block_rows],
                                     keys, lengths[first:first + split.block_rows])
                   for first in range(0, split.window_rows_count, split.block_rows)]
        partitions.append(pack(torch.cat(outputs), layout))
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


@pytest.mark.parametrize("capacity,context", [(3072, 4096), (3584, 6144), (640, 2048)])
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


def test_window_overlap_rows_come_from_their_owner():
    """640 rows: windows of two blocks; rank 2's window starts in rank 1's block 3. Rows a rank computes without owning
    them are dropped, even where that rank's inputs differ."""
    capacity, context = 640, 2048
    query, weights, keys, lengths = _chunk(capacity, context // 2, context, seed=9)
    layout = ExchangeLayout(RowSplit(capacity, WORLD), LANES)
    assert layout.split.window_start(2) == 3 and layout.split.owned_blocks(1) == range(2, 4)
    inputs = []
    for rank in range(WORLD):
        q = query.clone()
        if rank == 2:
            q[384:512] = -q[384:512]  # block 3: in rank 2's window, owned by rank 1
        inputs.append((q, weights, lengths))
    full = _replicated(query, weights, keys, lengths)
    for selected in _split(inputs, layout, keys):
        assert torch.equal(selected, full)


# --- the prepared exchange's call path -----------------------------------------------------------------------------

class _ChannelStub:
    world_size, heads_per_rank, query_head_dim = WORLD, LANES // 64, 64

    def __init__(self, rows):
        self.max_batch_size = rows


class _StubGather:
    """PreparedAllGather planning stand-in (module level: compiled-forward guards serialize the objects they read)."""
    handle = 4101
    rows = 0

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        assert source.dtype == output.dtype == torch.bfloat16
        assert source.shape == (self.rows, LANES) and output.shape == (self.rows, WORLD * LANES)

    def functional(self, *, session):
        return self


def _stub_channel(monkeypatch, rows, partitions_by_rank, ranks):
    """PreparedAllGather planning replaced by a recorder; the column transport by a copy of the emulated gather of
    every rank's partition (``ranks[0]`` = the calling rank, the stand-in publishes its source there)."""
    from ignition.components import collectives

    def transport(source, output, handle):
        assert handle == _StubGather.handle and source.dtype == output.dtype == torch.bfloat16
        partitions_by_rank[ranks[0]] = source.view(torch.int32).clone()
        output.view(torch.int32).copy_(emulate_gather(partitions_by_rank))

    monkeypatch.setattr(_StubGather, "rows", rows)
    monkeypatch.setattr(collectives, "PreparedAllGather", _StubGather)
    monkeypatch.setattr(collectives, "_gather_columns", transport)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: False)


def _exchanges(monkeypatch, split, full):
    """Three ranks' PreparedIndexExchange on a stand-in channel whose other partitions hold ``full``'s windows."""
    layout = ExchangeLayout(split, LANES)
    partitions = [pack(window, layout) for window in _windows(full, split)]
    calling = [0]
    _stub_channel(monkeypatch, layout.channel_rows, partitions, calling)
    exchanges = []
    for rank in range(WORLD):
        context = IndexRowSplitContext(split, rank, _ChannelStub(layout.channel_rows), coordinator=None, examples={})
        exchange = split_module.PreparedIndexExchange(context, "cpu", session=None, name="layer2.index_exchange")
        exchanges.append(exchange.functional(session=None))
    return exchanges, calling


def test_prepared_exchange_packs_gathers_and_unpacks(monkeypatch):
    capacity = 4096
    full = _patterns(capacity, 11)
    split = RowSplit(capacity, WORLD)
    exchanges, calling = _exchanges(monkeypatch, split, full)
    for rank, exchange in enumerate(exchanges):
        calling[0] = rank
        assert exchange.rows.dtype == torch.int64 and torch.equal(exchange.rows, split.window_rows(rank))
        assert torch.equal(exchange(exchange.window(full)), full)
    # Traced as a serving forward traces it: the custom op's output mutation and the int32 transforms compile.
    compiled = torch.compile(lambda value: exchange(exchange.window(value)), fullgraph=True)
    assert torch.equal(compiled(full.clone()), full)


class _OwnedOrdinalsForward:
    """The first commit's pattern: the rank's owned selector ordinals as Python ints in the traced forward."""

    def __init__(self, split, rank, scores):
        self.owned, self.scores = tuple(split.owned_blocks(rank)), scores

    def __call__(self):
        return torch.cat([self.scores[o * 128:(o + 1) * 128].topk(512, dim=1).indices for o in self.owned])


class _WindowForward:
    """The split's traced shape since the D1 fix: window gather through the rank's row-id tensor, the same number of
    128-row "selector" launches on every rank (a top-512 stand-in), then the exchange."""

    def __init__(self, exchange, scores):
        self.exchange, self.scores = exchange, scores

    def __call__(self):
        window = self.exchange.window(self.scores)
        blocks = [window[first:first + 128].topk(512, dim=1).indices.sort(dim=1).values.to(torch.int32)
                  for first in range(0, window.shape[0], 128)]
        return self.exchange(torch.cat(blocks))


def _stand_in_selection(scores):
    return torch.cat([scores[first:first + 128].topk(512, dim=1).indices.sort(dim=1).values.to(torch.int32)
                      for first in range(0, scores.shape[0], 128)])


@torch.no_grad()
def test_split_forward_binds_one_compiled_artifact_on_every_rank(monkeypatch):
    """Review D1: the export compiles one forward per prefill graph on one rank and every rank binds it
    (``aot_forward`` + ``guard_check``, as ``ForwardCache._load``). Ownership ordinals as Python ints fail the guards on
    the other ranks; the row-id tensor window passes on all three and each rank's compiled call returns the owners'
    selection, identical to the replicated one."""
    from ignition.preparation.aot import aot_forward

    capacity = 4096
    split = RowSplit(capacity, WORLD)
    scores = torch.randn((capacity, 1024), generator=torch.Generator().manual_seed(3))
    old = [_OwnedOrdinalsForward(split, rank, scores) for rank in range(WORLD)]
    compiled_old = aot_forward(old[0])
    assert compiled_old.guard_check(old[0])
    assert not compiled_old.guard_check(old[1]) and not compiled_old.guard_check(old[2])

    full = _stand_in_selection(scores)
    exchanges, calling = _exchanges(monkeypatch, split, full)
    new = [_WindowForward(exchange, scores) for exchange in exchanges]
    compiled_new = aot_forward(new[0])
    for rank, forward in enumerate(new):
        assert compiled_new.guard_check(forward), f"rank {rank} cannot bind rank 0's compiled split forward"
        calling[0] = rank
        assert torch.equal(compiled_new(forward), full)
        assert torch.equal(forward(), full)


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


# --- check mode (C1: transport checksums + structural rows; G0 ruled out a second-launch reference) ------------------

@pytest.mark.parametrize("capacity,rows", [(4096, 17), (3584, 16), (3072, 13)])
def test_check_mode_checksums_fit_the_padding(capacity, rows):
    plain, checked = ExchangeLayout(RowSplit(capacity, WORLD), LANES), ExchangeLayout(RowSplit(capacity, WORLD), LANES,
                                                                                         check=True)
    assert plain.channel_rows == checked.channel_rows == rows  # same channel and plans as the export
    assert checked.check_words == checked.split.window_rows_count
    assert exchange_rows(ROW_SPLIT_DEFAULT, WORLD, lanes=LANES, check=True) == 17
    # A layout without spare lanes grows by the checksum words.
    tight = ExchangeLayout(RowSplit(384, WORLD), 1024, check=True)
    assert tight.channel_rows == ExchangeLayout(RowSplit(384, WORLD), 1024).channel_rows + 1


def test_checksums_are_integer_exact_and_position_weighted():
    rows = _patterns(64, 21)
    sums = split_module.row_checksums(rows)
    assert sums.dtype == torch.int32
    expected = [sum((int(v) & 0xFFFFFFFF) * (2 * j + 1) for j, v in enumerate(row.tolist())) % (2 ** 31 - 1)
                for row in rows]
    assert sums.tolist() == expected
    swapped = rows.clone()
    swapped[3, [5, 6]] = swapped[3, [6, 5]]
    assert (split_module.row_checksums(swapped) != sums).nonzero().flatten().tolist() == [3]


@pytest.mark.parametrize("capacity", [4096, 3584, 3072])
def test_checked_transport_counts_corrupted_rows_per_source_rank(capacity):
    layout = ExchangeLayout(RowSplit(capacity, WORLD), LANES, check=True)
    full = _patterns(capacity, capacity + 1)
    partitions = [pack(window, layout) for window in _windows(full, layout.split)]
    gathered = emulate_gather(partitions)
    assert torch.equal(unpack(gathered, layout), full)  # checksums never leak into the winners
    assert split_module.transport_mismatches(gathered, layout).tolist() == [0, 0, 0]
    columns, width = LANES // 2, 512
    bad = gathered.clone()
    flat = bad[:, columns:2 * columns].reshape(-1).clone()  # rank 1's partition
    flat[7 * width + 3] ^= 1        # one bit of window row 7
    flat[100 * width:101 * width] = flat[100 * width:101 * width].flip(0)  # row 100 reversed
    bad[:, columns:2 * columns] = flat.view(layout.channel_rows, columns)
    assert split_module.transport_mismatches(bad, layout).tolist() == [0, 2, 0]
    with pytest.raises(ValueError, match="check-mode"):
        split_module.transport_mismatches(gathered, ExchangeLayout(layout.split, LANES))


def _valid_selection(lengths, seed):
    """A selector-shaped [rows, 512] int32: min(length, 512) distinct ascending ids in [0, length), then -1."""
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for length in lengths.tolist():
        used = min(length, 512)
        ids = torch.randperm(max(length, 1), generator=generator)[:used].sort().values if used else torch.zeros(0)
        rows.append(torch.cat((ids.to(torch.int32), torch.full((512 - used,), -1, dtype=torch.int32))))
    return torch.stack(rows)


def test_structural_check_finds_invalid_prefixes_and_counts_dirty_tails():
    lengths = torch.tensor([0, 1, 5, 511, 512, 513, 4000, 70000] * 4, dtype=torch.int32)
    clean = _valid_selection(lengths, 1)
    assert split_module.structural_counts(clean, lengths).tolist() == [0, 0]
    shuffled = clean.clone()
    shuffled[6, :512] = shuffled[6, torch.randperm(512)]  # order does not matter, the set does
    assert split_module.structural_counts(shuffled, lengths).tolist() == [0, 0]
    bad = clean.clone()
    bad[2, 1] = bad[2, 0]           # duplicate inside the prefix
    bad[3, 0] = 511                 # = length: out of range
    bad[6, 9] = -1                  # sentinel inside the prefix
    bad[8, 0] = 7                   # length 0: tail entry, informational only
    bad[10, 7] = 99                 # length 5: entry past the prefix, informational only
    assert split_module.structural_counts(bad, lengths).tolist() == [3, 2]


def _checked_exchanges(monkeypatch, split, full, transport_fault=None):
    layout = ExchangeLayout(split, LANES, check=True)
    partitions = [pack(window, layout) for window in _windows(full, split)]
    calling = [0]
    _stub_channel(monkeypatch, layout.channel_rows, partitions, calling)
    if transport_fault is not None:
        from ignition.components import collectives

        clean = collectives._gather_columns

        def faulty(source, output, handle):
            clean(source, output, handle)
            transport_fault(output.view(torch.int32))

        monkeypatch.setattr(collectives, "_gather_columns", faulty)
    counters = {}
    monkeypatch.setattr(split_module, "_CHECK_STATE", counters)
    exchanges = []
    for rank in range(WORLD):
        context = IndexRowSplitContext(split, rank, _ChannelStub(layout.channel_rows), None, {}, check=True)
        exchanges.append(split_module.PreparedIndexExchange(context, "cpu", session=None, name="l2.index_exchange",
                                                            layer=2).functional(session=None))
    return exchanges, calling


def test_check_mode_exchange_counts_through_the_prepared_call(monkeypatch):
    split = RowSplit(4096, WORLD)
    lengths = (torch.arange(4096, dtype=torch.int32) * 7 + 100) // 4  # prefill: lengths grow with the row
    full = _valid_selection(lengths, 2)
    exchanges, calling = _checked_exchanges(monkeypatch, split, full)
    counters = exchanges[0].counters
    assert all(exchange.counters is counters for exchange in exchanges)  # one per device ("cpu" here)
    for rank, exchange in enumerate(exchanges):
        calling[0] = rank
        assert torch.equal(exchange(exchange.window(full), lengths=lengths), full)
    assert counters[2].tolist() == [3, 0, 0, 0] and counters.sum().item() == 3
    compiled = torch.compile(lambda value, lengths: exchanges[1](exchanges[1].window(value), lengths=lengths),
                             fullgraph=True)
    calling[0] = 1
    assert torch.equal(compiled(full.clone(), lengths), full)
    assert counters[2].tolist() == [4, 0, 0, 0]
    with pytest.raises(ValueError, match="lengths"):
        exchanges[0](exchanges[0].window(full))


def test_check_mode_counts_a_corrupted_exchange(monkeypatch):
    split = RowSplit(4096, WORLD)
    lengths = torch.full((4096,), 3000, dtype=torch.int32)
    full = _valid_selection(lengths, 3)

    columns = LANES // 2
    owned = split.window_offsets()[2] * 512  # rank 2's first owned row inside its window (block 22 of 21-31)

    def fault(gathered):  # rank 2's partition: one bit of its first owned row, and its window row 0 (not owned)
        gathered[owned // columns, 2 * columns + owned % columns] ^= 4
        gathered[0, 2 * columns] ^= 4

    exchanges, calling = _checked_exchanges(monkeypatch, split, full, transport_fault=fault)
    calling[0] = 0
    selected = exchanges[0](exchanges[0].window(full), lengths=lengths)
    assert (selected != full).any(dim=1).nonzero().flatten().tolist() == [22 * 128]
    calls, transport, invalid, tail = exchanges[0].counters[2].tolist()
    assert (calls, transport, tail) == (1, 2, 0)  # the transport check covers rows the unpack drops too


def test_check_mode_reset_and_dump_requests(tmp_path, monkeypatch):
    monkeypatch.setenv("FLARE_DS41_INDEX_ROW_SPLIT_CHECK_DIR", str(tmp_path))
    request, path, ack = split_module.check_files(1, pid=77)
    assert request == str(tmp_path / "flare_idxsplit_check.request")
    assert path == str(tmp_path / "flare_idxsplit_check-dev1-pid77.pt") and ack.endswith("-dev1-pid77.ack")
    counters = torch.arange(split_module.CHECK_LAYERS * 4, dtype=torch.int64).view(-1, 4)
    request, path, ack = split_module.check_files(1)
    assert split_module.check_poll(counters, 1, "") == ""  # no request yet
    open(request, "w").write("reset 1\n")
    assert split_module.check_poll(counters, 1, "") == "reset 1" and counters.sum().item() == 0
    assert open(ack).read() == "reset 1\n"
    counters[2] = torch.tensor([5, 0, 0, 1])
    assert split_module.check_poll(counters, 1, "reset 1") == "reset 1"  # the same line is handled once
    open(request, "w").write("dump 2")
    assert split_module.check_poll(counters, 1, "reset 1") == "dump 2" and open(ack).read() == "dump 2\n"
    dumped = torch.load(path)
    assert dumped["fields"] == ("calls", "transport_rows", "invalid_rows", "tail_rows") and dumped["token"] == "dump 2"
    assert torch.equal(dumped["counters"], counters)
    open(request, "w").write("bogus 3")
    assert split_module.check_poll(counters, 1, "dump 2") == "bogus 3" and open(ack).read() == "dump 2\n"


def test_check_switch_needs_the_split_and_never_reaches_an_export():
    import os
    import subprocess

    from tests.recipes.test_deepseek41_tp3_config import EXPORT_SH, OVERLAY_STUB, RELEASE_CKPT

    assert not TP3Config().index_row_split_check and "index_row_split_check" not in TP3Config().to_metadata()
    on = TP3Config.from_environment({"FLARE_DS41_INDEX_ROW_SPLIT": "1", "FLARE_DS41_INDEX_ROW_SPLIT_CHECK": "1"})
    assert on.index_row_split_check and on.to_metadata()["index_row_split_check"] is True
    with pytest.raises(ValueError, match="index_row_split_check needs index_row_split"):
        TP3Config.from_environment({"FLARE_DS41_INDEX_ROW_SPLIT_CHECK": "1"})
    script = EXPORT_SH.read_text()
    body = script[:script.index("\nexec ")]
    result = subprocess.run(["bash", "-c", body, str(EXPORT_SH)], capture_output=True, text=True,
                            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "CKPT": RELEASE_CKPT,
                                 "B12X_OVERLAY": OVERLAY_STUB, "IDX_SPLIT": "1",
                                 "FLARE_DS41_INDEX_ROW_SPLIT_CHECK": "1"})
    assert result.returncode == 2 and "debug-serve only" in result.stderr
    mini = (EXPORT_SH.parent.parent / "mini-export.sh").read_text()
    assert mini.index("FLARE_DS41_INDEX_ROW_SPLIT_CHECK") < mini.index("ignition export")


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
    assert 'split_here = index_split is not None and mode == "prefill" and layer < candidate_source' in init
    assert "selector_rows = index_split.split.window_rows_count if split_here else tokens" in init
    run = inspect.getsource(attention.PreparedAttention._run)
    assert "index_owned" not in run and ".rank" not in run  # rank-symmetric trace: only the row-id tensor differs
    assert "pages, lengths = exchange.window(pages), exchange.window(lengths)" in run
    assert (run.index("selected = torch.cat(outputs, dim=0)")
            < run.index("selected = self.index_exchange(selected, lengths=geometry.lengths)"))
    assert attention.INDEXER_QUERY_ROWS == split_module.BLOCK_ROWS
    layer = inspect.getsource(decoder.PreparedDecoder.__init__)
    assert "index_split=index_split if layer_metadata is metadata else None" in layer  # never the 128-row suffix


def test_export_script_knob_reaches_the_switch():
    from tests.recipes.test_deepseek41_tp3_config import export_environment

    assert "FLARE_DS41_INDEX_ROW_SPLIT" not in export_environment()  # default export: replicated indexer
    assert TP3Config.from_environment(export_environment(IDX_SPLIT="1")).index_row_split == ROW_SPLIT_DEFAULT
    assert TP3Config.from_environment(export_environment(IDX_SPLIT="4096")).index_row_split == (4096,)
