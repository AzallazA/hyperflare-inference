# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""TP3 indexer row split for long prefill chunks (``TP3Config.index_row_split``, FLARE_DS41_INDEX_ROW_SPLIT).

Today every rank runs the sparse-attention indexer (all 32 index heads, top-512 per row) over every row of a prefill
chunk: the selection is replicated three times. Heads cannot be split (their scores are summed before the top-512),
but rows can: prefill already runs one selector per 128-row block (``attention.INDEXER_QUERY_ROWS``). With the split,
each rank runs the selectors of a contiguous, uneven share of the blocks (4096 rows = 32 blocks: 11 / 11 / 10), and
the int32 winners are exchanged so every rank holds the whole ``selected`` [rows, 512] again before attention and
the layers that share the index source read it.

The exchange reuses the FP32 vocabulary all-gather channel (``bootstrap.vocabulary_channels``; a column gather of
``lanes`` BF16 lanes per rank and row, ``PreparedAllGather``). A rank's winners are packed bit for bit into
``channel_rows`` rows of its column partition (int32 pairs travel as BF16 lanes; a gather is a pure copy, so 0x8000
halves and -1 sentinels survive), gathered, and unpacked into row order. Every value a rank reads is the value its
owner's selector wrote: ``selected`` equals the owner rank's replicated selection exactly. It equals every rank's own
replicated selection only where the indexer inputs are rank-identical (see PLAN.md, "identity precondition").

Pure geometry and the pack/unpack transforms live here (CPU-testable, traceable torch ops); ``PreparedIndexExchange``
binds them to the channel.
"""

from dataclasses import dataclass

import torch

# Prefill capacities split when FLARE_DS41_INDEX_ROW_SPLIT=1: the long-prompt chunk sizes, where the indexer is
# 5.7 ms (32K) to 101 ms (1M) per chunk; below ~3K rows the three exchanges (~0.2-0.4 ms each) eat the saving.
ROW_SPLIT_DEFAULT = (3072, 3584, 4096)
# DS41 index_topk: winners per row.
SELECTED_WIDTH = 512
# Rows per selector launch (attention.INDEXER_QUERY_ROWS; attention checks the two agree).
BLOCK_ROWS = 128


@dataclass(frozen=True)
class RowSplit:
    """Contiguous uneven ownership of a chunk's 128-row selector blocks: the first ``blocks % world`` ranks take one
    block more (4096 rows: 11 / 11 / 10 blocks). The same on every rank (pure function of capacity and world)."""

    capacity: int
    world: int
    block_rows: int = BLOCK_ROWS

    def __post_init__(self):
        if (type(self.capacity) is not int or type(self.world) is not int or type(self.block_rows) is not int
                or self.capacity <= 0 or self.world < 2 or self.block_rows <= 0):
            raise ValueError("an indexer row split needs a positive capacity, two or more ranks and positive blocks")
        if self.blocks < self.world:
            raise ValueError(f"an indexer row split of {self.capacity} rows has fewer {self.block_rows}-row blocks "
                             f"than {self.world} ranks")

    @property
    def blocks(self) -> int:
        return -(-self.capacity // self.block_rows)

    def owned_blocks(self, rank: int) -> range:
        """Selector ordinals this rank runs (``attention.PreparedAttention.selectors`` indices)."""
        if type(rank) is not int or not 0 <= rank < self.world:
            raise ValueError("rank outside the row split's world")
        base, extra = divmod(self.blocks, self.world)
        first = rank * base + min(rank, extra)
        return range(first, first + base + (rank < extra))

    def rows(self, rank: int) -> tuple[int, int]:
        """[first, last) chunk rows this rank selects for (the last block may be short)."""
        owned = self.owned_blocks(rank)
        return owned.start * self.block_rows, min(self.capacity, owned.stop * self.block_rows)

    def row_counts(self) -> tuple[int, ...]:
        return tuple(last - first for first, last in map(self.rows, range(self.world)))

    @property
    def max_rows(self) -> int:
        return max(self.row_counts())


@dataclass(frozen=True)
class ExchangeLayout:
    """How one rank's winners ride a column gather of ``lanes`` BF16 lanes per rank and row.

    Rank r's ``rows(r)`` winners ([n_r, 512] int32 = n_r * 1024 BF16 lanes) fill its partition row-major from lane 0
    of channel row 0; ``channel_rows`` covers the largest share. Lanes past a rank's winners are padding."""

    split: RowSplit
    lanes: int
    width: int = SELECTED_WIDTH

    def __post_init__(self):
        if type(self.lanes) is not int or self.lanes <= 0 or self.lanes % 2:
            raise ValueError("an index exchange needs a positive, even lane count per rank (int32 = two BF16 lanes)")

    @property
    def channel_rows(self) -> int:
        return -(-self.split.max_rows * self.width * 2 // self.lanes)

    @property
    def payload_bytes(self) -> int:
        """Bytes one rank publishes per exchange (its padded partition)."""
        return self.channel_rows * self.lanes * 2


def channel_lanes(world: int, vocabulary: int = 129280) -> int:
    """BF16 lanes per rank and row of the FP32 vocabulary all-gather channel (bootstrap ``_collective_runtimes``:
    ``columns = local_vocab * world * 2``)."""
    from ignition.components.parallel import TensorParallelShard

    first, last = TensorParallelShard(0, world).padded_partition(vocabulary, alignment=64)
    return (last - first) * 2


def exchange_rows(capacities, world: int, *, lanes: int | None = None) -> int:
    """Channel rows the widest split capacity needs (0 when nothing is split)."""
    lanes = channel_lanes(world) if lanes is None else lanes
    return max((ExchangeLayout(RowSplit(capacity, world), lanes).channel_rows for capacity in capacities), default=0)


def split_capacities(requested, prefill_capacities, *, chunk) -> tuple[int, ...]:
    """The prefill capacities whose graphs split the indexer rows (``tp3.index_row_split``); every requested size
    must be a captured prefill capacity. None or () -> nothing split."""
    if not requested:
        return ()
    missing = [capacity for capacity in requested if capacity not in prefill_capacities]
    if missing:
        raise ValueError(f"FLARE_DS41_INDEX_ROW_SPLIT {','.join(map(str, requested))}: "
                         f"{','.join(map(str, missing))} not captured prefill capacities "
                         f"(ladder {','.join(map(str, prefill_capacities))} for chunk {chunk})")
    return tuple(sorted(requested))


def pack(own: torch.Tensor, layout: ExchangeLayout, rank: int) -> torch.Tensor:
    """This rank's winners [n_r, 512] int32 -> its gather partition [channel_rows, lanes / 2] int32 (bit copy).

    Everything stays int32 in traced code: the BF16 reinterpretation the channel needs happens inside the exchange's
    custom op (``_exchange``), never in an Inductor kernel, whose BF16 copies may round through FP32 and canonicalize
    NaN halves (-1 is 0xFFFF 0xFFFF)."""
    rows = layout.split.row_counts()[rank]
    if own.dtype != torch.int32 or tuple(own.shape) != (rows, layout.width):
        raise ValueError(f"rank {rank} packs [{rows}, {layout.width}] int32 winners, not {tuple(own.shape)} {own.dtype}")
    columns = layout.lanes // 2
    flat = own.reshape(-1)
    padding = flat.new_full((layout.channel_rows * columns - flat.numel(),), -1)
    return torch.cat((flat, padding)).view(layout.channel_rows, columns)


def unpack(gathered: torch.Tensor, layout: ExchangeLayout) -> torch.Tensor:
    """The gathered channel rows [channel_rows, world * lanes / 2] int32 -> ``selected`` [capacity, 512] int32.

    Row i of the gather holds rank 0's columns, then rank 1's, ...; rank r's partition, read row-major, is its
    winners followed by padding."""
    split, rows, columns = layout.split, layout.channel_rows, layout.lanes // 2
    if gathered.dtype != torch.int32 or tuple(gathered.shape) != (rows, split.world * columns):
        raise ValueError("an index exchange unpacks [channel_rows, world * lanes / 2] int32")
    parts = []
    for rank, count in enumerate(split.row_counts()):
        partition = gathered[:, rank * columns:(rank + 1) * columns].reshape(-1)
        parts.append(partition[:count * layout.width].view(count, layout.width))
    return torch.cat(parts)


def emulate_gather(partitions) -> torch.Tensor:
    """CPU stand-in for the channel's column gather: every output row is rank 0's columns, then rank 1's, ..."""
    return torch.cat(tuple(partitions), dim=1)


@torch.library.custom_op("flare::index_row_exchange", mutates_args=("output",))
def _exchange(source: torch.Tensor, output: torch.Tensor, handle: int) -> None:
    """Column gather of int32 partitions through a ``PreparedAllGather``'s BF16 channel (``handle``): the int32 rows
    are reinterpreted as BF16 lane pairs here, in eager code, so the bytes travel untouched."""
    from ignition.components import collectives

    collectives._gather_columns(source.view(torch.bfloat16), output.view(torch.bfloat16), handle)


@_exchange.register_fake
def _exchange_fake(source, output, handle):
    pass


@dataclass(frozen=True)
class IndexRowSplitContext:
    """What a prefill decoder hands its layers when its capacity is split: the rank, the geometry and the channel.
    ``examples`` is the graph's declaration-example dict (shared send/receive declarations by role)."""

    split: RowSplit
    rank: int
    runtime: object
    coordinator: object
    examples: dict | None = None

    def layout(self, lanes: int) -> ExchangeLayout:
        return ExchangeLayout(self.split, lanes)


def prefill_split_context(*, mode, capacity, scoring, tp3, shard, runtime, coordinator, examples=None):
    """The split context a ``PreparedDecoder`` hands its full-row layers, or None (replicated indexer): prefill
    graphs whose capacity ``tp3.index_row_split`` lists, without prompt scoring, on more than one rank."""
    if (mode != "prefill" or scoring is not None or not tp3.index_row_split
            or capacity not in tp3.index_row_split or shard.world_size < 2):
        return None
    return IndexRowSplitContext(RowSplit(capacity, shard.world_size), shard.rank, runtime, coordinator, examples)


class PreparedIndexExchange:
    """All-gather of the split winners through the vocabulary channel; ``__call__(own) -> selected``."""

    def __init__(self, context: IndexRowSplitContext, device, *, session, name):
        from ignition.components.collectives import PreparedAllGather, _roce
        from ignition.preparation.examples import declaration_tensor

        runtime = context.runtime
        if runtime is None or _roce(runtime):
            raise ValueError("the indexer row split exchanges over a single-host PCIe vocabulary channel")
        if runtime.world_size != context.split.world:
            raise ValueError("the indexer row split's world must match the vocabulary channel's")
        lanes = runtime.heads_per_rank * runtime.query_head_dim
        self.layout = context.layout(lanes)
        self.rank = context.rank
        rows = self.layout.channel_rows
        if rows > runtime.max_batch_size:
            raise ValueError(f"the indexer row split needs {rows} vocabulary channel rows, the channel has "
                             f"{runtime.max_batch_size} (bootstrap.vocabulary_channels sizes it)")
        # The gather is planned on BF16 views of int32 declarations (the bytes the forward exchanges).
        send = declaration_tensor(context.examples, "attention.index_exchange_send", (rows, lanes // 2),
                                  torch.int32, device)
        receive = declaration_tensor(context.examples, "attention.index_exchange_receive",
                                     (rows, lanes // 2 * context.split.world), torch.int32, device)
        self.gather = PreparedAllGather(send.view(torch.bfloat16), receive.view(torch.bfloat16), runtime=runtime,
                                        session=session, coordinator=context.coordinator, name=name)
        self.receive_shape = tuple(receive.shape)

    def functional(self, *, session):
        self.gather.functional(session=session)
        return self

    def __call__(self, own: torch.Tensor) -> torch.Tensor:
        """Traced: this rank's winners [n_r, 512] -> every row's winners [capacity, 512] (owners' bits)."""
        source = pack(own, self.layout, self.rank)
        output = torch.empty(self.receive_shape, dtype=torch.int32, device=own.device)
        _exchange(source, output, self.gather.handle)
        return unpack(output, self.layout)
