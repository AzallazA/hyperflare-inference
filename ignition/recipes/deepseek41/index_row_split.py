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

Rank-symmetric trace (review D1): the export compiles ONE forward per prefill graph and every rank binds it, so the
traced code must not depend on the rank through Python values (Dynamo guards them). Every rank therefore runs the same
number of selectors, ``window_blocks`` (the largest share: 11 at 4096 rows), on a window of rows it gathers through a
per-rank int64 row-id TENSOR (a persistent buffer, bound like any other input). A rank with a smaller share widens its
window into its neighbour's blocks (rank 2 at 4096 runs blocks 21-31 and owns 22-31); the unpack takes every row from
its owner at fixed offsets that are the same on every rank. The slowest rank keeps 11 blocks: the gain is unchanged.

Pure geometry and the pack/unpack transforms live here (CPU-testable, traceable torch ops); ``PreparedIndexExchange``
binds them to the channel.

Check mode (``TP3Config.index_row_split_check``, FLARE_DS41_INDEX_ROW_SPLIT_CHECK=1; debug-serve only, never export with
it: its counter op is plain Python). G0 (2026-11-18, G0-ANALYSIS.md) showed the b12x selector picks different members of
the top-512 cut's tie class run to run, so a second selector launch cannot serve as a reference. The check is therefore
(a) a pure transport proof: every rank appends a checksum per window row to its partition (in the padding: the channel
row count does not change at 3072 / 3584 / 4096), and after the gather every rank recomputes the checksums of every
partition it received and counts the rows that differ (must be 0); and (b) a structural check of every assembled row
against this rank's own lengths: the first ``min(length, 512)`` winners distinct and in [0, length) (must be 0), plus an
informational count of rows whose tail past that prefix is not all -1. Persistent per-device counters
[layer, (calls, transport rows, invalid rows, tail rows)], dumped like probe.py (``check_dump``).
"""

import os
import threading
import time
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
    block more (4096 rows: 11 / 11 / 10 blocks). The same on every rank (pure function of capacity and world).

    Every rank computes a window of ``window_blocks`` blocks that contains its owned blocks (rank-symmetric trace)."""

    capacity: int
    world: int
    block_rows: int = BLOCK_ROWS

    def __post_init__(self):
        if (type(self.capacity) is not int or type(self.world) is not int or type(self.block_rows) is not int
                or self.capacity <= 0 or self.world < 2 or self.block_rows <= 0):
            raise ValueError("an indexer row split needs a positive capacity, two or more ranks and positive blocks")
        if self.capacity % self.block_rows:
            raise ValueError(f"an indexer row split needs whole {self.block_rows}-row selector blocks "
                             f"(capacity {self.capacity})")
        if self.blocks < self.world:
            raise ValueError(f"an indexer row split of {self.capacity} rows has fewer {self.block_rows}-row blocks "
                             f"than {self.world} ranks")

    @property
    def blocks(self) -> int:
        return self.capacity // self.block_rows

    def owned_blocks(self, rank: int) -> range:
        """Blocks whose winners this rank publishes (it is their owner)."""
        if type(rank) is not int or not 0 <= rank < self.world:
            raise ValueError("rank outside the row split's world")
        base, extra = divmod(self.blocks, self.world)
        first = rank * base + min(rank, extra)
        return range(first, first + base + (rank < extra))

    def rows(self, rank: int) -> tuple[int, int]:
        """[first, last) chunk rows this rank owns."""
        owned = self.owned_blocks(rank)
        return owned.start * self.block_rows, owned.stop * self.block_rows

    def row_counts(self) -> tuple[int, ...]:
        return tuple(last - first for first, last in map(self.rows, range(self.world)))

    @property
    def window_blocks(self) -> int:
        """Selector launches per rank: the largest share, the same on every rank."""
        return -(-self.blocks // self.world)

    @property
    def window_rows_count(self) -> int:
        return self.window_blocks * self.block_rows

    def window_start(self, rank: int) -> int:
        """First block of this rank's window: its first owned block, moved back so the window ends inside the chunk."""
        return min(self.owned_blocks(rank).start, self.blocks - self.window_blocks)

    def window_offsets(self) -> tuple[int, ...]:
        """Per rank: rows from the start of its window to its first owned row (rank-independent constants)."""
        return tuple((self.owned_blocks(rank).start - self.window_start(rank)) * self.block_rows
                     for rank in range(self.world))

    def window_rows(self, rank: int, device=None) -> torch.Tensor:
        """Chunk row ids [window_rows_count] int64 of this rank's window (the per-rank gather index)."""
        first = self.window_start(rank) * self.block_rows
        return torch.arange(first, first + self.window_rows_count, dtype=torch.int64, device=device)

    @property
    def max_rows(self) -> int:
        return max(self.row_counts())


@dataclass(frozen=True)
class ExchangeLayout:
    """How one rank's window of winners rides a column gather of ``lanes`` BF16 lanes per rank and row.

    A rank's window ([window_rows_count, 512] int32 = 1024 BF16 lanes per row) fills its partition row-major from
    lane 0 of channel row 0; lanes past it are padding. Every rank packs the same count (rank-symmetric)."""

    split: RowSplit
    lanes: int
    width: int = SELECTED_WIDTH
    check: bool = False  # check mode: one checksum word per window row after the winners (transport proof)

    def __post_init__(self):
        if type(self.lanes) is not int or self.lanes <= 0 or self.lanes % 2:
            raise ValueError("an index exchange needs a positive, even lane count per rank (int32 = two BF16 lanes)")

    @property
    def check_words(self) -> int:
        return self.split.window_rows_count if self.check else 0

    @property
    def channel_rows(self) -> int:
        return -(-(self.split.window_rows_count * self.width + self.check_words) * 2 // self.lanes)

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


def exchange_rows(capacities, world: int, *, lanes: int | None = None, check: bool = False) -> int:
    """Channel rows the widest split capacity needs (0 when nothing is split)."""
    lanes = channel_lanes(world) if lanes is None else lanes
    return max((ExchangeLayout(RowSplit(capacity, world), lanes, check=check).channel_rows for capacity in capacities),
               default=0)


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


def pack(window: torch.Tensor, layout: ExchangeLayout) -> torch.Tensor:
    """A rank's window of winners [window_rows_count, 512] int32 -> its gather partition [channel_rows, lanes / 2]
    int32 (bit copy). No rank enters: every rank packs the same shape.

    Everything stays int32 in traced code: the BF16 reinterpretation the channel needs happens inside the exchange's
    custom op (``_exchange``), never in an Inductor kernel, whose BF16 copies may round through FP32 and canonicalize
    NaN halves (-1 is 0xFFFF 0xFFFF)."""
    rows = layout.split.window_rows_count
    if window.dtype != torch.int32 or tuple(window.shape) != (rows, layout.width):
        raise ValueError(f"an index exchange packs [{rows}, {layout.width}] int32 winners, "
                         f"not {tuple(window.shape)} {window.dtype}")
    columns = layout.lanes // 2
    flat = window.reshape(-1)
    parts = (flat, row_checksums(window)) if layout.check else (flat,)
    used = flat.numel() + layout.check_words
    padding = flat.new_full((layout.channel_rows * columns - used,), -1)
    return torch.cat((*parts, padding)).view(layout.channel_rows, columns)


# Check mode checksum: per row sum_j (unsigned value_j) * (2 j + 1) mod (2^31 - 1), exact in int64 (< 2^52 for 512
# lanes), so the traced pack (Inductor) and the eager check compute the same int32 bits.
_CHECK_MODULUS = 2 ** 31 - 1


def row_checksums(rows: torch.Tensor) -> torch.Tensor:
    """[n, width] int32 -> [n] int32 position-weighted checksums (traceable; integer-exact)."""
    weights = torch.arange(1, 2 * rows.shape[1], 2, dtype=torch.int64, device=rows.device)
    unsigned = rows.to(torch.int64) & 0xFFFFFFFF
    return torch.remainder((unsigned * weights).sum(dim=1), _CHECK_MODULUS).to(torch.int32)


def transport_mismatches(gathered: torch.Tensor, layout: ExchangeLayout) -> torch.Tensor:
    """Check mode: per source rank, the window rows whose received checksum differs from the one recomputed from the
    received winners -> int64 [world]. Zero on every rank = the exchange moved every partition bit for bit."""
    if not layout.check:
        raise ValueError("transport checksums need a check-mode layout")
    split, columns, rows = layout.split, layout.lanes // 2, layout.split.window_rows_count
    counts = []
    for rank in range(split.world):
        partition = gathered[:, rank * columns:(rank + 1) * columns].reshape(-1)
        winners = partition[:rows * layout.width].view(rows, layout.width)
        sent = partition[rows * layout.width:rows * layout.width + rows]
        counts.append((row_checksums(winners) != sent).sum())
    return torch.stack(counts)


def structural_counts(selected: torch.Tensor, lengths: torch.Tensor, width: int = SELECTED_WIDTH) -> torch.Tensor:
    """Check mode: rows of an assembled ``selected`` [rows, width] int32 against this rank's own ``lengths`` [rows]
    -> int64 [2]: (rows whose first min(length, width) winners are not distinct values in [0, length): must be 0,
    rows whose entries past that prefix are not all -1: informational; the attention reads only the prefix)."""
    lengths = lengths.to(torch.int64)
    used = lengths.clamp(min=0, max=width)
    column = torch.arange(width, device=selected.device)
    inside = column[None, :] < used[:, None]
    values = selected.to(torch.int64)
    ordered = torch.where(inside, values, torch.iinfo(torch.int64).max).sort(dim=1).values  # prefix first, ascending
    bad = inside & ((ordered < 0) | (ordered >= lengths[:, None]))
    bad[:, 1:] |= inside[:, 1:] & (ordered[:, 1:] == ordered[:, :-1])
    tail = ~inside & (values != -1)
    return torch.stack((bad.any(dim=1).sum(), tail.any(dim=1).sum()))


# Check mode counters: [layer, (calls, transport rows, invalid rows, tail rows)] int64 per device, persistent storage.
CHECK_LAYERS = 64
CHECK_FIELDS = ("calls", "transport_rows", "invalid_rows", "tail_rows")
_CHECK_STATE = {}


def check_state(device) -> torch.Tensor:
    """This device's check counters (allocated once, persistent; starts the dumper thread)."""
    device = torch.device(device)
    key = (device.type, device.index)
    if key not in _CHECK_STATE:
        from ignition.preparation.persistent import persistent_zeros

        _CHECK_STATE[key] = persistent_zeros((CHECK_LAYERS, len(CHECK_FIELDS)), dtype=torch.int64, device=device)
        if device.type == "cuda":
            threading.Thread(target=_check_dumper, args=(_CHECK_STATE[key], device.index), daemon=True).start()
    return _CHECK_STATE[key]


def check_paths(index) -> tuple[str, str]:
    """(request file, dump file) of device ``index``: touch the request, wait for it to vanish, read the dump."""
    root = os.environ.get("FLARE_DS41_INDEX_ROW_SPLIT_CHECK_DIR", "/tmp")
    return os.path.join(root, f"flare_idxsplit_check_req-dev{index}"), os.path.join(root, f"flare_idxsplit_check-dev{index}.pt")


def check_dump(counters: torch.Tensor, path: str) -> None:
    torch.save({"fields": CHECK_FIELDS, "counters": counters.detach().cpu()}, path + ".tmp")
    os.replace(path + ".tmp", path)


def _check_dumper(counters, index):
    request, path = check_paths(index)
    while True:
        if os.path.exists(request):
            try:
                check_dump(counters, path)
            finally:
                os.remove(request)
        time.sleep(0.2)


@torch.library.custom_op("flare_debug::index_split_check", mutates_args=("counters",))
def _check(gathered: torch.Tensor, selected: torch.Tensor, lengths: torch.Tensor, counters: torch.Tensor, layer: int,
           capacity: int, world: int, lanes: int) -> None:
    """Check mode (device-side only, no host sync: safe inside a captured graph): add one call's counts to row
    ``layer`` of ``counters``."""
    layout = ExchangeLayout(RowSplit(capacity, world), lanes, check=True)
    row = counters[layer]
    row[0] += 1
    row[1] += transport_mismatches(gathered, layout).sum()
    row[2:4] += structural_counts(selected, lengths, layout.width)


@_check.register_fake
def _check_fake(gathered, selected, lengths, counters, layer, capacity, world, lanes):
    pass


def unpack(gathered: torch.Tensor, layout: ExchangeLayout) -> torch.Tensor:
    """The gathered channel rows [channel_rows, world * lanes / 2] int32 -> ``selected`` [capacity, 512] int32.

    Row i of the gather holds rank 0's columns, then rank 1's, ...; rank r's partition, read row-major, is its window
    followed by padding; its owned rows start ``window_offsets()[r]`` rows into the window. Rows a rank computed but
    does not own are dropped (their owner's copy is used), so every rank assembles identical bits."""
    split, rows, columns = layout.split, layout.channel_rows, layout.lanes // 2
    if gathered.dtype != torch.int32 or tuple(gathered.shape) != (rows, split.world * columns):
        raise ValueError("an index exchange unpacks [channel_rows, world * lanes / 2] int32")
    parts = []
    for rank, (count, offset) in enumerate(zip(split.row_counts(), split.window_offsets(), strict=True)):
        partition = gathered[:, rank * columns:(rank + 1) * columns].reshape(-1)
        parts.append(partition[offset * layout.width:(offset + count) * layout.width].view(count, layout.width))
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
    check: bool = False  # TP3Config.index_row_split_check (debug-serve only)

    def layout(self, lanes: int) -> ExchangeLayout:
        return ExchangeLayout(self.split, lanes, check=self.check)


def prefill_split_context(*, mode, capacity, scoring, tp3, shard, runtime, coordinator, examples=None):
    """The split context a ``PreparedDecoder`` hands its full-row layers, or None (replicated indexer): prefill
    graphs whose capacity ``tp3.index_row_split`` lists, without prompt scoring, on more than one rank."""
    if (mode != "prefill" or scoring is not None or not tp3.index_row_split
            or capacity not in tp3.index_row_split or shard.world_size < 2):
        return None
    return IndexRowSplitContext(RowSplit(capacity, shard.world_size), shard.rank, runtime, coordinator, examples,
                                check=tp3.index_row_split_check)


class PreparedIndexExchange:
    """The rank's window and the all-gather of its winners through the vocabulary channel.

    ``rows`` is this rank's window as an int64 row-id tensor in persistent storage: the only rank-specific value the
    traced forward reads, and a tensor, so every rank binds the same compiled forward. ``window(tensor)`` gathers a
    [capacity, ...] operand's window rows; ``__call__(window_winners) -> selected`` exchanges. In check mode the call
    also adds this layer's counts (``layer``) to the device's check counters (``lengths`` = the chunk's own lengths)."""

    def __init__(self, context: IndexRowSplitContext, device, *, session, name, layer=0):
        from ignition.components.collectives import PreparedAllGather, _roce
        from ignition.preparation.examples import declaration_tensor
        from ignition.preparation.persistent import persistent_copy

        runtime = context.runtime
        if runtime is None or _roce(runtime):
            raise ValueError("the indexer row split exchanges over a single-host PCIe vocabulary channel")
        if runtime.world_size != context.split.world:
            raise ValueError("the indexer row split's world must match the vocabulary channel's")
        lanes = runtime.heads_per_rank * runtime.query_head_dim
        self.layout = context.layout(lanes)
        rows = self.layout.channel_rows
        if rows > runtime.max_batch_size:
            raise ValueError(f"the indexer row split needs {rows} vocabulary channel rows, the channel has "
                             f"{runtime.max_batch_size} (bootstrap.vocabulary_channels sizes it)")
        self.rows = persistent_copy(context.split.window_rows(context.rank, device=device))
        # The gather is planned on BF16 views of int32 declarations (the bytes the forward exchanges).
        send = declaration_tensor(context.examples, "attention.index_exchange_send", (rows, lanes // 2),
                                  torch.int32, device)
        receive = declaration_tensor(context.examples, "attention.index_exchange_receive",
                                     (rows, lanes // 2 * context.split.world), torch.int32, device)
        self.gather = PreparedAllGather(send.view(torch.bfloat16), receive.view(torch.bfloat16), runtime=runtime,
                                        session=session, coordinator=context.coordinator, name=name)
        self.receive_shape = tuple(receive.shape)
        self.layer, self.lanes = layer, lanes
        self.counters = check_state(device) if self.layout.check else None
        if self.counters is not None and not 0 <= layer < CHECK_LAYERS:
            raise ValueError("check-mode counters cover layers below CHECK_LAYERS")

    def functional(self, *, session):
        self.gather.functional(session=session)
        return self

    def window(self, tensor: torch.Tensor) -> torch.Tensor:
        """Traced: this rank's window rows of a [capacity, ...] operand (same shape on every rank)."""
        return tensor.index_select(0, self.rows)

    def __call__(self, window: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Traced: this rank's window winners [window_rows_count, 512] -> every row's winners [capacity, 512] (owners'
        bits). ``lengths`` (check mode only): this rank's compressed lengths of the whole chunk."""
        source = pack(window, self.layout)
        output = torch.empty(self.receive_shape, dtype=torch.int32, device=window.device)
        _exchange(source, output, self.gather.handle)
        selected = unpack(output, self.layout)
        if self.counters is not None:
            if lengths is None:
                raise ValueError("check mode needs the chunk's lengths")
            split = self.layout.split
            _check(output, selected, lengths, self.counters, self.layer, split.capacity, split.world, self.lanes)
        return selected
