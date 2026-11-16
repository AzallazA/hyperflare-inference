# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""File-backed FP8 or MXFP4 n-gram tables, compact GPU inputs, and residual injection."""

from dataclasses import dataclass
from functools import cached_property

import torch
from b12x.norm import hyperconnection
from b12x.preparation import PreparedCall
from b12x.sequence import engram

from ignition.components.engram import EngramInputs, PreparedEngramLookup, PreparedEngramRequests
from ignition.components.readiness import ReadinessTimings
from ignition.components.embedding_partition import EmbeddingHeadPartition
from ignition.components.linear import QuantizedProjection
from ignition.components.collectives import PreparedChunkedAllReduce
from ignition.components.state import gather_rows, save_hidden
from ignition.preparation.runtime_core import FileSource
from ignition.preparation.persistent import persistent_copy
from ignition.preparation.tuning import declare
from ignition.progress import phase


@dataclass(frozen=True)
class Tables:
    geometry: object
    token_map: tuple[int, ...]
    sources: dict
    inputs: dict
    mailboxes: dict
    sequence: torch.Tensor
    shard: object
    row_mapping: dict
    columns: torch.Tensor
    readiness_runtime: object = None
    # TP3 fork: packed MXFP4 E2M1 rows (128 bytes + 8 scale bytes per row) instead of FP8 E4M3 rows (256 + 8).
    packed_fp4: bool = False
    # TP3 fork: every rank keeps the whole tables (TP3Config.engram_tables); False = upstream #220's per-rank head
    # partitions. Each rank projects only its own heads' columns either way.
    shared: bool = False
    # TP3 fork (tp3.share_engram_token_map): Engram hash plans read ``device_token_map`` instead of their own copy.
    share_token_map: bool = False

    @classmethod
    def prepare(cls, manifest, config, *, tokens, device, sequence, frame, host_allocator, progress=None, shard=None,
                shared=None):
        """``shared`` keeps whole tables on every rank (None: only a single rank does, as upstream)."""
        from transformers import PreTrainedTokenizerFast
        from ignition.components.parallel import TensorParallelShard

        shard = shard or TensorParallelShard(0, 1)
        shared = shard.world_size == 1 if shared is None else bool(shared)

        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(manifest.metadata_path / "tokenizer.json"))
        token_map, size = engram.build_compressed_token_map(tokenizer)
        text = config.text
        if size != text["engram_compressed_vocab_size"]:
            raise ValueError(f"Engram compressed vocabulary differs: {size}")
        geometry = engram.build_geometry(layer_ids=tuple(text["engram_layer_ids"]),
            base_table_size=text["engram_vocab_size"], compressed_vocab_size=size)
        if geometry.num_embeddings != tuple(text["engram_num_embeddings"]):
            raise ValueError("Engram hash geometry differs from checkpoint tables")
        sources, inputs, mailboxes, row_mapping = {}, {}, {}, {}
        packed_fp4 = None
        timings = ReadinessTimings.allocate(frame, len(geometry.layer_ids))
        with phase(progress, "Prepare embedding tables") as stage:
            for index, layer in enumerate(geometry.layer_ids):
                rows = geometry.num_embeddings[index]
                pair = []
                # FP8 E4M3 rows (stock checkpoint) or packed MXFP4 E2M1 rows (engram-nvidia checkpoint).
                weight = manifest.tensors[f"layers.{layer}.engram.embed.weight"]
                packed = weight.dtype == "U8" and weight.shape == (rows, 128)
                if packed_fp4 is None:
                    packed_fp4 = packed
                elif packed_fp4 != packed:
                    raise ValueError("Engram tables mix FP8 and MXFP4 rows")
                for suffix, columns, dtype in (("weight", 128 if packed else 256, "U8" if packed else "F8_E4M3"),
                                               ("scale", 8, "F8_E8M0")):
                    name = f"layers.{layer}.engram.embed.{suffix}"
                    metadata = manifest.tensors[name]
                    if metadata.shape != (rows, columns) or metadata.dtype != dtype:
                        raise ValueError(f"{name}: invalid Engram table geometry or dtype")
                    pair.append(FileSource(manifest.snapshot / metadata.shard, metadata.offset, rows * columns))
                sources[layer] = tuple(pair)
                # Shared tables keep canonical rows: no request remapping (the pre-#220 request path).
                row_mapping[layer] = None if shared else torch.tensor(EmbeddingHeadPartition.for_rank(
                    geometry.offsets[index], rows, shard.rank, shard.world_size).row_mapping,
                    dtype=torch.int64, device=device)
                inputs[layer] = EngramInputs.allocate(tokens, device, timings=timings, interval=index,
                                                      packed_fp4=packed)
                mailboxes[layer] = host_allocator(f"engram_{layer}_requests", (tokens * 24 + 1,), torch.int64)
                mailboxes[layer].host_view.zero_()
            stage.update(f"{sum(source.nbytes for pair in sources.values() for source in pair) / (1 << 30):.1f} GiB in checkpoint spans")
        columns = persistent_copy(torch.tensor([head * 256 + column for head in range(shard.rank, 24, shard.world_size)
                                for column in range(256)], dtype=torch.int64, device=device))
        return cls(geometry, tuple(token_map), sources, inputs, mailboxes, sequence, shard, row_mapping, columns,
                   packed_fp4=bool(packed_fp4), shared=shared)

    @property
    def row_bytes(self):
        """Bytes of one stored row in the weight and scale planes."""
        return (128 if self.packed_fp4 else 256), 8

    @cached_property
    def device_token_map(self):
        device = next(iter(self.inputs.values())).hashes.device
        return persistent_copy(torch.tensor(self.token_map, dtype=torch.int64, device=device))

    def buffers(self):
        return {**{name: tensor for layer, inputs in self.inputs.items() for name, tensor in inputs.buffers(layer).items()},
                **{f"engram_{layer}_requests": mailbox.device_view for layer, mailbox in self.mailboxes.items()},
                **next(iter(self.inputs.values())).timings.buffers()}

    def events(self):
        return {f"engram_{layer}_ready": inputs.ready.cuda_event for layer, inputs in self.inputs.items()}

    def resident_catalog(self):
        """Source spans and native bindings for shared RAM delivery without core export."""
        return {"vocabulary": len(self.token_map), "tables": [
            {"planes": [{"segments": [
                            {"path": str(source.path), "offset": source.offset + first * row_bytes,
                             "nbytes": (last - first) * row_bytes}
                            for first, last in self.source_rows(ordinal, self.shard.rank)], "row_bytes": row_bytes}
                        for source, row_bytes in zip(self.sources[layer], self.row_bytes, strict=True)],
             "heads": 24, "weights": f"engram_{layer}_weights", "scales": f"engram_{layer}_scales",
             "hashes": f"engram_{layer}_hashes", "event": f"engram_{layer}_ready",
             "request": f"engram_{layer}_requests"}
            for ordinal, layer in enumerate(self.geometry.layer_ids)
            for partition in (self.partition(ordinal, self.shard.rank),)]}

    def partition(self, ordinal, rank):
        """The rows ``rank`` stores for table ``ordinal``: its heads' rows, or every row with shared tables."""
        if self.shared:
            return EmbeddingHeadPartition.for_rank(self.geometry.offsets[ordinal],
                self.geometry.num_embeddings[ordinal], 0, 1)
        return EmbeddingHeadPartition.for_rank(self.geometry.offsets[ordinal],
            self.geometry.num_embeddings[ordinal], rank, self.shard.world_size)

    def source_rows(self, ordinal, rank):
        """Checkpoint row ranges ``rank`` stores; shared tables are one whole-table span (the pre-#220 layout)."""
        if self.shared:
            return ((0, self.geometry.num_embeddings[ordinal]),)
        return self.partition(ordinal, rank).source_rows

    def rank_bytes(self, rank):
        """Host bytes of the table rows ``rank`` stores (weight and scale planes)."""
        return sum(self.partition(ordinal, rank).rows * sum(self.row_bytes) for ordinal in range(len(self.sources)))

    def export_metadata(self):
        """Describe table geometry and checkpoint spans for native publication."""
        tables = []
        for ordinal, layer in enumerate(self.geometry.layer_ids):
            table = dict(layer=layer, rows=self.geometry.num_embeddings[ordinal],
                embedding_width=256, scale_block=32,
                encoding="MXFP4_E2M1_E8M0" if self.packed_fp4 else "FP8_E4M3_E8M0",
                weight_buffer=f"engram_{layer}_weights", scale_buffer=f"engram_{layer}_scales",
                hash_buffer=f"engram_{layer}_hashes", ready_event=f"engram_{layer}_ready",
                request_buffer=f"engram_{layer}_requests")
            offsets = [0] * 24
            first = 0
            shards = []
            for rank in range(1 if self.shared else self.shard.world_size):
                partition = self.partition(ordinal, rank)
                for head in partition.heads:
                    offsets[head] = first + partition.row_mapping[head][2]
                planes = dict(rank=rank, first_row=first, rows=partition.rows)
                for name, source, width in zip(("weights", "scales"), self.sources[layer], self.row_bytes, strict=True):
                    planes[name] = [dict(path=str(source.path), offset=source.offset + start * width,
                                         nbytes=(end - start) * width)
                                    for start, end in self.source_rows(ordinal, rank)]
                if self.shared or self.shard.world_size == 1:
                    table.update(weights=planes["weights"], scales=planes["scales"])
                else:
                    shards.append(planes)
                first += partition.rows
            table["hash"] = dict(method="XOR_MODULO", orders=[order for order in (2, 3, 4) for _ in range(8)],
                multipliers=list(self.geometry.multipliers[ordinal]), primes=list(self.geometry.primes[ordinal]),
                offsets=offsets)
            if shards:
                table["shards"] = shards
            tables.append(table)
        return dict(token_map=list(self.token_map), pad_token_id=2,
                    compressed_vocabulary=self.geometry.compressed_vocab_size, tables=tables)


class PreparedTableRequests:
    def __init__(self, tables, metadata, pool, *, session, name):
        self.pool = pool
        history = torch.zeros((metadata.starts.numel(), 3), dtype=torch.int64, device=metadata.starts.device)
        self.token_map = tables.device_token_map
        self.operations = tuple(PreparedEngramRequests(inputs, metadata.token_ids, metadata.query_start_loc,
            history, metadata.live_counts, tables.mailboxes[layer].device_view, tables.sequence,
            layer=layer, token_map=tables.token_map, geometry=tables.geometry, session=session,
            name=f"{name}.{layer}", row_mapping=tables.row_mapping[layer],
            # TP3 tp3.share_engram_token_map: the hash plans read this rank's one persistent map (same values).
            shared_token_map=self.token_map if tables.share_token_map else None).functional()
            for layer, inputs in tables.inputs.items())

    def __call__(self, metadata):
        m = metadata
        raw_history = torch.zeros((m.starts.numel(), 3), dtype=torch.int64, device=m.starts.device)
        gather_rows(self.pool.region("engram_history"), m.state_rows, raw_history)
        columns = torch.arange(3, device=m.starts.device)
        raw = torch.where(columns >= 3 - m.starts[:, None], raw_history, 2)
        # Image token types stop n-grams at image boundaries, including across prefill chunks.
        text = (raw >= 0) & (raw < self.token_map.numel())
        history = torch.where(text, self.token_map[raw.clamp(0, self.token_map.numel() - 1)], -1)
        for operation in self.operations:
            operation(m.token_ids, m.query_start_loc, history, m.live_counts)
        return raw_history

    def commit(self, counts, raw_history, metadata):
        m = metadata
        columns = torch.arange(3, device=m.starts.device)
        relative = counts[:, None].long() + columns - 3
        indices = (m.query_start_loc[:-1, None] + relative).clamp(0, m.token_ids.numel() - 1)
        history = torch.where(relative >= 0, m.token_ids[indices],
                              raw_history.gather(1, (relative + 3).clamp(0, 2)))
        save_hidden(history, self.pool.region("engram_history"), m.state_rows, counts > 0)


@dataclass(frozen=True)
class EngramScratch:
    """Declaration operands of serialized Engram layers (the traced forward allocates its own values).

    ``mask`` and ``gathered`` hold every row of the graph. ``local``, ``partial`` and ``projected`` hold one projection
    chunk: the whole graph by default, ``chunk_rows`` rows when the TP3 fork chunks the projection
    (tp3.engram_chunk_rows). PreparedEngram derives its chunking from these shapes, so all ranks, given one config,
    chunk alike.
    """

    mask: torch.Tensor
    gathered: torch.Tensor
    local: torch.Tensor
    partial: torch.Tensor
    projected: torch.Tensor

    @classmethod
    def allocate(cls, tokens, device, *, world=1, chunk_rows=0):
        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)
        if type(chunk_rows) is not int or chunk_rows < 0:
            raise ValueError("Engram chunk rows must be a non-negative integer")
        rows = min(tokens, chunk_rows) if chunk_rows else tokens
        gathered = zeros((tokens, 6144), torch.bfloat16)
        # Unsharded chunks project row views of ``gathered`` directly (rows are contiguous), so no chunk copy.
        local = gathered[:rows] if world == 1 else zeros((rows, 6144 // world), torch.bfloat16)
        partial = zeros((rows, 25600), torch.bfloat16)
        projected = partial if world == 1 else zeros((rows, 25600), torch.bfloat16)
        return cls(zeros(tokens, torch.bool), gathered, local, partial, projected)


class PreparedEngram:
    def __init__(self, layer, weights, tables, source, output, metadata, pool, *, scratch, session, name, workspace,
                 runtime=None, coordinator=None, examples=None, max_part_rows=0):
        # max_part_rows (TP3Config.engram_part_rows): the projection all-reduce's parts are cut at that many rows of its
        # (rows x 5, 5120) view (a 1024-token chunk at 768: 6 x 768 + 512) instead of halving (8 x 640); 0 = halving.
        self.pool = pool
        tokens, device = metadata.token_ids.numel(), source.device
        mask, gathered, projected = scratch.mask, scratch.gathered, scratch.projected
        local, partial = scratch.local, scratch.partial
        self.columns = tables.columns
        self.sharded = tables.shard.world_size > 1
        if self.sharded and runtime is None:
            raise ValueError("sharded Engram projection requires a collective runtime")
        # TP3 fork (tp3.engram_chunk_rows): projection operands narrower than the graph run the projection, its
        # all-reduce and the mix in row chunks of that height. Each chunk's local/partial/projected values are dead
        # before the next chunk allocates its own, so Inductor gives every chunk the same pool offsets and the
        # Engram layer's planned peak holds one chunk of them instead of every row. Default: one pass.
        chunk = partial.shape[0]
        if not 0 < chunk <= tokens:
            raise ValueError("Engram projection scratch must hold between one row and the graph's rows")
        self.chunked = chunk < tokens
        self.bounds = tuple((begin, min(begin + chunk, tokens)) for begin in range(0, tokens, chunk))
        for tensor, shape, dtype in ((mask, (tokens,), torch.bool),
                (gathered, (tokens, 6144), torch.bfloat16),
                (local, (chunk, 6144 // tables.shard.world_size), torch.bfloat16),
                (partial, (chunk, 25600), torch.bfloat16),
                (projected, (chunk, 25600), torch.bfloat16)):
            if tensor.shape != shape or tensor.dtype != dtype or tensor.device != device or not tensor.is_contiguous():
                raise ValueError("Engram scratch must match the contiguous lookup and projection geometry")
        self.lookup = PreparedEngramLookup(tables.inputs[layer], gathered, metadata.live_counts[:1],
            layer=layer, token_map=tables.token_map, geometry=tables.geometry,
            session=session, name=name + ".lookup", runtime=tables.readiness_runtime, coordinator=coordinator).functional(session=session)
        # One projection and one reduction per distinct chunk height. A shorter tail chunk gets its own pair and its
        # own declaration role (one backing per role is shared across graph capacities).
        self.chunk_ops = {}
        for begin, end in self.bounds:
            rows = end - begin
            if rows in self.chunk_ops:
                continue
            label = "" if rows == chunk else ".tail"
            projection = QuantizedProjection(local[:rows], weights["projection"], partial[:rows],
                session=session, name=name + ".projection" + label, workspace=workspace, activation_mode="auto",
                examples=examples, example_role="engram.projection" + label).functional(session=session)
            reduce = (PreparedChunkedAllReduce(partial[:rows].view(-1, 5120), projected[:rows].view(-1, 5120),
                runtime=runtime, session=session, coordinator=coordinator, name=name + ".reduce" + label,
                max_part_rows=max_part_rows).functional(session=session) if self.sharded else None)
            self.chunk_ops[rows] = (projection, reduce)
        self.projection, self.reduce = self.chunk_ops[chunk]
        self.norm_weight = weights["weight"].view(-1)
        self.eps = pool.config.text["rms_norm_eps"]
        self.mix_plan = hyperconnection.plan(hyperconnection.Caps(device=device, max_tokens=chunk, hidden_size=5120),
            invocation={"operation": "engram_mix", "eps": self.eps, "token_mask": True})
        declare(session, (self.mix_plan.request(name=name + ".mix", prepare_call=lambda state: PreparedCall(
            run=lambda: state.launch(source.view(tokens, -1)[:chunk], projected, self.norm_weight,
                                      mask[:chunk], output.view(tokens, -1)[:chunk], eps=self.eps))),))

    # --- TP3 (tp3.engram_side_layers): the hidden-state-free part on a side stream, started one layer early ---
    # Readiness wait + lookup + column gather + projection + all-reduce read only this round's staged rows (hashed
    # at the start of the target forward) and constants, never the hidden state; only the mix needs it. The decoder
    # starts them at the beginning of the previous layer (right after its predecessor's tail) and joins them after
    # that layer's attention, before its BRRR tail / attention reduction: the window between two tails runs no
    # collective on the main stream, so the side stream may use the shared all-reduce runtime (one channel, every
    # use still totally ordered and in the same order on every rank) and a spinning reduction never shares the GPU
    # with BRRR's persistent kernel. Same kernels, plans and arguments; the mix then runs where it always did.

    SIDE_PURPOSE = "ds41-engram"

    def enable_side(self):
        """Register the side task (decode/verify graphs: one projection chunk)."""
        from ignition.components.side_streams import register

        if self.chunked or len(self.bounds) != 1:
            raise ValueError("the Engram side stream needs a graph that projects in one chunk (decode/verify)")
        if not self.sharded:
            raise ValueError("the Engram side stream is for tensor-parallel ranks (it moves the reduction too)")
        self.side_task = register(self._run_side, self.SIDE_PURPOSE, self.columns.device)
        return self

    side_task = None

    def _run_side(self, inputs, outputs):
        """Side stream: exactly the kernels of ``__call__``'s one-chunk path up to the mix."""
        from ignition.components.engram import _lookup

        anchor, count = inputs
        gathered, barrier_input, barrier_output, local, partial, workspace, projected, *bases = outputs
        lookup, wait = self.lookup, self.lookup.wait
        _lookup(anchor, lookup.weights, lookup.scales, lookup.hashes, count, gathered, wait.timings.timestamps,
                wait.timings.frame, barrier_input, barrier_output, wait.interval, wait.handle, lookup.plan.handle,
                lookup.handle)
        torch.index_select(gathered, 1, self.columns, out=local)
        self.projection.run_into(local, partial, workspace, encoded_bases=tuple(bases) if bases else None)
        flat_in, flat_out = partial.view(-1, 5120), projected.view(-1, 5120)
        for (begin, end), part in zip(self.reduce.bounds, self.reduce.parts, strict=True):
            part(flat_in[begin:end], flat_out[begin:end])

    def start_side(self, anchor, metadata):
        """Traced, main stream: allocate every side buffer (as ``__call__`` would), fork after ``anchor`` (the
        previous layer's input). Returns the pending operands for ``finish_side``."""
        tokens, device = anchor.shape[0], anchor.device
        shape, dtype = self.lookup.output_layout
        gathered = torch.zeros(shape, dtype=dtype, device=device)
        barrier_input, barrier_output = self.lookup.wait.operands(anchor)
        local = torch.empty((tokens, self.columns.numel()), dtype=gathered.dtype, device=device)
        partial, workspace, bases = self.projection.allocate(device)
        projected = torch.empty((tokens, 25600), dtype=partial.dtype, device=device)
        outputs = [gathered, barrier_input, barrier_output, local, partial, workspace, projected, *(bases or ())]
        inputs = [anchor, metadata.live_counts[:1]]
        self.side_task.launch(anchor, inputs, outputs)
        return inputs, outputs

    def finish_side(self, pending, attended):
        """Traced, main stream: join after ``attended``'s producer and before its readers (it is declared written,
        so the BRRR tail / attention reduction that reads it is ordered after the join). Returns the projection."""
        inputs, outputs = pending
        self.side_task.join(inputs[0], inputs, [*outputs, attended])
        return outputs[6]

    def __call__(self, source, metadata, *, projected=None):
        """``projected`` (TP3 tp3.engram_side_layers): the reduced projection the side stream produced; only the mix
        runs here then. None = the whole Engram layer here (base)."""
        tokens = source.shape[0]
        if tokens != self.bounds[-1][1]:
            # The chunks tile exactly the declared rows: a wider source would leave its extra rows zero, a narrower one
            # would slice past its end (the one-pass mix used to refuse it by its capacity check).
            raise ValueError(f"Engram forward got {tokens} rows; it was declared for {self.bounds[-1][1]}")
        mask = (metadata.positions >= 0) & (metadata.token_ids < 129280)
        if projected is not None:
            if self.side_task is None or len(self.bounds) != 1:
                raise ValueError("a precomputed Engram projection needs the side stream (one chunk)")
            output = torch.zeros_like(source)
            (begin, end), = self.bounds  # the same operands as the one-chunk loop below
            hyperconnection.run_engram_mix(source.view(tokens, -1)[begin:end], projected, self.norm_weight,
                eps=self.eps, plan=self.mix_plan, out=output.view(tokens, -1)[begin:end], token_mask=mask[begin:end])
            return output
        gathered = self.lookup(source, metadata.live_counts[:1])
        output = torch.zeros_like(source)
        state, mixed = source.view(tokens, -1), output.view(tokens, -1)
        for begin, end in self.bounds:
            rows = end - begin
            projection, reduce = self.chunk_ops[rows]
            local = gathered[begin:end]
            if self.sharded:
                local = torch.index_select(local, 1, self.columns)
            projected = projection(local)
            if self.sharded:
                projected = reduce(projected.view(-1, 5120)).view(rows, 25600)
            hyperconnection.run_engram_mix(state[begin:end], projected, self.norm_weight, eps=self.eps,
                plan=self.mix_plan, out=mixed[begin:end], token_mask=mask[begin:end])
        return output
