# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DS41 TP3 side streams on one GPU with real checkpoint weights: switch off vs on write the same bytes.

Each test captures the serial forward and the side-stream forward of the same prepared operators in one session,
replays both on the same inputs (rows 1/6/12/21/43; three repeat replays of the side graph) and compares outputs and
every cache byte. FLARE_STREAMS_TIMING=1 also times FLARE_STREAMS_REPS (default 300) replays of each graph and prints
``STREAMS item=<index|engram|context> rows=N serial_us=.. side_us=.. saved_us=..``.

- index: index-source attention layers 2 (ratio-2 KV source: gates GEMV + query chain on the side), 20 (ratio-1 KV
  and candidate source) and 24 (index-only), TP4 head shard as the other DS41 layer tests; when the native extension
  has the qualification replay worker (PreparedGraphBundle), it replays both graphs too (else a STREAMS-NOTE line).
- engram: layer 14's Engram (TP3 rank-0 projection columns, synthetic staged rows) started before layer 13's
  attention and joined after it; the all-reduce is a single-rank copy here (one GPU), so the timing leaves out the
  ~18 us reduction that the side stream also hides at TP3.
- context: DSpark's context projections (TP4) beside a stand-in verification tail of FLARE_STREAMS_TAIL (100)
  dependent small kernels (c1 measured ~135 us of latency-bound verify/commit work after the LM head).

Requires FLARE_RUN_CUDA_TESTS=1 and FLARE_TEST_DS41_CHECKPOINT (a DS41 checkpoint view).
"""

import os
import statistics
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import triton
import triton.language as tl

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.aot import aot_forward
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.attention import AttentionScratch, PreparedAttention, Rotary
from ignition.recipes.deepseek41.pool import ContinuationPool
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or not os.environ.get("FLARE_TEST_DS41_CHECKPOINT"), reason="requires an assigned GPU and a DS41 checkpoint")

ROWS = (1, 6, 12, 21, 43)
START = 600  # positions 600..605: 5 pages of 128 tokens per request, compressed lengths below the 512 top-k


def _shape(rows):
    """(mode, per-request lengths) for a row count: one decode row, else verification blocks of up to 6 rows."""
    if rows == 1:
        return "decode", (1,)
    return "verify", (6,) * (rows // 6) + ((rows % 6,) if rows % 6 else ())


def _metadata(rows, device):
    mode, lengths = _shape(rows)
    batch, width = len(lengths), max(lengths)
    metadata = Metadata.allocate(rows, batch, 5, device, max_query_tokens=width)
    counts = torch.tensor(lengths, dtype=torch.int32, device=device)
    starts = torch.full((batch,), START, dtype=torch.int64, device=device)
    states = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
    pages = torch.arange(1, batch * 5 + 1, dtype=torch.int32, device=device).view(batch, 5)
    inputs = torch.zeros((batch, width), dtype=torch.int64, device=device)
    metadata.set(starts, counts, states, pages, inputs, torch.tensor([batch], dtype=torch.int32, device=device))
    return mode, batch, metadata, counts


def _attention_weights(loader, layer, text):
    prefix = f"layers.{layer}.attn"
    weights = dict(qkv_a=loader.load_linear_group((prefix + ".wq_a", prefix + ".wkv")),
        q_b=loader.load_linear(prefix + ".wq_b", rows=(0, 8192)),
        output=loader.load_attention_output(prefix, (0, 2048)),
        q_norm=loader.load_tensor(prefix + ".q_norm.weight"), kv_norm=loader.load_tensor(prefix + ".kv_norm.weight"),
        sink=loader.load_tensor(prefix + ".attn_sink", extent=slice(0, 16)))
    if layer in text["kv_source_layer_ids"]:
        weights["compressor"] = {"values": loader.load_tensor(prefix + ".compressor.wkv.weight"),
                                 "norm": loader.load_tensor(prefix + ".compressor.norm.weight", dtype=torch.float32)}
        if text["compress_ratios"][layer] == 2:
            weights["compressor"]["gates"] = loader.load_tensor(prefix + ".compressor.wgate.weight")
        weights["index_k"] = loader.load_tensor(prefix + ".indexer.wk.weight")
        weights["index_norm"] = loader.load_tensor(prefix + ".indexer.k_norm.weight")
    if layer in text["index_source_layer_ids"]:
        weights["index_q"] = loader.load_linear(prefix + ".indexer.wq_b")
        weights["index_weights"] = loader.load_tensor(prefix + ".indexer.weights_proj.weight")
    return weights


@pytest.fixture(scope="module")
def checkpoint():
    return inspect_indexed_checkpoint(Path(os.environ["FLARE_TEST_DS41_CHECKPOINT"]))


@pytest.fixture(scope="module")
def attention_weights(checkpoint):
    layers = (2, 13, 20, 24)
    with DeepSeekWeightLoader(checkpoint, torch.device("cuda:0"), shard=TensorParallelShard(0, 4),
            prefixes=tuple(f"layers.{layer}.attn." for layer in layers)) as loader:
        config = loader.config
        return {layer: _attention_weights(loader, layer, config.text) for layer in layers}, config


def _replay_us(graph, reps):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(reps):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3)
    return statistics.median(samples)


def _timing(item, rows, graphs):
    if os.environ.get("FLARE_STREAMS_TIMING") != "1":
        return
    reps = int(os.environ.get("FLARE_STREAMS_REPS", "300"))
    for graph in graphs.values():
        _replay_us(graph, 30)
    serial, side = (_replay_us(graphs[name], reps) for name in ("serial", "side"))
    print(f"\nSTREAMS item={item} rows={rows} serial_us={serial:.2f} side_us={side:.2f} saved_us={serial - side:.2f}")


def _capture(session, forwards, owners):
    graphs = {}
    for name, forward in forwards.items():
        compiled = aot_forward(forward)
        compiled(forward)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with session.capture(), torch.cuda.graph(graph):
            compiled(forward)
        graph.instantiate()
        graphs[name] = graph
        owners.extend((forward, compiled))
    return graphs


def _replay_and_compare(graphs, reset, observe, order=("serial", "side", "side", "side", "side", "serial")):
    """Replay in ``order`` from the same state (``reset``); every replay must write the first one's bytes."""
    expected = None
    for name in order:
        reset()
        torch.cuda.synchronize()
        graphs[name].replay()
        torch.cuda.synchronize()
        actual = tuple(value.clone() for value in observe())
        if expected is None:
            expected = actual
            continue
        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            assert torch.equal(value.view(torch.uint8), reference.view(torch.uint8)), (name, index)


# --------------------------------------------------------------------------------------------------------------------
# index


class IndexForward:
    """Layers 2 -> 20 -> 24 attention (each its own source), selections chained as in the decoder."""

    def __init__(self, layers, sources, metadata, outputs, selected):
        self.layers, self.sources, self.metadata = layers, sources, metadata
        self.outputs, self.selected = outputs, selected

    def __call__(self):
        selection = (torch.full_like(self.selected[0], -1), None, None)
        for index, (attention, source) in enumerate(zip(self.layers, self.sources, strict=True)):
            if attention.layer == 24:
                selection = (torch.full_like(self.selected[0], -1), *selection[1:])
            output, _, selection, _ = attention._run(source, self.metadata, selection)
            self.outputs[index].copy_(output)
            self.selected[index].copy_(selection[0])


@pytest.mark.parametrize("rows", ROWS)
@torch.no_grad()
def test_index_side_stream_is_bit_identical(attention_weights, rows):
    from ignition.preparation.graphs import PreparedGraphBundle

    torch._dynamo.reset()
    weights, config = attention_weights
    device = torch.device("cuda:0")
    torch.manual_seed(41152 + rows)
    mode, batch, metadata, _ = _metadata(rows, device)
    pool = ContinuationPool.allocate(config, batch + 1, batch * 5 + 1, device)
    scratch = AttentionScratch.allocate(rows, 4, device)
    sources = [torch.randn((rows, 5120), dtype=torch.bfloat16, device=device) * .1 for _ in range(3)]
    outputs = [torch.empty((rows, 5120), dtype=torch.bfloat16, device=device) for _ in range(3)]
    selected = [torch.empty((rows, 512), dtype=torch.int32, device=device) for _ in range(3)]
    arena = ScratchArena(device, overlap=True)
    owners = []
    with RecordedPreparationSession(device=device, checkpoint="ds41-index-side-stream",
            workload_schema=f"index-side-rows{rows}", numerical_policy="checkpoint", compile_workers=16) as session:
        session.register_scratch(arena)
        side = [PreparedAttention(layer, weights[layer], sources[i], scratch, metadata, pool, Rotary(config, device),
                    mode=mode, world=4, session=session, name=f"layer{layer}", workspace=arena,
                    index_side_stream=True) for i, layer in enumerate((2, 20, 24))]
        assert all(attention.index_side is not None for attention in side)
        assert side[0].index_side.gates_task is not None and side[1].index_side.gates_task is None
        serial = []
        for attention in side:
            plain = copy(attention)
            plain.index_side = None
            serial.append(plain)
        session.tune(show_progress=False)
        graphs = _capture(session, {"serial": IndexForward(serial, sources, metadata, outputs, selected),
                                    "side": IndexForward(side, sources, metadata, outputs, selected)}, owners)
        session.freeze()
        saved = pool.backing.clone()

        def reset():
            pool.backing.copy_(saved)
            arena.tensor.fill_(0xA5)
            for value in (*outputs, *selected):
                value.fill_(-1)

        try:
            _replay_and_compare(graphs, reset, lambda: (*outputs, *selected, pool.backing))
            import flare_heliopause
            if not hasattr(flare_heliopause, "GraphReplayWorker"):
                # The venv's extension is built without the qualification feature: torch replay only.
                print(f"\nSTREAMS-NOTE rows={rows} native replay worker not built (qualification feature): skipped")
                reset()
                _timing("index", rows, graphs)
                return
            bundle = PreparedGraphBundle(device=0, graphs=graphs, buffers={"source": sources[0]},
                resources=(weights, pool, scratch, metadata, arena, session, owners))
            worker = bundle.start_worker()
            try:  # native replay (the served runtime's path) writes the same bytes
                results = []
                for name in ("serial", "side", "side"):
                    reset()
                    torch.cuda.synchronize()
                    ticket = worker.submit(bundle.graph_indices[name])
                    result = worker.poll(timeout_ms=120000)
                    assert result is not None and result.ticket == ticket and result.error is None
                    results.append(tuple(value.clone() for value in (*outputs, *selected, pool.backing)))
                for other in results[1:]:
                    for value, reference in zip(other, results[0], strict=True):
                        assert torch.equal(value.view(torch.uint8), reference.view(torch.uint8))
            finally:
                worker.close()
            reset()
            _timing("index", rows, graphs)
        finally:
            for graph in graphs.values():
                graph.reset()


# --------------------------------------------------------------------------------------------------------------------
# engram


class EngramForward:
    def __init__(self, attention, engram, source13, residual14, metadata, attended, mixed, side):
        self.attention, self.engram, self.metadata = attention, engram, metadata
        self.source13, self.residual14, self.attended, self.mixed, self.side = (source13, residual14, attended,
                                                                                mixed, side)

    def __call__(self):
        selection = (torch.full((self.source13.shape[0], 512), -1, dtype=torch.int32, device=self.source13.device),
                     None, None)
        pending = self.engram.start_side(self.residual14, self.metadata) if self.side else None
        output, _, _, _ = self.attention._run(self.source13, self.metadata, selection)
        projected = None if pending is None else self.engram.finish_side(pending, output)
        self.attended.copy_(output)
        self.mixed.copy_(self.engram(self.residual14, self.metadata, projected=projected))


@pytest.mark.parametrize("rows", ROWS)
@torch.no_grad()
def test_engram_side_stream_is_bit_identical(attention_weights, checkpoint, rows, monkeypatch):
    from b12x.sequence import engram as b12x_engram

    from ignition.components.engram import EngramInputs
    from ignition.components.readiness import ReadinessTimings
    from ignition.recipes.deepseek41 import engram as engram_module

    torch._dynamo.reset()
    weights, config = attention_weights
    device = torch.device("cuda:0")
    torch.manual_seed(41153 + rows)
    _, batch, metadata, _ = _metadata(rows, device)
    with DeepSeekWeightLoader(checkpoint, device, shard=TensorParallelShard(0, 3),
            prefixes=("layers.14.engram.",)) as loader:
        engram_weights = {"projection": loader.load_linear_columns("layers.14.engram.wkv",
                ((head * 256, (head + 1) * 256) for head in range(0, 24, 3))),
            "weight": (loader.load_tensor("layers.14.engram.q_weight", dtype=torch.float32)
                       * loader.load_tensor("layers.14.engram.k_weight", dtype=torch.float32))}
    # One GPU: the reduction of the projection is a single-rank copy (the TP3 reduction is a b12x PCIe collective).
    real_reduce = engram_module.PreparedChunkedAllReduce
    monkeypatch.setattr(engram_module, "PreparedChunkedAllReduce",
                        lambda source, output, *, runtime, **kwargs: real_reduce(source, output, runtime=None, **kwargs))
    geometry = b12x_engram.build_geometry(layer_ids=(1, 14), base_table_size=101, compressed_vocab_size=32)
    frame = torch.zeros(1, dtype=torch.int32, device=device)
    inputs = EngramInputs.allocate(rows, device, timings=ReadinessTimings.allocate(frame, 1))
    staged = torch.randint(0, 0x77, inputs.weights.shape, dtype=torch.uint8, device=device)
    inputs.weights.copy_(staged | (torch.randint(0, 2, staged.shape, device=device, dtype=torch.uint8) << 7))
    inputs.scales.copy_(torch.randint(124, 128, inputs.scales.shape, dtype=torch.uint8, device=device))
    inputs.hashes.copy_(torch.arange(rows * 24, device=device).view(rows, 24) % geometry.num_embeddings[1])
    tables = SimpleNamespace(inputs={14: inputs}, token_map=tuple(range(32)), geometry=geometry,
        shard=TensorParallelShard(0, 3), readiness_runtime=None,
        columns=torch.tensor([head * 256 + column for head in range(0, 24, 3) for column in range(256)],
                             dtype=torch.int64, device=device))
    pool = ContinuationPool.allocate(config, batch + 1, batch * 5 + 1, device)
    scratch = AttentionScratch.allocate(rows, 4, device)
    source13 = torch.randn((rows, 5120), dtype=torch.bfloat16, device=device) * .1
    residual14 = torch.randn((rows, 4, 5120), dtype=torch.bfloat16, device=device) * .1
    engram_out = torch.empty_like(residual14)
    attended = torch.empty((rows, 5120), dtype=torch.bfloat16, device=device)
    mixed = torch.empty_like(residual14)
    arena = ScratchArena(device, overlap=True)
    owners = []
    with RecordedPreparationSession(device=device, checkpoint="ds41-engram-side-stream",
            workload_schema=f"engram-side-rows{rows}", numerical_policy="checkpoint", compile_workers=16) as session:
        session.register_scratch(arena)
        mode = "decode" if rows == 1 else "verify"
        attention = PreparedAttention(13, weights[13], source13, scratch, metadata, pool, Rotary(config, device),
            mode=mode, world=4, session=session, name="layer13", workspace=arena)
        engram = engram_module.PreparedEngram(14, engram_weights, tables, residual14, engram_out, metadata,
            SimpleNamespace(config=config), scratch=engram_module.EngramScratch.allocate(rows, device, world=3),
            session=session, name="layer14.engram", workspace=arena, runtime=object(), coordinator=None)
        engram.enable_side()
        session.tune(show_progress=False)
        graphs = _capture(session, {
            "serial": EngramForward(attention, engram, source13, residual14, metadata, attended, mixed, False),
            "side": EngramForward(attention, engram, source13, residual14, metadata, attended, mixed, True)},
            owners)
        session.freeze()
        saved = pool.backing.clone()

        def reset():
            pool.backing.copy_(saved)
            arena.tensor.fill_(0xA5)
            attended.fill_(float("nan"))
            mixed.fill_(float("nan"))

        try:
            _replay_and_compare(graphs, reset, lambda: (attended, mixed, pool.backing))
            assert torch.isfinite(mixed).all()
            _timing("engram", rows, graphs)
        finally:
            for graph in graphs.values():
                graph.reset()


# --------------------------------------------------------------------------------------------------------------------
# context


@triton.jit
def _tail_step(value):
    tl.store(value, tl.load(value) * 1.0001 + 1)


class ContextForward:
    def __init__(self, context, auxiliary, counts, metadata, tail, side):
        self.context, self.auxiliary, self.counts, self.metadata = context, auxiliary, counts, metadata
        self.tail, self.side = tail, side

    def __call__(self):
        pending = self.context.start(self.auxiliary) if self.side else None
        for _ in range(self.tail.shape[0]):  # stand-in verification tail: dependent one-CTA launches (unfused)
            torch.library.wrap_triton(_tail_step)[(1,)](self.tail)
        if pending is None:
            self.context(self.auxiliary, self.counts, self.metadata)
        else:
            self.context(self.auxiliary, self.counts, self.metadata, pending=pending)


@pytest.mark.parametrize("rows", ROWS)
@torch.no_grad()
def test_context_side_stream_is_bit_identical(checkpoint, rows):
    from ignition.components.dspark import PreparedContext

    torch._dynamo.reset()
    device = torch.device("cuda:0")
    torch.manual_seed(41154 + rows)
    prefixes = ("mtp.0.main_proj", "mtp.0.main_norm", *(f"mtp.{i}.attn.{part}"
                for i in range(3) for part in ("wkv", "kv_norm")))
    with DeepSeekWeightLoader(checkpoint, device, shard=TensorParallelShard(0, 4), prefixes=prefixes,
            dspark=True) as loader:
        config = loader.config
        draft = SimpleNamespace(main_projection=loader.load_linear("mtp.0.main_proj"),
            context_kv=loader.load_linear_group(tuple(f"mtp.{i}.attn.wkv" for i in range(3))),
            main_norm=loader.load_tensor("mtp.0.main_norm.weight"), layers=tuple(
                SimpleNamespace(attention={"kv_norm": loader.load_tensor(f"mtp.{i}.attn.kv_norm.weight")})
                for i in range(3)))
    weights = SimpleNamespace(config=config, draft=draft)
    _, batch, metadata, counts = _metadata(rows, device)
    pool = ContinuationPool.allocate(config, batch + 1, batch * 5 + 1, device, dspark=True)
    auxiliary = torch.randn((rows, 3 * 5120), dtype=torch.bfloat16, device=device) * .1
    tail = torch.ones(int(os.environ.get("FLARE_STREAMS_TAIL", "100")), device=device)  # one launch per entry
    arena = ScratchArena(device, overlap=True)
    owners = []
    with RecordedPreparationSession(device=device, checkpoint="ds41-context-side-stream",
            workload_schema=f"context-side-rows{rows}", numerical_policy="checkpoint", compile_workers=16) as session:
        session.register_scratch(arena)
        context = PreparedContext(weights, auxiliary, metadata, pool, session=session, name="context",
                                  workspace=arena, side_stream=True)
        assert context.side_task is not None
        session.tune(show_progress=False)
        graphs = _capture(session, {"serial": ContextForward(context, auxiliary, counts, metadata, tail, False),
                                    "side": ContextForward(context, auxiliary, counts, metadata, tail, True)}, owners)
        session.freeze()
        saved = pool.backing.clone()

        def reset():
            pool.backing.copy_(saved)
            arena.tensor.fill_(0xA5)
            tail.fill_(1)

        try:
            windows = lambda: (*(pool.window(i, draft=True) for i in range(3)), tail)
            _replay_and_compare(graphs, reset, windows)
            assert any(bool((window != 0).any()) for window in windows()[:3])  # the rings were written
            _timing("context", rows, graphs)
        finally:
            for graph in graphs.values():
                graph.reset()
