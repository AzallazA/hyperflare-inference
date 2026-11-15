# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DS41 TP3 mma_pad_once on one GPU with real checkpoint weights: per-layer fills vs one fill per forward write the
same bytes.

Attention layers 2 (ratio-2 KV source), 13 (plain), 20 (ratio-1 KV + candidate source) and 24 (index-only), TP3
rank-0 geometry (24 heads, BF16 MLA as served), chained as in the decoder, at rows 1/6/12/21/43. Four graphs of the
same prepared operators: {per-layer fills, one fill per forward} x {serial indexer chain, tp3.index_side_stream}.
Every graph is replayed from the same state (pool, poisoned arena) and must write the first one's outputs, selections
and every cache byte; the pad graphs run three times. FLARE_LAUNCHMERGE_TIMING=1 also times
FLARE_LAUNCHMERGE_REPS (default 300) replays of each graph and prints
``LAUNCHMERGE item=mma_pad rows=N side=0|1 base_us=.. pads_us=.. saved_us=.. layers=4``
(the served round has 40 target + 3 draft layers: per-round saving ~= saved_us / 4 x 43; the pad graph's own two
fills are already inside pads_us, and a round has two forwards, target and draft, so subtract ~2 more fills).

Requires FLARE_RUN_CUDA_TESTS=1 and FLARE_TEST_DS41_CHECKPOINT (a DS41 checkpoint view).
"""

import os
import statistics
from copy import copy
from pathlib import Path

import pytest
import torch

from ignition.components.paged_batch import Metadata
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.aot import aot_forward
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from ignition.recipes.deepseek41.attention import AttentionScratch, PreparedAttention, Rotary
from ignition.recipes.deepseek41.decoder import allocate_mma_pads, mma_pad_layouts
from ignition.recipes.deepseek41.pool import ContinuationPool
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or not os.environ.get("FLARE_TEST_DS41_CHECKPOINT"), reason="requires an assigned GPU and a DS41 checkpoint")

ROWS = (1, 6, 12, 21, 43)
WORLD = 3
MLA = "bf16"
START = 600
LAYERS = (2, 13, 20, 24)


def _shape(rows):
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
    return mode, batch, metadata


def _attention_weights(loader, layer, text):
    """weights.DeepSeekWeightLoader.load_layer's attention part for TP3 rank 0 (24 query heads, padded)."""
    prefix = f"layers.{layer}.attn"
    first, last = TensorParallelShard(0, WORLD).padded_partition(8)
    weights = dict(qkv_a=loader.load_linear_group((prefix + ".wq_a", prefix + ".wkv")),
        q_b=loader.load_linear_group((prefix + ".wq_b",), row_ranges=((first * 8 * 512, last * 8 * 512),), pad=True),
        output=loader.load_attention_output(prefix, (first * 1024, last * 1024), pad=True),
        q_norm=loader.load_tensor(prefix + ".q_norm.weight"), kv_norm=loader.load_tensor(prefix + ".kv_norm.weight"),
        sink=loader.load_padded_tensor(prefix + ".attn_sink", rows=(first * 8, last * 8)))
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
def attention_weights():
    checkpoint = inspect_indexed_checkpoint(Path(os.environ["FLARE_TEST_DS41_CHECKPOINT"]))
    with DeepSeekWeightLoader(checkpoint, torch.device("cuda:0"), shard=TensorParallelShard(0, WORLD),
            prefixes=tuple(f"layers.{layer}.attn." for layer in LAYERS)) as loader:
        config = loader.config
        return {layer: _attention_weights(loader, layer, config.text) for layer in LAYERS}, config


class ChainForward:
    """Layers 2 -> 13 -> 20 -> 24 attention, selections chained as in the decoder; ``layouts`` = the forward's
    once-zeroed MMA-scale tiles (tp3.mma_pad_once) or () for today's per-layer fills."""

    def __init__(self, layers, sources, metadata, outputs, selected, layouts):
        self.layers, self.sources, self.metadata = layers, sources, metadata
        self.outputs, self.selected, self.layouts = outputs, selected, layouts

    def __call__(self):
        pads = allocate_mma_pads(self.layouts, self.sources[0].device)
        selection = (torch.full_like(self.selected[0], -1), None, None)
        for index, (attention, source) in enumerate(zip(self.layers, self.sources, strict=True)):
            if attention.layer == 24:
                selection = (torch.full_like(self.selected[0], -1), *selection[1:])
            output, _, selection, _ = attention._run(source, self.metadata, selection,
                                                     **({} if pads is None else {"mma_pads": pads}))
            self.outputs[index].copy_(output)
            self.selected[index].copy_(selection[0])


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


@pytest.mark.parametrize("rows", ROWS)
@torch.no_grad()
def test_mma_pad_once_is_bit_identical(attention_weights, rows):
    torch._dynamo.reset()
    weights, config = attention_weights
    device = torch.device("cuda:0")
    torch.manual_seed(41155 + rows)
    mode, batch, metadata = _metadata(rows, device)
    pool = ContinuationPool.allocate(config, batch + 1, batch * 5 + 1, device)
    scratch = AttentionScratch.allocate(rows, WORLD, device)
    sources = [torch.randn((rows, 5120), dtype=torch.bfloat16, device=device) * .1 for _ in LAYERS]
    outputs = [torch.empty((rows, 5120), dtype=torch.bfloat16, device=device) for _ in LAYERS]
    selected = [torch.empty((rows, 512), dtype=torch.int32, device=device) for _ in LAYERS]
    arena = ScratchArena(device, overlap=True)
    owners, graphs = [], {}
    with RecordedPreparationSession(device=device, checkpoint="ds41-mma-pad-once",
            workload_schema=f"mma-pad-rows{rows}", numerical_policy="checkpoint", compile_workers=16) as session:
        session.register_scratch(arena)
        side = [PreparedAttention(layer, weights[layer], sources[i], scratch, metadata, pool, Rotary(config, device),
                    mode=mode, world=WORLD, session=session, name=f"layer{layer}", workspace=arena,
                    mla_compute_mode=MLA, index_side_stream=True) for i, layer in enumerate(LAYERS)]
        serial = []
        for attention in side:
            plain = copy(attention)
            plain.index_side = None
            serial.append(plain)
        session.tune(show_progress=False)
        layouts = mma_pad_layouts(side)
        assert len(layouts) == 2  # qkv_a (K 5120) and q_b (K 1280) tiles; rows below 128 always pad
        forwards = {"base": ChainForward(serial, sources, metadata, outputs, selected, ()),
                    "pads": ChainForward(serial, sources, metadata, outputs, selected, layouts),
                    "base-side": ChainForward(side, sources, metadata, outputs, selected, ()),
                    "pads-side": ChainForward(side, sources, metadata, outputs, selected, layouts)}
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
        session.freeze()
        saved = pool.backing.clone()

        def reset():
            pool.backing.copy_(saved)
            arena.tensor.fill_(0xA5)
            for value in (*outputs, *selected):
                value.fill_(-1)

        try:
            expected = None
            for name in ("base", "pads", "pads", "pads", "base-side", "pads-side", "pads-side", "base"):
                reset()
                torch.cuda.synchronize()
                graphs[name].replay()
                torch.cuda.synchronize()
                actual = tuple(value.clone() for value in (*outputs, *selected, pool.backing))
                if expected is None:
                    expected = actual
                    continue
                for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
                    assert torch.equal(value.view(torch.uint8), reference.view(torch.uint8)), (name, index)
            if os.environ.get("FLARE_LAUNCHMERGE_TIMING") == "1":
                reps = int(os.environ.get("FLARE_LAUNCHMERGE_REPS", "300"))
                for graph in graphs.values():
                    _replay_us(graph, 30)
                for suffix, flag in (("", 0), ("-side", 1)):
                    base, pads = (_replay_us(graphs[name + suffix], reps) for name in ("base", "pads"))
                    print(f"\nLAUNCHMERGE item=mma_pad rows={rows} side={flag} base_us={base:.2f} pads_us={pads:.2f} "
                          f"saved_us={base - pads:.2f} layers={len(LAYERS)}")
        finally:
            for graph in graphs.values():
                graph.reset()
