# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DS41 TP3 glue_pass2 on one GPU with real checkpoint weights: switch off vs on write the same bytes.

Four arms per case, all captured in one session from the same prepared operators and replayed from the same state:
  off-clean (reference) | off-poison | on-clean | on-poison (0xFF, then 0x7F)
"poison" traces the forward with every ``torch.empty`` / ``empty_like`` / ``empty_strided`` followed by a fill of
the poison byte, so the captured graph re-poisons each uninitialised buffer on EVERY replay (0xFF = NaN in FP8,
BF16, FP32 and E8M0; 0x7F = NaN in FP8 E4M3, 3.4e38 in BF16/FP32, 2^0 in E8M0). Every arm must write the
reference's bytes: attention outputs, selections and (verification) the committed pool rings. An off-poison
failure is not the switch's (it would be an older uninitialised buffer); it is reported separately.

- verification / decode: attention layers 2 (ratio-2 KV source), 20 (candidate source) and 24, chained as in the
  decoder, TP3 rank-0 geometry (24 heads, BF16 MLA as served), rows 1 (decode) / 6 / 21 / 43 (verification
  blocks of up to 6); the verification staging is allocated per forward as the decoder does (zeros off, empty on)
  and the accepted prefixes (all / all but one, alternating) are committed (attention.commit).
- draft: three draft-mode attention layers (layer 13's projections, no compressor or indexer, as DSpark's) over
  blocks of 6 rows for 1 and 8 requests (one request idle at 8), window cache allocated by ``_run``.
What the switch changes here: the verification staging fill and the draft window-cache fill (every other item is
value-identical by construction; CPU: tests/recipes/test_deepseek41_glue_pass2.py). The poison arms also cover the
older uninitialised buffers of these layers (first glue pass) in the same replays.

Requires FLARE_RUN_CUDA_TESTS=1 and FLARE_TEST_DS41_CHECKPOINT (a DS41 checkpoint view).
"""

import os
from contextlib import contextmanager
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
from ignition.recipes.deepseek41.pool import ContinuationPool
from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1"
    or not os.environ.get("FLARE_TEST_DS41_CHECKPOINT"), reason="requires an assigned GPU and a DS41 checkpoint")

WORLD = 3
MLA = "bf16"
START = 600
BLOCK = 6
PAGES = 7  # 896 tokens per request >= the last start 600 + 37 x 7 + 6 (5 pages clamped late rows onto the last page)
ARMS = (("off", None), ("off", 0xFF), ("on", None), ("on", 0xFF), ("on", 0x7F))


_PATTERNS = {}


def _pattern(dtype, byte):
    """The value whose bytes are ``byte`` repeated, computed on the host OUTSIDE the trace (no .item() in the graph).
    A dtype-view fill (``tensor.view(torch.uint8).fill_``) of a graph-local buffer makes flare's storage lowering
    (ignition/preparation/compilation.py, StorageBackend) overlap live buffers: that was seq226's verify alias error
    and its poison-arm draft differences, with or without the switch (CPU probe poison_alias_flare_probe.py, 11-17)."""
    key = (dtype, byte)
    if key not in _PATTERNS:
        if dtype == torch.bool:
            _PATTERNS[key] = True
        else:
            size = torch.empty(0, dtype=dtype).element_size()
            _PATTERNS[key] = torch.full((size,), byte, dtype=torch.uint8).view(dtype).item()
    return _PATTERNS[key]


def _poisoned(tensor, byte):
    if tensor.numel() == 0:
        return tensor
    return tensor.fill_(_pattern(tensor.dtype, byte))  # NaN payloads may be canonicalised; still NaN


@contextmanager
def _poison_allocations(byte):
    if byte is None:
        yield
        return
    for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e8m0fnu,
                  torch.uint8, torch.int8, torch.int32, torch.int64, torch.bool):
        _pattern(dtype, byte)  # warm the table before Dynamo traces the patched allocators
    saved = torch.empty, torch.empty_like, torch.empty_strided
    empty, empty_like, empty_strided = saved
    torch.empty = lambda *args, **kwargs: _poisoned(empty(*args, **kwargs), byte)
    torch.empty_like = lambda *args, **kwargs: _poisoned(empty_like(*args, **kwargs), byte)
    torch.empty_strided = lambda *args, **kwargs: _poisoned(empty_strided(*args, **kwargs), byte)
    try:
        yield
    finally:
        torch.empty, torch.empty_like, torch.empty_strided = saved


def _attention_weights(loader, layer, text, *, draft=False):
    """TP3 rank-0 attention weights (as tests/recipes/test_deepseek41_side_streams_cuda.py); draft: no KV source."""
    prefix = f"layers.{layer}.attn"
    first, last = TensorParallelShard(0, WORLD).padded_partition(8)
    weights = dict(qkv_a=loader.load_linear_group((prefix + ".wq_a", prefix + ".wkv")),
        q_b=loader.load_linear_group((prefix + ".wq_b",), row_ranges=((first * 8 * 512, last * 8 * 512),), pad=True),
        output=loader.load_attention_output(prefix, (first * 1024, last * 1024), pad=True),
        q_norm=loader.load_tensor(prefix + ".q_norm.weight"), kv_norm=loader.load_tensor(prefix + ".kv_norm.weight"),
        sink=loader.load_padded_tensor(prefix + ".attn_sink", rows=(first * 8, last * 8)))
    if draft:
        return weights
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
def weights():
    checkpoint = inspect_indexed_checkpoint(Path(os.environ["FLARE_TEST_DS41_CHECKPOINT"]))
    layers = (2, 13, 20, 24)
    with DeepSeekWeightLoader(checkpoint, torch.device("cuda:0"), shard=TensorParallelShard(0, WORLD),
            prefixes=tuple(f"layers.{layer}.attn." for layer in layers)) as loader:
        config = loader.config
        target = {layer: _attention_weights(loader, layer, config.text) for layer in (2, 20, 24)}
        draft = _attention_weights(loader, 13, config.text, draft=True)
    return target, draft, config


def _metadata(lengths, device, *, tokens=None):
    """Packed requests (lengths; 0 = idle row) in ``tokens`` rows (default: packed, sum of lengths)."""
    batch, width = len(lengths), max(max(lengths), 1)
    rows = sum(lengths) if tokens is None else tokens
    metadata = Metadata.allocate(rows, batch, PAGES, device, max_query_tokens=width)
    counts = torch.tensor(lengths, dtype=torch.int32, device=device)
    starts = torch.full((batch,), START, dtype=torch.int64, device=device) + torch.arange(batch, device=device) * 37
    states = torch.arange(1, batch + 1, dtype=torch.int64, device=device)
    pages = torch.arange(1, batch * PAGES + 1, dtype=torch.int32, device=device).view(batch, PAGES)
    inputs = torch.zeros((batch, width), dtype=torch.int64, device=device)
    metadata.set(starts, counts, states, pages, inputs, torch.tensor([batch], dtype=torch.int32, device=device))
    return rows, batch, metadata, counts


def _switched(attention, on):
    """The same prepared operators with glue_pass2 on or off (read only by PreparedAttention._run)."""
    view = copy(attention)
    view.glue_pass2 = on
    return view


class TargetForward:
    """Layers 2 -> 20 -> 24 as the decoder runs them; verification staging per forward, then the accepted commit."""

    def __init__(self, layers, sources, metadata, accepted, outputs, selected, staging_shape, on):
        self.layers, self.sources, self.metadata, self.accepted = layers, sources, metadata, accepted
        self.outputs, self.selected, self.staging_shape, self.on = outputs, selected, staging_shape, on

    def __call__(self):
        staging = None if self.staging_shape is None else (torch.empty if self.on else torch.zeros)(
            self.staging_shape, dtype=torch.uint8, device=self.sources[0].device)
        selection = (torch.full_like(self.selected[0], -1), None, None)
        records = []
        for index, (attention, source) in enumerate(zip(self.layers, self.sources, strict=True)):
            if attention.layer == 24:
                selection = (torch.full_like(self.selected[0], -1), *selection[1:])
            output, record, selection, _ = attention._run(source, self.metadata, selection,
                window_cache=None if staging is None else staging[index])
            self.outputs[index].copy_(output)
            self.selected[index].copy_(selection[0])
            records.append(record)
        if staging is not None:
            for attention, record in zip(self.layers, records, strict=True):
                attention.commit(self.accepted, record, self.metadata)


class DraftForward:
    def __init__(self, layers, sources, metadata, outputs):
        self.layers, self.sources, self.metadata, self.outputs = layers, sources, metadata, outputs

    def __call__(self):
        selection = (torch.full((self.sources[0].shape[0], 512), -1, dtype=torch.int32,
                                device=self.sources[0].device), None, None)
        for index, (attention, source) in enumerate(zip(self.layers, self.sources, strict=True)):
            output, _, selection, _ = attention._run(source, self.metadata, selection)
            self.outputs[index].copy_(output)


def _capture_arms(session, make, owners):
    graphs = {}
    for name, byte in ARMS:  # clean arms first: a poisoned trace patches the allocators while it compiles
        print(f"\nGLUE2 capture arm={name} poison={'clean' if byte is None else hex(byte)}", flush=True)
        torch._dynamo.reset()
        forward = make(name == "on")
        with _poison_allocations(byte):
            compiled = aot_forward(forward)
            compiled(forward)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with session.capture(), torch.cuda.graph(graph):
                compiled(forward)
        graph.instantiate()
        graphs[(name, byte)] = graph
        owners.extend((forward, compiled))
    return graphs


def assert_same_selection(value, reference, history, label):
    """``value`` / ``reference`` [rows, 512] int32 selections of one layer; ``history`` per row (or one int): the
    compressed entries ``0 <= index < history`` this test never writes (the pool's KV pages start zeroed), so they all
    carry an all-zero index key and latent row: identical scores and identical attention contributions. The MXFP4
    radix top-k picks an arbitrary subset of such ties when there are more entries than 512 (layer 20, ratio 1: ~600),
    run to run (seq226 / seq320 ('pads', 6), 11-17). Equal means: each row the same multiset outside the history range
    (this forward's tokens, -1 sentinels) and the same number of distinct history entries."""
    assert value.shape == reference.shape, label
    rows = value.shape[0]
    history = [history] * rows if isinstance(history, int) else list(history)
    for row, (mine, theirs) in enumerate(zip(value.cpu().tolist(), reference.cpu().tolist(), strict=True)):
        h = history[row]
        tied = [sorted(i for i in x if 0 <= i < h) for x in (mine, theirs)]
        fixed = [sorted(i for i in x if not 0 <= i < h) for x in (mine, theirs)]
        assert fixed[0] == fixed[1] and len(tied[0]) == len(tied[1]), (*label, row)
        for picked in tied:
            assert len(set(picked)) == len(picked), (*label, row, "duplicate history index")


def _same(value, reference, index, selections, label):
    if selections is not None and index in selections:
        try:
            assert_same_selection(value, reference, selections[index], label)
        except AssertionError:
            return False
        return True
    return torch.equal(value.view(torch.uint8), reference.view(torch.uint8))


def _compare(graphs, reset, observe, selections=None):
    """``selections``: {observed index: per-row history} for selection outputs (compared up to zero-history ties,
    assert_same_selection); everything else byte for byte."""
    expected, failures = None, []
    for arm in (*ARMS, *ARMS):  # each arm twice: a replay after another arm's replay writes the same bytes
        reset()
        torch.cuda.synchronize()
        graphs[arm].replay()
        torch.cuda.synchronize()
        actual = tuple(value.clone() for value in observe())
        if expected is None:
            expected = actual
            continue
        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            if not _same(value, reference, index, selections, (arm, index)):
                failures.append((arm, index))
    base = [failure for failure in failures if failure[0][0] == "off"]
    switch = [failure for failure in failures if failure[0][0] == "on"]
    print(f"\nGLUE2 arms={len(ARMS)} off-failures={sorted(set(base))} on-failures={sorted(set(switch))}")
    assert not base, f"switch OFF differs from the clean reference (not glue_pass2's): {sorted(set(base))}"
    assert not switch, f"glue_pass2 differs from the clean reference: {sorted(set(switch))}"


@pytest.mark.parametrize("lengths", ((1,), (6,), (6, 6, 6, 3), (6, 6, 6, 6, 6, 6, 6, 1)),
                         ids=("decode-rows1", "verify-rows6", "verify-rows21", "verify-rows43"))
@torch.no_grad()
def test_target_layers_switch_on_writes_the_same_bytes(weights, lengths):
    target, _, config = weights
    device = torch.device("cuda:0")
    torch.manual_seed(41152 + sum(lengths))
    rows, batch, metadata, counts = _metadata(lengths, device)
    mode = "decode" if rows == 1 else "verify"
    pool = ContinuationPool.allocate(config, batch + 1, batch * PAGES + 1, device)
    pool.backing.random_(0, 120)  # finite history (no 0x7F / 0xFF bytes), so the reference output is finite
    scratch = AttentionScratch.allocate(rows, WORLD, device)
    sources = [torch.randn((rows, 5120), dtype=torch.bfloat16, device=device) * .1 for _ in range(3)]
    outputs = [torch.empty((rows, 5120), dtype=torch.bfloat16, device=device) for _ in range(3)]
    selected = [torch.empty((rows, 512), dtype=torch.int32, device=device) for _ in range(3)]
    accepted = torch.where(torch.arange(batch, device=device) % 2 == 1, (counts - 1).clamp_min(0), counts)
    staging_shape = None if mode == "decode" else (3, batch, (128 + metadata.max_query_tokens) * 528)
    arena = ScratchArena(device, overlap=True)
    owners = []
    with RecordedPreparationSession(device=device, checkpoint="ds41-glue-pass2",
            workload_schema=f"glue2-target-rows{rows}", numerical_policy="checkpoint", compile_workers=16) as session:
        session.register_scratch(arena)
        layers = [PreparedAttention(layer, target[layer], sources[i], scratch, metadata, pool, Rotary(config, device),
                      mode=mode, world=WORLD, session=session, name=f"layer{layer}", workspace=arena,
                      mla_compute_mode=MLA, glue_pass2=True) for i, layer in enumerate((2, 20, 24))]
        session.tune(show_progress=False)
        graphs = _capture_arms(session, lambda on: TargetForward([_switched(layer, on) for layer in layers], sources,
            metadata, accepted, outputs, selected, staging_shape, on), owners)
        session.freeze()
        saved = pool.backing.clone()
        pages = [pool.pages[name] for name in sorted(pool.pages)]  # KV pages (index keys, latent rows): not in backing
        saved_pages = [page.clone() for page in pages]
        request_start = [START + 37 * request for request, count in enumerate(lengths) for _ in range(count)]
        selections = {len(outputs) + i: [start // layer.ratio for start in request_start]
                      for i, layer in enumerate(layers)}  # (2, 20, 24): ratios 2, 1, 1

        def reset():
            pool.backing.copy_(saved)
            for page, value in zip(pages, saved_pages, strict=True):
                page.copy_(value)
            arena.tensor.fill_(0xA5)
            for value in (*outputs, *selected):
                value.fill_(-1)

        try:
            _compare(graphs, reset, lambda: (*outputs, *selected, pool.backing, *pages), selections)
        finally:
            for graph in graphs.values():
                graph.reset()


@pytest.mark.parametrize("lengths", ((BLOCK,), (BLOCK,) * 7 + (0,)), ids=("draft-batch1", "draft-batch8"))
@torch.no_grad()
def test_draft_layers_switch_on_writes_the_same_bytes(weights, lengths):
    _, draft, config = weights
    device = torch.device("cuda:0")
    torch.manual_seed(41153 + len(lengths))
    rows, batch, metadata, _ = _metadata(lengths, device, tokens=len(lengths) * BLOCK)
    pool = ContinuationPool.allocate(config, batch + 1, batch * PAGES + 1, device, dspark=True)
    pool.backing.random_(0, 120)
    scratch = AttentionScratch.allocate(rows, WORLD, device)
    sources = [torch.randn((rows, 5120), dtype=torch.bfloat16, device=device) * .1 for _ in range(3)]
    outputs = [torch.empty((rows, 5120), dtype=torch.bfloat16, device=device) for _ in range(3)]
    arena = ScratchArena(device, overlap=True)
    owners = []
    with RecordedPreparationSession(device=device, checkpoint="ds41-glue-pass2",
            workload_schema=f"glue2-draft-rows{rows}", numerical_policy="checkpoint", compile_workers=16) as session:
        session.register_scratch(arena)
        layers = [PreparedAttention(index, draft, sources[index], scratch, metadata, pool, Rotary(config, device),
                      mode="draft", world=WORLD, session=session, name=f"draft{index}", workspace=arena,
                      mla_compute_mode=MLA, glue_pass2=True) for index in range(3)]
        session.tune(show_progress=False)
        graphs = _capture_arms(session, lambda on: DraftForward([_switched(layer, on) for layer in layers], sources,
            metadata, outputs), owners)
        session.freeze()
        saved = pool.backing.clone()

        def reset():
            pool.backing.copy_(saved)
            arena.tensor.fill_(0xA5)
            for value in outputs:
                value.fill_(-1)

        try:
            _compare(graphs, reset, lambda: (*outputs, pool.backing))
        finally:
            for graph in graphs.values():
                graph.reset()
