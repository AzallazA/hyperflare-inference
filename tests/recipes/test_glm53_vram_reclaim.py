# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fork-only GLM export-layout VRAM switches (2026-11-15, VRAM map T1/W1, T2, W2, T3).

    FLARE_GLM53_TRIAL_POOL=separate  trial ("Measure graph memory") graphs in their own pool, released before
                                     the serving capture (package exports only)
    FLARE_GLM53_SLAB_SEGMENT=1       the preparation-only 1M-position prefill slab in whole allocator segments
    FLARE_GLM53_ONESHOT_FIT=1        two/four ranks: one-shot all-reduce sized for the rows it carries (512)
    FLARE_GLM53_AR_ROWS=N            three ranks: one-shot channel sized for N rows, prefill reduced in parts

CPU checks cover parsing, plumbing into PreparationConfig, the channel sizes, the loud failure for an oversize
reduction, the slab's storage, and that the unset switches keep the upstream sizes. CUDA checks
(FLARE_RUN_CUDA_TESTS=1, one GPU, no checkpoint) show the allocator effects the switches rely on:

    FLARE_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q -p no:cacheprovider \\
        tests/recipes/test_glm53_vram_reclaim.py
"""

import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.prompt_logprobs import prefill_modes, scored_prefill_capacities
from ignition.recipes.glm53 import bootstrap
from ignition.recipes.glm53 import recipe as glm_recipe
from ignition.recipes.glm53.bootstrap import PreparationConfig, one_shot_tokens, trial_graph_pool
from ignition.recipes.glm53.mla import _SEGMENT_BYTES, MLAScratch

_CUDA = os.environ.get("FLARE_RUN_CUDA_TESTS") == "1"
_SWITCHES = ("TRIAL_POOL", "SLAB_SEGMENT", "ONESHOT_FIT", "AR_ROWS")
_OTHERS = ("EXPERT_ACTIVATION", "EXPAND_SCALES", "EXPERT_COLUMNS", "KDA_SIDE_CHAIN", "DRAFT_HEAD", "DRAFT_EH_PROJ",
           "EMBED_HOST", "VISION_STAGE", "DCP3")
# One-shot channel bytes per row: 4096 BF16 hidden values.
_ROW = 4096 * 2


def _config(world, *, concurrency=None, **options):
    """The shipped export's settings: TP3 c32, TP2 c8, DCP, MTP-only k5 (widths 1-6, token capacity 128)."""
    concurrency = concurrency or {2: 8, 3: 32}[world]
    options.setdefault("mtp_only", True)
    return PreparationConfig(devices=tuple(range(world)), max_active=concurrency, mtp=True,
                             context=1048576, prefill_chunk_size=4096, speculative_widths=(1, 2, 3, 4, 5, 6),
                             speculative_token_capacity=128, context_parallel=True, **options)


def _plan(world):
    """The plan fields recipe.prepare reads (as tests/recipes/test_glm53_vram_options.py)."""
    return SimpleNamespace(
        speculation=SimpleNamespace(widths=(1, 2, 3, 4, 5, 6), fixed_width=None, mtp_only=False),
        speculative_token_capacity=None, collective_tuning="static", collective_ladder=None, prompt_logprobs=False,
        compile_workers=None, autotune_effort="default", tuning_replica=None, native_dense_activations=False,
        lock_gpu_clocks=False, world_size=world, gpus=(SimpleNamespace(capability=(12, 0)),) * world,
        image_concurrency=None, language_model_only=True, concurrency=8, roce_ring=False, roce_chain=False,
        context_parallel=world != 3, stream_overlap=True, recipe=SimpleNamespace(context=1048576), prefill_chunk=4096,
        minimum_pages=None, snapshots=1, costs=SimpleNamespace(state_row_bytes=1 << 20), headroom_bytes=0,
        reserve_bytes=0, checkpoint=None, compile_headroom_bytes=8 << 30)


def _prepared_config(monkeypatch, world, environment):
    import ignition.preparation.rank_group as rank_group

    for name in (*_SWITCHES, *_OTHERS):
        monkeypatch.delenv(f"FLARE_GLM53_{name}", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(f"FLARE_GLM53_{name}", value)
    seen = {}
    monkeypatch.setattr(rank_group, "prepare_group",
                        lambda snapshot, config, function, **kwargs: seen.setdefault("config", config))
    glm_recipe.prepare("/nonexistent", _plan(world), compile_forwards=False, kernel_cache=None, progress=None)
    return seen["config"]


@pytest.mark.parametrize("world", [2, 3])
def test_unset_switches_keep_the_upstream_layout(monkeypatch, world):
    config = _prepared_config(monkeypatch, world, {})
    assert (config.trial_pool, config.slab_segment, config.oneshot_fit, config.ar_rows) == ("shared", False, False, 0)
    assert config.oneshot_rows == config.token_capacity
    # Upstream: two ranks size the one-shot channel for token_capacity, three for at least a whole chunk.
    assert one_shot_tokens(config, prefill_ring=True) == config.token_capacity
    assert one_shot_tokens(config, prefill_ring=False) == max(config.token_capacity, config.prefill_chunk_size)


@pytest.mark.parametrize("world, environment, expected", [
    (2, {"TRIAL_POOL": "separate"}, ("separate", False, False, 0)),
    (3, {"TRIAL_POOL": "Separate"}, ("separate", False, False, 0)),
    (3, {"SLAB_SEGMENT": "1"}, ("shared", True, False, 0)),
    (2, {"ONESHOT_FIT": "1"}, ("shared", False, True, 0)),
    (3, {"AR_ROWS": "1024"}, ("shared", False, False, 1024)),
    (2, {"TRIAL_POOL": "separate", "SLAB_SEGMENT": "1", "ONESHOT_FIT": "1"}, ("separate", True, True, 0)),
    (3, {"TRIAL_POOL": "separate", "SLAB_SEGMENT": "1"}, ("separate", True, False, 0)),
])
def test_export_environment_reaches_the_preparation_config(monkeypatch, world, environment, expected):
    config = _prepared_config(monkeypatch, world, environment)
    assert (config.trial_pool, config.slab_segment, config.oneshot_fit, config.ar_rows) == expected
    # Nothing else the export prepares changes.
    assert (config.kda_side_chain, config.draft_head, config.draft_eh_proj, config.expert_columns, config.embed_host,
            config.vision_stage) == (False, "target", "bf16", None, False, False)


@pytest.mark.parametrize("name, value", [("TRIAL_POOL", "own"), ("SLAB_SEGMENT", "yes"), ("ONESHOT_FIT", "2"),
                                         ("AR_ROWS", "1000"), ("AR_ROWS", "64"), ("AR_ROWS", "8192"),
                                         ("AR_ROWS", "1k")])
def test_invalid_switch_values_fail_the_export(monkeypatch, name, value):
    with pytest.raises(ValueError, match=name):
        _prepared_config(monkeypatch, 3, {name: value})


def test_switches_refuse_layouts_they_do_not_apply_to():
    with pytest.raises(ValueError, match="copy-engine prefill ring"):
        _config(3, oneshot_fit=True)
    with pytest.raises(ValueError, match="three ranks"):
        _config(2, ar_rows=1024)
    with pytest.raises(ValueError, match="RoCE"):
        PreparationConfig(devices=(0, 0), hosts=("a", "b"), oneshot_fit=True)
    with pytest.raises(ValueError, match="trial graph pool"):
        _config(2, trial_pool="none")
    with pytest.raises(TypeError, match="slab_segment"):
        _config(2, slab_segment=1)
    with pytest.raises(ValueError, match="part rows"):
        _config(3, ar_rows=100)


def test_two_ranks_fit_the_one_shot_channel_to_512_rows():
    """W2: 4096 -> 512 rows, 32 -> 4 MiB per call (b12x keeps 2 eager buffers x shards of that size)."""
    upstream, fit = _config(2), _config(2, oneshot_fit=True)
    assert upstream.decode_rows == 48 and fit.decode_rows == 48
    assert one_shot_tokens(upstream, prefill_ring=True) == 4096
    assert one_shot_tokens(fit, prefill_ring=True) == fit.oneshot_rows == 512
    assert one_shot_tokens(fit, prefill_ring=True) * _ROW == 4 << 20
    # Prompt scoring adds scored-prefill graphs (128, 512, 2048, 4096): still 512 rows on the one-shot channel.
    assert _config(2, oneshot_fit=True, prompt_logprobs=True).oneshot_rows == 512
    # A chunk smaller than the ring's threshold keeps the whole chunk on the one-shot channel.
    small = PreparationConfig(devices=(0, 1), max_active=8, prefill_chunk_size=512, context=1048576, oneshot_fit=True)
    assert small.oneshot_rows == 512 == small.token_capacity


def test_three_ranks_size_the_channel_for_ar_rows():
    """T3: 4096 -> 1024 rows (32 -> 8 MiB per call); decode rounds (128 rows) always fit."""
    assert one_shot_tokens(_config(3), prefill_ring=False) == 4096
    parts = _config(3, ar_rows=1024)
    assert parts.decode_rows == 128
    assert one_shot_tokens(parts, prefill_ring=False) == 1024
    assert one_shot_tokens(_config(3, ar_rows=128), prefill_ring=False) == 128


def _routes(config):
    """(graph, rows, one_shot) for every prefill-type graph of the catalog, routed as prepare_rank does."""
    scoring = object() if config.prompt_logprobs else None
    for enabled in (True, False) if not config.mtp_only else (True,):
        decodes = () if enabled else config.batch_capacities
        for kind, mode, capacities in prefill_modes(config.prefill_capacities, decodes, scoring):
            for capacity in capacities:
                yield f"{kind}{capacity}", capacity, not (mode == "prefill"
                                                          and capacity >= bootstrap._DMA_PREFILL_TOKENS)


@pytest.mark.parametrize("options", [{}, {"oneshot_fit": True}, {"oneshot_fit": True, "prompt_logprobs": True},
                                     {"oneshot_fit": True, "mtp_only": False}])
def test_every_catalog_graph_fits_the_fitted_two_rank_channel(options):
    config = _config(2, **options)
    for name, rows, one_shot in _routes(config):
        assert bootstrap._oneshot_parts(config, name, rows, one_shot=one_shot) is False, name
    assert scored_prefill_capacities(4096) == (128, 512, 2048, 4096)


def test_an_oversize_one_shot_reduction_fails_loudly():
    fit = _config(2, oneshot_fit=True)
    assert bootstrap._oneshot_parts(fit, "prefill512", 512, one_shot=True) is False
    # The ring carries 1024+ rows, so the same rows on the ring are fine.
    assert bootstrap._oneshot_parts(fit, "prefill1024", 1024, one_shot=False) is False
    with pytest.raises(ValueError, match="prefill1024.*1024 rows.*sized for 512.*FLARE_GLM53_ONESHOT_FIT"):
        bootstrap._oneshot_parts(fit, "prefill1024", 1024, one_shot=True)


def test_three_rank_prefill_reductions_run_in_parts_only_with_ar_rows():
    upstream, parts = _config(3), _config(3, ar_rows=1024)
    for name, rows, _ in _routes(upstream):
        # Three ranks have no ring: every graph reduces on the one-shot channel.
        assert bootstrap._oneshot_parts(upstream, name, rows, one_shot=True) is False
        assert bootstrap._oneshot_parts(parts, name, rows, one_shot=True) is (rows > 1024)


def test_chunked_reduction_is_selected_only_when_asked(monkeypatch):
    """The layer's default reduction stays PreparedAllReduce (no new graph code at the shipped settings)."""
    import ignition.recipes.glm53.decoder as decoder

    source = __import__("inspect").getsource(decoder.PreparedGlmLayer.__init__)
    assert "reduce = PreparedChunkedAllReduce if chunked_reduce else PreparedAllReduce" in source
    assert "chunked_reduce=False" in source


@pytest.mark.parametrize("own_segment", [False, True])
def test_prefill_slab_storage(own_segment):
    """T2: same [positions, 528] slab at offset 0; with own_segment its storage is whole 2 MiB granules."""
    blocks = -(-1048576 // 768)  # three-rank DCP pages of 768 positions
    scratch = MLAScratch.allocate(4, 24, blocks, "prefill", torch.device("cpu"), context_ranks=3,
                                  own_segment=own_segment)
    slab = scratch.slab
    assert slab.shape == (1049088, 528) and slab.dtype == torch.uint8 and slab.is_contiguous()
    assert slab.storage_offset() == 0
    storage = slab.untyped_storage().nbytes()
    if own_segment:
        assert storage % _SEGMENT_BYTES == 0 and storage - slab.nbytes < _SEGMENT_BYTES
        assert storage == 530 << 20  # 528.26 MiB of slab, 1.74 MiB of padding nobody else can take
    else:
        assert storage == slab.nbytes == 553918464
    # The views the prefill path takes (pull_context int32 rows, sparse MLA pages) are unchanged.
    assert slab.view(torch.int32).shape == (1049088, 132)
    assert slab.view(-1, 64, 528).shape == (16392, 64, 528)
    # rows() hands every graph the same slab.
    assert scratch.rows(2, "prefill").slab is slab


def test_two_rank_slab_is_already_whole_granules():
    """TP2 (DCP2, 1,048,576 positions = 528 MiB) has no tail: the switch changes nothing there."""
    scratch = MLAScratch.allocate(4, 32, 2048, "prefill", torch.device("cpu"), context_ranks=2, own_segment=True)
    assert scratch.slab.nbytes == scratch.slab.untyped_storage().nbytes() == 528 << 20


def test_forward_slab_never_takes_a_padded_storage():
    with pytest.raises(ValueError, match="persistent scratch only"):
        MLAScratch.allocate(4, 24, 1366, "prefill", torch.device("cpu"), context_ranks=3, forward=True,
                            own_segment=True)
    # The forward's own slab stays a whole local tensor (Inductor dtype-view lesson).
    forward = MLAScratch.allocate(4, 24, 1366, "prefill", torch.device("cpu"), context_ranks=3, forward=True)
    assert forward.slab.untyped_storage().nbytes() == forward.slab.nbytes


def test_decoder_scratch_plumbs_the_slab_switch():
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.glm53.stack import DecoderScratch

    for switch, storage in ((False, 553918464), (True, 530 << 20)):
        scratch = DecoderScratch.allocate(4, 1366, "prefill", torch.device("cpu"), shard=TensorParallelShard(0, 3),
                                          sample_rows=1, context_ranks=3, slab_segment=switch)
        assert scratch.mla.slab.untyped_storage().nbytes() == storage
        # Decode scratch has no slab either way.
        decode = DecoderScratch.allocate(4, 1366, "decode", torch.device("cpu"), shard=TensorParallelShard(0, 3),
                                         sample_rows=4, context_ranks=3, slab_segment=switch)
        assert decode.mla.slab.numel() == 0


class _Storage:
    def __init__(self):
        self.pools = []

    def trial_graph_pool(self):
        self.pools.append(object())
        return self.pools[-1]


def test_trial_pool_is_separate_only_for_package_exports():
    storage = _Storage()
    assert trial_graph_pool(_config(3), storage) is None and not storage.pools
    assert trial_graph_pool(_config(3, trial_pool="separate"), None) is None
    pool = trial_graph_pool(_config(3, trial_pool="separate"), storage)
    assert pool is storage.pools[0]


def test_core_storage_registers_the_trial_pool(monkeypatch):
    from ignition.preparation.core_storage import CoreStorage

    created = []

    class Pool:
        def __init__(self, allocator):
            self.allocator, self.id = allocator, (7, len(created))
            created.append(self)

    monkeypatch.setattr(torch.cuda, "MemPool", Pool)
    storage = SimpleNamespace(_allocator=SimpleNamespace(allocator=lambda: "core"), _preparation_pools={(1, 0)})
    pool = CoreStorage.trial_graph_pool(storage)
    assert pool.allocator == "core" and pool.id in storage._preparation_pools
    assert CoreStorage.trial_graph_pool(storage).id != pool.id


# ---- CUDA: the allocator behaviour the switches rely on ----------------------------------------------------------------

def _pool_segments(pool):
    return [s for s in torch.cuda.memory_snapshot(include_traces=False) if tuple(s["segment_pool_id"]) == tuple(pool.id)]


@pytest.mark.skipif(not _CUDA, reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")
@pytest.mark.parametrize("own_segment", [False, True])
def test_slab_segment_leaves_no_free_tail_for_later_tensors(own_segment):
    """Without own_segment a later 1.5 MiB tensor lands in the slab's segment; with it, in a segment of its own."""
    device = torch.device("cuda", torch.cuda.current_device())
    pool = torch.cuda.MemPool()
    with torch.cuda.use_mem_pool(pool, device=device):
        scratch = MLAScratch.allocate(4, 24, 1366, "prefill", device, context_ranks=3, own_segment=own_segment)
        later = torch.empty(3 << 19, dtype=torch.uint8, device=device)  # 1.5 MiB: fits the 1.74 MiB tail
    slab_address = scratch.slab.untyped_storage().data_ptr()
    segment = next(s for s in _pool_segments(pool) if s["address"] == slab_address)
    assert segment["total_size"] == 530 << 20
    shares = segment["address"] <= later.data_ptr() < segment["address"] + segment["total_size"]
    assert shares is (not own_segment)
    del scratch, later


@pytest.mark.skipif(not _CUDA, reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")
@pytest.mark.parametrize("separate", [False, True])
def test_serving_capture_reuses_trial_segments_only_from_a_shared_pool(separate):
    """T1's mechanism: a reset trial graph's segments stay in a shared pool and the serving capture is carved out of
    them; a separate trial pool returns them, and the serving pool holds only what serving graphs allocate."""
    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.Stream(device=device)
    serving = torch.cuda.MemPool()
    trial = torch.cuda.MemPool() if separate else serving
    source = torch.ones(1 << 20, device=device)

    def capture(pool, megabytes):
        graph = torch.cuda.CUDAGraph()
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.graph(graph, pool=pool.id, stream=stream):
            scratch = torch.empty((megabytes << 20) // 4, device=device)
            scratch[: source.numel()].copy_(source)
            out = scratch[: source.numel()].sum()
        return graph, out

    trial_graph, _ = capture(trial, 96)  # the eager trial is larger than the serving plan
    trial_graph.reset()
    del trial_graph
    if separate:
        del trial
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    graph, out = capture(serving, 8)
    graph.replay()
    torch.cuda.synchronize(device)
    assert out.item() == source.numel()
    reserved = sum(s["total_size"] for s in _pool_segments(serving))
    if separate:
        assert reserved < 96 << 20
    else:
        assert reserved >= 96 << 20
