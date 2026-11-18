# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 expert banks and host parking on CPU: resident, parked whole, and merged VRAM/host banks.

The CUDA test (test_deepseek41_expert_parking_cuda.py on tp3-main) checks numerics on a real checkpoint; this one
checks the wiring with stubs, without a GPU:

* weights.py ``DeepSeekWeightLoader.load_layer`` (its real ``__init__`` and ``load_experts``; the checkpoint reads and
  ``load_mxfp4_experts`` are stubbed) builds the bank layout TP3Config asks for: the hot set comes from the route counts
  in expert-id space; a merged bank's ``hot_map`` is indexed by expert id and holds slots (``bank[map[e]] == e``);
* the bytes the loader hands the host allocator (``host_allocate`` regions plus ``mixed_allocate`` host tails) equal
  recipe.py ``_parked_bytes``, the number the planner subtracts from the VRAM budget;
* decoder.py ``PreparedLayer._run`` (the functional forward) remaps router ids through ``hot_map`` as a forward-local
  value and the experts receive every route exactly once in the bank slot holding that expert (one-hot stub MoE);
* ``ModelWeights.load`` parks the token embedding and the vision tower through temporary device loads (the persistent
  arena never releases space), and keeps upstream's loads otherwise.

The split hot + cold bank of tp3-main was not ported (TP3Config requires ``merged_bank`` with hot experts). Each
check has a negative control: a merged map stored slot -> id instead of id -> slot, a hot set read in slot space, a
placement the byte count does not cover.
"""

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.deepseek41 import decoder, recipe, weights
from ignition.recipes.deepseek41.tp3_config import TP3Config

WORLD, EXPERTS, TOP_K, HIDDEN, INTERMEDIATE = 3, 384, 6, 5120, 2304
HOT = 64
PARKED = (20, 21)   # parked layers loaded in the loader tests
RESIDENT = 5        # a layer outside host_expert_layers


def _hot_ids():
    """Experts 1, 7, 13, ... (e % 6 == 1): 64 hot ids that are not a prefix of 0..383."""
    return [e for e in range(EXPERTS) if e % 6 == 1]


def _counts(layer):
    """Route counts in expert-id space: hot ids get large, distinct counts; the rest small ones (layer-dependent)."""
    hot = set(_hot_ids())
    return [10_000 + 3 * e + layer if e in hot else (e * 7 + layer) % 97 for e in range(EXPERTS)]


@pytest.fixture
def counts_dir(tmp_path):
    for layer in (*PARKED, RESIDENT):
        (tmp_path / f"expert-routes-rank0-layer{layer}.json").write_text(json.dumps({"counts": _counts(layer)}))
    return tmp_path


# --------------------------------------------------------------------------------------------------------------------
# Loader stubs


class FakeBank:
    """What the stubbed ``load_mxfp4_experts`` returns: the expert ids in slot order and where they were placed."""

    def __init__(self, ids, local, host):
        self.ids, self.local, self.host = tuple(ids), local, host

    def __len__(self):
        return len(self.ids)


class FakeStorage:
    """CoreStorage stand-in: counts the host bytes of host regions and of mixed regions' host tails."""

    def __init__(self):
        self.host_bytes = 0
        self.regions = []

    def host_allocate(self, name, shape, dtype):
        self.host_bytes += math.prod(shape) * dtype.itemsize
        self.regions.append(("host", name, tuple(shape), None))
        return SimpleNamespace(device_view=torch.zeros(shape, dtype=dtype))

    def mixed_allocate(self, name, shape, dtype, *, device_rows):
        assert 0 <= device_rows <= shape[0]
        self.host_bytes += (shape[0] - device_rows) * math.prod(shape[1:]) * dtype.itemsize
        self.regions.append(("mixed", name, tuple(shape), device_rows))
        return SimpleNamespace(device_view=None)


def _loader(monkeypatch, tp3, storage):
    """A real DeepSeekWeightLoader (its __init__ and load_experts run) over stubbed checkpoint reads."""
    text = dict(hidden_size=HIDDEN, moe_intermediate_size=INTERMEDIATE, n_routed_experts=EXPERTS, swiglu_limit=10.,
                kv_source_layer_ids=(), index_source_layer_ids=(), engram_layer_ids=(), compress_ratios={})
    config = SimpleNamespace(text=text, vision=None, draft=SimpleNamespace(experts=EXPERTS))
    monkeypatch.setattr(weights.BlockFP8WeightLoader, "__init__",
                        lambda self, manifest, device, **kwargs: setattr(self, "device", torch.device(device)))
    monkeypatch.setattr(weights, "ModelConfig", SimpleNamespace(load=lambda path, mini: config))
    loader = weights.DeepSeekWeightLoader(SimpleNamespace(metadata_path=None), "cpu",
        shard=TensorParallelShard(0, WORLD), expert_scale_compression="csf",
        host_allocator=storage.host_allocate, tp3=tp3)

    def load_mxfp4_experts(prefixes, *, hidden_size, intermediate_range, host_allocator, **kwargs):
        local = intermediate_range[1] - intermediate_range[0]
        ids = [int(prefix.rsplit(".", 1)[1]) for prefix in prefixes]
        if host_allocator is not None:  # the packed w13 / w2 values move to host memory (scales stay on the GPU)
            host_allocator(f"{prefixes[0]}.w13", (len(ids), 2 * local, hidden_size // 2), torch.uint8)
            host_allocator(f"{prefixes[0]}.w2", (len(ids), hidden_size, local // 2), torch.uint8)
        return FakeBank(ids, local, host_allocator is not None)

    loader.load_mxfp4_experts = load_mxfp4_experts
    for name in ("load_tensor", "load_padded_tensor", "load_linear", "load_linear_group", "load_linear_columns",
                 "load_attention_output"):
        setattr(loader, name, lambda *args, **kwargs: object())
    return loader


# --------------------------------------------------------------------------------------------------------------------
# Bank checks (shared by the loader tests, the layer tests and the negative controls)


def _map(tensor):
    assert tensor is not None and tensor.dtype == torch.int32 and tuple(tensor.shape) == (EXPERTS,)
    return tensor.tolist()


def _assert_bank(experts):
    """Check one ExpertWeights layout; return the expert ids held in VRAM slots."""
    if experts.hot_map is None:  # one bank, router ids used as-is
        assert experts.experts.ids == tuple(range(EXPERTS)) and experts.hot_rows is None
        return set() if experts.experts.host else set(range(EXPERTS))
    # merged: one hot-first bank, VRAM prefix + host tail
    hot_map = _map(experts.hot_map)
    assert experts.experts.host
    assert sorted(hot_map) == list(range(EXPERTS)), "merged hot_map must be a permutation of the slots"
    assert all(experts.experts.ids[hot_map[e]] == e for e in range(EXPERTS)), "hot_map disagrees with slot order"
    rows = experts.hot_rows
    assert type(rows) is int and 0 < rows < EXPERTS
    return set(experts.experts.ids[:rows])


def _assert_hot_set(vram_ids, layer):
    """The VRAM experts are the HOT most-routed ids, counts read in expert-id space."""
    counts = _counts(layer)
    expected = set(sorted(range(EXPERTS), key=lambda e: (-counts[e], e))[:HOT])
    assert vram_ids == expected
    assert sum(counts[e] for e in vram_ids) == sum(sorted(counts, reverse=True)[:HOT])


def _per_expert_bytes(local):
    return 2 * local * HIDDEN // 2 + HIDDEN * local // 2  # MXFP4 w13 + w2 values


CONFIGS = {
    "resident": dict(),
    "park_all": dict(host_expert_layers=frozenset(PARKED)),
    "merged": dict(host_expert_layers=frozenset(PARKED), hot_experts=HOT, merged_bank=True),
    "all_hot": dict(host_expert_layers=frozenset(PARKED), hot_experts=EXPERTS, merged_bank=True),
}


def _load(monkeypatch, counts_dir, name):
    tp3 = TP3Config(**CONFIGS[name], expert_counts_dir=counts_dir)
    storage = FakeStorage()
    loader = _loader(monkeypatch, tp3, storage)
    layers = {layer: loader.load_layer(layer).experts for layer in (*PARKED, RESIDENT)}
    return tp3, storage, layers


# --------------------------------------------------------------------------------------------------------------------
# Loader: layout, maps, hot set, parked bytes


@pytest.mark.parametrize("name", CONFIGS)
def test_loader_builds_the_configured_bank_and_parked_bytes_match(monkeypatch, counts_dir, name):
    tp3, storage, layers = _load(monkeypatch, counts_dir, name)
    local = INTERMEDIATE // WORLD
    host_experts = 0
    for layer, experts in layers.items():
        vram = _assert_bank(experts)
        parked = layer in PARKED and name != "resident"
        if not parked or name == "all_hot":
            assert experts.hot_map is None and vram == set(range(EXPERTS))
            continue
        if name == "park_all":
            assert vram == set() and experts.hot_map is None
            host_experts += EXPERTS
            continue
        _assert_hot_set(vram, layer)
        host_experts += EXPERTS - HOT
        first = f"layers.{layer}.ffn.experts.{experts.experts.ids[0]}.w13"
        region = next(r for r in storage.regions if r[1] == first)
        assert region[0] == "mixed" and region[3] == experts.hot_rows == HOT
    assert all(experts.experts.local == local for experts in layers.values())
    assert storage.host_bytes == host_experts * _per_expert_bytes(local)
    assert storage.host_bytes == recipe._parked_bytes(WORLD, tp3)
    if name in ("resident", "all_hot"):
        assert recipe._parked_bytes(WORLD, tp3) == 0 and not storage.regions


def test_hot_experts_without_a_merged_bank_are_refused(counts_dir):
    with pytest.raises(ValueError, match="hot_experts needs merged_bank"):
        TP3Config(host_expert_layers=frozenset(PARKED), hot_experts=HOT, expert_counts_dir=counts_dir)


def test_merged_bank_needs_core_storage_mixed_allocation(monkeypatch, counts_dir):
    tp3 = TP3Config(**CONFIGS["merged"], expert_counts_dir=counts_dir)
    loader = _loader(monkeypatch, tp3, FakeStorage())
    loader._host_allocator = lambda name, shape, dtype: None  # a plain function: no storage owns it
    with pytest.raises(ValueError, match="CoreStorage.host_allocate"):
        loader.load_layer(PARKED[0])


def test_negative_controls_for_the_bank_checks(monkeypatch, counts_dir):
    tp3, storage, layers = _load(monkeypatch, counts_dir, "merged")
    merged = layers[PARKED[0]]
    with pytest.raises(AssertionError):  # merged map stored slot -> id (the bank order) instead of id -> slot
        _assert_bank(replace(merged, hot_map=torch.tensor(merged.experts.ids, dtype=torch.int32)))
    with pytest.raises(AssertionError):  # hot set read in slot space: slots 0..63 taken for ids 0..63
        _assert_hot_set(set(range(HOT)), PARKED[0])
    with pytest.raises(AssertionError):  # bytes of the hot experts too would be parked
        assert storage.host_bytes + HOT * _per_expert_bytes(INTERMEDIATE // WORLD) == \
            recipe._parked_bytes(WORLD, tp3)


# --------------------------------------------------------------------------------------------------------------------
# PreparedLayer: the merged-bank remap inside the functional forward


class Component:
    """Construction stub with a unique plan; ``functional`` keeps it as is."""

    def __init__(self, *args, **kwargs):
        self.plan = object()
        self.args, self.kwargs = args, kwargs

    def functional(self, **kwargs):
        return self


class OneHotExperts(Component):
    """Stub MoE: for each route with slot >= 0, add its weight to column ``bank.ids[slot]`` (the global expert id)."""

    def __init__(self, inputs, bank, ids, route_weights, output, **kwargs):
        super().__init__(inputs, bank, ids, route_weights, output, **kwargs)
        self.bank, self.example_ids = bank, ids
        self.calls = []

    def __call__(self, source, ids, route_weights, **kwargs):
        self.calls.append(ids)
        output = torch.zeros((source.shape[0], HIDDEN))
        for row, (slots, values) in enumerate(zip(ids.tolist(), route_weights.tolist())):
            for slot, value in zip(slots, values):
                if slot >= 0:
                    assert slot < len(self.bank), "route outside the bank"
                    output[row, self.bank.ids[slot]] += value
        return output


def _layer(monkeypatch, experts, *, rows, mode="decode"):
    made = []

    def routed(*args, **kwargs):
        made.append(OneHotExperts(*args, **kwargs))
        return made[-1]

    for name in ("PreparedMHC", "PreparedAttention", "PreparedChunkedAllReduce", "UnquantizedProjection",
                 "PreparedRouter", "PreparedMLP", "PreparedMHCPost", "ConcurrentSharedExpert"):
        monkeypatch.setattr(decoder, name, Component)
    monkeypatch.setattr(decoder, "PreparedRoutedExperts", routed)
    monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)
    text = dict(rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20,
                scoring_func="sigmoid", norm_topk_prob=True, routed_scaling_factor=1., swiglu_limit=10.)
    scratch = decoder.Scratch.allocate(rows, 1, WORLD, "cpu")
    layer_weights = SimpleNamespace(engram=None, attention=None, experts=experts)
    layer = decoder.PreparedLayer(PARKED[0], layer_weights, None, scratch,
        SimpleNamespace(token_ids=torch.zeros(rows, dtype=torch.int64), positions=torch.zeros(rows, dtype=torch.int64)),
        SimpleNamespace(config=SimpleNamespace(text=text)), None,
        shard=TensorParallelShard(0, WORLD), mode=mode, runtime=None, coordinator=None, session=object(),
        name="layer", workspace=SimpleNamespace(overlap=False), carry=False, defer_post=True)
    assert len(made) == 1 and layer.experts is made[0]
    # The experts are declared on the router-id example (same shape and dtype as the remapped ids).
    assert made[0].example_ids is scratch.router.ids and made[0].bank is experts.experts
    assert layer.hot_map is experts.hot_map
    return layer, made[0]


def _routes(rows, vram, *, seed):
    """Router ids mixing VRAM and host experts in every row; weights are multiples of 1/8 (exact in BF16)."""
    g = torch.Generator().manual_seed(seed)
    vram, host = sorted(vram), sorted(set(range(EXPERTS)) - set(vram))
    ids = torch.empty((rows, TOP_K), dtype=torch.int32)
    for row in range(rows):
        picks = [vram[int(torch.randint(len(vram), (1,), generator=g))],
                 host[int(torch.randint(len(host), (1,), generator=g))]]
        for e in torch.randperm(EXPERTS, generator=g).tolist():
            if len(picks) == TOP_K:
                break
            if e not in picks:
                picks.append(e)
        ids[row] = torch.tensor(picks, dtype=torch.int32)
    route_weights = torch.randint(1, 9, (rows, TOP_K), generator=g).float() / 8
    return ids, route_weights


def _run_layer(layer, ids, route_weights):
    """Drive ``PreparedLayer._run`` with pass-through stubs around the expert section; return the MoE output."""
    rows = ids.shape[0]
    hidden = torch.zeros((rows, HIDDEN))
    layer.attention_hc = lambda *incoming, **kwargs: (incoming[0], None, None, hidden)
    layer.attention = SimpleNamespace(_run=lambda normalized, metadata, selection, **kwargs:
                                      (normalized, None, selection, {}))
    layer.attention_reduce = lambda source: source
    layer.ffn_hc = lambda output, residual, post, comb, **kwargs: (residual, post, comb, output)
    layer.router_projection = lambda source: source
    layer.router = lambda logits, **kwargs: SimpleNamespace(ids=ids, weights=route_weights, selected_logits=None)
    layer.shared = lambda source: torch.zeros_like(source)
    layer.mlp_reduce = lambda source: source
    layer.post = None
    metadata = SimpleNamespace(token_ids=torch.zeros(rows, dtype=torch.int64))
    incoming, _, _, _ = layer((torch.zeros((rows, 4, HIDDEN)),), torch.zeros((rows, 4)), metadata, None,
                              (None, None, None))
    reference = torch.zeros(rows, HIDDEN)
    reference.scatter_add_(1, ids.long(), route_weights)
    return incoming[0], reference


@pytest.mark.parametrize("name", ("resident", "merged"))
@pytest.mark.parametrize("rows", (1, 5, 48, 129))
def test_layer_remaps_router_ids_and_delivers_every_route_once(monkeypatch, counts_dir, name, rows):
    _, _, layers = _load(monkeypatch, counts_dir, name)
    experts = layers[PARKED[0]]
    vram = _assert_bank(experts)
    layer, moe = _layer(monkeypatch, experts, rows=rows)
    ids, route_weights = _routes(rows, vram if 0 < len(vram) < EXPERTS else set(range(HOT)), seed=rows)
    routed, reference = _run_layer(layer, ids, route_weights)
    assert torch.equal(routed, reference)
    (seen,) = moe.calls
    assert seen.dtype == ids.dtype and seen.shape == ids.shape
    assert (seen is ids) == (name == "resident")  # resident banks take the router ids as they are


def test_negative_control_for_the_layer_remap(monkeypatch, counts_dir):
    _, _, layers = _load(monkeypatch, counts_dir, "merged")
    merged = layers[PARKED[0]]
    wrong = replace(merged, hot_map=torch.tensor(merged.experts.ids, dtype=torch.int32))  # slot -> id
    layer, _ = _layer(monkeypatch, wrong, rows=8)
    routed, reference = _run_layer(layer, *_routes(8, _assert_bank(merged), seed=4))
    assert not torch.equal(routed, reference)


# --------------------------------------------------------------------------------------------------------------------
# ModelWeights.load: token embedding and vision tower parking


@pytest.mark.parametrize("embed_host,vision_host", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("allocator", [True, False])
def test_embedding_and_vision_parking_load_temporaries(monkeypatch, embed_host, vision_host, allocator):
    loads = []

    class Loader:
        _sources = {"vision.blocks.0.norm1.weight": None, "aligner.w1.weight": None, "image_start": None}

        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def load_tensor(self, name, *, temporary=False, **kwargs):
            loads.append((name, temporary))
            return torch.full((2, 3), 7, dtype=torch.bfloat16)

        def load_tensors(self, names, *, temporary=False):
            return tuple(self.load_tensor(name, temporary=temporary) for name in names)

        def load_padded_tensor(self, name, **kwargs):
            return self.load_tensor(name)

        def load_layer(self, index):
            return None

    config = SimpleNamespace(layers=1, vision={"present": True})
    monkeypatch.setattr(weights, "DeepSeekWeightLoader", Loader)
    monkeypatch.setattr(weights, "ModelConfig", SimpleNamespace(load=lambda path, mini: config))
    monkeypatch.setattr(weights, "allocated_tensor_bytes", lambda device: 0)
    storage = FakeStorage()
    tp3 = TP3Config(embed_host=embed_host, vision_host=vision_host)
    model = weights.ModelWeights.load(SimpleNamespace(metadata_path=None), "cpu", shard=TensorParallelShard(0, WORLD),
                                      host_allocator=storage.host_allocate if allocator else None, tp3=tp3)
    loaded = dict(loads)
    parked = {name for _, name, _, _ in storage.regions}
    park_embedding, park_vision = embed_host and allocator, vision_host and allocator
    assert loaded["embed.weight"] is park_embedding and loaded["norm.weight"] is False
    assert loaded["head.weight"] is False
    for name in Loader._sources:
        assert loaded[name] is park_vision
    assert parked == ({"embed.weight"} if park_embedding else set()) | (set(Loader._sources) if park_vision else set())
    # Parked tensors are the host regions' views, holding the loaded values.
    assert torch.equal(model.embedding, torch.full((2, 3), 7, dtype=torch.bfloat16))
    assert all(torch.equal(tensor, torch.full((2, 3), 7, dtype=torch.bfloat16)) for tensor in model.vision.values())


# --------------------------------------------------------------------------------------------------------------------
# DSparkWeights.load: Markov-head embedding parking (TP3 tp3.markov_embed_host, ds41-tp3-next 2026-11-17)


@pytest.mark.parametrize("park", [False, True])
def test_markov_embedding_parking_loads_a_temporary_into_a_host_region(park):
    loads, embed = [], "mtp.0.markov_head.embed.weight"

    class Loader:
        config = SimpleNamespace(draft=SimpleNamespace(layers=1))
        shard = TensorParallelShard(0, WORLD)

        def load_tensor(self, name, *, temporary=False, **kwargs):
            loads.append((name, temporary))
            return torch.full((4, 2), 3, dtype=torch.bfloat16)

        def load_padded_tensor(self, name, *, temporary=False, **kwargs):
            return self.load_tensor(name, temporary=temporary)

        def load_layer(self, index, *, draft=False):
            loads.append((f"layer{index}", draft))

        def load_linear(self, name):
            loads.append((name, False))

        def load_linear_group(self, names):
            loads.append((names, False))

    storage = FakeStorage()
    draft = weights.DSparkWeights.load(Loader(), vocabulary_parallel_sampling=True, markov_head_quantization="fp8",
                                       park_markov_embedding=storage.host_allocate if park else None)
    # Same load order as before the switch (layers, main_proj, wkv group, norms, embedding, head).
    assert [name for name, _ in loads] == ["layer0", "mtp.0.main_proj", ("mtp.0.attn.wkv",), "mtp.0.main_norm.weight",
                                           "mtp.0.norm.weight", embed, "mtp.0.markov_head.head.weight"]
    assert dict((n, t) for n, t in loads if isinstance(n, str))[embed] is park
    assert [name for _, name, _, _ in storage.regions] == ([embed] if park else [])
    assert torch.equal(draft.markov_embedding, torch.full((4, 2), 3, dtype=torch.bfloat16))


def test_markov_and_vision_arena_switches_parse_and_default_off():
    assert TP3Config().markov_embed_host is False and TP3Config().vision_arena_host is False
    on = TP3Config.from_environment({"FLARE_DS41_MARKOV_EMBED_HOST": "1", "FLARE_DS41_VISION_ARENA_HOST": "1"})
    assert on.markov_embed_host is True and on.vision_arena_host is True
    record = on.to_metadata()
    assert record["markov_embed_host"] is True and record["vision_arena_host"] is True
    with pytest.raises(TypeError):
        TP3Config(markov_embed_host=1)


# --------------------------------------------------------------------------------------------------------------------
# TP3 tp3.draft_vocab (FR-Spec style draft-head trim, ds41-tp3-next 2026-11-17)


def test_draft_vocab_columns_equal_width_on_every_rank_and_keep_every_listed_token():
    tokens = [0, 5, 7, 43135, 43136, 50000, 86272, 129279, 129000]
    width = 43136
    ranking = [9, 3, 43200, 43137, 129100, 86300] + list(range(129280))  # frequency order, then everything
    per = [weights.draft_vocab_columns(tokens, 3, r, k=600, ranking=ranking) for r in range(3)]
    assert {len(c) for c in per} == {256}  # max(share 4, 600 / 3 = 200) rounded up to 128
    for r, columns in enumerate(per):
        assert columns == sorted(set(columns)) and all(0 <= c < width for c in columns)
        assert all(t - r * width in columns for t in tokens if r * width <= t < (r + 1) * width)
        assert all(r * width + c < 129280 for c in columns)  # never a padding row
    assert TP3Config().draft_vocab is None
    assert "draft_vocab" not in TP3Config().to_metadata()  # recorded only when set: default records unchanged


def test_draft_vocab_fill_follows_the_ranking_not_the_id_order():
    """Review H1 (2026-11-17): the fill took each rank's lowest unused ids, so ranks 1-2 got ~8K arbitrary rare tokens
    and common rank-0 tokens just above the fill ("So", " agent") were dropped. The fill now takes the ranking."""
    width = 43136
    tokens = [1, 2, 43136 + 7]
    ranking = [43136 + 40000, 30000, 20000] + [43136 + 500 + i for i in range(200)] + list(range(1000, 1300))
    rank0 = weights.draft_vocab_columns(tokens, 3, 0, k=3 * 128, ranking=ranking)
    assert len(rank0) == 128 and {1, 2, 30000, 20000} <= set(rank0)
    assert set(rank0) - {1, 2, 30000, 20000} == set(range(1000, 1124))  # in ranking order, not ids 0, 3, 4, ...
    rank1 = weights.draft_vocab_columns(tokens, 3, 1, k=3 * 128, ranking=ranking)
    assert rank1 == sorted([7, 40000] + [500 + i for i in range(126)])  # 43136 + 40000 ranked first on rank 1
    with pytest.raises(ValueError, match="rebuild the table"):  # no ranking: refused instead of an id-order fill
        weights.draft_vocab_columns(tokens, 3, 0, k=3 * 128)
    exact = list(range(128)) + list(range(width, width + 128)) + list(range(2 * width, 2 * width + 128))
    assert weights.draft_vocab_columns(exact, 3, 2, k=3 * 128) == list(range(128))  # exact width: no fill needed


def _draft_vocab_tool():
    import importlib.util
    path = Path(__file__).resolve().parents[2] / "tp3-tools" / "ds41_draft_vocab.py"
    spec = importlib.util.spec_from_file_location("ds41_draft_vocab", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_draft_vocab_builder_keeps_the_most_frequent_tokens_per_rank_and_measures_held_out_misses():
    tool = _draft_vocab_tool()
    width = tool.shard_width()
    assert width == 43136 and tool.columns_for(32768) == 11008 and tool.columns_for(24576) == 8192
    assert tool.columns_for(10 ** 6) == width and all(tool.columns_for(k) % 128 == 0 for k in (1, 24576, 40960, 65536))
    records = {
        "code": {"train": {5: 50, 6: 30, width + 1: 10, 2 * width + 3: 5}, "heldout": {5: 8, 6: 1, 7: 1}},
        "prose": {"train": {9: 40, 8: 20, width + 2: 20}, "heldout": {9: 5, width + 9: 5}},
    }
    order = tool.ranking(records, {"code": 0.5, "prose": 0.5})
    assert order[:3] == [5, 9, 6] and len(order) == 129280 and len(set(order)) == 129280
    specials = [0, 1, 2, 128000, 129279]
    keep = tool.select(order, specials, 128)
    per_rank = [sum(r * width <= t < (r + 1) * width for t in keep) for r in range(3)]
    assert per_rank == [128, 128, 128] and set(specials) <= set(keep)
    assert {5, 6, 8, 9, width + 1, width + 2, 2 * width + 3} <= set(keep)
    miss = tool.miss_rates(records, keep)
    assert miss["code"] == 0.0  # 7 is unseen but inside the id-order tail of rank 0's 128 columns
    assert miss["prose"] == 0.0  # width + 9: rank 1 tail in id order
    tight = tool.miss_rates(records, tool.select(order, specials, 128)[:3])
    assert tight["code"] == 1.0
    assert tool.heldout("doc-a", 0.2) == tool.heldout("doc-a", 0.2)  # deterministic per document group
    assert 0.15 < sum(tool.heldout(f"d{i}", 0.2) for i in range(4000)) / 4000 < 0.25


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_draft_vocab_fill_is_exact_zero_probability_under_blockwise_softmax(dtype):
    from ignition.recipes.deepseek41.dspark import DRAFT_FILL
    torch.manual_seed(0)
    width, keep = 1024, torch.tensor(sorted(torch.randperm(1024)[:256].tolist()))
    sub = torch.randn(3, 256).to(dtype)
    full = torch.full((3, width), DRAFT_FILL, dtype=dtype).index_copy(1, keep, sub).float()
    bias = torch.randn(3, width)  # Markov transition bias added on top
    logits = full + bias
    blocks = logits.view(3, 4, 256)  # block maxima / sums as the fused Markov kernel combines them
    m = blocks.amax(-1)
    s = torch.exp(blocks - m[..., None]).sum(-1)
    lse = m.amax(-1) + torch.log((s * torch.exp(m - m.amax(-1, keepdim=True))).sum(-1))
    p = torch.exp(logits - lse[:, None])
    assert torch.isfinite(p).all() and torch.all(p[:, [c for c in range(width) if c not in set(keep.tolist())]] == 0)
    reference = torch.softmax(sub.float() + bias[:, keep], -1)
    assert torch.allclose(p[:, keep], reference, atol=1e-6)


def test_ds41_tp3_next_switches_are_absent_from_a_default_record():
    record = TP3Config().to_metadata()
    for name in ("mma_pad_once", "vision_arena_host", "markov_embed_host", "draft_vocab"):
        assert name not in record
    assert TP3Config(mma_pad_once=True).to_metadata()["mma_pad_once"] is True
