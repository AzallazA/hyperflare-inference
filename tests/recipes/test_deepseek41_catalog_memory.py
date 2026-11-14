# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 catalog memory (G1 2026-11-14): shared vision declaration scratch and the parked-copy load pool."""

from contextlib import contextmanager, nullcontext
from dataclasses import fields, is_dataclass

import torch

from ignition.components.deepseek_vision import VisionScratch
from ignition.preparation.persistent import _ACTIVE_STORAGE
from ignition.recipes.deepseek41.contract import PATCH_CAPACITIES
from ignition.recipes.deepseek41.weights import _parking_pool


def _leaves(value):
    if isinstance(value, torch.Tensor):
        return [value]
    return [leaf for item in fields(value) for leaf in _leaves(getattr(value, item.name))]


def test_vision_capacities_are_row_prefixes_of_the_largest_scratch():
    largest = VisionScratch.allocate(max(PATCH_CAPACITIES), "meta", hidden=5120)
    for capacity in PATCH_CAPACITIES:
        shared = largest.rows(capacity)
        expected = VisionScratch.allocate(capacity, "meta", hidden=5120)
        assert is_dataclass(shared) and type(shared) is type(expected)
        for view, own, base in zip(_leaves(shared), _leaves(expected), _leaves(largest), strict=True):
            # Same geometry the per-capacity allocation had, carved from the largest scratch.
            assert (view.shape, view.dtype, view.stride()) == (own.shape, own.dtype, own.stride())
            assert view.is_contiguous() and (view is base or view._base is base)


def test_parked_copies_load_in_a_pool_of_their_own():
    entered = []

    class Storage:
        @contextmanager
        def transient(self):
            entered.append(True)
            yield

    assert isinstance(_parking_pool(), nullcontext)
    token = _ACTIVE_STORAGE.set(Storage())
    try:
        with _parking_pool():
            pass
    finally:
        _ACTIVE_STORAGE.reset(token)
    assert entered == [True]


# --------------------------------------------------------------------------------------------------------------------
# S6/S7 preparation memory at full scale (2026-11-14 audit after G1): shapes of ds41-1017 (checkpoint headers).

from types import SimpleNamespace

import pytest

from ignition.components import deepseek_vision as vision

VISION = {"norm1.weight": (1024,), "attn.wqkv.weight": (3072, 1024), "attn.wqkv.bias": (3072,),
          "attn.wo.weight": (1024, 1024), "attn.wo.bias": (1024,), "norm2.weight": (1024,),
          "mlp.w1.weight": (5632, 1024), "mlp.w2.weight": (1024, 2816)}
RESIDENT = {"vision.patch_embed.proj.weight": (1024, 588), "vision.patch_embed.proj.bias": (1024,),
            "vision.norm.weight": (1024,)}
ALIGNER = {"aligner.w1.weight": (5120, 9216), "aligner.w1.bias": (5120,), "aligner.w2.weight": (5120, 5120),
           "aligner.w2.bias": (5120,)}
# Per set: the VisionScratch fields its kernels read or write (examples must not overlap them).
OPERANDS = {"block": ("residual", "normalized", "qkv", "query", "key", "value", "attended", "projected", "mlp"),
            "resident": ("patches", "residual", "normalized"),
            "aligner": ("merged", "aligned", "activated", "output")}


def _meta(shapes):
    return {name: torch.empty(shape, dtype=torch.bfloat16, device="meta") for name, shape in shapes.items()}


def _span(view):
    base = view if view._base is None else view._base
    start = view.storage_offset() * view.element_size() - base.storage_offset() * base.element_size()
    return base, start, start + view.nbytes


def test_vision_staging_examples_cost_no_fresh_preparation_memory_at_full_scale():
    scratch = VisionScratch.allocate(max(PATCH_CAPACITIES), "meta", hidden=5120)
    sets = vision.staging_example_donors(scratch, _meta(VISION), _meta(RESIDENT), _meta(ALIGNER))
    total_before = sum(t.nbytes for roles, _ in sets for t in roles.values())
    assert total_before == 173_731_840  # 165.7 MiB the S7 commit allocated afresh (block 24.5, resident 1.2, aligner 140.0)
    fresh = 0
    for (roles, donors), label in zip(sets, ("block", "resident", "aligner")):
        views, extra = vision.carve_examples(roles, donors)
        fresh += extra
        assert set(views) == set(roles)
        forbidden = [leaf for name in OPERANDS[label] for leaf in _leaves(getattr(scratch, name))]
        spans = [_span(view) for view in views.values()]
        for base, start, end in spans:
            assert all(base is not (leaf if leaf._base is None else leaf._base) for leaf in forbidden), label
            assert start % 256 == 0 and end <= base.nbytes
        for i, (a, s0, e0) in enumerate(spans):  # a set's examples never overlap each other
            assert all(a is not b or e0 <= s1 or e1 <= s0 for b, s1, e1 in spans[i + 1:])
    assert fresh == 0  # after the audit: every example is a view of the vision declaration scratch


def test_vision_staging_examples_fall_back_to_own_memory_when_the_scratch_is_too_small():
    scratch = VisionScratch.allocate(min(PATCH_CAPACITIES), "meta", hidden=5120)  # negative control
    (_, _), (_, _), (aligner, donors) = vision.staging_example_donors(scratch, _meta(VISION), _meta(RESIDENT),
                                                                      _meta(ALIGNER))
    views, fresh = vision.carve_examples(aligner, donors)
    assert fresh > 0 and all(views[name].shape == aligner[name].shape for name in aligner)


# BRRR: what Engine.register keeps per layer beyond Flare's and b12x's own tensors (CSF descriptors aside).
CHECKPOINT = {"attn.wo_a.scale": ((256, 128), torch.uint8), "attn.wo_b.scale": ((160, 256), torch.uint8),
              "ffn.gate.bias": ((384,), torch.float32), "ffn.shared_experts.w1.scale": ((72, 160), torch.uint8),
              "ffn.shared_experts.w3.scale": ((72, 160), torch.uint8),
              "ffn.shared_experts.w2.scale": ((160, 72), torch.uint8), "hc_ffn_fn": ((24, 20480), torch.float32),
              "hc_ffn_scale": ((3,), torch.float32), "hc_ffn_base": ((24,), torch.float32),
              "ffn_norm.weight": ((5120,), torch.bfloat16)}


@pytest.mark.parametrize("owned", [True, False])
def test_brrr_registration_bytes_per_rank_at_full_scale(monkeypatch, owned):
    from ignition.recipes.deepseek41 import brrr
    from ignition.recipes.deepseek41.tp3_config import TP3Config

    def tensor(name):  # a distinct value per checkpoint name: a name swapped in register() reads other bytes
        key = name.split(".", 2)[2]
        shape, dtype = CHECKPOINT[key]
        return torch.full(shape, list(CHECKPOINT).index(key) + 1, dtype=dtype)

    class Checkpoint:
        def shape(self, name):
            return CHECKPOINT[name.split(".", 2)[2]][0]

        def get(self, name, rows=slice(None), cols=slice(None)):
            value = tensor(name)
            return (value[rows, cols] if value.ndim == 2 else value[rows]).contiguous()

    class Ext:
        INTER, OA, EXPERTS, SH2_LD = 768, 3072, 384, 24

        def add_layer(self, handle, **kwargs):
            return 0

    kept = []
    monkeypatch.setattr(brrr, "persistent_copy", lambda t: kept.append(t.nbytes) or t.clone())
    monkeypatch.setattr(brrr, "_rows2d", lambda weight: weight)
    rep = SimpleNamespace(n64_repack=False, intermediate_size=768, num_experts=384, w13_layout="w31",
                          w13_rp=torch.zeros(4), w13_sfb=torch.zeros(4), w2_rp=torch.zeros(4), w2_sfb=torch.zeros(4))
    impl = SimpleNamespace(representation_for=lambda mode: rep, nvfp4_csf=None)
    flare = {name: tensor("layers.0." + name) * (1 if owned else 2) for name in
             ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm.weight", "ffn.gate.bias")}
    dense = torch.zeros(4, 32, dtype=torch.uint8)
    engine = object.__new__(brrr.Engine)
    engine.ext, engine.rank, engine.device = Ext(), 1, torch.device("cpu")
    engine.tp3 = TP3Config(brrr_layers=frozenset(range(40)), brrr_ckpt="/c")
    engine.local_handle, engine.slots, engine.keep, engine.checkpoint = 0, {}, [], Checkpoint()
    for layer in range(40):
        weights = SimpleNamespace(
            experts=SimpleNamespace(experts=SimpleNamespace(_impl=impl), hot_map=None, hot_rows=None, router=dense,
                                    correction=flare["ffn.gate.bias"], shared=SimpleNamespace(gate_up=dense, down=dense)),
            attention={"output": SimpleNamespace(wo_a=dense, wo_b=dense)},
            residual={name: flare[name] for name in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm.weight")})
        engine.register(layer, weights)
        kwargs = engine.keep[-1]
        for arg, name in (("fn", "hc_ffn_fn"), ("hc_scale", "hc_ffn_scale"), ("hc_base", "hc_ffn_base"),
                          ("norm", "ffn_norm.weight"), ("gate_b", "ffn.gate.bias")):
            assert (kwargs[arg] is flare[name]) == owned  # in place exactly when Flare's bytes are the checkpoint's
            assert torch.equal(kwargs[arg], tensor(f"layers.{layer}.{name}"))  # each argument holds its own name
    scales = "copied=160 [sh13_s=40 sh2_s=40 woa_s=40 wob_s=40]"
    if owned:  # after the audit: only the four block-scale copies (39,168 B/layer, 1.49 MiB for BRRR-40)
        assert sum(kept) == 40 * 39_168
        assert brrr.placement_line(engine) == (
            "[brrr] placement rank 1: layers=40 in_place=200 [fn=40 gate_b=40 hc_base=40 hc_scale=40 norm=40] "
            f"{scales} copied_bytes={40 * 39_168}")
    else:  # tp3-main / S6 behaviour (negative control: Flare's bytes differ): 2,017,132 B/layer, 76.95 MiB
        assert brrr.placement_line(engine) == (
            "[brrr] placement rank 1: layers=40 in_place=0 [-] copied=360 [fn=40 gate_b=40 hc_base=40 hc_scale=40 "
            f"norm=40 sh13_s=40 sh2_s=40 woa_s=40 wob_s=40] copied_bytes={40 * 2_017_132}")
        assert sum(kept) == 40 * 2_017_132


def test_brrr_in_place_tensors_must_live_in_the_persistent_arena(monkeypatch):
    from ignition.recipes.deepseek41 import brrr

    arena = SimpleNamespace(closed=False, used=4096, region=SimpleNamespace(address=1 << 40))
    monkeypatch.setattr(brrr, "active_core_storage", lambda: SimpleNamespace(_persistent_arena=arena))

    def fake(address, nbytes, kind="cuda"):
        return SimpleNamespace(device=SimpleNamespace(type=kind), data_ptr=lambda: address, nbytes=nbytes)
    brrr._require_persistent(fake((1 << 40) + 256, 1024), "fn")  # inside the used arena prefix
    for outside in (fake((1 << 40) + 3584, 1024), fake(5 << 20, 64)):  # past the used prefix / transient memory
        with pytest.raises(RuntimeError, match="not in the persistent arena"):
            brrr._require_persistent(outside, "fn")
    brrr._require_persistent(fake(5 << 20, 64, kind="cpu"), "fn")  # host tensors (unit tests): not checked
    monkeypatch.setattr(brrr, "active_core_storage", lambda: SimpleNamespace(_persistent_arena=None))
    with pytest.raises(RuntimeError, match="not in the persistent arena"):
        brrr._require_persistent(fake((1 << 40) + 256, 1024), "fn")
    monkeypatch.setattr(brrr, "active_core_storage", lambda: None)
    brrr._require_persistent(fake(5 << 20, 64), "fn")  # no core storage (eager runs): not checked


def test_shared_declarations_are_built_largest_first_in_the_speculative_catalog(monkeypatch):
    """The verification graphs' shared declarations, replayed through the real _prepare_speculation_catalog build
    order with the decoder's staging and window-cache shapes: one storage per role at its largest request."""
    from ignition.components import capacity, packed_speculation, parallel_draft
    from ignition.preparation.examples import declaration_summary, declaration_tensor
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig, _prepare_speculation_catalog

    layers = 40
    examples = {}

    class Round:
        def __init__(self, speculation, batch, width, name, proposal_only=False, generate_proposals=True,
                     target_token_capacity=None):
            self.batch, self.width, self.proposal_only = batch, width, proposal_only

        def prepare(self):
            if self.proposal_only:
                return
            page = (128 + self.width) * 528  # decoder.py verification staging / attention window cache geometry
            declaration_tensor(examples, "verification.window_staging", (layers, self.batch, page), torch.uint8, "meta")
            declaration_tensor(examples, "attention.window_cache", (self.batch, page), torch.uint8, "meta")

    monkeypatch.setattr(parallel_draft, "PreparedParallelDraftRound", Round)
    monkeypatch.setattr(packed_speculation, "PreparedPackedSpeculativeRound", lambda *args, **kwargs: None)
    config = PreparationConfig(max_active=8)  # export.sh: --max-concurrency 8, 5 drafts (widths 1-6)
    added = []

    def add(name, *args, **kwargs):
        added.append(name)

    def prefill_first():  # the prefill graphs come first; their window cache (1 row, page 128 + 4096) is the largest
        examples.clear()
        declaration_tensor(examples, "attention.window_cache", (1, (128 + 4096) * 528), torch.uint8, "meta")

    def run():
        prefill_first()
        added.clear()
        _prepare_speculation_catalog(config, object(), add=add, buffers={}, forward_inputs=("query_lengths",),
            host_allocate=lambda name, shape, dtype: SimpleNamespace(device_view=torch.zeros(shape, dtype=dtype)),
            device="cpu")
        return declaration_summary(examples)

    summary = run()
    largest_staging = layers * 8 * (128 + 6) * 528
    assert summary["superseded"] == 0 and summary["grown"] == {}
    assert summary["allocated"] == summary["largest"] == largest_staging + (128 + 4096) * 528
    # Graphs are still added in the catalog order: the width-1 rounds first, then the packed rounds.
    x1 = [f"speculative{batch}x1" for batch, _ in reversed(config.speculative_shapes)]
    assert added[:len(x1)] == x1 and len(added) == len(x1) + len(config.packed_speculative_shapes)
    assert all(name.startswith("packed-speculative") for name in added[len(x1):])
    # Negative control: the previous order (x1 rounds built first) supersedes the 8x1 staging, 20.79 MiB per rank.
    prefill_first()
    for batch, _ in reversed(config.speculative_shapes):
        Round(None, batch, 1, "x1").prepare()
    for batch, _ in sorted(config.packed_speculative_shapes, key=lambda shape: (shape[1], shape[0]), reverse=True):
        Round(None, batch, max(config.speculative_widths), "packed").prepare()
    old = declaration_summary(examples)
    assert old["grown"] == {"verification.window_staging": layers * 8 * 129 * 528}
    assert old["superseded"] == 21_795_840  # 20.79 MiB/rank, the item-2 saving


@pytest.mark.parametrize("parked", [True, False])
def test_weight_loading_parks_embedding_and_vision_inside_the_parking_pool(monkeypatch, parked):
    """ModelWeights.load itself (not just _parking_pool): the parked loads and their host copies happen inside the
    short pool, temporary; nothing else does. Negative control: without parking nothing enters the pool."""
    from ignition.components import weights as component_weights
    from ignition.components.parallel import TensorParallelShard
    from ignition.recipes.deepseek41 import weights as ds41_weights
    from ignition.recipes.deepseek41.tp3_config import TP3Config

    state = {"pool": 0}
    events = []

    @contextmanager
    def pool():
        state["pool"] += 1
        try:
            yield
        finally:
            state["pool"] -= 1

    class Loader:
        _sources = {"vision.patch_embed.proj.weight": None, "aligner.w1.weight": None, "layers.0.x": None}

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def load_tensor(self, name, temporary=False, **_):
            events.append(("load", name, temporary, state["pool"]))
            return torch.zeros(2)

        def load_tensors(self, names):
            events.extend(("load", name, False, state["pool"]) for name in names)
            return tuple(torch.zeros(2) for _ in names)

        def load_padded_tensor(self, name, rows):
            events.append(("load", name, False, state["pool"]))
            return torch.zeros(2)

    def park(allocator, name, tensor):
        events.append(("park", name, None, state["pool"]))
        return tensor

    monkeypatch.setattr(ds41_weights, "_parking_pool", pool)
    monkeypatch.setattr(ds41_weights, "DeepSeekWeightLoader", Loader)
    monkeypatch.setattr(ds41_weights, "allocated_tensor_bytes", lambda device: 0)
    monkeypatch.setattr(ds41_weights.ModelConfig, "load", classmethod(
        lambda cls, path, mini=False: SimpleNamespace(layers=0, vision=object())))
    monkeypatch.setattr(component_weights, "_park_on_host", park)
    tp3 = TP3Config(embed_host=parked, vision_host=parked)
    ds41_weights.ModelWeights.load(SimpleNamespace(metadata_path=None), "cpu", shard=TensorParallelShard(0, 3),
                                   host_allocator=object(), tp3=tp3)
    inside = {(kind, name) for kind, name, _, depth in events if depth}
    if parked:
        assert inside == {("load", "embed.weight"), ("park", "embed.weight"),
                          ("load", "vision.patch_embed.proj.weight"), ("park", "vision.patch_embed.proj.weight"),
                          ("load", "aligner.w1.weight"), ("park", "aligner.w1.weight")}
        assert all(temporary for kind, name, temporary, depth in events if kind == "load" and depth)
    else:
        assert inside == set() and not any(kind == "park" for kind, *_ in events)
    outside = {name for kind, name, _, depth in events if kind == "load" and not depth}
    assert {"norm.weight", "head.weight"} <= outside
