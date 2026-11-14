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

    def tensor(name):
        shape, dtype = CHECKPOINT[name.split(".", 2)[2]]
        return torch.ones(shape, dtype=dtype)

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
    for layer in range(40):
        engine = object.__new__(brrr.Engine)
        engine.ext, engine.rank, engine.device = Ext(), 0, torch.device("cpu")
        engine.tp3 = TP3Config(brrr_layers=frozenset(range(40)), brrr_ckpt="/c")
        engine.local_handle, engine.slots, engine.keep, engine.checkpoint = 0, {}, [], Checkpoint()
        weights = SimpleNamespace(
            experts=SimpleNamespace(experts=SimpleNamespace(_impl=impl), hot_map=None, hot_rows=None, router=dense,
                                    correction=flare["ffn.gate.bias"], shared=SimpleNamespace(gate_up=dense, down=dense)),
            attention={"output": SimpleNamespace(wo_a=dense, wo_b=dense)},
            residual={name: flare[name] for name in ("hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "ffn_norm.weight")})
        engine.register(layer, weights)
    if owned:  # after the audit: only the four block-scale copies (39,168 B/layer, 1.49 MiB for BRRR-40)
        assert sum(kept) == 40 * 39_168
    else:  # tp3-main / S6 behaviour (negative control: Flare's bytes differ): 2,017,132 B/layer, 76.95 MiB
        assert sum(kept) == 40 * 2_017_132
