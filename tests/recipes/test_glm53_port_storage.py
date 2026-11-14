# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fork TP3 deltas that depend on upstream's compiler-owned storage (port 2026-11-14).

The persistent arena never releases space, so every tensor that the fork replaces after loading (an MXFP8
eh_proj, a parked embedding, the KDA convolution parts) loads into temporary device memory, and derived
forward constants (the BF16 decoding of an MXFP8 kv_b_proj) are made once per weight.
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.glm53.mla import MXFP8_BMM_HEADS, decoded_kv_b
from ignition.recipes.glm53.shards import Shards
from ignition.recipes.glm53.weights import GlmWeightLoader


def _packed(heads, generator):
    values = (torch.randn((heads * 512, 512), generator=generator) * 4).to(torch.float8_e4m3fn)
    exponents = torch.randint(120, 134, (heads * 512, 16), generator=generator, dtype=torch.uint8)
    return SimpleNamespace(values=values, scale_rows=exponents)


def test_mxfp8_kv_b_decodes_exactly_once_per_weight():
    heads = Shards.of(3).heads
    assert heads not in MXFP8_BMM_HEADS
    generator = torch.Generator().manual_seed(3)
    packed = _packed(heads, generator)
    decoded = decoded_kv_b(packed, heads)
    scale = torch.pow(2.0, packed.scale_rows.int() - 127).float()
    reference = (packed.values.float().view(heads * 512, 16, 32) * scale[..., None]).view(heads, 512, 512)
    assert decoded.shape == (heads, 512, 512) and decoded.dtype == torch.bfloat16
    assert torch.equal(decoded.float(), reference)
    # Every prepared draft pass of the same layer shares one copy; a different weight gets its own.
    assert decoded_kv_b(packed, heads) is decoded
    other = _packed(heads, generator)
    assert decoded_kv_b(other, heads) is not decoded


def _recording_loader(world=3, rank=2):
    loader = object.__new__(GlmWeightLoader)
    loader.shard = SimpleNamespace(world_size=world, rank=rank)
    loader.calls = []

    def load_tensor(name, *, dtype=None, extent=None, temporary=False):
        loader.calls.append(("tensor", name, extent, temporary))
        return torch.zeros(1)

    def load_padded_tensor(name, *, rows, temporary=False):
        loader.calls.append(("padded", name, rows, temporary))
        return torch.zeros(1)

    loader.load_tensor, loader.load_padded_tensor = load_tensor, load_padded_tensor
    loader.load_attention = lambda layer: {}
    loader.load_mlp = lambda layer, *, scale_scratch=None: None
    loader._layer = lambda layer: f"layers.{layer}"
    return loader


@pytest.mark.parametrize("temporary", [False, True])
def test_only_a_replaced_eh_proj_loads_into_temporary_memory(temporary):
    loader = _recording_loader()
    loader.load_mtp(None, temporary_eh_proj=temporary)
    flags = {name: flag for kind, name, _, flag in loader.calls}
    assert flags.pop("layers.45.eh_proj.weight") is temporary
    assert len(flags) == 5 and not any(flags.values())


def test_padded_rows_forward_the_temporary_flag():
    loader = _recording_loader()
    loader._sources = {"short": torch.empty(64), "long": torch.empty(128)}
    loader._load_rows("short", (48, 72), temporary=True)
    loader._load_rows("long", (48, 72), temporary=True)
    loader._load_rows("long", (0, 24))
    assert loader.calls == [("padded", "short", (48, 72), True), ("tensor", "long", slice(48, 72), True),
                            ("tensor", "long", slice(0, 24), False)]
