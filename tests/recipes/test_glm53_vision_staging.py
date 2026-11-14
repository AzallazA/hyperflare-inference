# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Fork-only GLM vision weight staging (FLARE_GLM53_VISION_STAGE=1|blocks), DeepSeek's TP3 `vision_stage` for GLM.

Port 2026-11-14 (compiler-owned storage): the 24 blocks are parked and staged through two persistent slots; in the
default mode (VISION_STAGE=1) the tail (downsample and merger projections) is parked too and copied per image into
forward-local tensors (`stage_copies`, DeepSeek's S7 scheme) that the compiler places in the vision graph's pool;
VISION_STAGE=blocks keeps the tail resident.

CPU checks cover the packed byte layout (bytes, geometry and aliasing survive packing and binding), the split of
the tower into parked blocks, a parked tail and resident weights at the checkpoint's real geometry, slot binding
(block ``i`` reads slot ``i % 2``), the staging schedule, the staged tail (bit-equal to the resident tail with stub
kernels, negative controls, the copy ordered after the final norm and before every tail GEMM, each GEMM reading
its own part of the copy, the MXFP8/NVFP4/SM103 parts rebinding), and the export switch's plumbing. They run
without a GPU:

    CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q tests/recipes/test_glm53_vision_staging.py

The CUDA check (FLARE_RUN_CUDA_TESTS=1, one GPU, the GLM checkpoint) parks the tower in pinned host memory and
compares staged replay with the resident tower bit for bit (rewritten for functional forwards; not yet run):

    FLARE_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 FLARE_GLM53_SNAPSHOT=/m2-2/flare-ckpt/glm53-fp4-local \\
        .venv/bin/python -m pytest -q tests/recipes/test_glm53_vision_staging.py
"""

from __future__ import annotations

import dataclasses
import os
from types import SimpleNamespace

import pytest
import torch

from ignition.components.vision_staging import PackedLayout, map_tensors, pack_tree, tensor_leaves
from ignition.recipes.glm53 import recipe as glm_recipe
from ignition.recipes.glm53.bootstrap import PreparationConfig
from ignition.recipes.glm53 import vision as glm_vision
from ignition.recipes.glm53.vision import VISION_BLOCKS, VisionStaging, VisionWeights, staged_blocks

_CUDA = os.environ.get("FLARE_RUN_CUDA_TESTS") == "1"

if _CUDA:
    from tests.recipes.test_glm53_vision import manifest  # noqa: F401  (checkpoint fixture for the CUDA check)

# The checkpoint's vision_config (glm53-fp4-local config.json).
VISION_CONFIG = dict(attention_bias=True, depth=24, hidden_act="silu", hidden_size=1024, image_size=448,
                     in_channels=3, intermediate_size=4096, num_heads=16, out_hidden_size=4096, patch_size=14,
                     projection_intermediate_size=10240, rms_norm_eps=1e-5, spatial_merge_size=2,
                     swiglu_limit=10.0, temporal_patch_size=2)


def _bytes(tensor):
    """Raw bytes of a tensor (NaN-safe comparison of uninitialized storage)."""
    return tensor.contiguous().view(-1).view(torch.uint8)


def _same_bytes(a, b):
    return all(torch.equal(_bytes(x), _bytes(y)) and x.shape == y.shape and x.stride() == y.stride()
               and x.dtype == y.dtype for x, y in zip(tensor_leaves(a), tensor_leaves(b), strict=True))


# Packed layout ------------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class _Rows:
    values: torch.Tensor
    scale_rows: torch.Tensor | None
    scale_mma: torch.Tensor


@dataclasses.dataclass(frozen=True)
class _Packed:
    weight: _Rows
    in_features: int
    out_features: int


def test_packed_layout_keeps_bytes_geometry_aliasing_and_fields():
    shared = torch.arange(4096, dtype=torch.int32).view(torch.uint8)
    tree = {"projection": _Packed(_Rows(torch.randn(48, 64).to(torch.float8_e4m3fn),
                                        shared[1000:1384].view(48, 8), shared[2048:2432].view(4, 96).t()), 64, 48),
            "bias": torch.randn(48).bfloat16(), "norm": torch.randn(7, dtype=torch.float32)[1:]}
    layout, flat, bound = pack_tree(tree, lambda n: torch.zeros(n, dtype=torch.uint8))
    assert flat.numel() == layout.nbytes and layout.nbytes % 256 == 0
    assert _same_bytes(tree, bound)
    # Non-tensor fields survive; views of one storage keep sharing it at the same relative offset.
    assert (bound["projection"].in_features, bound["projection"].out_features) == (64, 48)
    rows, mma = bound["projection"].weight.scale_rows, bound["projection"].weight.scale_mma
    assert rows.untyped_storage().data_ptr() == mma.untyped_storage().data_ptr() == flat.data_ptr()
    assert (mma.data_ptr() - rows.data_ptr()) == 2048 - 1000
    # Only the covered span of a shared storage is packed, from a 256-byte-aligned start (alignment kept).
    assert layout.spans[0] == (0, 64 * 48)
    assert layout.spans[1] == (768, 2048 + 384)
    assert (rows.data_ptr() - flat.data_ptr()) % 256 == 1000 % 256
    # Rebinding to another buffer (a VRAM slot) gives the same geometry; filling it reproduces the tree.
    slot = torch.zeros(layout.nbytes + 512, dtype=torch.uint8)[256:]
    again = layout.bind(bound, slot)
    slot[:layout.nbytes].copy_(flat)
    assert _same_bytes(tree, again)
    # The layout keeps no reference to the source storage.
    assert not any(isinstance(value, torch.UntypedStorage) for value in vars(layout).values())


def test_packed_layout_rejects_mismatched_trees_and_buffers():
    tree = {"a": torch.zeros(16, dtype=torch.bfloat16), "b": torch.zeros(3, dtype=torch.float32)}
    layout = PackedLayout(tree)
    with pytest.raises(ValueError, match="layout"):
        layout.bind({"a": torch.zeros(17, dtype=torch.bfloat16), "b": torch.zeros(3)}, torch.zeros(layout.nbytes, dtype=torch.uint8))
    with pytest.raises(ValueError, match="flat uint8"):
        layout.bind(tree, torch.zeros(layout.nbytes - 1, dtype=torch.uint8))
    with pytest.raises(ValueError, match="256-byte"):
        layout.bind(tree, torch.zeros(layout.nbytes + 8, dtype=torch.uint8)[8:])
    with pytest.raises(ValueError, match="layout"):
        layout.pack({"a": torch.zeros(16, dtype=torch.bfloat16)}, torch.zeros(layout.nbytes, dtype=torch.uint8))
    assert PackedLayout(map_tensors(tree, torch.clone)).signature == layout.signature


# Checkpoint geometry ------------------------------------------------------------------------------------------

def _mxfp8(n, k):
    from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows
    from b12x.gemm.blockscaled._linear import MXFP8LinearWeight

    scales = torch.empty((n, k // 32), dtype=torch.uint8)
    return MXFP8LinearWeight(MXFP8Rows(torch.empty((n, k), dtype=torch.float8_e4m3fn), scales.view(1, n, k // 32),
                                       torch.empty(n * k // 32, dtype=torch.uint8)), k, k, n)


def _nvfp4(n, k):
    from b12x.gemm.blockscaled._a16 import NVFP4LinearWeight

    return NVFP4LinearWeight(torch.empty((n, k // 2), dtype=torch.uint8), torch.empty(n * k // 16, dtype=torch.uint8),
                             torch.ones(1, dtype=torch.float32), "multiplier", k, n)


def _projection(path, shape):
    if path.startswith("blocks.") and ".attn." in path:
        return _mxfp8(*shape)
    return _nvfp4(*shape)


def _checkpoint_weights():
    """VisionWeights as `VisionWeights.load` builds them, at the checkpoint's shapes (uninitialized bytes)."""
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    with torch.device("meta"):
        module = Glm5NextVisionModel(Glm5NextVisionConfig(**VISION_CONFIG))
    quantized = {f"blocks.{i}.{p}" for i in range(24) for p in
                 ("attn.qkv", "attn.proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
    quantized |= {f"merger.{p}" for p in ("proj", "gate_proj", "up_proj", "down_proj")}
    parameters, projections = {}, {}
    for path, parameter in module.named_parameters():
        prefix = path.removesuffix(".weight")
        if path.endswith(".weight") and prefix in quantized:
            projections[prefix] = _projection(prefix, tuple(parameter.shape))
        else:
            parameters[path] = torch.randn(parameter.shape).to(torch.bfloat16)
    # VisionWeights.load drops the convolution-layout weight after transposing it.
    downsample = parameters.pop("downsample.weight").permute(0, 2, 3, 1).reshape(4096, 4096).contiguous()
    return VisionWeights(parameters, projections, downsample)


class _RecordingLoader:
    """VisionWeights.load's loader interface on CPU: records each load's `temporary` flag."""

    def __init__(self, module):
        self.shapes = {"model.visual." + path: tuple(p.shape) for path, p in module.named_parameters()}
        quantized = {f"blocks.{i}.{p}" for i in range(24) for p in
                     ("attn.qkv", "attn.proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
        quantized |= {f"merger.{p}" for p in ("proj", "gate_proj", "up_proj", "down_proj")}
        self.quantization = {"model.visual." + name for name in quantized}
        self.temporary = {}

    def load_linear(self, prefix, *, temporary=False):
        self.temporary[prefix + ".weight"] = temporary
        return _nvfp4(*self.shapes[prefix + ".weight"])

    def load_tensor(self, name, *, temporary=False):
        self.temporary[name] = temporary
        return torch.zeros(self.shapes[name], dtype=torch.bfloat16)


@pytest.mark.parametrize("park_blocks, park_tail", [(False, False), (True, False), (True, True)])
def test_load_keeps_only_the_transposed_downsample_weight(park_blocks, park_tail):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

    with torch.device("meta"):
        module = Glm5NextVisionModel(Glm5NextVisionConfig(**VISION_CONFIG))
    loader = _RecordingLoader(module)
    weights = VisionWeights.load(module, loader, park_blocks=park_blocks, park_tail=park_tail)
    # The convolution-layout weight (32 MiB) loads into temporary memory and is dropped after the transpose.
    assert loader.temporary.pop("model.visual.downsample.weight") is True
    assert "downsample.weight" not in weights.parameters and weights.downsample.shape == (4096, 4096)
    assert weights.downsample.is_contiguous()
    tail = {f"model.visual.{key}.weight" for key in glm_vision.TAIL_PROJECTIONS}
    assert all(flag is (park_blocks and name.startswith("model.visual.blocks.") or park_tail and name in tail)
               for name, flag in loader.temporary.items())


class _Host:
    """Stand-in for core storage host regions: flat uint8 buffers named like the export's regions."""

    def __init__(self):
        self.regions = {}

    def __call__(self, name, nbytes):
        flat = torch.zeros(nbytes, dtype=torch.uint8)
        self.regions[name] = flat
        return flat


@pytest.fixture(scope="module")
def checkpoint():
    weights = _checkpoint_weights()
    host = _Host()
    return weights, weights.park(host), host


def test_parking_splits_the_tower_into_blocks_and_a_resident_set(checkpoint):
    weights, parked, host = checkpoint
    parking = parked.parking
    assert list(host.regions) == [f"model.visual.blocks.{i}" for i in range(VISION_BLOCKS)]
    # Every block shares one layout of ~11 MiB (262 MiB parked in all).
    assert len({flat.nbytes for flat in parking.block_sources}) == 1
    assert 10 << 20 < parking.block_sources[0].nbytes < 12 << 20
    assert parking.parked_bytes == VISION_BLOCKS * parking.block_sources[0].nbytes
    assert set(parked.parameters) == set(weights.parameters)
    assert set(parked.projections) == set(weights.projections)

    def parked_tensor(value):
        return any(f.data_ptr() <= value.data_ptr() < f.data_ptr() + f.nbytes for f in host.regions.values())

    resident = {k for k, v in parked.parameters.items() if not parked_tensor(v)}
    assert resident == {k for k in weights.parameters if not k.startswith("blocks.")}
    assert {k for k, v in parked.projections.items() if not any(map(parked_tensor, tensor_leaves(v)))} == {
        k for k in weights.projections if not k.startswith("blocks.")}
    # Resident weights are the loaded tensors themselves; parked bytes equal the loaded ones.
    assert all(parked.parameters[k] is weights.parameters[k] for k in resident)
    assert parked.downsample is weights.downsample
    assert _same_bytes(weights.projections, parked.projections)
    assert _same_bytes(weights.parameters, parked.parameters)
    # The parked geometry tree holds no loaded tensors.
    loaded = {t.untyped_storage().data_ptr() for t in tensor_leaves((weights.parameters, weights.projections))
              if not any(t is weights.parameters.get(k) for k in resident)}
    assert not loaded & {t.untyped_storage().data_ptr() for t in tensor_leaves(parking.block_tree)}


def test_staging_binds_blocks_to_two_alternating_persistent_slots(checkpoint):
    weights, parked, _ = checkpoint
    with pytest.raises(ValueError, match="parked"):
        VisionStaging(weights, "cpu")
    staging = VisionStaging(parked, "cpu")
    assert staging.allocated_bytes == 2 * parked.parking.block_sources[0].nbytes
    assert len(staging.slots()) == 2 and all(slot.dtype == torch.uint8 for slot in staging.slots())
    assert not staging.staged_tail and staging.tail_bytes == 0
    parameters, projections, downsample = staging.bound(parked)
    assert set(parameters) == set(parked.parameters) and set(projections) == set(parked.projections)
    assert downsample is parked.downsample

    def within(tensor, flat):
        return flat.data_ptr() <= tensor.data_ptr() < flat.data_ptr() + flat.nbytes

    for index in range(VISION_BLOCKS):
        slot = staging.blocks.slot(index)["block"]
        assert slot is staging.slots()[index % 2]
        assert all(within(t, slot) for k, t in {**parameters, **projections}.items()
                   for t in tensor_leaves(t) if k.startswith(f"blocks.{index}."))
    # The tail and the resident set are bound unchanged.
    assert all(parameters[k] is parked.parameters[k] for k in parameters if not k.startswith("blocks."))
    assert all(projections[k] is parked.projections[k] for k in projections if not k.startswith("blocks."))
    # Emulate the copies: after staging block i the bound weights equal the loaded ones.
    for index in (0, 1, 7, 23):
        staging.blocks.slot(index)["block"].copy_(staging.block_sources[index]["block"])
        prefix = f"blocks.{index}."
        assert _same_bytes({k: v for k, v in projections.items() if k.startswith(prefix)},
                           {k: v for k, v in weights.projections.items() if k.startswith(prefix)})
        assert _same_bytes({k: v for k, v in parameters.items() if k.startswith(prefix)},
                           {k: v for k, v in weights.parameters.items() if k.startswith(prefix)})


class _Recorder:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def stage(self, index, sources):
        self.log.append((self.name, "stage", index))

    def join(self, index, dependency):
        self.log.append((self.name, "join", index, dependency))


def test_block_schedule_overlaps_the_next_copy_with_the_current_block():
    log = []
    staging = SimpleNamespace(blocks=_Recorder(log, "blocks"), block_sources=tuple(range(VISION_BLOCKS)))

    def run(block, hidden):
        log.append(("run", block))
        return hidden + 1

    assert staged_blocks(list(range(VISION_BLOCKS)), 0, run, staging) == VISION_BLOCKS
    assert log[0] == ("blocks", "stage", 0)
    for index in range(VISION_BLOCKS):
        position = log.index(("run", index))
        # Block i waits for its own copy, which followed block i - 1's output ...
        assert log.index(("blocks", "join", index, index)) < position
        # ... and block i + 1's copy (into the slot block i - 1 read) is issued before block i runs.
        if index + 1 < VISION_BLOCKS:
            stage = log.index(("blocks", "stage", index + 1))
            assert log.index(("blocks", "join", index, index)) < stage < position
            if index:
                assert log.index(("run", index - 1)) < stage
    assert ("blocks", "stage", VISION_BLOCKS) not in log
    # Without staging the tower runs unchanged.
    assert staged_blocks([1, 2, 3], 0, lambda block, hidden: hidden + block, None) == 6


# Staged tail ---------------------------------------------------------------------------------------------------

def _tail_weights():
    """Checkpoint-geometry weights with random tail bytes (NVFP4 merger values/scales, BF16 downsample)."""
    weights = _checkpoint_weights()
    generator = torch.Generator().manual_seed(1114)
    for key in glm_vision.TAIL_PROJECTIONS:
        for leaf in tensor_leaves(weights.projections[key]):
            if leaf.dtype == torch.uint8:
                leaf.copy_(torch.randint(0, 256, leaf.shape, dtype=torch.uint8, generator=generator))
    return weights


@pytest.fixture(scope="module")
def tail_checkpoint():
    weights = _tail_weights()
    host = _Host()
    return weights, weights.park(host, tail=True), host


def test_parking_with_the_tail_moves_downsample_and_merger_projections(tail_checkpoint):
    weights, parked, host = tail_checkpoint
    parking = parked.parking
    assert list(host.regions) == [*(f"model.visual.blocks.{i}" for i in range(VISION_BLOCKS)), "model.visual.tail"]
    tail = host.regions["model.visual.tail"]
    assert parking.tail_source is tail and 108 << 20 < tail.nbytes < 110 << 20  # 32 MiB downsample + 76.5 merger
    assert parking.parked_bytes == VISION_BLOCKS * parking.block_sources[0].nbytes + tail.nbytes

    def within(value):
        return tail.data_ptr() <= value.data_ptr() < tail.data_ptr() + tail.nbytes

    assert within(parked.downsample) and torch.equal(_bytes(parked.downsample), _bytes(weights.downsample))
    for key in glm_vision.TAIL_PROJECTIONS:
        assert all(map(within, tensor_leaves(parked.projections[key])))
        assert _same_bytes(parked.projections[key], weights.projections[key])
    # Biases and the merger norm stay resident (the loaded tensors themselves).
    assert all(parked.parameters[k] is v for k, v in weights.parameters.items() if not k.startswith("blocks."))
    assert not any(map(within, tensor_leaves(parking.block_tree)))


class _Stub:
    """A prepared projection stand-in: reads ``self.weight`` at call time (as UnquantizedProjection and the plain
    NVFP4 path of QuantizedProjection do) and logs which bytes it read."""

    def __init__(self, name, weight, log):
        self.name, self.weight, self.log = name, weight, log

    def __call__(self, source):
        if self.log is not None:  # eager runs only (traced tensors have no addresses)
            self.log.append((self.name, tuple(t.data_ptr() for t in tensor_leaves(self.weight))))
        if isinstance(self.weight, torch.Tensor):  # downsample, BF16 [out, in]
            return torch.mm(source.float(), self.weight.float().t()).to(torch.bfloat16)
        values, scales = self.weight.values, self.weight.scale_mma  # NVFP4 [out, in / 2] bytes
        product = torch.mm(source.float()[:, :values.shape[1]], values.float().t() / 256)
        return (product + scales.float().mean() / 256).to(torch.bfloat16)


class _Merger(torch.nn.Module):
    """Glm5NextVisionPatchMerger's projection order (proj -> gate/up -> down) with stub children."""

    def forward(self, hidden):
        hidden = self.proj(hidden)
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


def _tail_encoder(projections, downsample, *, staging=None, log=None):
    """PreparedVision's tail state as __init__ builds it, over stub kernels declared on the given weights."""
    log = [] if log is None else log
    encoder = glm_vision.PreparedVision.__new__(glm_vision.PreparedVision)
    encoder.staging = staging
    encoder.staged_tail = staging is not None and staging.staged_tail
    encoder.downsample = glm_vision._Linear(_Stub("downsample", downsample, log), torch.ones(4096).bfloat16())
    merger = _Merger()
    for key in glm_vision.MERGER_PROJECTIONS:
        setattr(merger, key, glm_vision._Linear(_Stub(key, projections[f"merger.{key}"], log)))
    encoder.module = SimpleNamespace(merger=merger)
    return encoder, log


def _normed(rows=4):
    return torch.randn((rows * 4, 1024), generator=torch.Generator().manual_seed(7)).to(torch.bfloat16)


def _resident_tail(weights):
    encoder, _ = _tail_encoder(weights.projections, weights.downsample)
    return encoder.tail(_normed())


@pytest.mark.parametrize("kernel_copy", [False, True])
def test_staged_tail_matches_the_resident_tail_and_reads_its_own_copy(tail_checkpoint, kernel_copy):
    weights, parked, _ = tail_checkpoint
    staging = VisionStaging(parked, "cpu", kernel_copy=kernel_copy)
    assert staging.staged_tail and staging.tail_bytes == parked.parking.tail_source.nbytes
    # The slots stay the block pair: the tail copy is forward-local.
    assert staging.allocated_bytes == 2 * parked.parking.block_sources[0].nbytes
    _, projections, downsample = staging.bound(parked)
    example = staging.tail_example
    assert example is not None and example.device.type == "cpu" and example.nbytes == staging.tail_layout.nbytes
    declared = [downsample, *(projections[key] for key in glm_vision.TAIL_PROJECTIONS)]
    assert all(example.data_ptr() <= t.data_ptr() < example.data_ptr() + example.nbytes
               for t in tensor_leaves(declared))
    encoder, log = _tail_encoder(projections, downsample, staging=staging)
    expected = _resident_tail(weights)

    staged = encoder.tail(_normed())
    assert torch.equal(_bytes(staged), _bytes(expected))
    # One forward-local copy (neither the declaration example nor the parked bytes) feeds the five GEMMs; each
    # reads its own part of it, at the place the packed layout gives that weight.
    copies = []
    original = glm_vision.stage_copies

    def recording(token, sources, **kwargs):
        copies.append(original(token, sources, **kwargs)["tail"])
        return {"tail": copies[-1]}

    log.clear()
    glm_vision.stage_copies, previous = recording, glm_vision.stage_copies
    try:
        again = encoder.tail(_normed())
    finally:
        glm_vision.stage_copies = previous
    assert torch.equal(_bytes(again), _bytes(expected)) and len(copies) == 1
    copy = copies[0]
    assert copy.data_ptr() not in (example.data_ptr(), parked.parking.tail_source.data_ptr())
    bound = staging.tail_layout.bind(staging.tail_tree, copy)
    want = {"downsample": tuple(t.data_ptr() for t in tensor_leaves(bound["downsample"]))}
    want.update({key: tuple(t.data_ptr() for t in tensor_leaves(bound["projections"][f"merger.{key}"]))
                 for key in glm_vision.MERGER_PROJECTIONS})
    assert sorted(log) == sorted(want.items()) and len({ptrs for _, ptrs in log}) == 5
    # The prepared kernels keep their declaration weights; the forward rebinds copies of them.
    assert encoder.downsample.projection.weight is downsample
    # After tuning the examples are dropped; later forwards still rebind every tail kernel to the copy.
    encoder._release_tail_examples()
    assert staging.tail_example is None and encoder.downsample.projection.weight is None
    assert all(getattr(encoder.module.merger, key).projection.weight is None for key in glm_vision.MERGER_PROJECTIONS)
    assert torch.equal(_bytes(encoder.tail(_normed())), _bytes(expected))


def test_staged_tail_negative_controls(tail_checkpoint, monkeypatch):
    weights, parked, _ = tail_checkpoint
    expected = _resident_tail(weights)
    staging = VisionStaging(parked, "cpu")
    _, projections, downsample = staging.bound(parked)
    encoder, _ = _tail_encoder(projections, downsample, staging=staging)
    # Kernels left on the (zero) declaration example: the result differs.
    monkeypatch.setattr(glm_vision, "rebound", lambda projection, weight: projection)
    assert not torch.equal(_bytes(encoder.tail(_normed())), _bytes(expected))
    monkeypatch.undo()
    # A copy that skips the merger's bytes: the result differs.
    monkeypatch.setattr(glm_vision, "stage_copies", lambda token, sources, **kwargs: {
        "tail": torch.cat((sources["tail"][:32 << 20], torch.zeros_like(sources["tail"][32 << 20:])))})
    assert not torch.equal(_bytes(encoder.tail(_normed())), _bytes(expected))


def _ancestors(node):
    seen, stack = set(), [node]
    while stack:
        for parent in stack.pop().all_input_nodes:
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def test_functionalized_tail_copies_after_the_final_norm_and_before_every_tail_gemm(tail_checkpoint):
    """The graph AOTAutograd hands Inductor: one functionalized copy into a graph-local empty whose token is the
    final norm's output, ordered before all five tail GEMMs, each GEMM's weight a view of that copy."""
    from torch._functorch.aot_autograd import aot_function

    _, parked, _ = tail_checkpoint
    staging = VisionStaging(parked, "cpu")
    _, projections, downsample = staging.bound(parked)
    encoder, _ = _tail_encoder(projections, downsample, staging=staging)
    encoder._release_tail_examples()
    for linear in (encoder.downsample, *encoder.module.merger.children()):
        linear.projection.log = None
    captured = []

    class Captured(Exception):
        pass

    def compiler(graph, inputs):
        captured.append(graph)
        raise Captured

    def forward(hidden, source, bias):
        # Constants enter as graph inputs (fake mode refuses real constants).
        staging.tail_source, encoder.downsample.bias = source, bias
        return encoder.tail(torch.nn.functional.rms_norm(hidden, (1024,)))

    source = staging.tail_source
    try:
        with pytest.raises(Captured):
            aot_function(forward, fw_compiler=compiler)(_normed(), source, encoder.downsample.bias)
    finally:
        staging.tail_source = source
    graph = captured[0].graph
    stage = torch.ops.flare.stage_parked_weights.default
    copies = [node for node in graph.nodes if node.op == "call_function"
              and "auto_functionalized" in str(node.target) and node.args[0] is stage]
    assert len(copies) == 1
    (node,) = copies
    assert node.kwargs["stream"] == 0
    assert all(str(base.target).startswith("aten.empty") for base in node.kwargs["_all_bases"])
    # The copy follows the final norm's output (computed from the hidden input), not a graph input it could be
    # hoisted to; the downsample GEMM consumes that same tensor.
    token = node.kwargs["token"]
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    assert token.op == "call_function" and placeholders[0] in _ancestors(token)
    mms = [n for n in graph.nodes if n.op == "call_function" and str(n.target).startswith("aten.mm")]
    assert len(mms) == 5
    order = list(graph.nodes)
    assert all(order.index(node) < order.index(mm) and node in _ancestors(mm.args[1]) for mm in mms)
    assert token is mms[0].args[0] or token in _ancestors(mms[0].args[0])
    # No GEMM reads the parked bytes directly.
    assert not any(placeholders[1] in _ancestors(mm.args[1]) - _ancestors(node) for mm in mms)


def _quantized(weight, *, encoded=False, sm103=False):
    from ignition.components.linear import QuantizedProjection

    projection = QuantizedProjection.__new__(QuantizedProjection)
    projection.weight, projection.sm103, projection.activation_scale = weight, sm103, None
    projection.encoded, projection.encoded_layout = None, (((4, 64), torch.float8_e4m3fn),) if encoded else None
    projection.parts = ("declared",)
    projection.plan = "selected"
    return projection


def test_rebound_rederives_the_parts_quantized_projection_caches(monkeypatch):
    from b12x.gemm.blockscaled._a16 import _weight_parts
    from b12x.quantization import mxfp8

    declared = _nvfp4(64, 128)
    staged = dataclasses.replace(_nvfp4(64, 128), activation_global_scale=torch.full((1,), 2.0))
    for sm103 in (False, True):
        projection = _quantized(declared, sm103=sm103)
        result = glm_vision.rebound(projection, staged)
        values, scales, gain, _ = _weight_parts(staged)
        if sm103 and scales.dtype == torch.float8_e8m0fnu:
            scales = scales.view(torch.uint8)
        assert result.weight is staged and result.plan == "selected"
        assert [t.data_ptr() if t is not None else None for t in result.parts] == [
            t.data_ptr() if t is not None else None for t in (values, scales, gain)]
        assert result.activation_scale is staged.activation_global_scale
        assert projection.weight is declared and projection.parts == ("declared",)
    # MXFP8 pre-encoded projections (`encoded=` / functional `encoded_layout`) cache the packed values and MMA scales.
    # (dense_scale_storage needs CUDA tensors; on CPU it is checked to receive the staged weight's MMA scales.)
    monkeypatch.setattr(mxfp8, "dense_scale_storage", lambda scale_mma: ("storage", scale_mma))
    declared, staged = _mxfp8(64, 128), _mxfp8(64, 128)
    for encoded in (True, False):  # functional (`encoded_layout`) and eager (`encoded`) pre-encoded projections
        projection = _quantized(declared, encoded=encoded)
        if not encoded:
            projection.encoded = "eager rows"
        result = glm_vision.rebound(projection, staged)
        assert result.parts[0] is staged.weight.values and result.parts[2] is None
        assert result.parts[1] == ("storage", staged.weight.scale_mma) and projection.parts == ("declared",)
    # Unquantized projections read `weight` at call time; only it changes.
    plain = SimpleNamespace(weight=torch.zeros(2), bias=None, plan="selected")
    swapped = glm_vision.rebound(plain, torch.ones(2))
    assert torch.equal(swapped.weight, torch.ones(2)) and torch.equal(plain.weight, torch.zeros(2))


# Export switch ------------------------------------------------------------------------------------------------

def test_vision_stage_defaults_off_and_is_validated(monkeypatch):
    assert PreparationConfig(devices=(0, 1)).vision_stage is False
    # Staging the tail is the default mode once staging is on.
    assert PreparationConfig(devices=(0, 1)).vision_stage_tail is True
    assert PreparationConfig(devices=(0, 1), vision_stage=True).vision_stage is True
    with pytest.raises(TypeError, match="vision_stage"):
        PreparationConfig(devices=(0, 1), vision_stage=1)
    with pytest.raises(TypeError, match="vision_stage_tail"):
        PreparationConfig(devices=(0, 1), vision_stage_tail="blocks")
    monkeypatch.setenv("FLARE_GLM53_VISION_STAGE", "yes")
    with pytest.raises(ValueError, match="VISION_STAGE"):
        glm_recipe._tp3_override("VISION_STAGE", ("1", "blocks", "0"), "0")


@pytest.mark.parametrize("world", [2, 3])
@pytest.mark.parametrize("value, expected, tail", [(None, False, True), ("0", False, True), ("1", True, True),
                                                   ("blocks", True, False)])
def test_export_environment_reaches_the_preparation_config(monkeypatch, world, value, expected, tail):
    import ignition.preparation.rank_group as rank_group
    from tests.recipes.test_glm53_vram_options import _plan

    for name in ("EXPERT_ACTIVATION", "EXPAND_SCALES", "EXPERT_COLUMNS", "KDA_SIDE_CHAIN", "DRAFT_HEAD",
                 "DRAFT_EH_PROJ", "EMBED_HOST", "VISION_STAGE"):
        monkeypatch.delenv(f"FLARE_GLM53_{name}", raising=False)
    if value is not None:
        monkeypatch.setenv("FLARE_GLM53_VISION_STAGE", value)
    seen = {}
    monkeypatch.setattr(rank_group, "prepare_group",
                        lambda snapshot, config, function, **kwargs: seen.setdefault("config", config))
    plan = _plan(world)
    plan.language_model_only = False
    glm_recipe.prepare("/nonexistent", plan, compile_forwards=False, kernel_cache=None, progress=None)
    config = seen["config"]
    assert config.vision_stage is expected and config.vision_stage_tail is tail
    assert (config.embed_host, config.kda_side_chain, config.language_model_only) == (False, False, False)


# GPU: staged replay equals the resident tower ------------------------------------------------------------------

@pytest.mark.skipif(not _CUDA, reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")
@torch.no_grad()
@pytest.mark.parametrize("compiler", ["none", "worker"])
@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("tail", [True, False])  # VISION_STAGE=1 (blocks + tail) and VISION_STAGE=blocks
def test_staged_replay_matches_the_resident_tower(manifest, compiler, overlap, tail, tmp_path):
    from functools import partial

    from b12x.preparation import PreparationSession
    from transformers.vision_utils import get_vision_position_ids

    from ignition.components.weights import DirectWeightLoader
    from ignition.preparation.scratch import ScratchArena
    from ignition.recipes.glm53.vision import PreparedVision, VisionMetadata, VisionScratch
    from tests.recipes.test_glm53_vision import _input, _module
    from tests.recipes.test_glm53_vram_options import _PinnedHost

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    with DirectWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        weights = VisionWeights.load(_module(manifest), loader)
    with DirectWeightLoader(manifest, device, prefixes=("model.visual.",)) as loader:
        temporary = VisionWeights.load(_module(manifest), loader, park_blocks=True, park_tail=tail)
    host = _PinnedHost(device)
    parked = temporary.park(lambda name, nbytes: host(name, (nbytes,), torch.uint8).device_view, tail=tail)
    assert (parked.parking.tail_source is not None) is tail
    del temporary
    capacity = 4096
    inputs = _input([(14 * 56, 14 * 56)], 0)
    count = inputs.pixel_values.shape[0]
    outputs = {}
    for variant, source in (("resident", weights), ("staged", parked)):
        module = _module(manifest)
        scratch = VisionScratch.allocate(16384, device)
        metadata = VisionMetadata.allocate(capacity, 1, device)
        pixels = torch.zeros((capacity, 1176), dtype=torch.float32, device=device)
        arena = ScratchArena(device, overlap=overlap)
        with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
            staging = (VisionStaging(parked, device, stream=arena.auxiliary_stream if overlap else None)
                       if variant == "staged" else None)
            # Workspaces as in the native replay test (none); the arena only supplies the copy stream.
            prepared = PreparedVision(module, scratch.rows(capacity), metadata, source, session=session,
                                      staging=staging)
            prepared.functional(session=session)
            session.freeze()
            forward = lambda: prepared(pixels.to(torch.bfloat16), metadata)  # noqa: E731
            if compiler == "worker":
                from ignition.preparation.aot import ForwardCache
                from ignition.preparation.execution import GraphSignature

                cache = ForwardCache(namespace={"vision": "glm53-staged"}, device=device, root=tmp_path / variant)
                forward = cache.prepare_many({"vision": partial(forward)},
                                             {"vision": GraphSignature("vision", 1, capacity)})["vision"]
            metadata.position_ids[:count].copy_(get_vision_position_ids(inputs.image_grid_thw, 2))
            metadata.cu_seqlens.copy_(torch.tensor([0, count], dtype=torch.int32))
            metadata.num_patches.fill_(count)
            pixels[:count].copy_(inputs.pixel_values)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                forward()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = forward()
            results = []
            for _ in range(3):
                graph.replay()
                torch.cuda.synchronize(device)
                results.append(output[:count // 4].clone())
            outputs[variant] = results
    for result in (*outputs["resident"], *outputs["staged"]):
        torch.testing.assert_close(result, outputs["resident"][0], rtol=0, atol=0)


@pytest.mark.parametrize("language_model_only, value, warned", [
    (True, "1", True), (True, "blocks", True), (True, "0", False), (False, "1", False)])
def test_vision_stage_without_a_vision_tower_warns(monkeypatch, language_model_only, value, warned):
    import warnings

    import ignition.preparation.rank_group as rank_group
    from tests.recipes.test_glm53_vram_options import _plan

    for name in ("EXPERT_ACTIVATION", "EXPAND_SCALES", "EXPERT_COLUMNS", "KDA_SIDE_CHAIN", "DRAFT_HEAD",
                 "DRAFT_EH_PROJ", "EMBED_HOST"):
        monkeypatch.delenv(f"FLARE_GLM53_{name}", raising=False)
    monkeypatch.setenv("FLARE_GLM53_VISION_STAGE", value)
    monkeypatch.setattr(rank_group, "prepare_group", lambda snapshot, config, function, **kwargs: config)
    plan = _plan(3)
    plan.language_model_only = language_model_only
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        glm_recipe.prepare("/nonexistent", plan, compile_forwards=False, kernel_cache=None, progress=None)
    messages = [str(w.message) for w in caught if "VISION_STAGE" in str(w.message)]
    assert messages == (["FLARE_GLM53_VISION_STAGE ignored: a language-model-only export has no vision tower"]
                        if warned else [])
