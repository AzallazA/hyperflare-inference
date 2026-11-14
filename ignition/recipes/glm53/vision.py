# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured GLM vision tower with canonical blocks and prepared projections."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass, replace

import torch
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextVisionRotaryEmbedding,
    apply_rotary_pos_emb_vision,
)

from ignition.components.linear import QuantizedProjection, UnquantizedProjection
from ignition.components.varlen_attention import PreparedVarlenAttention
from ignition.components.vision_staging import PackedLayout, WeightStaging, pack_tree, stage_copies
from ignition.preparation.persistent import persistent_copy, persistent_zeros
from ignition.preparation.tensor_metadata import record_numeric_tensor, unrecorded_numeric_tensors
from ignition.preparation.tuning import register_finalizer

VISION_BLOCKS = 24
# Weight staging (fork-only, FLARE_GLM53_VISION_STAGE=1|blocks): the 24 blocks (262 MiB) share one geometry and
# are parked in device-visible host memory and staged through two persistent VRAM slots (22 MiB) per image.
# With the tail (VISION_STAGE=1, the default mode) the downsample and merger projection weights (109 MiB) are
# parked too and copied once per image, after the last block, into forward-local tensors the compiler places in
# the vision graph's pool (tp3-main aliased a tail slot onto dead vision scratch by hand). The patch embedding,
# the final norm and the small tail parameters (biases, merger norm) stay resident.
MERGER_PROJECTIONS = ("proj", "gate_proj", "up_proj", "down_proj")
TAIL_PROJECTIONS = tuple(f"merger.{key}" for key in MERGER_PROJECTIONS)


@dataclass(frozen=True)
class VisionMetadata:
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    num_patches: torch.Tensor

    @classmethod
    def allocate(cls, patches, images, device):
        if patches <= 0 or patches % 4 or images <= 0:
            raise ValueError("GLM vision requires positive image capacity and patch capacity divisible by four")
        return cls(torch.zeros((patches, 2), dtype=torch.int64, device=device),
                   torch.zeros(images + 1, dtype=torch.int32, device=device),
                   torch.zeros(1, dtype=torch.int32, device=device))


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    hidden: torch.Tensor
    qkv: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention: torch.Tensor
    projected: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    merged: torch.Tensor
    merger_gate: torch.Tensor
    merger_up: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches, device):
        if patches <= 0 or patches % 4:
            raise ValueError("GLM vision patch capacity must be positive and divisible by four")
        def zeros(*shape):
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)
        return cls(zeros(patches, 1176), zeros(patches, 1024), zeros(patches, 3072),
                   zeros(patches, 16, 64), zeros(patches, 16, 64), zeros(patches, 16, 64),
                   zeros(patches, 1024), zeros(patches, 1024),
                   zeros(patches, 4096), zeros(patches, 4096),
                   zeros(patches // 4, 4096), zeros(patches // 4, 10240),
                   zeros(patches // 4, 10240), zeros(patches // 4, 4096))

    def rows(self, patches):
        if patches <= 0 or patches % 4 or patches > self.patches.shape[0]:
            raise ValueError("GLM vision view exceeds its patch capacity")
        merged = {"merged", "merger_gate", "merger_up", "output"}
        return type(self)(*(getattr(self, name)[:patches // 4 if name in merged else patches]
                            for name in self.__dataclass_fields__))


@dataclass(frozen=True)
class VisionParking:
    """Host-parked block weights: one flat region per block, all sharing `block_layout`; `block_tree` gives the
    geometry the layout binds (keys relative to a block). With the tail parked, `tail_source` holds the downsample
    weight and merger projections, bound by `tail_layout` to `tail_tree`'s geometry."""
    block_layout: PackedLayout
    block_sources: tuple
    block_tree: dict
    tail_layout: PackedLayout | None = None
    tail_source: torch.Tensor | None = None
    tail_tree: dict | None = None

    @property
    def parked_bytes(self):
        return sum(flat.nbytes for flat in self.block_sources) + (
            0 if self.tail_source is None else self.tail_source.nbytes)


def _block_tree(parameters, projections, index):
    prefix = f"blocks.{index}."
    return {"parameters": {k.removeprefix(prefix): v for k, v in parameters.items() if k.startswith(prefix)},
            "projections": {k.removeprefix(prefix): v for k, v in projections.items() if k.startswith(prefix)}}


@dataclass(frozen=True)
class VisionWeights:
    parameters: dict[str, torch.Tensor]
    projections: dict[str, object]
    downsample: torch.Tensor
    parking: VisionParking | None = None

    @classmethod
    def load(cls, module, loader, *, park_blocks=False, park_tail=False):
        """With `park_blocks` (`park_tail`) the block (tail projection) weights load into temporary device memory:
        `park` copies them to host memory, and the persistent arena never releases what it holds."""
        parameters, projections = {}, {}
        declarations = loader.quantization
        for path, _ in module.named_parameters():
            prefix = "model.visual." + path.removesuffix(".weight")
            # The convolution-layout downsample weight is read once, below, so it never needs persistent storage.
            temporary = (park_blocks and path.startswith("blocks.") or path == "downsample.weight"
                         or park_tail and path.removesuffix(".weight") in TAIL_PROJECTIONS)
            if path.endswith(".weight") and prefix in declarations:
                projections[path.removesuffix(".weight")] = loader.load_linear(prefix, temporary=temporary)
            else:
                parameters[path] = loader.load_tensor("model.visual." + path, temporary=temporary)
        # Patch rows are ordered [merge_h, merge_w, channel]. Conv2d stores
        # weights as [out, channel, merge_h, merge_w]; transpose once at load and drop the convolution layout
        # (only the transposed copy is read; tp3-main's parking dropped it too).
        transposed = parameters.pop("downsample.weight").permute(0, 2, 3, 1).reshape(4096, 4096)
        downsample = transposed.contiguous() if park_tail else persistent_copy(transposed)
        return cls(parameters, projections, downsample)

    def park(self, allocate, *, tail=False):
        """Move the block weights into ``allocate(name, nbytes)`` flat uint8 regions (device-visible host memory);
        with `tail` also the downsample weight and merger projections (one region). The returned weights view the
        parked bytes and keep the resident set (embedding, norms, biases) in place."""
        trees = tuple(_block_tree(self.parameters, self.projections, i) for i in range(VISION_BLOCKS))
        layout = PackedLayout(trees[0])
        if any(PackedLayout(tree).signature != layout.signature for tree in trees):
            raise ValueError("GLM vision staging requires every block to share one weight layout")
        blocks = tuple(f"blocks.{i}." for i in range(VISION_BLOCKS))
        if any(k.startswith("blocks.") and not k.startswith(blocks) for k in (*self.parameters, *self.projections)):
            raise ValueError("GLM vision weights name a block outside the 24-block tower")
        parked_parameters, parked_projections, sources, parked_blocks = {}, {}, [], []
        for index, tree in enumerate(trees):
            _, flat, parked = pack_tree(tree, lambda n, i=index: allocate(f"model.visual.blocks.{i}", n))
            sources.append(flat)
            parked_blocks.append(parked)
            parked_parameters.update({f"blocks.{index}.{k}": v for k, v in parked["parameters"].items()})
            parked_projections.update({f"blocks.{index}.{k}": v for k, v in parked["projections"].items()})
        if (set(parked_parameters) != {k for k in self.parameters if k.startswith("blocks.")}
                or set(parked_projections) != {k for k in self.projections if k.startswith("blocks.")}):
            raise ValueError("GLM vision weights must split into blocks and a resident set")
        downsample, tail_parking = self.downsample, {}
        if tail:
            tree = {"projections": {key: self.projections[key] for key in TAIL_PROJECTIONS},
                    "downsample": self.downsample}
            tail_layout, tail_source, tail_tree = pack_tree(tree, lambda n: allocate("model.visual.tail", n))
            parked_projections.update(tail_tree["projections"])
            downsample = tail_tree["downsample"]
            tail_parking = dict(tail_layout=tail_layout, tail_source=tail_source, tail_tree=tail_tree)
        # Same keys in the same order, parked entries viewing the parked bytes.
        parameters = {k: parked_parameters.get(k, v) for k, v in self.parameters.items()}
        projections = {k: parked_projections.get(k, v) for k, v in self.projections.items()}
        # Geometry trees view the parked bytes, so no reference to the loaded device copies survives.
        return replace(self, parameters=parameters, projections=projections, downsample=downsample,
                       parking=VisionParking(layout, tuple(sources), parked_blocks[0], **tail_parking))


class VisionStaging:
    """Two persistent VRAM slots that every GLM vision graph of one rank stages its block weights through, and
    (parked tail) the forward-local tail copy.

    Block ``i`` reads slot ``i % 2`` while block ``i + 1`` lands in the other. The slots and the staging token
    are persistent, refilled on every replay; graphs replay serially, so all patch capacities share them.
    The tail kernels are declared and tuned on one VRAM declaration example of the packed tail (shared by every
    capacity, released after tuning) and rebound per forward to `stage_tail`'s copy.
    """

    def __init__(self, weights, device, *, stream=None, kernel_copy=False):
        parking = weights.parking
        if parking is None:
            raise ValueError("GLM vision staging requires host-parked weights (VisionWeights.park)")
        self.block_sources = tuple({"block": flat} for flat in parking.block_sources)
        slots = tuple({"block": persistent_zeros((parking.block_layout.nbytes,), torch.uint8, device)}
                      for _ in range(2))
        self.blocks = WeightStaging(self.block_sources[0], storage=slots, stream=stream, kernel_copy=kernel_copy,
                                    token=persistent_zeros((1,), torch.int32, device))
        if not all(self.blocks.matches(sources) for sources in self.block_sources):
            raise ValueError("GLM vision staging requires every block to share one packed size")
        # The parked bytes are the checkpoint storage the package stores and the slots are filled from.
        for flat in parking.block_sources:
            record_numeric_tensor(flat)
        self.block_trees = tuple(parking.block_layout.bind(parking.block_tree, self.blocks.slot(i)["block"])
                                 for i in range(2))
        self.kernel_copy, self.device = bool(kernel_copy), device
        self.tail_layout, self.tail_source, self.tail_tree = parking.tail_layout, parking.tail_source, parking.tail_tree
        self.tail_example = None
        if self.tail_source is not None:
            # The parked bytes remain the checkpoint storage the package stores and the graph copies from.
            record_numeric_tensor(self.tail_source)

    @property
    def staged_tail(self):
        return self.tail_source is not None

    @property
    def allocated_bytes(self):
        """Persistent VRAM the staging holds: the two block slots (the tail copy is forward-local)."""
        return self.blocks.nbytes

    @property
    def tail_bytes(self):
        return 0 if self.tail_source is None else self.tail_source.nbytes

    def stage_tail(self, token):
        """The tail weights copied into a forward-local tensor after ``token`` (the last block's output) exists."""
        copied = stage_copies(token, {"tail": self.tail_source}, kernel_copy=self.kernel_copy)["tail"]
        return self.tail_layout.bind(self.tail_tree, copied)

    def tail_example_tree(self):
        """The tail geometry bound to one VRAM declaration example, shared by every vision graph prepared before
        tuning (`release_tail_example` drops it; a later graph declares on a fresh one)."""
        if self.tail_example is None:
            self.tail_example = torch.zeros((self.tail_layout.nbytes,), dtype=torch.uint8, device=self.device)
        return self.tail_layout.bind(self.tail_tree, self.tail_example)

    def release_tail_example(self):
        self.tail_example = None

    def slots(self):
        """Slot tensors, rewritten on every vision replay (export stores them as zero ranges)."""
        return tuple(slot["block"] for slot in self.blocks.slots)

    def bound(self, weights):
        """Parameters, projections and downsample weight with every block's weights read from its slot and the
        tail's from the declaration example."""
        staged_parameters, staged_projections = {}, {}
        for index in range(VISION_BLOCKS):
            tree = self.block_trees[index % 2]
            staged_parameters.update({f"blocks.{index}.{k}": v for k, v in tree["parameters"].items()})
            staged_projections.update({f"blocks.{index}.{k}": v for k, v in tree["projections"].items()})
        if (set(staged_parameters) != {k for k in weights.parameters if k.startswith("blocks.")}
                or set(staged_projections) != {k for k in weights.projections if k.startswith("blocks.")}):
            raise ValueError("staged GLM vision weights do not cover the parked weights")
        downsample = weights.downsample
        if self.staged_tail:
            example = self.tail_example_tree()
            staged_projections.update(example["projections"])
            downsample = example["downsample"]
        return ({k: staged_parameters.get(k, v) for k, v in weights.parameters.items()},
                {k: staged_projections.get(k, v) for k, v in weights.projections.items()}, downsample)


def rebound(projection, weight):
    """A prepared projection reading ``weight`` (same geometry as declared) in this forward.

    UnquantizedProjection and the plain QuantizedProjection path read ``weight`` at call time; the parts the
    SM103 and MXFP8-encoded paths cache are re-derived from it the way QuantizedProjection.__init__ does."""
    result = copy.copy(projection)
    result.weight = weight
    if isinstance(projection, QuantizedProjection):
        result.activation_scale = getattr(weight, "activation_global_scale", None)
        if projection.encoded is not None or projection.encoded_layout is not None:
            from b12x.quantization import mxfp8

            packed = weight.weight
            result.parts = packed.values, mxfp8.dense_scale_storage(packed.scale_mma), None
        else:
            from b12x.gemm.blockscaled._a16 import _weight_parts

            values, scales, gain, _ = _weight_parts(weight)
            if projection.sm103 and scales.dtype == torch.float8_e8m0fnu:
                scales = scales.view(torch.uint8)
            result.parts = values, scales, gain
    return result


def _with_children(module, children):
    """A shallow copy of ``module`` whose submodules ``children`` replace the named ones (the original keeps its)."""
    clone = copy.copy(module)
    clone._modules = {**module._modules, **children}
    return clone


class _Linear(torch.nn.Module):
    def __init__(self, projection, bias=None):
        super().__init__()
        self.projection, self.bias = projection, bias

    def forward(self, source):
        result = self.projection(source)
        if self.bias is not None:
            result.add_(self.bias)
        return result


class _Attention(torch.nn.Module):
    def __init__(self, module, scratch, metadata, weights, *, session, name, shared=None,
                 projection_workspace=None, attention_workspace=None):
        super().__init__()
        self.scratch = scratch
        self.q_norm, self.k_norm = module.q_norm, module.k_norm
        self.qkv = QuantizedProjection(scratch.hidden, weights["qkv"], scratch.qkv,
            session=session, name=f"{name}.qkv", bias=module.qkv.bias, workspace=projection_workspace)
        self.proj = QuantizedProjection(scratch.attention, weights["proj"], scratch.projected,
            session=session, name=f"{name}.proj", bias=module.proj.bias, workspace=projection_workspace)
        self.attention = shared.attention if shared is not None else PreparedVarlenAttention(
            scratch.query, scratch.key, scratch.value, metadata.cu_seqlens,
            scratch.attention.view_as(scratch.query), session=session, workspace=attention_workspace, name=name)
        self.owns_attention = shared is None

    def functional(self, *, session):
        self.qkv.functional(session=session)
        self.proj.functional(session=session)
        if self.owns_attention:
            self.attention.functional(session=session)
        self.scratch = None
        return self

    def forward(self, hidden_states, cu_seqlens, position_embeddings=None, **kwargs):
        projected = self.qkv(hidden_states).view(-1, 3, 16, 64)
        query, key = apply_rotary_pos_emb_vision(self.q_norm(projected[:, 0]), self.k_norm(projected[:, 1]),
                                               *position_embeddings)
        attended = self.attention(query.contiguous(), key.contiguous(), projected[:, 2].contiguous(),
                                  cu_seqlens)
        return self.proj(attended.view(hidden_states.shape[0], 1024))


def staged_blocks(blocks, hidden, run, staging):
    """Run ``hidden = run(block, hidden)`` over the tower, staging block ``i + 1`` while block ``i`` runs.

    Block ``i`` reads slot ``i % 2``. Its join follows the previous block's output (``hidden``); the next block's
    copies fork only after that join, when the slot's previous reader (block ``i - 1``) has been enqueued."""
    if staging is not None:
        staging.blocks.stage(0, staging.block_sources[0])
    for index, block in enumerate(blocks):
        if staging is not None:
            staging.blocks.join(index, hidden)
            if index + 1 < len(blocks):
                staging.blocks.stage(index + 1, staging.block_sources[index + 1])
        hidden = run(block, hidden)
    return hidden


class PreparedVision:
    """Encode packed patches with a fixed capacity and device-resident lengths.

    Canonical GLM blocks retain RMS normalization, Q/K normalization, rotary
    positions, clamped SwiGLU, and residual semantics. Prepared projections
    retain checkpoint precision. Each capacity owns a canonical module tree;
    weights and sequential scratch may be shared across capacities.
    """

    def __init__(self, module, scratch, metadata, weights, *, session, name="vision",
                 projection_workspace=None, attention_workspace=None, staging=None):
        config = module.config
        expected = dict(hidden_size=1024, num_heads=16, intermediate_size=4096, depth=24,
                        out_hidden_size=4096, projection_intermediate_size=10240,
                        patch_size=14, temporal_patch_size=2, spatial_merge_size=2, in_channels=3)
        if any(getattr(config, key) != value for key, value in expected.items()):
            raise ValueError("GLM vision geometry differs from the checkpoint contract")
        capacity, device = scratch.hidden.shape[0], scratch.hidden.device
        if metadata.position_ids.shape != (capacity, 2):
            raise ValueError("GLM vision metadata and scratch capacities differ")
        if metadata.num_patches.item() or bool(metadata.cu_seqlens.any().item()):
            raise ValueError("GLM vision preparation requires inactive metadata")
        self.module, self.scratch, self.metadata, self.staging = module, scratch, metadata, staging
        parameters, projections, downsample = ((weights.parameters, weights.projections, weights.downsample)
                                               if staging is None else staging.bound(weights))
        self.staged_tail = staging is not None and staging.staged_tail
        for path, tensor in parameters.items():
            parent, _, attribute = path.rpartition(".")
            setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(tensor, requires_grad=False))
        self.patch_embed = _Linear(UnquantizedProjection(scratch.patches,
            module.patch_embed.proj.weight.flatten(1), scratch.hidden,
            session=session, name=f"{name}.patch_embed"), module.patch_embed.proj.bias)
        module.rotary_pos_emb = Glm5NextVisionRotaryEmbedding(config).to(device)
        # Staged block kernels read slot tensors, rewritten on every replay rather than checkpoint storage, so they
        # are not recorded as numeric storage (VisionStaging recorded the parked sources).
        with unrecorded_numeric_tensors() if staging is not None else nullcontext():
            for index, block in enumerate(module.blocks):
                prefix = f"blocks.{index}"
                block.attn = _Attention(block.attn, scratch, metadata,
                    {key: projections[f"{prefix}.attn.{key}"] for key in ("qkv", "proj")},
                    session=session, name=f"{name}.{prefix}.attn",
                    shared=module.blocks[0].attn if index else None,
                    projection_workspace=projection_workspace, attention_workspace=attention_workspace)
                for key, source, output in (("gate_proj", scratch.hidden, scratch.gate),
                                           ("up_proj", scratch.hidden, scratch.up),
                                           ("down_proj", scratch.gate, scratch.projected)):
                    original = getattr(block.mlp, key)
                    setattr(block.mlp, key, _Linear(QuantizedProjection(source,
                        projections[f"{prefix}.mlp.{key}"], output,
                        session=session, name=f"{name}.{prefix}.mlp.{key}",
                        activation_mode="auto", bias=original.bias, workspace=projection_workspace)))
        # Staged tail kernels are declared on the tail's declaration example and rebound to the forward's copy.
        with unrecorded_numeric_tensors() if self.staged_tail else nullcontext():
            self.downsample = _Linear(UnquantizedProjection(scratch.hidden.view(-1, 4096),
                downsample, scratch.merged, session=session, name=f"{name}.downsample"), module.downsample.bias)
            for key, source, output in (("proj", scratch.merged, scratch.output),
                                       ("gate_proj", scratch.output, scratch.merger_gate),
                                       ("up_proj", scratch.output, scratch.merger_up),
                                       ("down_proj", scratch.merger_gate, scratch.output)):
                original = getattr(module.merger, key)
                setattr(module.merger, key, _Linear(QuantizedProjection(source,
                    projections[f"merger.{key}"], output,
                    session=session, name=f"{name}.merger.{key}",
                    activation_mode="auto", bias=original.bias, workspace=projection_workspace)))
        if self.staged_tail:
            register_finalizer(session, self._release_tail_examples)
        self.capacity = capacity

    def _release_tail_examples(self):
        """After tuning: drop every reference to the tail's declaration example (each forward rebinds the tail)."""
        tail = (self.downsample, *(getattr(self.module.merger, key) for key in MERGER_PROJECTIONS))
        for linear in tail:
            linear.projection.weight = None
            if hasattr(linear.projection, "parts"):
                linear.projection.parts = None
        self.staging.release_tail_example()

    def tail(self, normed):
        """Downsample and merge the final-normed patch rows; a staged tail first copies its weights (after
        ``normed`` exists) and runs the prepared kernels on that copy."""
        downsample, merger = self.downsample, self.module.merger
        if self.staged_tail:
            weights = self.staging.stage_tail(normed)
            downsample = _Linear(rebound(downsample.projection, weights["downsample"]), downsample.bias)
            merger = _with_children(merger, {key: _Linear(rebound(getattr(merger, key).projection,
                weights["projections"][f"merger.{key}"]), getattr(merger, key).bias) for key in MERGER_PROJECTIONS})
        return merger(downsample(normed.view(-1, 4096)))

    def functional(self, *, session):
        self.patch_embed.projection.functional(session=session)
        for block in self.module.blocks:
            block.attn.functional(session=session)
            for name in ("gate_proj", "up_proj", "down_proj"):
                getattr(block.mlp, name).projection.functional(session=session)
        self.downsample.projection.functional(session=session)
        for name in ("proj", "gate_proj", "up_proj", "down_proj"):
            getattr(self.module.merger, name).projection.functional(session=session)
        self.scratch = self.metadata = None
        return self

    def __call__(self, pixels, metadata):
        module = self.module
        hidden = self.patch_embed(pixels)
        rope = module.rotary_pos_emb(hidden, metadata.position_ids)
        hidden = staged_blocks(module.blocks, hidden, lambda block, value: block(
            value, cu_seqlens=metadata.cu_seqlens, position_embeddings=rope), self.staging)
        output = self.tail(module.post_layernorm(hidden))
        output.masked_fill_((torch.arange(self.capacity // 4, device=pixels.device) >= metadata.num_patches // 4)[:, None], 0)
        return output
