# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Native FP8/MXFP4 checkpoint transfers and tensor-parallel partitions."""

import os
from contextlib import nullcontext
from dataclasses import dataclass, replace

import torch

from ignition.components.weights import MLPWeights
from ignition.components.block_fp8_weights import BlockFP8WeightLoader
from ignition.components.vocabulary import mxfp8_draft_head, nvfp4_draft_head
from ignition.components.markov_sampling import fp8_markov_head
from ignition.preparation.persistent import active_core_storage, allocated_tensor_bytes, persistent_copy, persistent_empty

from .config import ModelConfig
from .tp3_config import TP3Config


@dataclass(frozen=True)
class ExpertWeights:
    router: torch.Tensor
    correction: torch.Tensor
    experts: object
    shared: MLPWeights
    vision_correction: torch.Tensor | None = None
    # TP3 merged expert bank: a parked layer keeps its hottest experts in the VRAM rows of one mixed VRAM/host bank
    # (hot rows first). ``hot_map`` translates a global router ID to its bank slot (indexed by expert ID, a
    # permutation of the slots); bank slots < ``hot_rows`` are VRAM rows, the rest pinned host memory.
    # None = one bank in router-ID order (resident, or parked whole).
    hot_map: torch.Tensor | None = None
    hot_rows: int | None = None


@dataclass(frozen=True)
class LayerWeights:
    attention: dict
    experts: ExpertWeights
    residual: dict
    engram: dict | None


@dataclass(frozen=True)
class DSparkWeights:
    layers: tuple[LayerWeights, ...]
    main_projection: object
    context_kv: object
    main_norm: torch.Tensor
    norm: torch.Tensor
    markov_embedding: torch.Tensor
    markov_head: object
    confidence_head: torch.Tensor | None = None
    draft_head: object | None = None
    draft_columns: torch.Tensor | None = None  # TP3 tp3.draft_vocab: local vocabulary columns of draft_head's rows

    @classmethod
    def load(cls, loader, *, confidence=False, vocabulary_parallel_sampling=False, markov_head_quantization=None,
             park_markov_embedding=None):
        config = loader.config.draft
        last = f"mtp.{config.layers - 1}"
        markov_name = last + ".markov_head.head.weight"
        temporary = markov_head_quantization in ("nvfp4", "fp8")
        layers = tuple(loader.load_layer(i, draft=True) for i in range(config.layers))
        main_projection = loader.load_linear("mtp.0.main_proj")
        context_kv = loader.load_linear_group(tuple(f"mtp.{i}.attn.wkv" for i in range(config.layers)))
        main_norm, norm = loader.load_tensor("mtp.0.main_norm.weight"), loader.load_tensor(last + ".norm.weight")
        embed_name = last + ".markov_head.embed.weight"
        if park_markov_embedding is None:
            markov_embedding = loader.load_tensor(embed_name)
        else:
            # TP3 tp3.markov_embed_host: load into a pool of its own, copy into device-visible host memory, drop the
            # device copy inside the pool (as the parked token embedding, ModelWeights.load).
            from ignition.components.weights import _park_on_host
            with _parking_pool():
                markov_embedding = _park_on_host(park_markov_embedding, embed_name,
                                                 loader.load_tensor(embed_name, temporary=True))
        return cls(layers, main_projection, context_kv, main_norm, norm, markov_embedding,
            loader.load_padded_tensor(markov_name, rows=loader.shard.padded_partition(129280, alignment=64),
                                      temporary=temporary) if vocabulary_parallel_sampling
            else loader.load_tensor(markov_name, temporary=temporary),
            loader.load_tensor(last + ".confidence_head.proj.weight") if confidence else None)


@dataclass(frozen=True)
class ModelWeights:
    config: ModelConfig
    shard: object
    embedding: torch.Tensor
    norm: torch.Tensor
    head: torch.Tensor
    layers: tuple[LayerWeights, ...]
    draft: DSparkWeights | None
    allocated_bytes: int
    vision: dict | None = None

    @classmethod
    def load(cls, manifest, device, *, shard, progress=None, dspark=False, confidence=False,
             expert_scale_compression=None, draft_head_quantization=None, markov_head_quantization=None,
             vocabulary_parallel_sampling=False, host_allocator=None, tp3=None):
        tp3 = TP3Config() if tp3 is None else tp3
        config = ModelConfig.load(manifest.metadata_path, mini=tp3.mini)
        before = allocated_tensor_bytes(device)
        # TP3 host parking: a parked tensor is copied into device-visible host memory after loading. The persistent
        # arena never releases space, so a parked tensor loads into temporary (released) device memory instead.
        park_embedding = tp3.embed_host and host_allocator is not None
        park_vision = tp3.vision_host and host_allocator is not None
        with DeepSeekWeightLoader(manifest, device, shard=shard, progress=progress, dspark=dspark,
                                 expert_scale_compression=expert_scale_compression,
                                 host_allocator=host_allocator, tp3=tp3) as loader:
            if park_embedding:
                # TP3: the 1.23 GiB token embedding is replicated on every rank but each step gathers only its
                # tokens' rows (<= 48 in decode, one prefill chunk otherwise); read it from device-visible host
                # memory and give the VRAM to hot experts / KV.
                from ignition.components.weights import _park_on_host
                with _parking_pool():
                    embedding = _park_on_host(host_allocator, "embed.weight",
                                              loader.load_tensor("embed.weight", temporary=True))
                norm = loader.load_tensor("norm.weight")
            else:
                embedding, norm = loader.load_tensors(("embed.weight", "norm.weight"))
            head = loader.load_padded_tensor("head.weight", rows=shard.padded_partition(129280, alignment=64))
            layers = tuple(loader.load_layer(i) for i in range(config.layers))
            draft = DSparkWeights.load(loader, confidence=confidence,
                vocabulary_parallel_sampling=vocabulary_parallel_sampling,
                markov_head_quantization=markov_head_quantization,
                park_markov_embedding=host_allocator if tp3.markov_embed_host else None) if dspark else None
            if draft is not None:
                if draft_head_quantization == "nvfp4" and tp3.draft_vocab is not None:
                    draft = replace(draft, **trimmed_draft_head(head, tp3.draft_vocab, shard,
                                                                vocabulary_parallel_sampling))
                elif draft_head_quantization == "nvfp4":
                    draft = replace(draft, draft_head=nvfp4_draft_head(head))
                elif draft_head_quantization == "mxfp8":
                    draft = replace(draft, draft_head=mxfp8_draft_head(head))
                if markov_head_quantization == "nvfp4":
                    draft = replace(draft, markov_head=nvfp4_draft_head(draft.markov_head))
                elif markov_head_quantization == "fp8":
                    draft = replace(draft, markov_head=fp8_markov_head(draft.markov_head))
            with _parking_pool() if park_vision else nullcontext():
                vision = ({name: loader.load_tensor(name, temporary=park_vision) for name in loader._sources
                           if name.startswith(("vision.", "aligner.", "image_"))}
                          if config.vision is not None else None)
                if vision and park_vision:
                    # TP3: the vision tower (~0.8 GiB, replicated) runs only for image prompts; its projections
                    # consume these tensors as-is, so read them from device-visible host memory and give the VRAM to
                    # KV (images ~3 s slower each unless tp3.vision_stage copies them into VRAM per block, PreparedVision).
                    from ignition.components.weights import _park_on_host
                    vision = {name: _park_on_host(host_allocator, name, tensor) for name, tensor in vision.items()}
        return cls(config, shard, embedding, norm, head, layers, draft,
                   allocated_tensor_bytes(device) - before, vision)


def draft_vocab_columns(tokens, world, rank, *, k=0, vocab=129280, alignment=64, multiple=128):
    """TP3 tp3.draft_vocab: this rank's sorted local columns of the draft vocabulary (the listed tokens in this rank's
    shard). Every rank keeps the same count, max(largest share, k / world) rounded up to ``multiple``; shares below it
    are filled with the rank's lowest unused real columns, so every rank prepares the same projection shape."""
    width = -(-vocab // (world * alignment)) * alignment
    shares = [sorted({t - r * width for t in tokens if r * width <= t < min((r + 1) * width, vocab)}) for r in range(world)]
    count = min(width, -(-max(max(len(share) for share in shares), -(-k // world)) // multiple) * multiple)
    real = min(width, vocab - rank * width)
    columns, have = list(shares[rank]), set(shares[rank])
    filler = (c for c in range(width) if c not in have)
    while len(columns) < count:
        column = next(filler)
        if column >= real:
            raise ValueError("draft vocabulary: a rank has fewer real columns than the trimmed width")
        columns.append(column)
    return sorted(columns)


def trimmed_draft_head(head, vocabulary_file, shard, vocabulary_parallel_sampling):
    """NVFP4 draft head of the draft vocabulary's rows of this rank's head shard + their local column ids."""
    import json

    if not vocabulary_parallel_sampling:
        raise ValueError("tp3.draft_vocab needs vocabulary-parallel sampling (the trimmed head is a rank-local projection)")
    record = json.loads(vocabulary_file.read_text())
    tokens, k = record["tokens"], int(record.get("k", 0))
    if not tokens or any(type(t) is not int or not 0 <= t < 129280 for t in tokens):
        raise ValueError(f"draft vocabulary {vocabulary_file.name}: token ids must be integers in [0, 129280)")
    columns = draft_vocab_columns(tokens, shard.world_size, shard.rank, k=k)
    if head.shape[0] != -(-129280 // (shard.world_size * 64)) * 64:
        raise ValueError("draft vocabulary: the head shard is not the padded vocabulary partition")
    index = persistent_empty((len(columns),), torch.int64, head.device)
    index.copy_(torch.tensor(columns, dtype=torch.int64))
    with _parking_pool():
        draft_head = nvfp4_draft_head(head.index_select(0, index).contiguous())
    return {"draft_head": draft_head, "draft_columns": index}


def _parking_pool():
    """TP3: a pool of its own for device copies that are parked on the host right after loading.

    The load pool stays active for the whole load, and ``torch.cuda.empty_cache`` does not release an active
    pool's cached segments. A freed parked copy (the 1.23 GiB embedding) therefore stayed cached there, a long-lived
    allocation (the routed experts' scale scratch) was carved from it, and the whole segment stayed mapped through
    tuning (G1 2026-11-14: 1.06 GiB/rank, graph catalog out of memory). This pool is destroyed on exit, so its
    segments return to the device; the parked tensors must be dropped inside it."""
    storage = active_core_storage()
    return nullcontext() if storage is None else storage.transient()


def vision_host_parked(vision: dict, host_regions) -> bool:
    """Whether every vision/aligner tensor lives in one of the package's mapped host regions (`_park_on_host`)."""
    spans = [(region.address, region.address + region.extent) for region in host_regions if not region.closed]
    tensors = [tensor for name, tensor in vision.items() if name.startswith(("vision.", "aligner."))]
    return bool(tensors) and all(any(start <= tensor.data_ptr() < end for start, end in spans) for tensor in tensors)


def hot_expert_ids(layer: int, experts: int, tp3: TP3Config) -> list | None:
    """TP3 expert promotion: the ``tp3.hot_experts`` hottest experts of a parked layer, by the route counts in
    ``tp3.expert_counts_dir``/expert-routes-rank0-layer{L}.json (vLLM route-monitor format). None = park all."""
    hot = tp3.hot_experts
    if hot <= 0:
        return None
    if hot >= experts:
        return list(range(experts))
    import json
    if tp3.expert_counts_dir is None:
        raise ValueError(f"TP3 config: hot_experts={hot} for layer {layer} needs expert_counts_dir")
    path = tp3.expert_counts_dir / f"expert-routes-rank0-layer{layer}.json"
    counts = json.loads(path.read_text())["counts"]
    if len(counts) != experts:
        raise ValueError(f"{path}: expected {experts} expert counts, got {len(counts)}")
    order = sorted(range(experts), key=lambda e: (-counts[e], e))
    return sorted(order[:hot])


# FLARE_DS41_EXL3_WOA_SINGLE=1 (exp/ds41-exl3-final-woa, default off): keep one WO-A layout. At TP2 the per-rank WO-A
# is (4 groups, 1024, 4096), a shape for which b12x also packs a tile-major copy (values_tiled, 16 MiB per attention
# layer incl. the drafter's); the WO plan (PreparedGroupedOutputProjection) never declares wo_a_tiled, so the decode
# WO-A GEMM reads `values` either way and the copy is retained but unread. On = the copy is not kept (export-baked:
# the packed storage is smaller, the graphs the same). TP3's padded 3-group WO-A never gets a copy (no-op there).
def woa_single() -> bool:
    value = os.environ.get("FLARE_DS41_EXL3_WOA_SINGLE", "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"FLARE_DS41_EXL3_WOA_SINGLE={value!r}: expected 0 or 1")
    return value == "1"


class DeepSeekWeightLoader(BlockFP8WeightLoader):
    def __init__(self, manifest, device, *, shard, progress=None, prefixes=None, dspark=False,
                 expert_scale_compression=None, host_allocator=None, tp3=None):
        if expert_scale_compression not in (None, "csf"):
            raise ValueError("DeepSeek V4.1 expert scale compression must be None or 'csf'")
        self.dspark = dspark
        self.expert_scale_compression = expert_scale_compression
        self.tp3 = TP3Config() if tp3 is None else tp3
        super().__init__(manifest, device, progress=progress, prefixes=prefixes)
        self.config = ModelConfig.load(manifest.metadata_path, mini=self.tp3.mini)
        self.shard = shard
        self._expert_scale_scratch = {}
        # TP3 expert parking: routed-expert banks of these layers live (wholly, or past their hot rows) in
        # device-visible host memory; without a host allocator nothing is parked.
        self._host_allocator = host_allocator
        self._host_layers = self.tp3.host_expert_layers if host_allocator is not None else frozenset()

    def load_experts(self, prefixes, interval, *, host=False):
        text = self.config.text
        # TP3 BRRR (tp3.brrr_native_scales): the fused layer kernel reads native E8M0 expert scales, so its layers
        # skip CSF.
        native = (self.tp3.brrr_native_scales and prefixes and prefixes[0].startswith("layers.")
                  and int(prefixes[0].split(".")[1]) in self.tp3.brrr_layers)
        return self.load_mxfp4_experts(prefixes, hidden_size=text["hidden_size"],
            intermediate_size=text["moe_intermediate_size"], intermediate_range=interval,
            swiglu_limit=text["swiglu_limit"], scale_name="scale", projection_names=("w3", "w1", "w2"),
            scale_compression=None if native else self.expert_scale_compression,
            scale_scratch=None if native else self._expert_scale_scratch,
            host_allocator=self._host_allocator if host else None)

    def _selected_names(self):
        return {name for name in super()._selected_names()
                if (name.startswith("layers.") or (self.dspark and name.startswith("mtp."))
                    or name.startswith(("vision.", "aligner.", "image_"))
                    or name in ("embed.weight", "norm.weight", "head.weight"))
                and ".engram.embed." not in name}

    def load_layer(self, layer, *, draft=False):
        prefix = f"mtp.{layer}" if draft else f"layers.{layer}"
        text, shard = self.config.text, self.shard
        attn = prefix + ".attn"
        first, last = shard.padded_partition(8)
        heads = first * 8 * 512, last * 8 * 512
        groups = first * 1024, last * 1024
        result = {
            "qkv_a": self.load_linear_group((attn + ".wq_a", attn + ".wkv")),
            "q_b": self.load_linear_group((attn + ".wq_b",), row_ranges=(heads,), pad=True),
            "output": self.load_attention_output(attn, groups, pad=True, tiled=not woa_single()),
            "q_norm": self.load_tensor(attn + ".q_norm.weight"),
            "kv_norm": self.load_tensor(attn + ".kv_norm.weight"),
            "sink": self.load_padded_tensor(attn + ".attn_sink", rows=(first * 8, last * 8)),
        }
        if not draft and layer in text["kv_source_layer_ids"]:
            ratio = text["compress_ratios"][layer]
            result["compressor"] = {
                "values": self.load_tensor(attn + ".compressor.wkv.weight"),
                "norm": self.load_tensor(attn + ".compressor.norm.weight", dtype=torch.float32),
            }
            if ratio == 2:
                result["compressor"]["gates"] = self.load_tensor(attn + ".compressor.wgate.weight")
            result["index_k"] = self.load_tensor(attn + ".indexer.wk.weight")
            result["index_norm"] = self.load_tensor(attn + ".indexer.k_norm.weight")
        if not draft and layer in text["index_source_layer_ids"]:
            # Replicated index heads avoid a context-sized TP score reduction.
            result["index_q"] = self.load_linear(attn + ".indexer.wq_b")
            result["index_weights"] = self.load_tensor(attn + ".indexer.weights_proj.weight")
        ffn = prefix + ".ffn"
        interval = shard.partition(text["moe_intermediate_size"], alignment=32)
        count = self.config.draft.experts if draft else text["n_routed_experts"]
        parked = not draft and layer in self._host_layers
        hot = hot_expert_ids(layer, count, self.tp3) if parked else None
        hot_map = hot_rows = None
        if getattr(self.config, "exl3", False):
            # exp/ds41-exl3: stacked EXL3 banks, rank slices of the intermediate axis; parked layers keep their hot
            # experts in the VRAM rows of mixed banks (hot first, like the MXFP4 merged bank).
            from .exl3.experts import load_exl3_experts
            order = list(range(count))
            mixed = None
            if parked:
                storage = getattr(self._host_allocator, "__self__", None)
                if storage is None or not hasattr(storage, "mixed_allocate"):
                    raise ValueError("EXL3 parking needs CoreStorage.host_allocate as the host allocator")
                from .exl3.experts import layer_hot_count
                per_layer = layer_hot_count(layer, self.tp3.hot_experts)
                if per_layer != self.tp3.hot_experts:          # FLARE_DS41_EXL3_HOT_LAYERS override for this layer
                    from dataclasses import replace
                    hot = hot_expert_ids(layer, count, replace(self.tp3, hot_experts=per_layer))
                hot = [] if hot is None else list(hot)
                hot_set = set(hot)
                order = hot + [e for e in range(count) if e not in hot_set]
                mixed = storage.mixed_allocate
                slots = torch.empty((count,), dtype=torch.int32)
                slots[order] = torch.arange(count, dtype=torch.int32)
                hot_map = persistent_copy(slots.to(self.device))
                hot_rows = len(hot)
            experts = load_exl3_experts(self, [f"{ffn}.experts.{i}" for i in order], rank=shard.rank,
                                        world=shard.world_size, hot_rows=hot_rows, mixed_allocate=mixed,
                                        codebook="mcg" if draft else "mul1")
        elif hot is None or len(hot) == count:
            experts = self.load_experts([f"{ffn}.experts.{i}" for i in range(count)], interval,
                                        host=parked and hot is None)
        else:
            # Merged bank (TP3Config requires merged_bank with hot experts): one bank, hot experts first, VRAM rows
            # for hot and host rows for cold, so each layer keeps one MoE call.
            storage = getattr(self._host_allocator, "__self__", None)
            if storage is None or not hasattr(storage, "mixed_allocate"):
                raise ValueError("TP3 merged expert banks need CoreStorage.host_allocate as the host allocator")
            hot_set = set(hot)
            order = list(hot) + [e for e in range(count) if e not in hot_set]
            host_allocator = self._host_allocator
            self._host_allocator = lambda name, shape, dtype: storage.mixed_allocate(
                name, shape, dtype, device_rows=len(hot))
            try:
                experts = self.load_experts([f"{ffn}.experts.{i}" for i in order], interval, host=True)
            finally:
                self._host_allocator = host_allocator
            slots = torch.empty((count,), dtype=torch.int32)
            slots[order] = torch.arange(count, dtype=torch.int32)
            hot_map = persistent_copy(slots.to(self.device))  # a forward constant: packed with the final weights
            hot_rows = len(hot)
        shared = MLPWeights(self.load_linear_group((ffn + ".shared_experts.w1", ffn + ".shared_experts.w3"),
            row_ranges=(interval, interval)), self.load_linear(ffn + ".shared_experts.w2", columns=interval))
        routed = ExpertWeights(self.load_tensor(ffn + ".gate.weight"),
            self.load_tensor(ffn + ".gate.bias"), experts, shared,
            self.load_tensor(ffn + ".gate.bias_vl") if self.config.vision is not None else None,
            hot_map=hot_map, hot_rows=hot_rows)
        residual = {name: self.load_tensor(prefix + "." + name) for name in (
            "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn", "hc_ffn_base", "hc_ffn_scale",
            "attn_norm.weight", "ffn_norm.weight")}
        engram = None
        if not draft and layer in text["engram_layer_ids"]:
            engram = {"projection": self.load_linear_columns(prefix + ".engram.wkv",
                ((head * 256, (head + 1) * 256) for head in range(shard.rank, 24, shard.world_size))),
                "weight": persistent_copy(self.load_tensor(prefix + ".engram.q_weight", dtype=torch.float32, temporary=True)
                          * self.load_tensor(prefix + ".engram.k_weight", dtype=torch.float32, temporary=True))}
        return LayerWeights(result, routed, residual, engram)
