# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel DeepSeek text stack with lagged hyperconnections."""

from dataclasses import dataclass, fields, is_dataclass, replace

import torch

from ignition.components.collectives import PreparedAllReduce, PreparedChunkedAllReduce
from ignition.components.embedding import PreparedEmbedding
from ignition.components.expert_overlap import ConcurrentSharedExpert
from ignition.components.linear import UnquantizedProjection
from ignition.components.mhc import MHCScratch, MHCWeights, PreparedMHC, PreparedMHCPost
from ignition.components.mlp import MLPScratch, PreparedMLP
from ignition.components.moe import PreparedRoutedExperts, PreparedRouter, RouterScratch
from ignition.components.parallel import TensorParallelShard
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tuning import register_finalizer

from .attention import (
    AttentionIndices,
    AttentionScratch,
    PreparedAttention,
    PreparedGlobalKV,
    Rotary,
    normalize,
)
from .decoder_prefill import PreparedDecoderSuffix
from .engram import EngramScratch, PreparedEngram, PreparedTableRequests
from .index_row_split import prefill_split_context
from .host_embedding import HostRowGather
from .probe import Probe
from .routecapture import RouteCapture
from .routecount import RouteCounter
from .steering import steer_attention, steer_shared_expert
from .tp3_config import TP3Config


# TP3 BRRR: each rank decides per layer and graph whether the fused tail runs, from its own TP3Config (brrr_layers,
# brrr_parked_max_rows, host_expert_layers, hot_experts, ...) and the graph's row count. The kernel's in-kernel
# all-reduce needs every rank to make the same decision: a rank that runs Flare's path while a peer waits in the kernel
# deadlocks. PreparedLayer records its decisions here and the bootstrap compares them across ranks
# (check_brrr_decisions) in plain construction code, before any capture.
_BRRR_DECISIONS = []
_BRRR_CHECKED = 0
_BRRR_DECISION_FIELDS = ("graph", "layer", "mode", "rows", "brrr", "gated")


def record_brrr_decision(name, index, mode, rows, uses_brrr, gated=False):
    _BRRR_DECISIONS.append((str(name), int(index), str(mode), int(rows), bool(uses_brrr), bool(gated)))


def check_brrr_decisions(group):
    """All-gather the BRRR decisions recorded since the previous check and raise when ranks disagree.

    Collective over ``group`` (the Gloo group create_engine used): every rank must call it at the same point, from plain
    construction code, never inside a traced or captured forward. No-op without a BRRR engine (no engine, no kernel to
    deadlock) or without a group. Returns the number of decisions compared.
    """
    global _BRRR_CHECKED
    from ignition.recipes.deepseek41.brrr import engine

    if group is None or engine() is None:
        return 0
    import torch.distributed as dist

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("check_brrr_decisions must run outside CUDA graph capture")
    pending, _BRRR_CHECKED = list(_BRRR_DECISIONS[_BRRR_CHECKED:]), len(_BRRR_DECISIONS)
    gathered = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, pending, group=group)
    root = gathered[0]
    for rank, other in enumerate(gathered[1:], 1):
        if other == root:
            continue
        if len(other) != len(root):
            detail = f"rank 0 recorded {len(root)} layer decisions, rank {rank} recorded {len(other)}"
        else:
            at = next(i for i, (a, b) in enumerate(zip(root, other)) if a != b)
            detail = (f"rank 0 decided {dict(zip(_BRRR_DECISION_FIELDS, root[at]))}, "
                      f"rank {rank} decided {dict(zip(_BRRR_DECISION_FIELDS, other[at]))}")
        raise RuntimeError("BRRR layer decisions differ between ranks (the fused kernel's all-reduce would deadlock): "
                           f"{detail}. Every rank must receive the same TP3Config (brrr_*, host_expert_layers, "
                           "hot_experts) and graph catalog.")
    return len(pending)


@dataclass(frozen=True)
class Scratch:
    """BF16 decoder activations and FP32 target vocabulary logits."""

    logits_dtype = torch.float32

    embedding: torch.Tensor
    residual: torch.Tensor
    engram_residual: torch.Tensor
    engram: EngramScratch
    pre_mix: torch.Tensor
    attn_mix: torch.Tensor
    attention_hc: MHCScratch
    ffn_hc: MHCScratch
    attention: AttentionScratch
    attention_output: torch.Tensor
    shared: MLPScratch
    router_logits: torch.Tensor
    router: RouterScratch
    routed: torch.Tensor
    mlp_partial: torch.Tensor
    mlp_output: torch.Tensor
    hidden: torch.Tensor
    local_logits: torch.Tensor
    logits: torch.Tensor

    def rows(self, tokens, samples):
        def select(value, count):
            if isinstance(value, torch.Tensor):
                return value[:count]
            if is_dataclass(value):
                return type(value)(*(select(getattr(value, field.name), count) for field in fields(value)))
            raise TypeError("decoder scratch must contain tensors and tensor dataclasses")
        return type(self)(*(select(getattr(self, field.name), samples if field.name in
            ("hidden", "local_logits", "logits") else tokens) for field in fields(self)))

    @classmethod
    def allocate(cls, tokens, samples, world, device, *, experts=384, top_k=6,
                 candidate_tokens=None, engram_chunk_rows=0, share_engram_residual=False):
        """Declaration operands (the traced forwards allocate their own values). TP3 fork switches:
        ``engram_chunk_rows`` sizes the Engram projection operands to one chunk (PreparedEngram chunks its
        projection, all-reduce and mix accordingly); ``share_engram_residual`` declares ``engram_residual`` on
        ``ffn_hc.residual``'s storage. The latter only trims preparation memory: in the functional forwards the
        Engram output is a value Inductor already places on dead storage (S5 report, 2026-11-14)."""
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        residual = zeros(tokens, 4, 5120)
        pre_mix, attn_mix = zeros(tokens, 4, dtype=torch.float32), zeros(tokens, 4, dtype=torch.float32)
        attention_hc = MHCScratch.allocate(tokens, 5120, device)
        ffn_hc = MHCScratch.allocate(tokens, 5120, device)
        # No declared operator takes both (the Engram mix writes engram_residual from residual; the Engram layer's
        # attention mHC reads it), but tuning calls may clobber each other's contents through the alias. That is
        # harmless only because no kernel touching this storage is data-dependent in its control flow: a future
        # data-dependent consumer of ffn_hc.residual or engram_residual must re-check this alias.
        engram_residual = ffn_hc.residual if share_engram_residual else zeros(tokens, 4, 5120)
        attention = AttentionScratch.allocate(tokens, world, device, candidate_tokens=candidate_tokens)
        attention_output, mlp_output = zeros(tokens, 5120), zeros(tokens, 5120)
        _, vocabulary = TensorParallelShard(0, world).padded_partition(129280, alignment=64)
        hidden = zeros(samples, 5120)
        local_logits = zeros(samples, vocabulary, dtype=cls.logits_dtype)
        logits = zeros(samples, 129280, dtype=cls.logits_dtype)
        embedding = zeros(tokens, 5120)
        engram = EngramScratch.allocate(tokens, device, world=world, chunk_rows=engram_chunk_rows)
        shared = MLPScratch.allocate(tokens, 5120, 2304 // world, device)
        router_logits = zeros(tokens, experts, dtype=torch.float32)
        router = RouterScratch.allocate(tokens, device, experts=experts, top_k=top_k)
        routed, partial = zeros(tokens, 5120), zeros(tokens, 5120)
        return cls(embedding, residual, engram_residual, engram, pre_mix, attn_mix, attention_hc, ffn_hc,
            attention, attention_output, shared, router_logits, router, routed, partial, mlp_output,
            hidden, local_logits, logits)



class DraftScratch(Scratch):
    """Retain BF16 proposal logits independently of FP32 target logits."""

    logits_dtype = torch.bfloat16


def _mhc_weights(weights, kind, norm):
    residual = weights.residual
    return MHCWeights(residual[f"hc_{kind}_fn"], residual[f"hc_{kind}_scale"],
                      residual[f"hc_{kind}_base"], residual[norm + ".weight"])


def initial_mix(rows, device, *, functional=False):
    """mHC pre-mix of the first layer: [rows, 4] float32 with column 0 one and the rest zero.

    Base: a zero fill plus an in-place fill of column 0 (a Triton fill and an eager kernel per forward).
    ``functional`` (TP3 glue_pass2): the same values from one pointwise expression (one Triton kernel).
    """
    if not functional:
        mix = torch.zeros((rows, 4), dtype=torch.float32, device=device)
        mix[:, 0].fill_(1)
        return mix
    return (torch.arange(4, device=device) == 0).to(torch.float32).expand(rows, 4).contiguous()


class PreparedDecoderBoundary:
    """Produce complete global KV, then gather the decoder's bounded replay rows."""

    def __init__(self, index, weights, source, target, packing, pool, rotary, *, carry,
                 session, name, workspace, window_metadata, source_indices):
        self.packing = packing
        incoming = ((source.mlp_output, source.ffn_hc.residual, source.ffn_hc.post, source.ffn_hc.comb)
                    if carry else (source.residual,))
        text = pool.config.text
        self.mhc = PreparedMHC(incoming, _mhc_weights(weights, "attn", "attn_norm"), source.attention_hc,
            session=session, name=name + ".attn_hc", workspace=workspace,
            pre_mix=source.pre_mix, pre_out=source.attn_mix, rms_eps=text["rms_norm_eps"],
            norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"], sinkhorn_iters=text["hc_sinkhorn_iters"]).functional()
        self.global_kv = PreparedGlobalKV(index, weights.attention, source.attention_hc.normalized, source.attention,
            source_indices.metadata, pool, rotary, session=session, name=name + ".global_kv",
            indices=source_indices)

    def __call__(self, incoming, pre_mix, metadata, indices, indexer):
        suffix = self.packing.begin(metadata)
        # Lagged mHC writes every [tokens, 4] coefficient of pre_out (b12x mhc finalize / lagged producer, one
        # program per row; shapes are validated equal to pre_mix), so the next-layer mix needs no zero fill.
        attn_mix = torch.empty_like(pre_mix)
        residual, post, comb, normalized = self.mhc(*incoming, pre_mix=pre_mix, pre_out=attn_mix)
        self.global_kv(normalized, indexer, metadata, indices.compressed(self.global_kv.ratio))
        return suffix, tuple(suffix.gather(value) for value in (residual, post, comb, normalized)), suffix.gather(attn_mix)

    def global_only(self, incoming, pre_mix, metadata, indices, indexer):
        """TP3 encoder-only prefill (``PreparedDecoder(encoder_only=True)``): the same mHC and global KV on every row,
        without packing or gathering the decoder suffix (nothing after the boundary runs)."""
        attn_mix = torch.empty_like(pre_mix)
        _, _, _, normalized = self.mhc(*incoming, pre_mix=pre_mix, pre_out=attn_mix)
        self.global_kv(normalized, indexer, metadata, indices.compressed(self.global_kv.ratio))


class PreparedLayer:
    # TP3 fork attributes; class defaults keep the upstream behaviour (one router-ID-ordered bank, no debug hooks).
    hot_map = None
    brrr = None
    no_engram = False
    # TP3 tp3.engram_side_layers: the NEXT layer's PreparedEngram whose hidden-state-free part this layer runs on a
    # side stream (started before this layer's attention, joined before its tail); None = none (base).
    engram_side = None
    probe, probe_index, probe_mode = Probe(), None, None
    route_counter, route_experts = RouteCounter(enabled=False), 0
    # Debug-serve only (FLARE_DS41_EXL3_ROUTE_CAPTURE, routecapture.py): per-round routes + prefetch predictors.
    # route_capture_next = (router weight, correction bias) of the next layer, set by PreparedDecoder; None = last.
    route_capture, route_capture_own, route_capture_next, route_capture_score = RouteCapture(enabled=False), None, None, None

    def __init__(self, index, weights, tables, scratch, metadata, pool, rotary, *, shard, mode,
                 runtime, coordinator, session, name, workspace, carry, defer_post,
                 boundary=None, replay_start=None, window_staging=None, mla_compute_mode="auto",
                 window_metadata=None, examples=None, tp3=None, probe=None, index_split=None, steering=None):
        tp3 = TP3Config() if tp3 is None else tp3  # TP3 fork switches; the decoder shares one instance per graph
        s, text = scratch, pool.config.text
        self.scratch, self.carry = s, carry
        # Debug (TP3Config.probe / no_engram, flare#193): per-layer residual probes; Engram handshake kept but its
        # contribution to the residual discarded.
        self.probe = Probe(tp3.probe) if probe is None else probe
        self.probe_index, self.probe_mode = index, mode
        self.probe.allocate(s.residual.device)  # eager, outside traced forwards (no-op unless enabled)
        self.route_capture = RouteCapture()
        if self.route_capture.active(mode):
            self.route_capture_own = (weights.experts.router, weights.experts.correction)
            self.route_capture_score = text["scoring_func"]
            self.route_capture.allocate(s.residual.device, mode, index, weights.experts.hot_map, weights.experts.hot_rows)
        # Debug-serve only (FLARE_DS41_ROUTE_COUNTS, routecount.py): live-row route counts per layer; off = untraced.
        self.route_counter = RouteCounter()
        self.route_experts = text["n_routed_experts"] if self.route_counter.enabled else 0
        if mode in ("prefill", "decode", "verify"):
            self.route_counter.allocate(s.residual.device, self.route_experts)
        self.no_engram = tp3.no_engram
        self.metadata = metadata
        self.steering = steering
        self.image_mask = (torch.zeros(metadata.token_ids.shape, dtype=torch.bool, device=metadata.token_ids.device)
                           if mode == "prefill" and weights.experts.vision_correction is not None else None)
        self.boundary = boundary
        self.engram = None
        if weights.engram is not None:
            self.engram = PreparedEngram(index, weights.engram, tables, s.residual, s.engram_residual,
                metadata, pool,
                scratch=s.engram, session=session, name=name + ".engram", workspace=workspace,
                runtime=runtime, coordinator=coordinator, examples=examples, max_part_rows=tp3.engram_part_rows)
        source = s.engram_residual if self.engram is not None else s.residual
        options = dict(rms_eps=text["rms_norm_eps"], norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"],
                       sinkhorn_iters=text["hc_sinkhorn_iters"])
        incoming = ((s.mlp_output, s.ffn_hc.residual, s.ffn_hc.post, s.ffn_hc.comb) if carry else (source,))
        self.attention_hc = None if boundary is not None else PreparedMHC(
            incoming, _mhc_weights(weights, "attn", "attn_norm"), s.attention_hc,
            session=session, name=name + ".attn_hc", workspace=workspace, pre_mix=s.pre_mix, pre_out=s.attn_mix, **options)
        self.attention = PreparedAttention(index, weights.attention, s.attention_hc.normalized, s.attention,
            metadata, pool, rotary, mode=mode, world=shard.world_size,
            session=session, name=name + ".attention", workspace=workspace, replay_start=replay_start,
            window_staging=window_staging, mla_compute_mode=mla_compute_mode,
            window_metadata=window_metadata, examples=examples, index_side_stream=tp3.index_side_stream,
            glue_pass2=tp3.glue_pass2 and mode != "prefill",
            pin_small_qproj=tp3.pin_small_qproj and mode != "prefill", index_split=index_split)
        # Chunked: a runtime sized for fewer rows than a prefill chunk (TP3Config.ar_rows) still reduces prefill
        # rows, in row chunks the runtime accepts; one part (a plain reduction) whenever the rows fit.
        # TP3Config.ar_part_rows caps the parts and cuts at the cap (0 = halving, base).
        self.attention_reduce = PreparedChunkedAllReduce(s.attention.output, s.attention_output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".attention_reduce",
            max_part_rows=tp3.ar_part_rows)
        self.ffn_hc = PreparedMHC((s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb),
            _mhc_weights(weights, "ffn", "ffn_norm"), s.ffn_hc, session=session, name=name + ".ffn_hc", workspace=workspace,
            pre_mix=s.attn_mix, pre_out=s.pre_mix, **options)
        self.router_projection = UnquantizedProjection(s.ffn_hc.normalized, weights.experts.router, s.router_logits,
            session=session, name=name + ".router")
        self.router = PreparedRouter(s.router_logits, s.router, session=session, name=name + ".topk",
            score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
            routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction,
            image_correction_bias=weights.experts.vision_correction if self.image_mask is not None else None,
            image_mask=self.image_mask)
        # TP3 merged bank: router IDs map to the bank's hot-first slots inside the forward (``_run``); the
        # declaration example has the router IDs' shape and dtype.
        self.hot_map = getattr(weights.experts, "hot_map", None)
        from .exl3.experts import Exl3Experts, PreparedExl3Experts
        exl3 = isinstance(weights.experts.experts, Exl3Experts)  # exp/ds41-exl3: diffbot EXL3 pack
        expert_class = PreparedExl3Experts if exl3 else PreparedRoutedExperts
        self.experts = expert_class(s.ffn_hc.normalized, weights.experts.experts, s.router.ids,
            s.router.weights, s.routed, session=session, name=name + ".experts", workspace=workspace,
            router_plan=self.router.plan)
        self.shared = PreparedMLP(s.ffn_hc.normalized, weights.experts.shared, s.shared.gate_up,
            s.shared.intermediate, s.shared.output, session=session, name=name + ".shared", workspace=workspace,
            swiglu_limit=text["swiglu_limit"], round_silu=False, activation_mode="auto")
        self.concurrent_shared = None
        self.shared_joins_before_experts = False
        # exp/ds41-exl3: no shared-expert overlap with the xmoe kernels (their grid barriers need every block resident).
        # FLARE_DS41_EXL3_SHARED_SIDE=1: the shared expert overlaps the router projection + top-k only and joins before
        # the xmoe call (ds41_exl3_moe_after orders it after the join).
        from .exl3.experts import shared_side
        if (exl3 and shared_side() and mode in ("decode", "verify", "draft")
                and s.ffn_hc.normalized.shape[0] <= 128 and workspace.overlap):
            self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                parallel_plans=(self.router_projection.plan, self.router.plan),
                session=session, workspace=workspace,
                defer_launch=s.ffn_hc.normalized.shape[0] > 4)
            self.shared_joins_before_experts = True
        # exp/ds41-exl3-final-merges (exl3/merges.py), each default off: MERGE_TOPK = the hot_map lookup folded into
        # the fused prep, which runs as its own op before the shared join; MERGE_EPILOGUE = routed + shared folded
        # into the MoE op's last copy (needs the early join; off while the probe records routed alone).
        self.exl3_merge_prep = exl3 and getattr(self.experts, "merge_prep", False)
        from .exl3 import merges
        self.exl3_merge_epilogue = (exl3 and merges.MERGE_EPILOGUE and self.shared_joins_before_experts
                                    and not self.probe.enabled)
        if (mode in ("decode", "verify", "draft") and s.ffn_hc.normalized.shape[0] <= 128 and workspace.overlap
                and not exl3):
            self.concurrent_shared = ConcurrentSharedExpert(self.shared,
                parallel_plans=(self.router_projection.plan, self.router.plan, self.experts.plan),
                session=session, workspace=workspace,
                defer_launch=s.ffn_hc.normalized.shape[0] > 4)
        self.mlp_reduce = PreparedChunkedAllReduce(s.mlp_partial, s.mlp_output, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".mlp_reduce",
            max_part_rows=tp3.ar_part_rows)
        self.post = None if defer_post else PreparedMHCPost(s.mlp_output, s.ffn_hc.residual,
            s.ffn_hc.post, s.ffn_hc.comb, s.residual, session=session, name=name + ".post")
        # TP3 BRRR: one fused kernel from the attention output to the MLP all-reduce in decode/verify graphs
        # (TP3Config.brrr_layers). tp3.brrr_parked_max_rows = N: parked (host-expert) layers use it only in graphs of at
        # most N rows; wide verification of parked layers stays on Flare's MoE path (BRRR-40 measured c8 -2.7% vs H300,
        # BRRR-26 level). Flare's tail operators above are still declared and tuned (as on tp3-main); _run skips them.
        if tp3.brrr_layers and exl3:
            raise ValueError("BRRR (TP3Config.brrr_layers) runs MXFP4 experts only; leave it off for EXL3 checkpoints")
        if tp3.brrr_layers:
            rows = metadata.positions.numel()
            # A layer counts as parked only if it has a merged bank with a host tail (hot_map). TP3Config refuses
            # brrr_parked_max_rows unless parked BRRR layers have one (0 < hot_experts < 384), so the gate is never
            # a silent no-op; without the gate a fully host-parked bank (hot_experts=0) is simply not gated.
            parked = index in tp3.host_expert_layers and self.hot_map is not None
            gated = bool(tp3.brrr_parked_max_rows) and parked and rows > tp3.brrr_parked_max_rows
            if index in tp3.brrr_layers and mode in ("decode", "verify") and not gated:
                if self.image_mask is not None or boundary is not None or self.attention.owns_window_metadata:
                    raise ValueError(f"BRRR layer {index} needs a decode/verify layer with shared window indices")
                from ignition.recipes.deepseek41.brrr import MAX_ROWS, PreparedBRRRTail
                if rows > MAX_ROWS:
                    raise ValueError(f"BRRR layer {index} in {name}: the graph has {rows} rows, the fused kernel "
                                     f"takes at most {MAX_ROWS}: lower the speculative widths or concurrency, or leave the layer "
                                     "out of brrr_layers")
                self.attention.external_tail = True
                self.brrr = PreparedBRRRTail(index, weights, metadata.positions.dtype,
                                             rotary.frequencies[bool(self.attention.ratio)], config=pool.config, tp3=tp3)
            record_brrr_decision(name, index, mode, rows, self.brrr is not None, gated)
        # TP3 steering port (feat/ds41-steering-sidecar): the BRRR fused tail has no steering projection yet (it would
        # skip the attention and shared-expert projection silently), and the EXL3 merged epilogue folds the shared
        # expert into the routed output before the projection could run. Refuse both combinations.
        if steering is not None and self.brrr is not None:
            raise ValueError(f"steering sidecar: layer {index} in {name} runs the BRRR fused tail, which has no steering "
                             "projection; leave steering layers out of tp3.brrr_layers")
        if steering is not None and self.exl3_merge_epilogue:
            raise ValueError("steering sidecar: the EXL3 merged epilogue (MERGE_EPILOGUE) cannot project the shared expert")

        if self.attention_hc is not None:
            self.attention_hc.functional()
        self.attention_reduce.functional(session=session)
        self.ffn_hc.functional()
        self.router_projection.functional(session=session)
        self.router.functional(session=session)
        self.experts.functional(session=session)
        if self.concurrent_shared is None:
            self.shared.functional(session=session)
        else:
            self.concurrent_shared.functional(session=session)
        self.mlp_reduce.functional(session=session)
        if self.post is not None:
            self.post.functional()
        self.has_image_mask = self.image_mask is not None
        register_finalizer(session, self.release_examples)

    def release_examples(self):
        self.scratch = self.metadata = self.image_mask = None
        if self.concurrent_shared is not None:
            self.shared = None

    def __call__(self, incoming, pre_mix, metadata, indices, selection, *, boundary=None, window_cache=None,
                 engram_projected=None, engram_handoff=None, mma_pads=None, token_scale=None):
        incoming, next_mix, records, selection, _ = self._run(incoming, pre_mix, metadata, indices,
            selection, boundary=boundary, window_cache=window_cache, engram_projected=engram_projected,
            engram_handoff=engram_handoff, mma_pads=mma_pads, token_scale=token_scale)
        return incoming, next_mix, records, selection

    def _run(self, incoming, pre_mix, metadata, indices, selection, *, boundary=None, window_cache=None,
             engram_projected=None, engram_handoff=None, mma_pads=None, token_scale=None):
        """TP3 tp3.engram_side_layers: ``engram_projected`` = this layer's Engram projection, already reduced on the
        side stream by the previous layer; ``engram_handoff`` = a list that receives the next layer's (when this
        layer is the one that starts it, ``engram_side``). TP3 tp3.mma_pad_once: ``mma_pads`` = the forward's
        once-zeroed MXFP8 MMA-scale tiles (``PreparedAttention._run``); None = per-layer fills."""
        probe, layer, mode = self.probe, self.probe_index, self.probe_mode
        if (engram_projected is not None) != (self.engram is not None and self.engram.side_task is not None):
            raise ValueError("an Engram layer started on the side stream must receive its projection (and only it)")
        if (engram_handoff is not None) != (self.engram_side is not None):
            raise ValueError("only the layer before a side-stream Engram layer hands its projection over")
        side = None if self.engram_side is None else self.engram_side.start_side(incoming[0], metadata)
        if self.engram is not None and self.no_engram:
            self.engram(incoming[0], metadata, projected=engram_projected)  # debug A/B: result dropped
        elif self.engram is not None:
            incoming = (self.engram(incoming[0], metadata, projected=engram_projected),)
        if len(incoming) == 1:
            probe.record(incoming[0], layer, 0, mode)
        steering = self.steering
        if steering is not None and token_scale is None:
            raise ValueError("steering-enabled decoder layer requires per-token scale")
        if boundary is None:
            attn_mix = torch.empty_like(pre_mix)  # written whole by the lagged mHC (see PreparedBoundary)
            residual, post, comb, normalized = self.attention_hc(*incoming, pre_mix=pre_mix, pre_out=attn_mix)
        else:
            (residual, post, comb, normalized), attn_mix = boundary
        attention_normalized = normalized
        attended, records, selection, observations = self.attention._run(normalized, metadata, selection,
            indices=indices, window_cache=window_cache, **({} if mma_pads is None else {"mma_pads": mma_pads}))
        if side is not None:
            # Before the tail: BRRR (persistent, cross-rank) and the attention reduction read ``attended``, which the
            # join declares written, so neither can be scheduled before it.
            engram_handoff.append(self.engram_side.finish_side(side, attended))
        if self.brrr is not None:
            # TP3 BRRR: ``attended`` is the attention output before inverse RoPE (attention.external_tail); the kernel
            # runs inverse RoPE .. MLP all-reduce and returns what Flare's path below would.
            output, (residual, post, comb), next_mix = self.brrr(attended, residual, post, comb, attn_mix,
                                                                 indices.position)
            incoming = ((self.post(output, residual, post, comb),) if self.post is not None
                        else (output, residual, post, comb))
            observations = {"attention." + key: value for key, value in observations.items()}
            observations.update({"attention_hc.normalized": attention_normalized, "output": incoming[0]})
            return incoming, next_mix, records, selection, observations
        attention_output = self.attention_reduce(attended)
        if steering is not None:
            attention_output = steer_attention(attention_output, steering, token_scale)
        probe.record(attention_output, layer, 1, mode)
        next_mix = torch.empty_like(attn_mix)  # written whole by the lagged mHC
        residual, post, comb, normalized = self.ffn_hc(attention_output, residual, post, comb,
            pre_mix=attn_mix, pre_out=next_mix)
        pending = None if self.concurrent_shared is None else self.concurrent_shared.start(normalized)
        image_mask = metadata.token_ids >= 129280 if self.has_image_mask else None
        router_logits = self.router_projection(normalized)
        route = self.router(router_logits, image_mask=image_mask)
        probe.router(route.ids, route.weights, layer, mode)
        if self.route_counter.enabled:
            self.route_counter.record(route.ids, metadata.positions, layer, mode, self.route_experts)
        if self.route_capture.enabled:
            self.route_capture.record(route.ids, normalized, attention_normalized, layer, mode,
                                      self.route_capture_own, self.route_capture_next, self.route_capture_score)
        probe.full(normalized, layer, "in", mode)
        ids = route.ids
        prepped = None
        if getattr(self, "exl3_merge_prep", False):
            prepped = self.experts.prep(normalized, ids, route.weights, self.hot_map)
            order = prepped.token_sorted    # the join (and with it the xmoe launches) after the prep
        elif self.hot_map is not None:
            ids = torch.index_select(self.hot_map, 0, ids.reshape(-1)).view(ids.shape)
        joins_early = getattr(self, "shared_joins_before_experts", False)
        summed = getattr(self, "exl3_merge_epilogue", False)
        if joins_early:
            shared = self.concurrent_shared.finish(pending, ids if prepped is None else order)
            extra = {} if prepped is None and not summed else {"prepped": prepped, "add_after": summed}
            routed = self.experts(normalized, ids, route.weights, router_logits=router_logits,
                                  selected_logits=route.selected_logits, image_mask=image_mask, after=shared, **extra)
        elif prepped is not None:
            routed = self.experts(normalized, ids, route.weights, router_logits=router_logits,
                                  selected_logits=route.selected_logits, image_mask=image_mask, prepped=prepped)
        else:
            routed = self.experts(normalized, ids, route.weights, router_logits=router_logits,
                                  selected_logits=route.selected_logits, image_mask=image_mask)
        probe.full(routed, layer, "out", mode)
        if not joins_early:
            shared = (self.shared(normalized) if self.concurrent_shared is None
                      else self.concurrent_shared.finish(pending, routed))
        if steering is not None:
            # Rank-local shared-expert partial: the projection is linear, so projecting each rank's partial before the
            # MLP all-reduce equals projecting the reduced sum (upstream PR #404; up to BF16 rounding of the partials).
            shared = steer_shared_expert(shared, steering, token_scale)
        probe.record(routed, layer, 3, mode)
        probe.record(shared, layer, 4, mode)
        partial = routed if summed else routed + shared    # MERGE_EPILOGUE: routed already holds routed + shared
        probe.record(partial, layer, 5, mode)
        output = self.mlp_reduce(partial)
        probe.record(output, layer, 2, mode)
        incoming = ((self.post(output, residual, post, comb),) if self.post is not None
                    else (output, residual, post, comb))
        observations = {"attention." + key: value for key, value in observations.items()}
        observations.update({"attention_hc.normalized": attention_normalized, "router.ids": route.ids,
                             "output": incoming[0]})
        return incoming, next_mix, records, selection, observations


def mma_pad_layouts(attentions):
    """TP3 tp3.mma_pad_once: the distinct padded MXFP8 MMA-scale layouts of a forward's attention layers (qkv_a and
    q_b activations; ``PreparedAttention.padded_mma_layouts``), in first-use order."""
    layouts = []
    for attention in attentions:
        for layout in attention.padded_mma_layouts():
            if layout not in layouts:
                layouts.append(layout)
    return tuple(layouts)


def allocate_mma_pads(layouts, device):
    """Inside a forward: one zero-filled tile per layout (a tuple in ``layouts`` order), shared by every layer (None when
    there is none; ``dense_bases_with_pads`` matches them by shape and dtype). Each layer's
    quantizer rewrites every real row's scales before its GEMM reads the tile; padding rows are written by nothing."""
    if not layouts:
        return None
    return tuple(torch.zeros(shape, dtype=dtype, device=device) for shape, dtype in layouts)


def configure_engram_side(layers, engram_side_layers, mode):
    """TP3 tp3.engram_side_layers: Engram layer L's readiness wait, lookup, projection and reduction run on a side
    stream during layer L-1's attention, in decode/verify graphs (prefill keeps them in place). Marks the starting
    layers; raises for a named layer that is not an Engram layer after the first."""
    if mode not in ("decode", "verify"):
        return
    for index in sorted(engram_side_layers):
        engram = layers[index].engram if index < len(layers) else None
        if engram is None or index == 0:
            raise ValueError(f"TP3 config: engram_side_layers names layer {index}, which is not an Engram layer "
                             "after the first (the side stream starts in the previous layer)")
        previous = layers[index - 1]
        if previous.boundary is not None or previous.engram_side is not None or previous.engram is not None:
            raise ValueError(f"TP3 config: engram_side_layers cannot start layer {index}'s Engram in layer {index - 1}")
        previous.engram_side = engram.enable_side()


def shared_rotary_angles(tp3, mode):
    """Whether ``mode``'s forwards read one shared RoPE table per call (TP3 tp3.rotary_once: decode/verify, the
    draft gets it through dspark.py; tp3.rotary_once_prefill: prefill, its encoder-only twins and suffix replays)."""
    return (tp3.rotary_once and mode in ("decode", "verify")) or (tp3.rotary_once_prefill and mode == "prefill")


class PreparedDecoder:
    host_rows = None  # TP3 tp3.embed_rows gather (HostRowGather); None = b12x's lookup
    glue_pass2 = concatenated_feedback = False  # TP3 tp3.glue_pass2; set per instance
    mma_pad_layouts = ()  # TP3 tp3.mma_pad_once: MMA-scale tiles zeroed once per forward; () = per-layer fills

    def __init__(self, weights, tables, scratch, metadata, pool, *, mode, runtime,
                 vocabulary_runtime, coordinator, session, name, workspace,
                 vision=None, mla_compute_mode="auto", scoring=None, vocabulary_parallel=False, examples=None,
                 tp3=None, encoder_only=False, index_split_variant=None):
        if scratch.local_logits.dtype != torch.float32 or scratch.logits.dtype != torch.float32:
            raise ValueError("DeepSeek target vocabulary logits require FP32 storage")
        tp3 = TP3Config() if tp3 is None else tp3  # TP3 fork switches, shared by every layer of this graph
        self.probe = Probe(tp3.probe)
        # TP3 tp3.glue_pass2 (see __call__): decode/verification graphs only; False = base graphs
        self.glue_pass2 = tp3.glue_pass2 and mode in ("decode", "verify")
        self.weights, self.scratch, self.metadata, self.pool = weights, scratch, metadata, pool
        self.mode = mode
        self.vision = vision if mode == "prefill" else None
        self.table_requests = PreparedTableRequests(tables, metadata, pool, session=session, name=name + ".table_requests")
        decoder_start = weights.config.decoder_start
        self.packing = None
        if mode == "prefill" and metadata.positions.numel() > 128 and decoder_start is not None and scoring is None:
            if (any(layer.engram is not None for layer in weights.layers[decoder_start:])
                    or any(source.layer > decoder_start for source in weights.config.sources)):
                raise ValueError("decoder suffix replay requires Engram and compressed KV sources before its boundary")
            self.packing = PreparedDecoderSuffix(metadata)
        # TP3 (tp3.prefill_skip_suffix): an encoder-only twin stops after the boundary's global KV; it builds the
        # layers through the boundary layer (whose first index selector writes the index keys) and nothing after.
        self.encoder_only = encoder_only
        if encoder_only and self.packing is None:
            raise ValueError("an encoder-only prefill needs a prefill chunk larger than the decoder suffix")
        self.output_metadata = metadata if self.packing is None else self.packing.metadata
        self.output_scratch = scratch
        if self.packing is not None:
            # TP3: suffix scratch is declaration-only, so the prefill graphs share one (released with the examples),
            # as tp3-main's suffix_scratch did; per graph it was ~0.4 GiB/rank of catalog operands (G1 2026-11-14).
            key = ("decoder.suffix_scratch", scratch.hidden.shape[0], weights.shard.world_size, scratch.embedding.device)
            self.output_scratch = None if examples is None else examples.get(key)
            if self.output_scratch is None:
                self.output_scratch = Scratch.allocate(128, *key[1:])
                if examples is not None:
                    examples[key] = self.output_scratch
        self.aux_layers = () if weights.draft is None else weights.config.draft.target_layers
        if self.packing is not None and any(layer <= decoder_start for layer in self.aux_layers):
            raise ValueError("bounded decoder feedback requires draft target layers after its global-KV boundary")
        self.auxiliary_shape = (self.output_metadata.token_ids.numel(), len(self.aux_layers) * 5120)
        self.auxiliary = None if encoder_only else torch.zeros(self.auxiliary_shape, dtype=torch.bfloat16,
                                                               device=metadata.positions.device)
        # TP3 glue_pass2: with every draft target layer in this stack and unpacked rows, the feedback slices are
        # concatenated once after the layers (one kernel) instead of copied into a buffer one slice at a time.
        self.concatenated_feedback = (self.glue_pass2 and self.packing is None and bool(self.aux_layers)
                                      and all(layer < len(weights.layers) for layer in self.aux_layers))
        self.safe_tokens = torch.zeros_like(metadata.token_ids) if self.vision is not None else metadata.token_ids
        # TP3 (tp3.embed_rows): decode and verification graphs gather the host-parked table's rows with one
        # vectorised launch (host_embedding.py, same values); prefill graphs and the default keep b12x's lookup.
        self.host_rows = (HostRowGather(weights.embedding, rows=metadata.token_ids.numel())
                          if tp3.embed_rows and mode in ("decode", "verify") else None)
        self.embedding = None if self.host_rows is not None else PreparedEmbedding(
            weights.embedding, self.safe_tokens, scratch.embedding, num_rows=metadata.live_counts[:1],
            session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        self.rotary = Rotary(weights.config, scratch.embedding.device)
        self.window_commit = None
        if mode == "verify":
            # TP3: the staging is only a declaration example (each call allocates its own), so every verification
            # graph shares one by role, as tp3-main's window_staging did; per graph it was 0.9 GiB/rank over the
            # 68 TP3 speculative graphs (G1 2026-11-14, graph catalog out of memory).
            staging = None if examples is None else declaration_tensor(examples, "verification.window_staging",
                (len(weights.layers), metadata.state_rows.numel(), (128 + metadata.max_query_tokens) * 528),
                torch.uint8, metadata.positions.device)
            self.window_commit = PreparedRingCommit(pool.reserved_backing,
                [pool.layout.region(f"layer{index}.window").offset for index in range(len(weights.layers))],
                metadata.state_rows, metadata.starts,
                token_capacity=metadata.max_query_tokens,
                window_tokens=128, record_bytes=528, staging=staging)
        ratios = weights.config.text["compress_ratios"][:len(weights.layers)]
        prefix_ratios = ratios if self.packing is None else ratios[:decoder_start + 1]
        # TP3 tp3.rotary_once: decode/verification forwards compute each RoPE table once (AttentionIndices.angles);
        # prefill keeps per-launch angles unless tp3.rotary_once_prefill (E3 d: past ~105.6K positions every per-launch
        # program takes sinf/cosf's slow range reduction; the table costs one [rows, 32, 2] FP32 write per forward).
        self.window_metadata = AttentionIndices(metadata, mode=mode,
            tokens=metadata.positions.numel(),
            page_size=128 if mode == "decode" else 128 + metadata.max_query_tokens, width=128,
            ratios=set(prefix_ratios) - {0},
            rotary=self.rotary if shared_rotary_angles(tp3, mode) else None)
        self.suffix_window_metadata = None
        if self.packing is not None:
            self.suffix_window_metadata = AttentionIndices(self.output_metadata, mode=mode,
                tokens=self.output_metadata.positions.numel(),
                page_size=128 + self.output_metadata.max_query_tokens, width=128,
                replay_start=self.packing.replay_start, ratios=set(ratios[decoder_start:]) - {0},
                rotary=self.rotary if tp3.rotary_once_prefill else None)
        # TP3 tp3.index_row_split: listed prefill capacities (and their encoder-only twins) split the indexer rows of the
        # full-row index layers across ranks and exchange the winners over the vocabulary channel; prompt-logprob
        # graphs keep the replicated indexer (index_row_split.py).
        index_split = prefill_split_context(mode=mode, capacity=metadata.positions.numel(), scoring=scoring, tp3=tp3,
            shard=weights.shard, runtime=vocabulary_runtime, coordinator=coordinator, examples=examples,
            variant=index_split_variant)
        self.index_split = index_split
        self.layers, carry = [], False
        built = weights.layers[:decoder_start + 1] if encoder_only else weights.layers
        for index, layer in enumerate(built):
            boundary = replay_start = None
            layer_scratch, layer_metadata = scratch, metadata
            if self.packing is not None and index >= decoder_start:
                layer_scratch, layer_metadata = self.output_scratch, self.output_metadata
                replay_start = self.packing.replay_start
                if index == decoder_start:
                    boundary = PreparedDecoderBoundary(index, layer, scratch, layer_scratch, self.packing,
                        pool, self.rotary, carry=carry, session=session, name=f"{name}.layer{index}", workspace=workspace,
                        window_metadata=self.suffix_window_metadata,
                        source_indices=self.window_metadata)
                    layer = replace(layer, attention={key: value for key, value in layer.attention.items()
                                                      if key not in ("compressor", "index_k", "index_norm")})
            defer = (index + 1 < len(weights.layers) and weights.layers[index + 1].engram is None
                     and index + 1 not in self.aux_layers)
            self.layers.append(PreparedLayer(index, layer, tables, layer_scratch, layer_metadata, pool, self.rotary,
                shard=weights.shard, mode=mode, runtime=runtime,
                coordinator=coordinator, session=session, name=f"{name}.layer{index}", workspace=workspace,
                carry=carry, defer_post=defer, boundary=boundary, replay_start=replay_start,
                window_staging=None if self.window_commit is None else self.window_commit.staging[index],
                mla_compute_mode=mla_compute_mode,
                window_metadata=(self.window_metadata if layer_metadata is metadata else self.suffix_window_metadata),
                examples=examples, tp3=tp3, probe=self.probe,
                index_split=index_split if layer_metadata is metadata else None,
                steering=None if weights.steering is None else weights.steering.layers.get(index)))
            carry = defer
        configure_engram_side(self.layers, tp3.engram_side_layers, mode)
        self.route_capture = RouteCapture()
        if self.route_capture.active(mode):   # debug-serve only: each layer's predictor uses the next layer's router
            self.route_capture.allocate(metadata.positions.device, mode)
            for index, layer in enumerate(self.layers[:-1]):
                following = weights.layers[index + 1].experts
                layer.route_capture_next = (following.router, following.correction)
        # TP3 (tp3.mma_pad_once): decode/verification forwards zero each padded MXFP8 MMA-scale tile shape once and
        # every layer's qkv_a/q_b quantizer reuses it (prefill and the packed suffix keep per-layer fills).
        self.mma_pad_layouts = (mma_pad_layouts(layer.attention for layer in self.layers)
                                if tp3.mma_pad_once and mode in ("decode", "verify") and self.packing is None else ())
        self.scoring = None if scoring is None else scoring.prepare(scratch.embedding, weights.head,
            vocabulary=129280, session=session, coordinator=coordinator, name=name + ".prompt",
            logits_dtype=torch.float32)
        self.head = None
        if not encoder_only:
            self._prepare_head(vocabulary_parallel=vocabulary_parallel, vocabulary_runtime=vocabulary_runtime,
                coordinator=coordinator, session=session, name=name, workspace=workspace)
        self.window_metadata.functional()
        if self.suffix_window_metadata is not None:
            self.suffix_window_metadata.functional()
            self.packing.functional()
        self.staging_shape = None if self.window_commit is None else tuple(self.window_commit.staging.shape)
        if self.window_commit is not None:
            self.window_commit.functional()
        register_finalizer(session, self.release_outputs)

    def _prepare_head(self, *, vocabulary_parallel, vocabulary_runtime, coordinator, session, name, workspace=None):
        scratch, weights = self.scratch, self.weights
        self.compact_head = scratch.hidden.shape[0] > self.output_metadata.token_ids.numel()
        head_rows = self.output_metadata.token_ids.numel() if self.compact_head else scratch.hidden.shape[0]
        self.head_hidden = scratch.hidden[:head_rows]
        head_output = scratch.logits
        if vocabulary_parallel and self.compact_head:
            # Declare expanded local logits independently of the physical projection rows.
            head_output = scratch.logits.view(-1)[:scratch.local_logits.numel()].view_as(scratch.local_logits)
        self.head = PreparedVocabularyProjection(self.head_hidden, weights.head, scratch.local_logits[:head_rows],
            head_output, runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head",
            row_indices=self.output_metadata.sample_indices if self.compact_head else None, gather=not vocabulary_parallel,
            workspace=workspace)

        self.local_logits = self.head.output if vocabulary_parallel else scratch.local_logits
        self.head.functional(session=session, dynamic_indices=True)

    def release_outputs(self):
        self.scratch = self.metadata = self.output_scratch = self.output_metadata = None
        self.auxiliary = self.safe_tokens = self.head_hidden = self.local_logits = None

    def __call__(self, metadata):
        source_metadata = metadata
        self.probe.step(metadata.positions.device, self.mode)
        self.probe.positions(metadata.positions, self.mode)
        if self.route_capture.enabled:
            self.route_capture.step(metadata.positions, self.mode)
        token_scale = metadata.steering_scale
        if self.weights.steering is not None and token_scale is None:
            raise ValueError("steering-enabled decoder requires per-token scale metadata")
        raw_history = self.table_requests(metadata)
        tokens = (metadata.token_ids.clamp(0, self.weights.embedding.shape[0] - 1)
                  if self.vision is not None else metadata.token_ids)
        embedding = (self.embedding(token_ids=tokens, num_rows=metadata.live_counts[:1]) if self.host_rows is None
                     else self.host_rows(tokens, metadata.live_counts[:1]))
        if self.vision is not None:
            self.vision.inject(metadata, embedding)
        residual = embedding[:, None, :].expand(-1, 4, -1).contiguous()
        pre_mix = initial_mix(embedding.shape[0], embedding.device, functional=self.glue_pass2)
        incoming = (residual,)
        indices = self.window_metadata(metadata)
        selection = (torch.full((embedding.shape[0], 512), -1, dtype=torch.int32, device=embedding.device), None, None)
        # TP3 glue_pass2: every layer's window writer copies the 128 snapshot records of each request with a valid
        # state row and stores each live token's record before attention reads its slice; attention reads only
        # those (window indices) and the ring commit only accepted live tokens of rows > 0, so the staging needs no
        # zero fill (2.8 MB at 1 request, 22.6 MB at 8, per verification round).
        staging = (None if self.staging_shape is None else (torch.empty if self.glue_pass2 else torch.zeros)(
            self.staging_shape, dtype=torch.uint8, device=embedding.device))
        # Each draft target layer's feedback is copied into its whole 5120-column slice of every row below
        # (rows match or broadcast); every slice is written when every target layer is in this stack.
        auxiliary = None if self.encoder_only or self.concatenated_feedback else (
            torch.empty if all(layer < len(self.layers) for layer in self.aux_layers) else torch.zeros)(
            self.auxiliary_shape, dtype=torch.bfloat16, device=embedding.device)
        feedbacks = {}
        suffix, records = None, []
        engram_projected = None  # TP3 tp3.engram_side_layers: projection handed from the previous layer
        mma_pads = allocate_mma_pads(self.mma_pad_layouts, embedding.device)  # TP3 tp3.mma_pad_once; None = off
        for index, layer in enumerate(self.layers):
            boundary = None
            if layer.boundary is not None and self.encoder_only:
                layer.boundary.global_only(incoming, pre_mix, metadata, indices, layer.attention.selectors[0])
                break
            if layer.boundary is not None:
                suffix, state, attn_mix = layer.boundary(incoming, pre_mix, metadata, indices,
                                                        layer.attention.selectors[0])
                if token_scale is not None:
                    token_scale = suffix.gather(token_scale)
                metadata = suffix.metadata
                indices = self.suffix_window_metadata(metadata, replay_start=suffix.replay_start)
                boundary = state, attn_mix
                selection = (torch.full((metadata.positions.numel(), 512), -1,
                                       dtype=torch.int32, device=embedding.device), None, None)
            if index in self.aux_layers:
                feedback = incoming[0].float().mean(1)
                if suffix is not None:
                    feedback = torch.where(suffix.live[:, None], feedback, 0)
                column = self.aux_layers.index(index)
                if self.concatenated_feedback:
                    feedbacks[column] = feedback
                else:
                    auxiliary[:, column * 5120:(column + 1) * 5120].copy_(feedback)
            handoff = None if layer.engram_side is None else []
            incoming, pre_mix, record, selection = layer(incoming, pre_mix, metadata, indices, selection,
                boundary=boundary, window_cache=None if staging is None else staging[index],
                engram_projected=engram_projected, engram_handoff=handoff,
                token_scale=token_scale, **({} if mma_pads is None else {"mma_pads": mma_pads}))
            engram_projected = handoff[0] if handoff else None
            records.append(record)
        if self.encoder_only:
            self.table_requests.commit(source_metadata.lengths, raw_history, source_metadata)
            return None
        residual, = incoming
        if self.concatenated_feedback:
            auxiliary = torch.cat([feedbacks[column] for column in range(len(self.aux_layers))], dim=1).to(torch.bfloat16)
            if tuple(auxiliary.shape) != self.auxiliary_shape:
                raise ValueError("concatenated draft feedback must match the auxiliary rows")
        samples = metadata.sample_indices
        head_residual = residual if self.compact_head else residual[samples]
        head_mix = pre_mix if self.compact_head else pre_mix[samples]
        collapsed = (head_residual.float() * head_mix[:, :, None]).sum(1).bfloat16()
        if self.scoring is not None:
            collapsed = (residual.float() * pre_mix[:, :, None]).sum(1).bfloat16()
            normalized = torch.zeros_like(collapsed)
            normalize(collapsed, self.weights.norm, normalized, self.weights.config.text["rms_norm_eps"])
            self.scoring(normalized)
            hidden = normalized[samples]
        else:
            hidden = torch.zeros_like(collapsed)
            normalize(collapsed, self.weights.norm, hidden, self.weights.config.text["rms_norm_eps"])
        logits = self.head(hidden, row_indices=samples if self.compact_head else None)
        if self.compact_head:
            hidden = hidden[samples]
        if self.mode != "verify":
            self.table_requests.commit(source_metadata.lengths, raw_history, source_metadata)
        return hidden, logits, auxiliary, (staging, tuple(records), raw_history)

    def commit(self, counts, records, metadata):
        if self.window_commit is None:
            raise ValueError("accepted decoder commits require verification staging")
        staging, layers, raw_history = records
        self.window_commit(counts, staging=staging, rows=metadata.state_rows, starts=metadata.starts)
        for layer, record in zip(self.layers, layers, strict=True):
            layer.attention.commit_pending(counts, record, metadata)
        self.table_requests.commit(counts, raw_history, metadata)
