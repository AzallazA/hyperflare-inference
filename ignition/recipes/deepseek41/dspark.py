# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Block drafts conditioned on accepted target states, with sequential Markov logits."""

import dataclasses
import json

import torch

from ignition.components.embedding import PreparedEmbedding
from ignition.components.vocabulary import PreparedVocabularyProjection
from ignition.preparation.tuning import register_finalizer

from .attention import AttentionIndices, Rotary, normalize
from .decoder import PreparedLayer, allocate_mma_pads, mma_pad_layouts
from .decoder import initial_mix
from .host_embedding import HostRowGather
from .tp3_config import TP3Config


# TP3 tp3.draft_vocab: logit of a token outside the draft vocabulary (finite: block-wise max/LSE stay defined)
DRAFT_FILL = -1e30


class PreparedDraft:
    host_rows = None  # TP3 tp3.embed_rows gather (HostRowGather); None = b12x's lookup
    glue_pass2 = False  # TP3 tp3.glue_pass2 (state rows once, functional pre-mix); set per instance
    mma_pad_layouts = ()  # TP3 tp3.mma_pad_once: MMA-scale tiles zeroed once per forward; () = per-layer fills
    draft_columns = None  # TP3 tp3.draft_vocab: local columns of the trimmed draft head; None = full head

    def __init__(self, weights, scratch, metadata, pool, *, runtime, vocabulary_runtime,
                 coordinator, session, name, workspace, mla_compute_mode="auto", vocabulary_parallel=False, examples=None,
                 tp3=None):
        if scratch.local_logits.dtype != torch.bfloat16 or scratch.logits.dtype != torch.bfloat16:
            raise ValueError("DSpark proposal vocabulary logits require BF16 storage")
        self.weights, self.scratch, self.metadata = weights, scratch, metadata
        # TP3 (tp3.embed_rows): one vectorised gather of the host-parked table's rows, and the block's noise-token rows
        # from a device copy instead of repeated reads of one host row (host_embedding.py, same values).
        self.host_rows = (HostRowGather(weights.embedding, rows=metadata.token_ids.numel(),
                                        noise_token=weights.config.draft.noise_token)
                          if tp3 is not None and tp3.embed_rows else None)
        self.embedding = None if self.host_rows is not None else PreparedEmbedding(
            weights.embedding, metadata.token_ids, scratch.embedding, num_rows=metadata.live_counts[:1],
            session=session, name=name + ".embedding").functional(session=session, dynamic_inputs=True)
        rotary = Rotary(weights.config, scratch.embedding.device)
        # TP3 glue_pass2 (draft layers get no other TP3 switch, as before): see __call__ and PreparedAttention._run.
        self.glue_pass2 = tp3 is not None and tp3.glue_pass2
        # TP3 tp3.pin_small_qproj / tp3.rotary_once also reach the draft layers (same declarations as the target's).
        pin_small_qproj = tp3 is not None and tp3.pin_small_qproj
        layer_tp3 = (TP3Config(glue_pass2=self.glue_pass2, pin_small_qproj=pin_small_qproj)
                     if self.glue_pass2 or pin_small_qproj else None)
        page_size = 128 + metadata.max_query_tokens
        self.window_metadata = AttentionIndices(metadata, mode="draft",
            tokens=metadata.positions.numel(), page_size=page_size, width=page_size,
            rotary=rotary if tp3 is not None and tp3.rotary_once else None)
        self.layers = tuple(PreparedLayer(i, layer, None, scratch, metadata, pool, rotary,
            shard=weights.shard, mode="draft", runtime=runtime, coordinator=coordinator,
            session=session, name=f"{name}.layer{i}", workspace=workspace, carry=i > 0,
            defer_post=i + 1 < len(weights.draft.layers), mla_compute_mode=mla_compute_mode,
            window_metadata=self.window_metadata, examples=examples, tp3=layer_tp3)
            for i, layer in enumerate(weights.draft.layers))
        # TP3 (tp3.mma_pad_once): the draft layers share one zero fill per padded MXFP8 MMA-scale tile shape.
        self.mma_pad_layouts = (mma_pad_layouts(layer.attention for layer in self.layers)
                                if tp3 is not None and tp3.mma_pad_once else ())
        self.head_hidden = torch.zeros_like(scratch.hidden)
        self.head_hidden_shape = tuple(scratch.hidden.shape)
        self.logit_shape, self.logit_dtype = tuple(scratch.logits.shape), scratch.logits.dtype
        self.local_logit_shape, self.local_logit_dtype = tuple(scratch.local_logits.shape), scratch.local_logits.dtype
        head = weights.head if weights.draft.draft_head is None else weights.draft.draft_head
        # TP3 tp3.draft_vocab: the trimmed head projects onto its kept columns only; __call__ places them in the rank's
        # full local logits, every other column a large negative constant (never proposed).
        self.draft_columns = getattr(weights.draft, "draft_columns", None)
        local_logits = scratch.local_logits
        if self.draft_columns is not None:
            if not vocabulary_parallel or local_logits.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError("tp3.draft_vocab needs vocabulary-parallel FP32/BF16 draft logits")
            local_logits = torch.empty((local_logits.shape[0], self.draft_columns.numel()), dtype=local_logits.dtype,
                                       device=local_logits.device)
        # TP3 tp3.draft_head_plans (prepared, not measured): an explicit head projection config for this row count.
        head_plan = None
        if tp3 is not None and tp3.draft_head_plans is not None:
            head_plan = next((json.loads(config) for rows, config in tp3.draft_head_plans
                              if rows == scratch.hidden.shape[0]), None)
        self.head = PreparedVocabularyProjection(scratch.hidden, head, local_logits, scratch.logits,
            runtime=vocabulary_runtime, coordinator=coordinator, session=session, name=name + ".head", workspace=workspace, gather=not vocabulary_parallel,
            **({} if head_plan is None else {"plan_override": head_plan})).functional(session=session)
        self.window_metadata.functional()
        register_finalizer(session, self.release_outputs)

    def release_outputs(self):
        self.scratch = self.metadata = self.head_hidden = None

    def __call__(self, metadata):
        if self.glue_pass2:
            # The draft metadata's state rows are a pointwise expression of the round inputs that every draft
            # layer's window writer reads through an opaque call; realised once instead of once per layer (same
            # values: snapshot is a copy).
            metadata = dataclasses.replace(metadata, state_rows=torch.ops.flare.snapshot(metadata.state_rows))
        embedding = (self.embedding(token_ids=metadata.token_ids, num_rows=metadata.live_counts[:1])
                     if self.host_rows is None else self.host_rows(metadata.token_ids, metadata.live_counts[:1]))
        incoming = (embedding[:, None, :].expand(-1, 4, -1).contiguous(),)
        pre_mix = initial_mix(embedding.shape[0], embedding.device, functional=self.glue_pass2)
        indices = self.window_metadata(metadata)
        selection = (torch.full((embedding.shape[0], 512), -1, dtype=torch.int32, device=embedding.device), None, None)
        mma_pads = allocate_mma_pads(self.mma_pad_layouts, embedding.device)  # TP3 tp3.mma_pad_once; None = off
        for layer in self.layers:
            incoming, pre_mix, _, selection = layer(incoming, pre_mix, metadata, indices, selection,
                                                    **({} if mma_pads is None else {"mma_pads": mma_pads}))
        residual, = incoming
        head_hidden = (residual[metadata.sample_indices].float()
                       * pre_mix[metadata.sample_indices, :, None]).sum(1).bfloat16()
        hidden = torch.zeros_like(head_hidden)
        normalize(head_hidden, self.weights.draft.norm, hidden, self.weights.config.text["rms_norm_eps"])
        features = {"head_hidden": head_hidden.view(metadata.starts.numel(), -1, head_hidden.shape[-1])}
        logits = self.head(hidden)
        if self.draft_columns is not None:
            full = torch.full((logits.shape[0], self.local_logit_shape[1]), DRAFT_FILL, dtype=logits.dtype, device=logits.device)
            logits = full.index_copy(1, self.draft_columns, logits)
        return logits, None, features
