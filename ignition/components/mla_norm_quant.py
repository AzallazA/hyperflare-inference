# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prepared BF16 MLA normalization with MXFP8 query encoding."""

import torch
from b12x.preparation import PreparedCall
from b12x.quantization import mla_norm_mxfp8, mxfp8

from ignition.components.linear import mxfp8_dense_bases
from ignition.preparation.fixtures.projection import input_factory
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.tensor_metadata import record_numeric_tensor
from ignition.preparation.tuning import declare, register_context_plan, register_finalizer


def dense_bases_with_pads(layout, device, mma_pads=None):
    """``mxfp8_dense_bases`` whose MMA scales come from ``mma_pads`` (DS41 TP3 ``mma_pad_once``).

    ``mma_pads`` is a tuple of buffers the forward zeroed once (one per padded MMA-scale layout) and shares between
    every layer whose quantizer has that layout (matched by shape and dtype, static while tracing). The quantizer
    still writes every real row's scales before its GEMM reads them; the padding rows of the 128-row tiles are
    written by nothing, so they keep that one fill's zero, as a fresh ``mxfp8_dense_bases`` tile would. A layout
    without a pad (or ``mma_pads`` None) allocates as before.
    """
    (values_shape, values_dtype), (rows_shape, rows_dtype), (mma_shape, mma_dtype) = layout
    pad = None
    for candidate in mma_pads or ():
        if tuple(candidate.shape) == tuple(mma_shape) and candidate.dtype == mma_dtype:
            pad = candidate
            break
    if pad is None:
        return mxfp8_dense_bases(layout, device)
    return (torch.empty(values_shape, dtype=values_dtype, device=device),
            torch.empty(rows_shape, dtype=rows_dtype, device=device), pad)


def padded_mma_layout(layout):
    """The MMA-scale layout ``(shape, dtype)`` of MXFP8 dense bases whose tiles have padding rows (the ones
    ``mxfp8_dense_bases`` zero-fills), else None."""
    (values_shape, _), _, (mma_shape, mma_dtype) = layout
    return None if values_shape[0] % 128 == 0 else (tuple(mma_shape), mma_dtype)


class PreparedMLANormQuant:
    """Normalize query and KV slices, then encode the rounded BF16 query."""

    def __init__(self, projected, weights, query, kv, *, eps, session, name, buffers=None,
                 release_dependents=False, examples=None, plan_override=None):
        rows = projected.shape[0]
        if (projected.shape != (rows, 1792) or projected.dtype != torch.bfloat16
                or not projected.is_contiguous() or query.shape != (rows, 1280)
                or kv.shape != (rows, 512) or query.dtype != torch.bfloat16
                or kv.dtype != torch.bfloat16 or not query.is_contiguous()
                or not kv.is_contiguous() or eps != 1e-20):
            raise ValueError("MLA normalization requires contiguous BF16 1280+512 rows and model epsilon")
        self.weights, self.query, self.kv = tuple(weights), query, kv
        for weight, width in zip(self.weights, (1280, 512), strict=True):
            if (weight.shape != (width,) or weight.dtype not in (torch.bfloat16, torch.float32)
                    or weight.device != projected.device or not weight.is_contiguous()):
                raise ValueError("MLA normalization affine weights must match their output slices")
            record_numeric_tensor(weight)
        def allocate(role, shape, dtype):
            if examples is not None:
                return declaration_tensor(examples, "mla.query." + role, shape, dtype, projected.device)
            return (torch.zeros(shape, dtype=dtype, device=projected.device) if buffers is None
                    else buffers.zeros(shape, dtype))

        values_shape, rows_shape, mma_shape = mxfp8.dense_rows_storage_shapes(rows, 1280)
        self.encoded_bases = (allocate("values", values_shape, torch.float8_e4m3fn),
                              allocate("scale_rows", rows_shape, torch.uint8),
                              allocate("scale_mma", mma_shape, torch.uint8))
        self.encoded = mxfp8.dense_rows_from_bases(*self.encoded_bases)
        from .linear import _overridden

        # plan_override (TP3 fork): a config mapping taken as an explicit override; None = tuned (base).
        self.plan = _overridden(mla_norm_mxfp8.plan, mla_norm_mxfp8.Query(
            rows=rows, query_weight_dtype=str(weights[0].dtype).removeprefix("torch."),
            kv_weight_dtype=str(weights[1].dtype).removeprefix("torch."),
            query_features=1280, kv_features=512, epsilon=eps, min_amax=1e-4,
            affine_prefetch_ready=True, release_dependents=release_dependents), plan_override)
        representative = input_factory(projected, projected,
            storage=getattr(session, "benchmark_storage", None), session=session)

        encoded_bases, encoded = self.encoded_bases, self.encoded

        def prepare(state, fixture=None):
            source = projected if fixture is None else fixture.source
            return PreparedCall(run=lambda: state.run(source, *self.weights, query, kv,
                *encoded_bases), produce=None if fixture is None else fixture.produce,
                output=(query, kv, *encoded_bases), owners=(fixture, encoded))

        declare(session, (self.plan.request(name=name,
            prepare_call=lambda state: prepare(state),
            benchmark_call=lambda state: prepare(state, representative())),))
        register_context_plan(session, name, self)

    def functional(self, *, session):
        self.layouts = tuple((tuple(t.shape), t.dtype) for t in (self.query, self.kv, *self.encoded_bases))
        register_finalizer(session, self._release_examples)
        return self

    def _release_examples(self):
        self.query = self.kv = self.encoded_bases = self.encoded = None

    @property
    def padded_mma_layout(self):
        """Functional: the q_b activation's MMA-scale layout when its tiles have padding rows, else None."""
        return padded_mma_layout(self.layouts[2:])

    def __call__(self, projected, *, mma_pads=None):
        if self.query is None:
            # The kernel (b12x mla_norm_mxfp8, grid = rows x {query, kv}) stores every element of the
            # query, KV, values and compact scales of each planned row, so those start uninitialised.
            # It stores MMA scales only for real rows; mxfp8_dense_bases keeps the padding rows the
            # q_b GEMM loads (whole 128-row scale tiles) zero.
            (query_shape, query_dtype), (kv_shape, kv_dtype), *encoded = self.layouts
            query = torch.empty(query_shape, dtype=query_dtype, device=projected.device)
            kv = torch.empty(kv_shape, dtype=kv_dtype, device=projected.device)
            # TP3 mma_pad_once: ``mma_pads`` supplies the MMA scales zeroed once per forward (dense_bases_with_pads).
            bases = dense_bases_with_pads(encoded, projected.device, mma_pads)
            mla_norm_mxfp8.run(projected, *self.weights, query, kv, *bases, plan=self.plan)
            return query, kv, tuple(bases)
        mla_norm_mxfp8.run(projected, *self.weights, self.query, self.kv,
                            *self.encoded_bases, plan=self.plan)
        return self.query, self.kv
