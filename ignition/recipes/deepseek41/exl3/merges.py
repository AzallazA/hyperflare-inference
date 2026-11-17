# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Small bit-exact kernel merges in the DS41 EXL3 decode tail (exp/ds41-exl3-final-merges, fused-tp2 REPORT.md §3 (c)).

Three switches, each default off, read at import (baked into captured graphs); every merged path writes bit for bit
what the op sequence it replaces writes:

``FLARE_DS41_EXL3_MERGE_TOPK=1`` (c-i, needs ``FLARE_DS41_EXL3_FUSED_PREP=1``; graphs within its row limit): the merged
  bank's ``hot_map`` lookup (decoder.py ``index_select``, one launch) folds into the fused routing prep, and the prep
  runs as its own op right after top-k, i.e. BEFORE the shared-expert join (``FLARE_DS41_EXL3_SHARED_SIDE``) instead
  of inside the MoE op after it. The s1 profile (seq296) has the router chain (GEMV, top-k, index_select) 5-9 us ahead
  of the shared expert at c1/c4, so the prep (1.5-1.6 us) leaves the critical path. The b12x top-k itself stays: it is
  off the critical path too, and a bit-exact copy of its Triton math (approximate exp/sqrt, reduction order of the
  renormalisation) would gain nothing measurable.
``FLARE_DS41_EXL3_MERGE_ROPEQ=1`` (c-ii, decode/verify attention): inverse RoPE (rotary.py ``_rotate``, in place,
  rounded to bf16) and b12x's WO-A input quantizer (``_quantize_grouped_tgd_to_tdg_kernel``) in one Triton kernel; the
  rest of b12x's prepared WO projection (WO-A GEMM, WO-B) runs unchanged on the same quantized bytes. The attention
  output is no longer rotated in place (nothing after the projection reads it; observations do not use it).
``FLARE_DS41_EXL3_MERGE_EPILOGUE=1`` (c-iii, needs ``FLARE_DS41_EXL3_SHARED_SIDE=1``): the op's ``out.copy_(acc)`` and
  the graph's ``routed + shared`` become one kernel (the op already reads the shared output as its ordering operand).
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import torch
import triton
import triton.language as tl

KERNELS = Path(__file__).resolve().parent / "kernels" / "merges"


def _switch(name: str) -> bool:
    value = os.environ.get(name, "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"{name}={value!r}: expected 0 or 1")
    return value == "1"


def merge_topk() -> bool:
    return _switch("FLARE_DS41_EXL3_MERGE_TOPK")


def merge_ropeq() -> bool:
    return _switch("FLARE_DS41_EXL3_MERGE_ROPEQ")


def merge_epilogue() -> bool:
    return _switch("FLARE_DS41_EXL3_MERGE_EPILOGUE")


MERGE_TOPK = merge_topk()
MERGE_ROPEQ = merge_ropeq()
MERGE_EPILOGUE = merge_epilogue()


def sources() -> list[Path]:
    return [KERNELS / "exl3_merges.cu"]


@cache
def extension():
    """The merges extension (own digest-named build; no --use_fast_math, see exl3_merges.cu)."""
    from ignition.recipes.deepseek41.brrr import _cuext

    return _cuext().build("ds41_exl3_merges", sources(), sources(), extra_cuda_cflags=("-lineinfo",),
                          extra_cflags=("-O3",))


# ---- c-i / c-iii: reference sequences (what the merged kernels must reproduce bit for bit) ----------------------------

def torch_mapped_prep(ids, hot_map, weights, hidden, num_experts: int, hot_rows: int):
    """decoder.py's hot_map index_select followed by the op's torch prep (experts.torch_prep)."""
    from .experts import torch_prep

    if hot_map is not None:
        ids = torch.index_select(hot_map, 0, ids.reshape(-1)).view(ids.shape)
    return torch_prep(ids, weights, hidden, num_experts, hot_rows)


def torch_epilogue(acc, add):
    """The op's ``out.copy_(acc)`` (round to bf16) then the graph's ``routed + shared``."""
    return acc.to(torch.bfloat16) + add


# ---- c-ii: inverse RoPE + WO-A input quantization -------------------------------------------------------------------

@triton.jit
def _inv_rope_quantize_kernel(
    source,
    positions,
    frequencies,
    values,
    scale_rows,
    scale_mma,
    tokens,
    groups: tl.constexpr,
    group_width: tl.constexpr,
    source_stride_t,
    source_stride_g,
    source_stride_d,
    values_stride_t,
    values_stride_d,
    values_stride_g,
    scale_mma_s0,
    scale_mma_s1,
    scale_mma_s2,
    scale_mma_s3,
    scale_mma_s4,
    scale_mma_s5,
    CHUNKS_PER_PROGRAM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    SIGN: tl.constexpr,
    WRITE_BACK: tl.constexpr,
) -> None:
    """b12x ``_quantize_grouped_tgd_to_tdg_kernel`` with rotary.py ``_rotate`` (SIGN -1: inverse) applied to the rope
    lanes first, exactly as the two kernels compute them: per pair (a, b) = (even, odd) element, angle = position *
    frequency, cos, sin * SIGN, a * cos - b * sin and a * sin + b * cos in fp32, rounded to the source dtype, then
    quantized from that rounded value. WRITE_BACK also stores the rotated lanes into the source (the in-place rotation;
    tests compare them, the decoder path does not need them). Each rope pair lies in one 32-wide chunk (NOPE_DIM and
    HEAD_DIM even)."""
    token = tl.program_id(0)
    group = tl.program_id(1)
    chunk_block = tl.program_id(2)
    chunk = chunk_block * CHUNKS_PER_PROGRAM + tl.arange(0, CHUNKS_PER_PROGRAM)
    d = chunk[:, None] * 32 + tl.arange(0, 32)[None, :]

    base = source + token * source_stride_t + group * source_stride_g
    raw = tl.load(base + d * source_stride_d)
    x = raw.to(tl.float32)
    head_d = d % HEAD_DIM
    is_rope = head_d >= NOPE_DIM
    odd = (d & 1) == 1
    partner = tl.load(base + (d ^ 1) * source_stride_d, mask=is_rope, other=0.0).to(tl.float32)
    pair = tl.maximum(head_d - NOPE_DIM, 0) >> 1
    angle = tl.load(positions + token).to(tl.float32) * tl.load(frequencies + pair, mask=is_rope, other=0.0)
    cos, sin = tl.cos(angle), tl.sin(angle) * SIGN
    a = tl.where(odd, partner, x)
    b = tl.where(odd, x, partner)
    first = a * cos - b * sin
    second = a * sin + b * cos
    rotated = tl.where(odd, second, first).to(source.dtype.element_ty)
    if WRITE_BACK:
        tl.store(base + d * source_stride_d, rotated, mask=is_rope)
    src = tl.where(is_rope, rotated, raw).to(tl.float32)

    max_abs = tl.max(tl.abs(src), axis=1)
    quant_scale = tl.where(max_abs > 0.0, max_abs / 448.0, 1.0)
    scale_exp = tl.minimum(tl.maximum(tl.ceil(tl.log2(quant_scale)), -127.0), 127.0)
    scale = tl.exp2(scale_exp)
    scale_u8 = (scale_exp + 127.0).to(tl.uint8)

    tl.store(
        values
        + token * values_stride_t
        + d * values_stride_d
        + group * values_stride_g,
        (src / scale[:, None]).to(tl.float8e4nv),
    )

    sf_cols = group_width // 32
    tl.store(
        scale_rows + group * tokens * sf_cols + token * sf_cols + chunk,
        scale_u8,
    )

    row32 = token % 32
    row4 = (token // 32) % 4
    tile_m = token // 128
    k4 = chunk % 4
    tile_k = chunk // 4
    tl.store(
        scale_mma
        + row32 * scale_mma_s0
        + row4 * scale_mma_s1
        + tile_m * scale_mma_s2
        + k4 * scale_mma_s3
        + tile_k * scale_mma_s4
        + group * scale_mma_s5,
        scale_u8,
    )


def _rope_args(source_tgd, positions, frequencies, out, chunks, head_dim, nope_dim, write_back=False):
    """Arguments in the order of b12x ``_WOQuantizers.quantize_a`` plus the rope operands."""
    return ((int(source_tgd.shape[0]), int(source_tgd.shape[1]), int(source_tgd.shape[2]) // 32 // chunks),
            (source_tgd, positions, frequencies, out.values, out.scale_rows.view(torch.uint8),
             out.scale_mma.view(torch.uint8), int(source_tgd.shape[0]), int(source_tgd.shape[1]),
             int(source_tgd.shape[2]), source_tgd.stride(0), source_tgd.stride(1), source_tgd.stride(2),
             out.values.stride(0), out.values.stride(1), out.values.stride(2) if out.values.ndim == 3 else 0,
             out.scale_mma.stride(0), out.scale_mma.stride(1), out.scale_mma.stride(2), out.scale_mma.stride(3),
             out.scale_mma.stride(4), out.scale_mma.stride(5)),
            dict(CHUNKS_PER_PROGRAM=chunks, HEAD_DIM=head_dim, NOPE_DIM=nope_dim, SIGN=-1, WRITE_BACK=bool(write_back),
                 num_warps=4))


def inv_rope_quantize(source_tgd, positions, frequencies, out, *, chunks: int, head_dim: int, nope_dim: int,
                      write_back: bool = False):
    """Launch the merged kernel into ``out`` (b12x MXFP8Rows): writes out's values/scales from ``source_tgd`` (tokens,
    groups, group_width) with its rope lanes inverse-rotated; ``write_back`` also rotates them in place."""
    grid, args, options = _rope_args(source_tgd, positions, frequencies, out, chunks, head_dim, nope_dim, write_back)
    key = (tuple(source_tgd.shape), source_tgd.dtype, source_tgd.device.index, bool(write_back), positions.dtype)
    capturing = source_tgd.is_cuda and torch.cuda.is_current_stream_capturing()
    if capturing and key not in _ROPE_WARM:
        raise RuntimeError("merged inverse-RoPE quantizer first launched inside a CUDA graph capture; "
                           "PreparedInvRopeOutput warms it up during preparation")
    _inv_rope_quantize_kernel[grid](*args, **options)
    if not capturing:
        _ROPE_WARM.add(key)


_ROPE_WARM: set = set()     # launch keys run eagerly; filled at preparation (warm), read by the guard above


class _RopeQuantizers:
    """b12x's WO quantizers with quantize_a replaced by the merged kernel (quantize_b unchanged)."""

    def __init__(self, base, positions, frequencies, head_dim, nope_dim):
        self.base, self.positions, self.frequencies = base, positions, frequencies
        self.head_dim, self.nope_dim = head_dim, nope_dim

    def quantize_a(self, source_tgd, out):
        inv_rope_quantize(source_tgd, self.positions, self.frequencies, out, chunks=self.base.a_chunks,
                          head_dim=self.head_dim, nope_dim=self.nope_dim)

    def __getattr__(self, name):
        return getattr(self.base, name)


class _StateView:
    """A prepared b12x WO state seen through other quantizers: ``type(state).run(view, binding)`` runs b12x's own
    sequence (quantize_a, WO-A GEMM, fused or quantized WO-B) with every attribute but ``quantizers`` from the state."""

    def __init__(self, state, quantizers):
        self._state, self.quantizers = state, quantizers

    def __getattr__(self, name):
        return getattr(self._state, name)


def _packed_weights(weights, tiled_a, tiled_b, query):
    from b12x.gemm._shared.wo_mxfp8 import MXFP8Rows, WOProjectionMXFP8Weights

    def matrix(parts, tiled):
        return MXFP8Rows(parts[0], parts[1].view(torch.float8_e8m0fnu), parts[2].view(torch.float8_e8m0fnu), tiled)

    return WOProjectionMXFP8Weights(matrix(weights[:3], tiled_a), matrix(weights[3:], tiled_b), query.groups,
                                    query.group_width, query.rank, query.hidden, query.sfb_k_replicated)


@torch.library.custom_op("flare::ds41_wo_inv_rope_out", mutates_args=("out", "scratch"))
def wo_inv_rope_out(source: torch.Tensor, positions: torch.Tensor, frequencies: torch.Tensor,
                    weights: list[torch.Tensor], tiled_a: torch.Tensor | None, tiled_b: torch.Tensor | None,
                    out: torch.Tensor, scratch: torch.Tensor, handle: int, head_dim: int, nope_dim: int) -> None:
    """b12x ``wo_projection_out`` (api.py ``_run_out``) with the merged inverse-RoPE quantizer. ``source`` is only read
    (the rotated lanes go straight into the quantized bytes; nothing after the projection reads the attention output)."""
    from b12x._lib.scratch import require_disjoint
    from b12x.preparation.types import plan_from_handle, require_prepared

    require_disjoint(out, (source, scratch, *weights, tiled_a, tiled_b), owner="WO output")
    plan = plan_from_handle(handle)
    state = require_prepared(plan, "gemm.wo_projection", source.device)
    query = plan.query
    binding = state.bind(source_tgd=source, weights=_packed_weights(weights, tiled_a, tiled_b, query), out=out,
                         scratch=scratch, return_3d=query.return_3d)
    quantizers = _RopeQuantizers(state.quantizers, positions, frequencies, head_dim, nope_dim)
    type(state).run(_StateView(state, quantizers), binding)


@wo_inv_rope_out.register_fake
def _wo_inv_rope_out_fake(source, positions, frequencies, weights, tiled_a, tiled_b, out, scratch, handle, head_dim,
                          nope_dim):
    pass


class PreparedInvRopeOutput:
    """attention.py's ``rotary.apply(attended, inverse=True)`` + ``output_projection(attended.view(t, -1, W))`` as
    one merged quantizer launch plus the projection's own GEMMs. Wraps the attention's PreparedGroupedOutputProjection
    (same plan, scratch and weights; its declarations and tuning are unchanged)."""

    def __init__(self, projection, frequencies, *, head_dim: int, session=None, warm_example=None):
        if head_dim % 2 or (head_dim - 64) % 2 or head_dim <= 64:
            raise ValueError("merged inverse RoPE needs an even head width with 64 trailing rope lanes")
        self.projection, self.frequencies = projection, frequencies
        self.head_dim, self.nope_dim = head_dim, head_dim - 64
        from ignition.preparation.tuning import RecordedPreparationSession, register_finalizer

        # Model preparation: warm up once plans are prepared and bound (session finalizers run after the scratch arena's,
        # so the projection's own rebind is done). A standalone session would call it right now, before preparation;
        # callers outside model preparation call warm() themselves before any capture.
        if isinstance(session, RecordedPreparationSession) and warm_example is not None:
            register_finalizer(session, lambda: self.warm(*warm_example))

    def warm(self, attended, positions):
        """Run the merged path once eagerly during preparation (on a copy of the example), so the merged quantizer is
        compiled for exactly this graph's layout before any capture."""
        with torch.no_grad():
            self(attended.detach().clone(), positions)

    def __call__(self, attended, positions):
        projection = self.projection
        output, scratch = projection.output, projection.scratch
        if output is None:
            shape, stride, dtype = projection.output_layout
            output = torch.empty_strided(shape, stride, dtype=dtype, device=attended.device)
            scratch, = (torch.empty(shape, dtype=dtype, device=attended.device)
                        for shape, dtype in projection.scratch_layout)
        source = attended.view(attended.shape[0], -1, projection.weights.group_width)
        wo_inv_rope_out(source, positions, self.frequencies, projection.operands, projection.weights.wo_a.values_tiled,
                        projection.weights.wo_b.values_tiled, output, scratch, projection.plan.handle,
                        self.head_dim, self.nope_dim)
        return output
