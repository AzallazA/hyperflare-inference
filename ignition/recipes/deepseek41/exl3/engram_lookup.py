# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Staged int4 Engram lookup (diffbot ``int4_fp16s32`` rows) on our staging interface, recipe-local (no b12x edit).

Helios stages each request's rows before the graph consumes them, exactly as for FP8/MXFP4 tables: ``weights`` uint8
``[tokens*24, 128]`` (nibble pairs, even column low) and ``scales`` uint8 ``[tokens*24, 16]`` (8 little-endian FP16 scales,
one per 32 columns), filled from RAM or from disk through the row cache (a serve-time choice; the package is the same).
The graph waits for the readiness event (the shared ``wait_for_readiness``) and expands rows into the BF16 ``[tokens,
6144]`` layer input: ``(nibble - 8) * scale`` in FP32, as diffbot's lookup computes it (``engram_int4.decode_rows``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ignition.components.readiness import wait_for_readiness

HEADS, WIDTH, ROW_BYTES, SCALE_BYTES = 24, 256, 128, 16


@triton.jit
def _lookup_int4(weight, scales, hashes, num_tokens, out, T: tl.constexpr):
    t, head = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, 256)
    nt = tl.load(num_tokens)
    live = (t < nt) & (nt >= 0) & (nt <= T)
    row = tl.load(hashes + t.to(tl.int64) * 24 + head, live, -1).to(tl.int64)
    local = live & (row >= 0)
    staged = t.to(tl.int64) * 24 + head
    byte = tl.load(weight + staged * 128 + (col // 2).to(tl.int64), local, 0x88).to(tl.int32)
    nib = tl.where(col % 2 == 0, byte & 0xF, (byte >> 4) & 0xF)
    lo = tl.load(scales + staged * 16 + (col // 32).to(tl.int64) * 2, local, 0).to(tl.int32)
    hi = tl.load(scales + staged * 16 + (col // 32).to(tl.int64) * 2 + 1, local, 0).to(tl.int32)
    scale = (lo | (hi << 8)).to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)
    result = ((nib - 8).to(tl.float32) * scale).to(tl.bfloat16)
    tl.store(out + t.to(tl.int64) * 6144 + head * 256 + col, tl.where(local, result, 0))


def lookup_int4(weight, scales, hashes, count, out):
    """Expand staged rows; ``hashes`` int64 ``[T, 24]`` (-1 = absent), ``count`` int32/64 ``[1]`` live tokens."""
    tokens = hashes.shape[0]
    if weight.shape[-1] != ROW_BYTES or scales.shape[-1] != SCALE_BYTES or out.shape[-1] != HEADS * WIDTH:
        raise ValueError("int4 Engram lookup needs [rows,128] weights, [rows,16] scales and [tokens,6144] output")
    _lookup_int4[(tokens, HEADS, 1)](weight, scales, hashes, count, out, T=tokens, num_warps=4)


@torch.library.custom_op("flare::staged_engram_int4", mutates_args=("out", "timings", "barrier_output"))
def staged_engram_int4(source: torch.Tensor, weight: torch.Tensor, scales: torch.Tensor,
                       hashes: torch.Tensor, count: torch.Tensor, out: torch.Tensor,
                       timings: torch.Tensor, frame: torch.Tensor, barrier_input: torch.Tensor,
                       barrier_output: torch.Tensor, interval: int, barrier: int, event: int) -> None:
    wait_for_readiness(source, event, timings, frame, interval, barrier_input, barrier_output, barrier)
    lookup_int4(weight, scales, hashes, count, out)


@staged_engram_int4.register_fake
def _staged_engram_int4_fake(source, weight, scales, hashes, count, out, timings, frame, barrier_input,
                             barrier_output, interval, barrier, event):
    pass


class PreparedEngramLookupInt4:
    """``PreparedEngramLookup`` for int4 rows: same inputs, readiness wait and call signature, no b12x plan."""

    def __init__(self, inputs, output, count, *, layer, token_map, geometry, session, name, runtime=None,
                 coordinator=None):
        from ignition.components.readiness import PreparedReadinessWait

        tokens = output.shape[0]
        self.weights = inputs.weights[:tokens * 24]
        self.scales = inputs.scales[:tokens * 24]
        self.hashes = inputs.hashes[:tokens]
        self.count, self.output = count, output
        self.handle = inputs.ready.cuda_event
        self.wait = PreparedReadinessWait(inputs.timings, inputs.interval, runtime=runtime, session=session,
                                         coordinator=coordinator, name=name + ".readiness")

    def functional(self, *, session):
        self.output_layout = tuple(self.output.shape), self.output.dtype
        self.count = self.output = None
        self.wait.functional(session=session)
        return self

    def __call__(self, source, count=None):
        output = self.output
        if output is None:
            shape, dtype = self.output_layout
            output = torch.zeros(shape, dtype=dtype, device=source.device)
        barrier_input, barrier_output = self.wait.operands(source)
        staged_engram_int4(source, self.weights, self.scales, self.hashes, self.count if count is None else count,
                           output, self.wait.timings.timestamps, self.wait.timings.frame, barrier_input,
                           barrier_output, self.wait.interval, self.wait.handle, self.handle)
        return output
