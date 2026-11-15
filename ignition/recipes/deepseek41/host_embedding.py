# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 fork (``tp3.embed_rows``): token-embedding rows from the host-parked table in decode, verification and draft
graphs.

With ``embed_host`` the 1.23 GiB BF16 table lives in device-visible host memory and every step reads its rows over
PCIe. b12x's embedding kernel copies a row with one dependent 2-byte load + store per thread and iteration (40 per
5120-wide row), so a lookup costs ~40 PCIe round trips (~26-28 us at any row count up to 48, 2026-11-15 trace). DSpark
draft blocks are the accepted token plus ``block - 1`` copies of one noise token, and concurrent reads of one host row
serialise: the draft lookup costs ~131 us at c4 (20 rows) and ~280-360 us at c8 (40 rows).

``HostRowGather`` replaces that lookup (and the zero fill before it) with one Triton launch that issues all of a row's
16-byte loads before its stores, and, given a noise token, reads that token's row from a 10 KB device copy instead of
the host table. Every output row is bit-identical to b12x's ``embedding_out`` after ``PreparedEmbedding``'s zero fill:
``table[ids[r]]`` for ``r < live``, zero for the rest. Two deliberate differences, both only for inputs b12x traps on
(no valid step produces them): an id outside the table is clamped into it, and a live count outside ``[0, rows]`` is
clamped into that range.
"""

import torch
import triton
import triton.language as tl
from torch.utils.weak import WeakIdKeyDictionary

from ignition.preparation.persistent import persistent_copy


@triton.jit
def _gather_host_rows(table, ids, count, noise_row, out, capacity, table_rows, noise_token,
                      WIDTH: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr,
                      HAS_COUNT: tl.constexpr, HAS_NOISE: tl.constexpr):
    row = tl.program_id(0)
    live = capacity
    if HAS_COUNT:
        live = tl.minimum(tl.maximum(tl.load(count).to(tl.int32), 0), capacity)
    active = row < live
    token = tl.load(ids + row).to(tl.int64)
    token = tl.minimum(tl.maximum(token, 0), table_rows - 1)
    cols = tl.arange(0, BLOCK)
    inside = cols < WIDTH
    if HAS_NOISE:
        noise = token == noise_token
        real = token != noise_token
        # One masked load each: a live noise row reads only the device copy, a live real row only the table.
        values = tl.load(table + token * STRIDE + cols, mask=inside & active & real, other=0.0)
        copies = tl.load(noise_row + cols, mask=inside & active & noise, other=0.0)
        values = tl.where(noise, copies, values)
    else:
        values = tl.load(table + token * STRIDE + cols, mask=inside & active, other=0.0)
    tl.store(out + row.to(tl.int64) * WIDTH + cols, values, mask=inside)


_NOISE_ROWS = WeakIdKeyDictionary()


def device_noise_row(table, token):
    """One device copy of ``table[token]`` per (table, token), shared by every graph of the rank (persistent)."""
    rows = _NOISE_ROWS.setdefault(table, {})
    if token not in rows:
        rows[token] = persistent_copy(table[token:token + 1].contiguous())
    return rows[token]


class HostRowGather:
    """Gather ``table`` rows into a fresh ``(ids.numel(), width)`` tensor inside a traced forward (b12x semantics)."""

    NUM_WARPS = 4

    def __init__(self, table, *, rows, noise_token=None):
        if table.ndim != 2 or table.stride(1) != 1 or table.stride(0) < table.shape[1]:
            raise ValueError("host row gather needs a [table_rows, width] table with contiguous rows")
        if table.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("host row gather copies BF16 or FP32 rows")
        if type(rows) is not int or rows <= 0:
            raise ValueError("host row gather needs a positive row capacity")
        if noise_token is not None and not 0 <= noise_token < table.shape[0]:
            raise ValueError("noise token outside the embedding table")
        self.table, self.rows, self.noise_token = table, rows, noise_token
        self.noise_row = None if noise_token is None else device_noise_row(table, int(noise_token))

    def __call__(self, ids, num_rows=None):
        if ids.ndim != 1 or ids.numel() != self.rows or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"host row gather expects {self.rows} flat Int32/Int64 ids")
        table = self.table
        width = table.shape[1]
        out = torch.empty((self.rows, width), dtype=table.dtype, device=ids.device)
        if out.is_cuda:
            noise = table if self.noise_row is None else self.noise_row  # dummy pointer when HAS_NOISE is off
            count = ids if num_rows is None else num_rows
            torch.library.wrap_triton(_gather_host_rows)[(self.rows,)](
                table, ids, count, noise, out, self.rows, table.shape[0],
                -1 if self.noise_token is None else self.noise_token,
                WIDTH=width, STRIDE=table.stride(0), BLOCK=triton.next_power_of_2(width),
                HAS_COUNT=num_rows is not None, HAS_NOISE=self.noise_row is not None, num_warps=self.NUM_WARPS)
            return out
        # CPU reference (tests): the same semantics with torch ops.
        live = self.rows if num_rows is None else int(num_rows.reshape(-1)[0].clamp(0, self.rows))
        tokens = ids.long().clamp(0, table.shape[0] - 1)
        out.zero_()
        values = table[tokens[:live]]
        if self.noise_row is not None:
            values = torch.where((tokens[:live] == self.noise_token)[:, None], self.noise_row, values)
        out[:live] = values
        return out
