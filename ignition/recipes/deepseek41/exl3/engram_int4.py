# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""int4 Engram rows (diffbot DS41 EXL3 pack, ``engram_embed_format = "int4_fp16s32"``): format and NumPy reference lookup.

Per table (layers 1 and 14): ``layers.L.engram.embed.weight`` U8 ``[rows, 128]`` and ``.scale`` F16 ``[rows, 8]``.
Row ``r`` holds 256 values; byte ``i`` packs column ``2i`` in the low nibble and ``2i+1`` in the high nibble; a nibble is
``q + 8`` with ``q`` in ``[-8, 7]`` (no zero point); the scale is per 32 columns (absmax / 7.5 at conversion). The looked-up
value is ``(nibble - 8) * scale`` computed in FP32 and stored as BF16, as diffbot's lookup kernel does. 144 bytes per row
(FP8: 264, our MXFP4: 136); 384,006,168 and 384,016,682 rows = 51.5 GiB each, 103 GiB both.

In packages this is Engram encoding ``INT4_FP16_BLOCK32`` (16): 0-2 are upstream's, 3 is our MXFP4 type and upstream PR
#161 also claims 3 for its TRELLIS_K4 rows, so the next free small number (4) is the one most likely to be taken by a later
upstream addition; 16 starts a fork-local range that neither side uses.
"""

from __future__ import annotations

import numpy as np

ENCODING_NAME = "INT4_FP16_BLOCK32"
ENCODING_VALUE = 16
WIDTH, BLOCK, ROW_BYTES, SCALE_BYTES = 256, 32, 128, 16


def _bf16(values: np.ndarray) -> np.ndarray:
    """Round FP32 to BF16 (round to nearest even), returned as FP32 values."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16 << 16
    return rounded.astype(np.uint32).view(np.float32)


def decode_rows(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """``packed`` uint8 ``[n, 128]``, ``scales`` float16 ``[n, 8]`` -> BF16-rounded FP32 ``[n, 256]``."""
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float16)
    if packed.shape[-1] != ROW_BYTES or scales.shape[-1] != WIDTH // BLOCK or packed.shape[:-1] != scales.shape[:-1]:
        raise ValueError(f"int4 Engram rows need [n, 128] bytes and [n, 8] scales, got {packed.shape} / {scales.shape}")
    q = np.empty((*packed.shape[:-1], WIDTH), dtype=np.float32)
    q[..., 0::2] = (packed & 0xF).astype(np.float32) - 8
    q[..., 1::2] = (packed >> 4).astype(np.float32) - 8
    return _bf16(q * np.repeat(scales.astype(np.float32), BLOCK, axis=-1))


def lookup(indices: np.ndarray, packed_table, scale_table) -> np.ndarray:
    """Reference lookup: ``indices`` int ``[...]`` into row-addressable tables (arrays or memmaps) -> ``[..., 256]``."""
    flat = np.asarray(indices, dtype=np.int64).reshape(-1)
    rows = decode_rows(np.asarray(packed_table[flat]), np.asarray(scale_table[flat]))
    return rows.reshape(*np.shape(indices), WIDTH)


def encode_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """diffbot's conversion (convert_engram_int4.py, described not copied): scale = fp16(absmax / 7.5) per 32,
    q = clamp(round(x / scale), -8, 7) (zero scale -> 1), nibble = q + 8. For tests."""
    x = np.asarray(values, dtype=np.float32).reshape(-1, WIDTH // BLOCK, BLOCK)
    s = (np.abs(x).max(-1, keepdims=True) / np.float32(7.5)).astype(np.float16)
    sf = s.astype(np.float32)
    sf = np.where(sf == 0, np.float32(1), sf)
    q = np.clip(np.rint(x / sf), -8, 7).astype(np.int32).reshape(-1, WIDTH) + 8
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).astype(np.uint8)
    return packed.reshape(*np.shape(values)[:-1], ROW_BYTES), s.reshape(*np.shape(values)[:-1], WIDTH // BLOCK)
