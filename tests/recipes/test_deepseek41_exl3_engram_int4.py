# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the int4 Engram row format (diffbot pack). The file test needs FLARE_DS41_EXL3_PACK with a complete
model-00047/00048 shard and FLARE_DS41_SNAPSHOT (official 1017 FP8 tables); it reads only sampled rows."""

import json
import mmap
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from ignition.recipes.deepseek41.exl3 import engram_int4 as e4


def test_nibble_order_and_values():
    packed = np.zeros((1, 128), dtype=np.uint8)
    packed[0, 0] = 0x0F          # column 0 = 15 - 8 = 7, column 1 = 0 - 8 = -8
    packed[0, 1:] = 0x88         # zeros
    scales = np.full((1, 8), 0.5, dtype=np.float16)
    row = e4.decode_rows(packed, scales)[0]
    assert row[0] == 3.5 and row[1] == -4.0 and not row[2:].any()


def test_encode_decode_roundtrip():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((64, 256)).astype(np.float32) * rng.uniform(0.01, 2, (64, 1)).astype(np.float32)
    packed, scales = e4.encode_rows(x)
    y = e4.decode_rows(packed, scales)
    step = np.repeat(scales.astype(np.float32), 32, axis=1)
    assert np.all(np.abs(y - x) <= step * 0.5 + np.abs(x) * 2 ** -7 + 1e-6)   # half a step + BF16 rounding


def _tensor_rows(path, name, idx):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        e = h[name]
        item = {"U8": 1, "F8_E4M3": 1, "F8_E8M0": 1, "F16": 2}[e["dtype"]]
        width = e["shape"][1] * item
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            raw = b"".join(mm[8 + n + e["data_offsets"][0] + int(r) * width:][:width] for r in idx)
    dt = np.float16 if e["dtype"] == "F16" else np.uint8
    return np.frombuffer(raw, dtype=np.uint8).view(dt).reshape(len(idx), -1)


def _e4m3(b):
    b = b.astype(np.int64)
    e, m = (b >> 3) & 0xF, b & 7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7))
    return np.where(b & 0x80, -v, v).astype(np.float32)


@pytest.mark.skipif(not (os.environ.get("FLARE_DS41_EXL3_PACK") and os.environ.get("FLARE_DS41_SNAPSHOT")),
                    reason="needs FLARE_DS41_EXL3_PACK and FLARE_DS41_SNAPSHOT")
@pytest.mark.parametrize("layer,shard", [(1, "model-00047-of-00048.safetensors"), (14, "model-00048-of-00048.safetensors")])
def test_pack_rows_are_the_described_conversion_of_fp8(layer, shard):
    pack = Path(os.environ["FLARE_DS41_EXL3_PACK"]) / shard
    snap = Path(os.environ["FLARE_DS41_SNAPSHOT"]) / shard
    if not pack.is_file():
        pytest.skip(f"{pack.name} not complete yet")
    name = f"layers.{layer}.engram.embed"
    idx = np.sort(np.random.default_rng(layer).choice(384_000_000, 2000, replace=False))
    packed, scales = _tensor_rows(pack, name + ".weight", idx), _tensor_rows(pack, name + ".scale", idx)
    w, s = _tensor_rows(snap, name + ".weight", idx), _tensor_rows(snap, name + ".scale", idx)
    fp8 = _e4m3(w) * np.repeat(np.exp2(s.astype(np.float32) - 127), 32, axis=1)
    my_packed, my_scales = e4.encode_rows(fp8)
    assert np.array_equal(my_scales, scales)
    assert np.mean(my_packed == packed) > 0.9999          # rounding of exact .5 cases may differ from torch division
    rows = e4.decode_rows(packed, scales)
    cos = np.sum(rows * fp8, 1) / np.maximum(np.linalg.norm(rows, axis=1) * np.linalg.norm(fp8, axis=1), 1e-30)
    print(f"{name}: identical bytes {np.mean(my_packed == packed):.6f}, row cosine mean {cos.mean():.5f} min {cos.min():.5f}")
    assert cos.mean() > 0.99
