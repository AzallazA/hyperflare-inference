# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the EXL3 reference decoder (torch-free).

The real-weight test needs FLARE_DS41_EXL3_PACK (diffbot DS41 EXL3 pack directory, or a directory of raw tensor
slices ``<name>.bin``) and FLARE_DS41_SNAPSHOT (official DeepSeek-V4.1-Flash-1017 snapshot); it is skipped otherwise.
"""

import json
import os
import struct
from pathlib import Path

import numpy as np
import pytest

from ignition.recipes.deepseek41.exl3 import reference as ref


def _emulate_reconstruct_tile(values_stream: np.ndarray) -> np.ndarray:
    """Tile placement written independently from reconstruct.cu's warp shuffle (one warp = one 16x16 tile)."""
    tile = np.zeros((16, 16), dtype=values_stream.dtype)
    frag = values_stream.reshape(32, 8)          # lane l decodes stream positions 8l..8l+7 (dq8 t_offset = lane*8)
    for lane in range(32):
        if lane & 4:
            continue
        mine, other = frag[lane], frag[lane + 4]  # __shfl_down_sync(..., 4)
        r0 = (lane % 4) * 2
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        c0 = lane // 8
        for half, cpair in ((0, c0), (1, c0 + 4)):
            for i, row in enumerate(rows):
                v = 4 * half + i
                tile[row, 2 * cpair] = mine[v]
                tile[row, 2 * cpair + 1] = other[v]
    return tile


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_pack_and_windows(bits):
    rng = np.random.default_rng(bits)
    symbols = rng.integers(0, 1 << bits, size=(5, 256), dtype=np.uint32)
    states = ref.tile_states(ref.pack_symbols(symbols, bits), bits)
    assert np.array_equal(states & ((1 << bits) - 1), symbols)
    # A window holds the previous symbols too (tail-biting): rebuild it from the symbols directly.
    per = -(-16 // bits)
    for p in (0, 1, 100, 255):
        value = 0
        for back in range(per - 1, -1, -1):
            value = (value << bits) | int(symbols[0, (p - back) % 256])
        assert states[0, p] == value & 0xFFFF


def test_codebook_laws():
    states = np.arange(1 << 16, dtype=np.uint32)
    mul1 = ref.codebook_values(states, "mul1").astype(np.float64)
    x = (states.astype(np.uint64) * ref.MUL1_MULTIPLIER) & 0xFFFFFFFF
    byte_sum = sum(((x >> s) & 0xFF) for s in (0, 8, 16, 24)).astype(np.float64)
    assert np.max(np.abs(mul1 - (byte_sum - 510.0) / 147.7)) < 0.02         # the law, up to fp16 constants
    for book in ("mul1", "mcg", "3inst"):
        v = ref.codebook_values(states, book).astype(np.float64)
        assert np.all(np.isfinite(v)) and abs(v.mean()) < 0.05 and 0.8 < v.std() < 1.6, book
    assert ref.codebook_values(np.array([0]), "mul1")[0] == np.float16(1024 * float(ref._MUL1_KINV) + float(ref._MUL1_KBIAS))


def test_tile_placement_matches_reconstruct_shuffle():
    stream = np.arange(256, dtype=np.float64)
    tile = np.empty(256)
    tile[ref.tensor_core_perm()] = stream
    assert np.array_equal(tile.reshape(16, 16), _emulate_reconstruct_tile(stream))
    assert sorted(ref.tensor_core_perm()) == list(range(256))


def test_decode_places_tiles_and_transform_identity():
    rng = np.random.default_rng(7)
    bits, rows, cols = 3, 256, 384
    symbols = rng.integers(0, 8, size=(rows // 16, cols // 16, 256), dtype=np.uint32)
    trellis = ref.pack_symbols(symbols, bits)
    w_hat = ref.decode_trellis(trellis, "mul1")
    values = ref.codebook_values(ref.tile_states(trellis, bits), "mul1")       # [R/16, C/16, 256] stream order
    for (tr, tc) in ((0, 0), (3, 7), (15, 23)):
        expect = _emulate_reconstruct_tile(values[tr, tc])
        assert np.array_equal(w_hat[tr * 16:(tr + 1) * 16, tc * 16:(tc + 1) * 16], expect)
    suh = rng.choice([-1.0, 1.0], rows) * rng.uniform(0.5, 1.5, rows)
    svh = rng.choice([-1.0, 1.0], cols) * rng.uniform(0.5, 1.5, cols)
    w = ref.effective_weight(trellis, suh, svh, "mul1")
    x = rng.standard_normal((4, rows))
    assert np.allclose(x @ w, ref.apply(x, w_hat, suh, svh), rtol=1e-10, atol=1e-9)
    h = ref.hadamard(128)
    assert np.allclose(h @ h, np.eye(128))


_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


def _safetensors_read(path: Path, name: str):
    with path.open("rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
        entry = header[name]
        begin, end = entry["data_offsets"]
        handle.seek(8 + length + begin)
        raw = handle.read(end - begin)
    dtype = {"I8": np.uint8, "F8_E8M0": np.uint8}[entry["dtype"]]
    return np.frombuffer(raw, dtype=dtype).reshape(entry["shape"])


def _official_mxfp4(snapshot: Path, name: str) -> np.ndarray:
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    shard = snapshot / weight_map[name + ".weight"]
    packed = _safetensors_read(shard, name + ".weight")
    scale = _safetensors_read(shard, name + ".scale")
    values = np.empty((packed.shape[0], packed.shape[1] * 2))
    values[:, 0::2] = _E2M1[packed & 0xF]
    values[:, 1::2] = _E2M1[packed >> 4]
    exp = np.exp2(scale.astype(np.float64) - 127.0)
    return values * np.repeat(exp, 32, axis=1)               # [out, in], torch Linear layout


def _load_pack_matrix(pack: Path, prefix: str):
    if (pack / "model.safetensors.index.json").is_file():
        from ignition.recipes.deepseek41.exl3.pack_reader import SafetensorsFile, read_matrix
        weight_map = json.loads((pack / "model.safetensors.index.json").read_text())["weight_map"]
        shard = pack / weight_map[prefix + ".trellis"]
        return read_matrix({shard.name: SafetensorsFile(shard)}, prefix)
    specs = {"trellis": np.int16, "suh": np.float16, "svh": np.float16, "mul1": np.int32}
    out = {part: np.fromfile(pack / f"{prefix}.{part}.bin", dtype=dtype) for part, dtype in specs.items()}
    return out


@pytest.mark.skipif(not (os.environ.get("FLARE_DS41_EXL3_PACK") and os.environ.get("FLARE_DS41_SNAPSHOT")),
                    reason="needs FLARE_DS41_EXL3_PACK and FLARE_DS41_SNAPSHOT")
@pytest.mark.parametrize("matrix,shape", [("w1", (320, 144)), ("w3", (320, 144)), ("w2", (144, 320))])
def test_real_expert_reconstructs_official_weights(matrix, shape):
    pack, snapshot = Path(os.environ["FLARE_DS41_EXL3_PACK"]), Path(os.environ["FLARE_DS41_SNAPSHOT"])
    prefix = f"layers.0.ffn.experts.0.{matrix}"
    parts = _load_pack_matrix(pack, prefix)
    trellis = parts["trellis"].reshape(*shape, 48)
    codebook = ref.codebook_from_markers(parts)
    assert codebook == "mul1"
    w = ref.effective_weight(trellis, parts["suh"], parts["svh"], codebook)    # [in, out]
    official = _official_mxfp4(snapshot, prefix).T                              # [in, out]
    rel = np.linalg.norm(w - official) / np.linalg.norm(official)
    cos = float(np.sum(w * official) / (np.linalg.norm(w) * np.linalg.norm(official)))
    print(f"{prefix}: relative L2 {rel:.4f}, cosine {cos:.5f}")
    assert cos > 0.95 and rel < 0.35


@pytest.mark.parametrize("matrix", ["w1", "w2"])
def test_tp2_rank_slices_compose(matrix):
    """Rank slices of the intermediate axis reproduce the full matrix's columns (w1/w3) or rows (w2) exactly."""
    from ignition.recipes.deepseek41.exl3 import xmoe

    rng = np.random.default_rng(11)
    hidden, inter = 256, 512                         # small stand-ins; slices are whole Hadamard-128 blocks
    rows, cols = (hidden, inter) if matrix != "w2" else (inter, hidden)
    parts = {"trellis": ref.pack_symbols(rng.integers(0, 8, (rows // 16, cols // 16, 256), dtype=np.uint32), 3),
             "suh": rng.uniform(-1, 1, rows).astype(np.float16), "svh": rng.uniform(-1, 1, cols).astype(np.float16)}
    full = ref.effective_weight(parts["trellis"], parts["suh"], parts["svh"], "mul1")
    pieces = []
    for rank in range(2):
        s = xmoe.shard_matrix(matrix, parts, rank, 2, intermediate=inter)
        pieces.append(ref.effective_weight(s["trellis"], s["suh"], s["svh"], "mul1"))
    joined = np.concatenate(pieces, axis=1 if matrix != "w2" else 0)
    assert np.allclose(joined, full, rtol=1e-12, atol=1e-12)
    assert xmoe.shard_interval(1, 2) == (1152, 2304)
    with pytest.raises(ValueError):
        xmoe.shard_interval(0, 2, 384)               # 192 per rank is not whole Hadamard-128 blocks
