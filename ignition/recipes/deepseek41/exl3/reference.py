# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""NumPy reference decoder for EXL3 trellis-coded weights (ExLlamaV3 format, as in the diffbot DS41 3-bit pack).

Written from the format as ExLlamaV3 defines it (turboderp-org/exllamav3 @ 5be88657, MIT, Copyright (c) 2025
Turboderp): ``exllamav3_ext/quant/codebook.cuh`` (codebook laws and constants), ``exl3_dq.cuh`` and ``reconstruct.cu``
(16-bit decode windows, tile placement), ``pack.cu`` (bit order), ``modules/quant/exl3_lib/quantize.py``
(``tensor_core_perm``, Hadamard size 128) and ``util/hadamard.py`` (Sylvester construction). No source is copied; the
constants below are format constants. The diffbot pack (MIT, Copyright (c) 2026 Diffbot) stores routed experts with
the MUL1 codebook at K=3 and the DSpark drafter experts with MCG at K=3.

Per weight matrix (``<prefix>.w{1,2,3}``) the pack holds:

- ``trellis``: int16 ``[in/16, out/16, 16*K]``: one 256-weight tile per (16-row, 16-column) block, ``K*256`` bits.
- ``suh``: fp16 ``[in]``, ``svh``: fp16 ``[out]``: input/output sign-and-scale vectors.
- ``mul1`` or ``mcg``: int32 scalar marker (the codebook multiplier).

Bit stream of one tile: uint32 word ``k`` = int16 element ``2k`` (low half) | element ``2k+1`` (high half), little
endian, and stream bit ``i`` is bit ``31 - i % 32`` of word ``i // 32`` (MSB first). Tile position ``p`` (0..255) decodes
the tail-biting 16-bit window ending at stream bit ``(p + 1) * K`` (mod ``256 * K``). Position ``p`` sits at row/column
``tensor_core_perm(p)`` inside the 16x16 tile (row = input channel, column = output channel).

Effective weight (``y = x @ W``, shape ``[in, out]``): ``W = diag(suh) . H . What . H . diag(svh)`` with ``H`` the
block-diagonal normalized Sylvester Hadamard of size 128 on each side; equivalently ``y = H((H(x * suh)) @ What) * svh``
applied per 128-channel block.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

HAD_DIM = 128
MUL1_MULTIPLIER = 0x83DCD12D
MCG_MULTIPLIER = 0xCBAC1FED
_LOP3_MASK = 0x8FFF8FFF
_LOP3_XOR = 0x3B603B60
_MUL1_KINV = np.uint16(0x1EEE).view(np.float16)   # ~1/147.7
_MUL1_KBIAS = np.uint16(0xC931).view(np.float16)  # ~(-1024 - 510) * kinv
CODEBOOKS = ("3inst", "mcg", "mul1")


def tensor_core_perm() -> np.ndarray:
    """Row-major tile index (row * 16 + column) of each of the 256 stream positions."""
    perm = np.empty(256, dtype=np.int64)
    for t in range(32):
        r0 = (t % 4) * 2
        rows = (r0, r0 + 1, r0 + 8, r0 + 9)
        c0 = t // 4
        for i in range(8):
            perm[t * 8 + i] = rows[i % 4] * 16 + (c0 if i < 4 else c0 + 8)
    return perm


@lru_cache(maxsize=None)
def _window_index(bits: int) -> np.ndarray:
    """[256, 16] stream-bit indices (MSB first) of each position's 16-bit window."""
    if not 1 <= bits <= 8:
        raise ValueError(f"EXL3 tiles use 1..8 bits per weight, not {bits}")
    total = 256 * bits
    p = np.arange(256)[:, None]
    j = np.arange(16)[None, :]
    return ((p + 1) * bits - 16 + j) % total


def tile_states(tiles: np.ndarray, bits: int) -> np.ndarray:
    """Decode windows of packed tiles: ``tiles`` int16/uint16 ``[..., 16*bits]`` -> uint32 ``[..., 256]`` (stream order)."""
    tiles = np.asarray(tiles)
    if tiles.shape[-1] != 16 * bits:
        raise ValueError(f"tile last dimension {tiles.shape[-1]} != 16 * bits ({16 * bits})")
    lead = tiles.shape[:-1]
    halves = tiles.reshape(-1, 8 * bits, 2).view(np.uint16).astype(np.uint32)
    words = halves[..., 0] | (halves[..., 1] << 16)                    # [n, 8K] uint32, LE pairs
    stream = np.unpackbits(words.astype(">u4").view(np.uint8), axis=-1)  # [n, 256K] MSB first per word
    window = stream[:, _window_index(bits)]                            # [n, 256, 16]
    weights = (1 << np.arange(15, -1, -1, dtype=np.uint32))
    return (window.astype(np.uint32) * weights).sum(-1, dtype=np.uint32).reshape(*lead, 256)


def _lop3_halves(x: np.ndarray) -> np.ndarray:
    y = ((x & _LOP3_MASK) ^ _LOP3_XOR).astype(np.uint32)
    lo = (y & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)
    hi = (y >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
    return (lo + hi).astype(np.float16)   # one fp16 rounding, as __hadd


def codebook_values(states: np.ndarray, codebook: str) -> np.ndarray:
    """fp16 value of each 16-bit decode window, bit-exact with ExLlamaV3's ``decode_3inst<cb>``."""
    s = np.asarray(states, dtype=np.uint64) & 0xFFFF
    if codebook == "mul1":
        x = (s * MUL1_MULTIPLIER) & 0xFFFFFFFF
        byte_sum = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)
        h = 1024.0 + byte_sum.astype(np.float64)                       # 0x6400 + sum as fp16 bits = 1024 + sum, exact
        exact = h * float(_MUL1_KINV) + float(_MUL1_KBIAS)            # exact in float64 (11-bit x 11-bit)
        return exact.astype(np.float16)                                # single rounding = fma.rn.f16
    if codebook == "mcg":
        return _lop3_halves((s * MCG_MULTIPLIER) & 0xFFFFFFFF)
    if codebook == "3inst":
        return _lop3_halves((s * 89226354 + 64248484) & 0xFFFFFFFF)
    raise ValueError(f"unknown EXL3 codebook {codebook!r}; expected one of {CODEBOOKS}")


def codebook_from_markers(markers: dict) -> str:
    """Codebook from the tensors present next to ``trellis`` (``mul1`` / ``mcg`` scalars, or none = 3INST)."""
    if "mul1" in markers:
        value = int(np.asarray(markers["mul1"]).astype(np.int64).reshape(-1)[0]) & 0xFFFFFFFF
        if value != MUL1_MULTIPLIER:
            raise ValueError(f"mul1 marker {value:#010x} != {MUL1_MULTIPLIER:#010x}")
        return "mul1"
    if "mcg" in markers:
        value = int(np.asarray(markers["mcg"]).astype(np.int64).reshape(-1)[0]) & 0xFFFFFFFF
        if value != MCG_MULTIPLIER:
            raise ValueError(f"mcg marker {value:#010x} != {MCG_MULTIPLIER:#010x}")
        return "mcg"
    return "3inst"


def decode_trellis(trellis: np.ndarray, codebook: str, *, chunk_tiles: int = 4096) -> np.ndarray:
    """``What`` as fp16 ``[in, out]`` from ``trellis`` ``[in/16, out/16, 16*K]`` (bit-exact with ``ext.reconstruct``)."""
    trellis = np.asarray(trellis)
    if trellis.ndim != 3 or trellis.shape[-1] % 16:
        raise ValueError(f"trellis must be [in/16, out/16, 16*K], got {trellis.shape}")
    bits = trellis.shape[-1] // 16
    rows, cols = trellis.shape[:2]
    flat = trellis.reshape(rows * cols, 16 * bits)
    values = np.empty((rows * cols, 256), dtype=np.float16)
    for start in range(0, flat.shape[0], chunk_tiles):
        stop = min(start + chunk_tiles, flat.shape[0])
        values[start:stop] = codebook_values(tile_states(flat[start:stop], bits), codebook)
    tiles = np.empty_like(values)
    tiles[:, tensor_core_perm()] = values                              # stream order -> row-major 16x16
    return tiles.reshape(rows, cols, 16, 16).transpose(0, 2, 1, 3).reshape(rows * 16, cols * 16)


@lru_cache(maxsize=None)
def hadamard(n: int = HAD_DIM) -> np.ndarray:
    """Normalized Sylvester Hadamard matrix (float64), the construction ExLlamaV3 uses for powers of two."""
    if n < 1 or n & (n - 1):
        raise ValueError(f"Sylvester Hadamard needs a power of two, not {n}")
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def _had_rows(x: np.ndarray) -> np.ndarray:
    k, n = x.shape
    return (hadamard() @ x.reshape(k // HAD_DIM, HAD_DIM, n)).reshape(k, n)


def _had_cols(x: np.ndarray) -> np.ndarray:
    k, n = x.shape
    return (x.reshape(k, n // HAD_DIM, HAD_DIM) @ hadamard()).reshape(k, n)


def effective_weight(trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray, codebook: str) -> np.ndarray:
    """``W = diag(suh) H What H diag(svh)`` in float64, ``[in, out]`` (``y = x @ W``)."""
    w = decode_trellis(trellis, codebook).astype(np.float64)
    suh = np.asarray(suh, dtype=np.float64)
    svh = np.asarray(svh, dtype=np.float64)
    if w.shape != (suh.size, svh.size):
        raise ValueError(f"trellis decodes to {w.shape}, suh/svh give {(suh.size, svh.size)}")
    return _had_cols(_had_rows(w) * suh[:, None]) * svh[None, :]


def apply(x: np.ndarray, w_hat: np.ndarray, suh: np.ndarray, svh: np.ndarray) -> np.ndarray:
    """``x @ W`` the way the kernels run it: rotate the activations, multiply by ``What``, rotate back (float64)."""
    x = np.asarray(x, dtype=np.float64)
    xin = _had_cols(x * np.asarray(suh, dtype=np.float64)[None, :])
    return _had_cols(xin @ np.asarray(w_hat, dtype=np.float64)) * np.asarray(svh, dtype=np.float64)[None, :]


def swiglu(gate: np.ndarray, up: np.ndarray, limit: float, *, order: str = "xmoe") -> np.ndarray:
    """SwiGLU with DS41's limit. ``xmoe``: ``min(silu(g), L) * clamp(u, -L, L)`` (the kernel's hadamard_inner.cuh);
    ``deepseek``: ``silu(min(g, L)) * clamp(u, -L, L)`` (the reference model). They differ only where g > ~L."""
    up = np.clip(up, -limit, limit)
    if order == "xmoe":
        return np.minimum(gate / (1.0 + np.exp(-gate)), limit) * up
    if order == "deepseek":
        g = np.minimum(gate, limit)
        return g / (1.0 + np.exp(-g)) * up
    raise ValueError(f"unknown SwiGLU order {order!r}")


def moe_reference(x: np.ndarray, topk_ids: np.ndarray, topk_weights: np.ndarray, experts: dict, *,
                  limit: float = 10.0, order: str = "xmoe") -> np.ndarray:
    """Float64 routed-expert output ``sum_k w_k * down(swiglu(gate(x), up(x)))``; ``experts[e] = {"w1","w2","w3"}``
    effective weights ``[in, out]`` (rank-sliced weights give this rank's partial sum)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros((x.shape[0], experts[next(iter(experts))]["w2"].shape[1]))
    for row in range(x.shape[0]):
        for e, w in zip(topk_ids[row], topk_weights[row]):
            m = experts[int(e)]
            act = swiglu(x[row] @ m["w1"], x[row] @ m["w3"], limit, order=order)
            out[row] += float(w) * (act @ m["w2"])
    return out


def pack_symbols(symbols: np.ndarray, bits: int) -> np.ndarray:
    """Inverse of the stream layout, for tests: K-bit symbols ``[..., 256]`` (stream order) -> int16 ``[..., 16*K]``.

    Follows ExLlamaV3's ``pack_trellis`` bit order (MSB-first symbols, 16-bit halves swapped inside each uint32)."""
    symbols = np.asarray(symbols, dtype=np.uint32) & ((1 << bits) - 1)
    lead = symbols.shape[:-1]
    flat = symbols.reshape(-1, 256)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    stream = ((flat[:, :, None] >> shifts) & 1).astype(np.uint8).reshape(flat.shape[0], 256 * bits)
    words = np.packbits(stream, axis=-1).view(">u4").astype(np.uint32)   # [n, 8K] MSB-first words
    halves = np.stack([words & 0xFFFF, words >> 16], axis=-1).astype(np.uint16)
    return halves.reshape(*lead, 16 * bits).view(np.int16)
