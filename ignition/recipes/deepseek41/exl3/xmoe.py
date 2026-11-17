# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DS41 routed experts on EXL3 3-bit weights: the xmoe kernel extension and one MoE layer around it.

``kernels/xmoe`` is diffbot's sm_120 rewrite of ExLlamaV3's ``exl3_moe`` (MIT; ``kernels/THIRD_PARTY.md``). One call
runs every routed expert of a layer: input rotation (``x * suh``, Hadamard-128), gate/up GEMMs with in-register trellis
decode, output rotation (Hadamard-128, ``* svh``), SwiGLU with the limit, the same for down, and the route-weighted sum
into an FP32 output. This module adds what the kernel leaves to its caller (routing sort, counts, scratch, pointer
tables) in a capture-safe form: fixed shapes per row bucket, no host reads of device data, no data-dependent Python.

Tensor parallelism: the intermediate axis is split. Rank ``r`` of ``w`` takes gate/up output columns and down input
rows ``[r*I/w, (r+1)*I/w)``; at I = 2304 that is 1152 per rank at TP2 (9 Hadamard-128 blocks; the 128-blocks never
straddle ranks, so the rotations stay rank-local). suh of gate/up and svh of down are hidden-axis vectors every rank
keeps whole. Each rank's output is a partial sum of the layer output and joins the layer's existing all-reduce; no new
exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

KERNELS = Path(__file__).resolve().parent / "kernels" / "xmoe"
HIDDEN = 5120
INTERMEDIATE = 2304
TOP_K = 6
TILE = 16
HAD = 128
MOE_SMS_PER_EXPERT = 8          # exl3_moe_common.cuh: group width, also the per-group scratch count
ACT_SILU = 0
SWIGLU_LIMIT = 10.0


def sources() -> list[Path]:
    """Translation units, as diffbot's build_xmoe.py lists them (bindings, launcher, device context, instances)."""
    quant = KERNELS / "quant"
    return [KERNELS / "bindings.cpp", quant / "exl3_moe.cu", quant / "exl3_devctx.cu",
            *sorted((quant / "comp_units").glob("exl3_moe_inst_*.cu")), quant / "exl3_dense_flat.cu",
            quant / "exl3_do.cu",    # exp/ds41-exl3: decode-once prefill (diffbot xmoe_do, ported)
            quant / "exl3_prep.cu",   # exp/ds41-exl3-next: fused decode routing prep
            quant / "exl3_moe_transfer.cu"]  # exp/ds41-exl3-final-bulk: diffbot xmoe_bulk hot loop (MIT)


def digest_files() -> list[Path]:
    return sorted(path for path in KERNELS.rglob("*") if path.suffix in (".cu", ".cuh", ".cpp", ".h"))


@cache
def extension():
    """Build/load the xmoe extension through the shared BRRR builder (same build dir convention, digest in the name)."""
    from ignition.recipes.deepseek41.brrr import _cuext

    return _cuext().build("ds41_xmoe", sources(), digest_files(),
                          extra_cuda_cflags=("--use_fast_math", "-lineinfo"), extra_cflags=("-O3",))


def shard_interval(rank: int, world: int, intermediate: int = INTERMEDIATE) -> tuple[int, int]:
    """Rank slice of the intermediate axis; must be whole Hadamard-128 blocks (TP2: 1152 = 9 x 128)."""
    if intermediate % world or (intermediate // world) % HAD:
        raise ValueError(f"intermediate {intermediate} does not split into Hadamard-128 blocks over {world} ranks")
    local = intermediate // world
    return rank * local, (rank + 1) * local


def shard_matrix(name: str, parts: dict, rank: int, world: int, intermediate: int = INTERMEDIATE) -> dict:
    """One expert matrix's rank slice (NumPy or torch arrays): ``w1``/``w3`` split columns, ``w2`` splits rows."""
    first, last = shard_interval(rank, world, intermediate)
    trellis, suh, svh = parts["trellis"], parts["suh"], parts["svh"]
    if name in ("w1", "w3"):
        trellis = trellis[:, first // TILE:last // TILE, :]
        svh = svh[first:last]
    elif name == "w2":
        trellis = trellis[first // TILE:last // TILE, :, :]
        suh = suh[first:last]
    else:
        raise ValueError(f"unknown expert matrix {name!r}")
    out = dict(parts)
    out.update(trellis=trellis, suh=suh, svh=svh)
    return out


@dataclass
class ExpertBank:
    """One layer's routed experts on one rank: stacked device tensors and the kernel's per-expert pointer tables."""

    trellis: dict          # name -> int16 [E, in/16, out/16, 16K] (contiguous per expert)
    suh: dict              # name -> fp16 [E, in]
    svh: dict              # name -> fp16 [E, out]
    pointers: dict         # (name, part) -> int64 [E] device addresses
    bits: int
    codebook: str          # "mul1" (routed) or "mcg" (drafter)

    @classmethod
    def from_arrays(cls, matrices: list[dict], device, *, bits: int = 3, codebook: str = "mul1"):
        """``matrices[e] = {"w1": parts, "w2": parts, "w3": parts}`` (already rank-sliced NumPy arrays)."""
        import torch

        trellis, suh, svh, pointers = {}, {}, {}, {}
        for name in ("w1", "w2", "w3"):
            trellis[name] = torch.from_numpy(np.ascontiguousarray(np.stack([m[name]["trellis"] for m in matrices]))).to(device)
            suh[name] = torch.from_numpy(np.ascontiguousarray(np.stack([m[name]["suh"] for m in matrices]))).to(device)
            svh[name] = torch.from_numpy(np.ascontiguousarray(np.stack([m[name]["svh"] for m in matrices]))).to(device)
            for part, tensor in (("trellis", trellis[name]), ("suh", suh[name]), ("svh", svh[name])):
                step = tensor.stride(0) * tensor.element_size()
                pointers[name, part] = torch.tensor([tensor.data_ptr() + e * step for e in range(tensor.shape[0])],
                                                    dtype=torch.int64, device=device)
        return cls(trellis, suh, svh, pointers, bits, codebook)

    @property
    def experts(self) -> int:
        return self.trellis["w1"].shape[0]

    @property
    def local_intermediate(self) -> int:
        return self.trellis["w1"].shape[2] * TILE


@dataclass
class Scratch:
    """Static per-bucket buffers (allocate once per row bucket, before capture)."""

    rows: int
    groups: int
    state_g: object
    state_u: object
    inter_g: object
    inter_u: object
    counts: object
    output: object

    @classmethod
    def allocate(cls, rows: int, local_intermediate: int, experts: int, device, *, groups: int | None = None,
                 tokens_per_expert: int | None = None):
        import torch

        if groups is None:
            groups = max(1, torch.cuda.get_device_properties(device).multi_processor_count // MOE_SMS_PER_EXPERT)
        per = rows if tokens_per_expert is None else tokens_per_expert
        half = torch.float16
        return cls(rows, groups,
                   torch.empty(groups, per, HIDDEN, dtype=half, device=device),
                   torch.empty(groups, per, HIDDEN, dtype=half, device=device),
                   torch.empty(groups, per, local_intermediate, dtype=half, device=device),
                   torch.empty(groups, per, local_intermediate, dtype=half, device=device),
                   torch.empty(experts + 1, dtype=torch.int64, device=device),
                   torch.empty(rows, HIDDEN, dtype=torch.float32, device=device))


def routed_experts(hidden, topk_ids, topk_weights, bank: ExpertBank, scratch: Scratch, *, limit: float = SWIGLU_LIMIT):
    """This rank's partial routed-expert output, FP32 ``[rows, 5120]``.

    ``hidden`` bf16/fp16 ``[rows, 5120]``, ``topk_ids`` int ``[rows, 6]``, ``topk_weights`` float ``[rows, 6]`` (route scale
    already applied). Capture-safe: only fixed-shape device ops; ``num_active = -1`` keeps the launch shape static."""
    import torch

    rows, top_k = topk_ids.shape
    flat = topk_ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    token_sorted = torch.div(order, top_k, rounding_mode="floor")
    weight_sorted = topk_weights.reshape(-1).index_select(0, order).to(torch.float16)
    counts = scratch.counts
    counts.zero_()
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    out = scratch.output[:rows]
    out.zero_()
    mul1 = bank.codebook == "mul1"
    p = bank.pointers
    extension().exl3_moe(hidden.to(torch.float16).contiguous(), out, counts, token_sorted, weight_sorted,
                         scratch.state_g, scratch.state_u, scratch.inter_g, scratch.inter_u, ACT_SILU,
                         bank.bits, bank.bits, bank.bits,
                         p["w1", "trellis"], p["w1", "suh"], p["w1", "svh"],
                         p["w3", "trellis"], p["w3", "suh"], p["w3", "svh"],
                         p["w2", "trellis"], p["w2", "suh"], p["w2", "svh"],
                         not mul1, mul1, not mul1, mul1, not mul1, mul1, float(limit), -1)
    return out
