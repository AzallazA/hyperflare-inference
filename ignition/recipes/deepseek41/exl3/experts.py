# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""EXL3 routed experts in the DS41 recipe graph (exp/ds41-exl3): loading, parking, and the routed-expert call.

Banks: per layer and matrix (w1 gate, w3 up, w2 down) one stacked int16 trellis tensor ``[E, in/16, out/16, 48]``
holding this rank's slice of the intermediate axis (``xmoe.shard_matrix``), plus FP16 ``suh``/``svh`` vectors and int64
pointer tables (one address per expert) the kernel takes. A parked layer (``TP3Config.host_expert_layers``) puts its
trellis banks in one mixed VRAM/host range per matrix, hot experts first (``hot_map`` translates router IDs to bank slots,
as the MXFP4 merged bank does); the vectors stay in VRAM. Pointers are device virtual addresses of the core storage's
fixed reservation, valid as long as the package maps weights where preparation put them (as BRRR relies on).

Call: ``flare::ds41_exl3_moe`` writes the routed partial sum (BF16 ``[rows, 5120]``). Resident layers take one launch
(the flat decode kernel up to 64 rows, the group kernel above). Parked layers take two: hot slots with cold counts
negated, then cold slots on the group kernel (``xmoe_set_force_group``; the flat kernel reads host memory badly) with
hot counts negated; both accumulate into one FP32 output. Rows per expert are bounded by the scratch: decode/verify
graphs use every group at the graph's row count, prefill graphs use PREFILL_GROUPS (default 4) wide groups at the chunk
size, so no expert is ever dropped (the kernel skips experts above ``max_tokens_per_expert``).

Prefill scratch (exp/ds41-exl3-final-scratch, both switches default off): FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS runs the
decode-once pass over ranges of the expert-sorted routes (it accumulates into the FP32 output), so its scratch holds one
range instead of the chunk's rows x top_k routes; FLARE_DS41_EXL3_PREFILL_GROUPS sets the prefill group count;
FLARE_DS41_EXL3_PREFILL_COLD_SIDE=1 runs a parked layer's cold prefill launch on a side stream beside the decode-once
kernels (disjoint scratch regions, one shared FP32 output, joined before the BF16 copy).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch

from . import xmoe

MATRICES = ("w1", "w3", "w2")
PREFILL_GROUP_WIDTH = 32
# Prefill MoE path (FLARE_DS41_EXL3_PREFILL): "group" = the fused group kernel (default), "do" = diffbot's decode-once
# kernel (quant/exl3_do.cu) for the VRAM experts of graphs with >= DO_MIN_ROWS rows; cold (host) experts of parked layers
# always take the group kernel's second launch. Read at import: the choice is baked into the captured graphs.
PREFILL_MODES = ("group", "do")
DO_MIN_ROWS = 128
DO_JOB_ROWS = 128          # rows per decode-once job (DO_MC in exl3_do.cu)


def prefill_mode() -> str:
    mode = os.environ.get("FLARE_DS41_EXL3_PREFILL", "group").strip() or "group"
    if mode not in PREFILL_MODES:
        raise ValueError(f"FLARE_DS41_EXL3_PREFILL={mode!r}: expected one of {PREFILL_MODES}")
    return mode


def do_job_capacity(routed_rows: int, experts: int) -> int:
    """Fixed job-table capacity for ``routed_rows`` expert-sorted rows: each expert's rows split into 128-row jobs, so
    sum(ceil(c_e / 128)) <= ceil(R / 128) + E."""
    return -(-routed_rows // DO_JOB_ROWS) + experts


# ---- prefill scratch and cold side stream (exp/ds41-exl3-final-scratch; read at import, baked into captured graphs) ----
# FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS=<n>: routed rows (routes) per decode-once pass, a multiple of 128. The decode-once
#   kernels take any contiguous range of the expert-sorted routes: the pass gets per-expert counts clipped to its range
#   (``pass_counts``), so its job table, gathers and scatter-adds cover exactly those routes. Per route every value up to
#   the FP32 scatter-add is computed by the same instructions as in one pass (a GEMM row does not depend on the other
#   rows of its 128-row job); only the order of the FP32 atomics into the output changes (they are unordered within one
#   pass too). 0/unset = the whole chunk in one pass (rows x top_k routes: 588 MiB at 4096 rows, TP2).
# FLARE_DS41_EXL3_PREFILL_GROUPS=<g>: expert groups of prefill group-kernel launches (cold launches of parked layers in
#   the decode-once mode; every prefill launch in the group mode). Default 4. Each group keeps the chunk's rows, so no
#   expert is dropped and the cold kernel's read-once stage copy still fits; fewer groups = fewer SMs reading host
#   experts at once (32 per group).
# FLARE_DS41_EXL3_PREFILL_COLD_SIDE=1 (decode-once mode only): the cold group launch of a parked layer runs on a side
#   stream (alternate lock buffer) while the decode-once kernels run on the main stream. The decode-once kernels have no
#   grid barriers or locks, so the cold kernel's group barriers only wait for its own blocks, which become resident as
#   decode-once blocks retire. Both add into the one FP32 accumulator with atomics (as the decode cold side stream does),
#   so no second accumulator is needed; the scratch holds the group views and one decode-once range side by side.
PREFILL_GROUPS_MAX = 4


def prefill_groups() -> int:
    value = os.environ.get("FLARE_DS41_EXL3_PREFILL_GROUPS", "").strip()
    groups = int(value) if value else 4
    if not 1 <= groups <= PREFILL_GROUPS_MAX:
        raise ValueError(f"FLARE_DS41_EXL3_PREFILL_GROUPS={value!r}: expected 1..{PREFILL_GROUPS_MAX}")
    return groups


def prefill_pass_routes() -> int:
    value = os.environ.get("FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS", "").strip()
    routes = int(value) if value else 0
    if routes < 0 or routes % DO_JOB_ROWS:
        raise ValueError(f"FLARE_DS41_EXL3_PREFILL_SCRATCH_ROWS={value!r}: expected 0 or a multiple of {DO_JOB_ROWS}")
    return routes


def prefill_cold_side() -> bool:
    value = os.environ.get("FLARE_DS41_EXL3_PREFILL_COLD_SIDE", "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"FLARE_DS41_EXL3_PREFILL_COLD_SIDE={value!r}: expected 0 or 1")
    return value == "1"


def do_passes(routed: int, pass_routes: int) -> list[tuple[int, int]]:
    """[start, stop) ranges of the expert-sorted routes, one per decode-once pass (one range when pass_routes is 0)."""
    step = pass_routes if 0 < pass_routes < routed else routed
    return [(start, min(routed, start + step)) for start in range(0, routed, step)] if routed else [(0, 0)]


def pass_counts(counts, start: int, stop: int):
    """Per-expert route counts clipped to the sorted-route range [start, stop) (the experts' routes are contiguous in
    the sorted order: expert e owns [sum(|c| below e), + |c_e|)); sums to stop - start. Fixed shapes, capture-safe."""
    size = counts.abs()
    ends = torch.cumsum(size, 0)
    return (ends.clamp(max=stop) - (ends - size).clamp(min=start)).clamp(min=0)


def scratch_halfs(rows: int, local: int, groups: int, *, top_k: int = 6, decode_once: bool = False,
                  pass_routes: int = 0, cold_side: bool = False) -> int:
    """fp16 elements of the shared per-device scratch: the group kernel's [groups, rows, ...] views, or for the
    decode-once path [routes, ...] (xg, xu over hidden; cg, cu over the local intermediate) with routes = rows * top_k
    or one pass's range. ``cold_side``: the cold group launch runs beside the decode-once pass, so both regions are
    held side by side (group views first) instead of overlapping."""
    per_row = 2 * xmoe.HIDDEN + 2 * local
    need = groups * rows * per_row
    if not decode_once:
        return need
    routed = rows * top_k
    do = (min(routed, pass_routes) if pass_routes else routed) * per_row
    return need + do if cold_side else max(need, do)


def scratch_report(chunk: int, local: int, decode_rows: int, num_sms: int = 188, *, groups: int | None = None,
                   pass_routes: int | None = None, cold_side: bool | None = None, mode: str | None = None) -> dict:
    """Bytes of the shared xmoe scratch per device (the largest graph's need: the prefill chunk or decode rows), for
    the configured switches and for the defaults, as the plan check prints it."""
    groups = PREFILL_GROUPS if groups is None else groups
    pass_routes = PREFILL_PASS_ROUTES if pass_routes is None else pass_routes
    cold_side = PREFILL_COLD_SIDE if cold_side is None else cold_side
    mode = PREFILL_MODE if mode is None else mode
    decode = scratch_halfs(decode_rows, local, max(1, num_sms // xmoe.MOE_SMS_PER_EXPERT))
    do = mode == "do" and chunk >= DO_MIN_ROWS
    now = max(decode, scratch_halfs(chunk, local, groups, decode_once=do, pass_routes=pass_routes, cold_side=cold_side))
    base = max(decode, scratch_halfs(chunk, local, 4, decode_once=do))
    return {"bytes": 2 * now, "default_bytes": 2 * base, "saved_bytes": 2 * (base - now), "groups": groups,
            "pass_routes": pass_routes or chunk * xmoe.TOP_K, "cold_side": bool(cold_side and do), "mode": mode}


# Declared device rows (FLARE_DS41_EXL3_EXPORT_HOT=<n>, second TP2 export): a parked bank's mixed region is mapped with
# only min(n, hot_rows) VRAM rows during the export, so kernel tuning keeps the export-time headroom, while the kernels
# (captured graphs) use hot_rows and the package declares hot_rows VRAM rows ("mixed-<index>-<declared bytes>"): Helios
# maps that prefix as VRAM at serve. The bank is one virtual range and the pointer tables hold its addresses, so they
# are the same whichever pages are VRAM; during the export the rows [n, hot_rows) are read from pinned host memory
# through the same addresses (correct, slower). Unset = every hot row is VRAM during the export too.
def export_hot() -> int | None:
    value = os.environ.get("FLARE_DS41_EXL3_EXPORT_HOT", "").strip()
    if not value:
        return None
    rows = int(value)
    if rows < 0:
        raise ValueError(f"FLARE_DS41_EXL3_EXPORT_HOT={value!r}: expected a nonnegative row count")
    return rows


def declare_device_rows(region, rows: int, row_bytes: int) -> None:
    """Record on a mixed region the VRAM prefix the package declares (whole granules, at least the mapped prefix,
    at most the extent)."""
    granularity = int(getattr(region, "granularity", 2 << 20))
    extent = int(getattr(region, "extent", -(-rows * row_bytes // granularity) * granularity))
    declared = min(-(-rows * row_bytes // granularity) * granularity, extent)
    mapped = int(getattr(region, "device_bytes", 0))
    if declared < mapped:
        raise ValueError(f"declared VRAM prefix {declared} below the mapped {mapped} bytes")
    region.declared_device_bytes = declared
    region.declared_device_rows = rows


# Per-layer resident counts (FLARE_DS41_EXL3_HOT_LAYERS="L:h,L:h,..." or "@/path/file.json" holding {"L": h}): parked
# layers listed get h hot experts instead of the uniform HOT (FLARE_DS41_HOT_EXPERTS); the hot set is still the top-h of
# the route counts. Every bank already carries its own hot_rows (mixed region, pointer tables, split launches, the
# decode-once job builder), so nothing else changes. Default: unset = uniform.
def hot_layers() -> dict:
    value = os.environ.get("FLARE_DS41_EXL3_HOT_LAYERS", "").strip()
    if not value:
        return {}
    if value.startswith("@"):
        import json

        raw = json.loads(open(value[1:]).read())
        items = [(int(k), int(v)) for k, v in raw.items()]
    else:
        items = []
        for part in filter(None, (p.strip() for p in value.split(","))):
            layer, sep, rows = part.partition(":")
            if not sep:
                raise ValueError(f"FLARE_DS41_EXL3_HOT_LAYERS: {part!r} is not LAYER:COUNT")
            items.append((int(layer), int(rows)))
    out = {}
    for layer, rows in items:
        if not 0 <= rows <= 384 or layer < 0 or layer in out:
            raise ValueError(f"FLARE_DS41_EXL3_HOT_LAYERS: bad or repeated entry {layer}:{rows}")
        out[layer] = rows
    return out


def layer_hot_count(layer: int, uniform: int) -> int:
    return hot_layers().get(layer, uniform)


# Cold side stream (FLARE_DS41_EXL3_COLD_SIDE=<k SMs>, decode rows <= 64, parked banks): the cold (host-expert) group
# launch runs on a side stream on k SMs with the alternate lock buffer and its own scratch, concurrently with the hot
# flat launch on the remaining SMs, so PCIe reads of cold experts overlap the hot experts' compute. Grids sum to at most
# the SM count, so every block of both kernels can be resident at once (the flat kernel's grid barrier and the group
# kernel's per-group locks cannot wait on each other). Both accumulate into the same FP32 output with atomics.
# 0/unset = sequential (hot, then cold on the same stream). k must be a multiple of 8 (one expert group).
def cold_side_sms() -> int:
    value = os.environ.get("FLARE_DS41_EXL3_COLD_SIDE", "").strip()
    k = int(value) if value else 0
    if k < 0 or k % xmoe.MOE_SMS_PER_EXPERT:
        raise ValueError(f"FLARE_DS41_EXL3_COLD_SIDE={value!r}: expected 0 or a multiple of {xmoe.MOE_SMS_PER_EXPERT} SMs")
    return k


# FLARE_DS41_EXL3_COLD_WIDTH: SMs per expert group for DECODE cold (host-resident) launches. Default 0 = the kernel's
# MOE_SMS_PER_EXPERT (8): the seq229 profile shows each cold launch as 23 groups x 8 SMs, and a launch with one cold expert
# takes ~308 us for 6.6 MB (~21 GB/s, one 8-SM group reading pinned host memory) against ~48 GB/s when several experts
# overlap, so a wider group may let a lone cold expert read at link speed. Read at import (baked into captured graphs).
COLD_WIDTHS = (0, 8, 16, 24, 32)


def cold_width() -> int:
    value = os.environ.get("FLARE_DS41_EXL3_COLD_WIDTH", "").strip()
    width = int(value) if value else 0
    if width not in COLD_WIDTHS:
        raise ValueError(f"FLARE_DS41_EXL3_COLD_WIDTH={value!r}: expected one of {COLD_WIDTHS}")
    return width


# FLARE_DS41_EXL3_FUSED_PREP (exp/ds41-exl3-next): 1 = the decode routing prep (stable sort, counts, hot/cold split,
# fp16 activations, zeroed accumulator) in one kernel (quant/exl3_prep.cu) instead of ~18 torch launches per layer
# (~20 us/layer, ~0.9 ms per c1 round in the seq229 profile). Its outputs are bit-identical to the torch path. Decode
# rows only (<= 64 rows; prefill keeps the torch path). Read at import (baked into captured graphs).
def fused_prep() -> bool:
    value = os.environ.get("FLARE_DS41_EXL3_FUSED_PREP", "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"FLARE_DS41_EXL3_FUSED_PREP={value!r}: expected 0 or 1")
    return value == "1"


def fused_prep_max_rows() -> int:
    """FLARE_DS41_EXL3_FUSED_PREP_MAX_ROWS: the fused prep only for graphs with <= this many rows (decode/verify row
    counts are static per captured graph, so the path is chosen at trace time); torch prep above. Default 64 (all decode
    rows); set from the nx row sweep."""
    value = os.environ.get("FLARE_DS41_EXL3_FUSED_PREP_MAX_ROWS", "").strip()
    rows = int(value) if value else 64
    if not 1 <= rows <= 64:
        raise ValueError(f"FLARE_DS41_EXL3_FUSED_PREP_MAX_ROWS={value!r}: expected 1..64")
    return rows


# FLARE_DS41_EXL3_BULK_MIN_ROWS (exp/ds41-exl3-final-bulk): decode/verify graphs with >= this many rows run their hot
# (flat) launch on diffbot's xmoe_bulk loop (quant/exl3_moe_transfer.cu: a producer warp bulk-copies weight tiles into
# three stage buffers, 8 warps compute with the flat loop's arithmetic). s1 microbench seq296 (layer 20, HOT 320): rows 6
# +4.5% / +0.6% (cold 0 / cold 2 + side 16), 14 +7.5% / +3.8%, 24 +9.8% / +15.0%. Not bit-identical: both kernels
# atomicAdd the expert contributions into the FP32 accumulator, so the summation order (and the BF16 rounding of the
# output) varies run to run for either kernel; outputs agree within 1 BF16 ulp per element (final/bulk_microtest.py). Default 0 = off (the regular flat kernel everywhere); 14 = the planned value
# once GPU-confirmed in the serve path. K=3 MUL1 only, rows <= 64. Read at import (baked into captured graphs).
def bulk_min_rows() -> int:
    value = os.environ.get("FLARE_DS41_EXL3_BULK_MIN_ROWS", "").strip()
    rows = int(value) if value.lstrip("-").isdigit() else (0 if not value else -1)
    if not 0 <= rows <= 64:
        raise ValueError(f"FLARE_DS41_EXL3_BULK_MIN_ROWS={value!r}: expected 0 (off) or 1..64")
    return rows


def use_bulk(rows: int) -> bool:
    """The hot launch of a graph with ``rows`` rows takes the xmoe_bulk loop (decode/verify rows only)."""
    return 0 < BULK_MIN_ROWS <= rows <= 64


def torch_prep(ids, weights, hidden, num_experts: int, hot_rows: int):
    """The op's original torch prep (reference for the fused kernel): token_sorted, weight_sorted, hot, cold, x16, acc."""
    rows, top_k = ids.shape
    flat = ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    token_sorted = torch.div(order, top_k, rounding_mode="floor")
    weight_sorted = weights.reshape(-1).index_select(0, order).to(torch.float16)
    counts = torch.zeros(num_experts + 1, dtype=torch.int64, device=hidden.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    slots = torch.arange(num_experts + 1, device=hidden.device)
    hot = torch.where(slots < hot_rows, counts, -counts)
    cold = torch.where(slots < hot_rows, -counts, counts)
    acc = torch.zeros((rows, xmoe.HIDDEN), dtype=torch.float32, device=hidden.device)
    return token_sorted, weight_sorted, hot, cold, hidden.to(torch.float16).contiguous(), acc


def fused_prep_outputs(ext, ids, weights, hidden, num_experts: int, hot_rows: int):
    rows, top_k = ids.shape
    dev = hidden.device
    token_sorted = torch.empty(rows * top_k, dtype=torch.int64, device=dev)
    weight_sorted = torch.empty(rows * top_k, dtype=torch.float16, device=dev)
    hot = torch.empty(num_experts + 1, dtype=torch.int64, device=dev)
    cold = torch.empty(num_experts + 1, dtype=torch.int64, device=dev)
    x16 = torch.empty((rows, xmoe.HIDDEN), dtype=torch.float16, device=dev)
    acc = torch.empty((rows, xmoe.HIDDEN), dtype=torch.float32, device=dev)
    ext.xmoe_decode_prep(ids.contiguous(), weights.contiguous(), hidden.contiguous(), int(hot_rows),
                         token_sorted, weight_sorted, hot, cold, x16, acc)
    return token_sorted, weight_sorted, hot, cold, x16, acc


def side_budgets(num_sms: int, k: int) -> tuple[int, int]:
    """(hot SMs, cold SMs) for a cold side launch; refuses a split that leaves the hot flat kernel < 3/4 of the SMs."""
    if not k:
        return num_sms, 0
    if 4 * (num_sms - k) < 3 * num_sms:
        raise ValueError(f"cold side stream: {k} of {num_sms} SMs leaves the hot kernel under 3/4 of the device")
    return num_sms - k, k


STAGING_EXPERTS = 16      # parked banks: experts per VRAM staging chunk (3 x 16 x 2.2 MB = ~106 MB transient at TP2)


@dataclass
class Exl3Experts:
    """One layer's EXL3 routed experts on one rank (the ``ExpertWeights.experts`` of an EXL3 checkpoint)."""

    trellis: dict           # matrix -> int16 [E, in/16, out/16, 16 * bits] (VRAM, or mixed VRAM/host when parked)
    suh: dict               # matrix -> fp16 [E, in]
    svh: dict               # matrix -> fp16 [E, out]
    pointers: dict          # (matrix, part) -> int64 [E] device addresses
    bits: int
    codebook: str           # "mul1" (routed) or "mcg" (drafter)
    hot_rows: int           # bank slots below this are VRAM rows (= num_experts when resident)
    mapped_rows: int | None = None   # VRAM rows mapped during the export when fewer than hot_rows (declared rows)

    @property
    def num_experts(self) -> int:
        return self.trellis["w1"].shape[0]

    @property
    def local_intermediate(self) -> int:
        return self.trellis["w1"].shape[2] * xmoe.TILE

    @property
    def parked(self) -> bool:
        return self.hot_rows < self.num_experts


def _pointer_table(tensor, device):
    step = tensor.stride(0) * tensor.element_size()
    values = torch.arange(tensor.shape[0], dtype=torch.int64) * step + tensor.data_ptr()
    from ignition.preparation.persistent import persistent_copy

    return persistent_copy(values.to(device))


def load_exl3_experts(loader, prefixes, *, rank, world, hot_rows=None, mixed_allocate=None, codebook="mul1"):
    """Load ``prefixes`` (bank order) into stacked rank-sliced banks. ``mixed_allocate(name, shape, dtype,
    device_rows)`` (CoreStorage) places trellis banks with ``hot_rows`` VRAM rows; None = all VRAM."""
    from ignition.preparation.persistent import persistent_empty

    device = loader.device
    count = len(prefixes)
    hot_rows = count if hot_rows is None else int(hot_rows)
    if not 0 <= hot_rows <= count or (hot_rows < count and mixed_allocate is None):
        raise ValueError(f"EXL3 parking needs a mixed allocator for {count - hot_rows} host experts")
    first = loader._sources[f"{prefixes[0]}.w1.trellis"]
    bits = first.shape[-1] // 16
    trellis, suh, svh, pointers, regions = {}, {}, {}, {}, {}
    limit = export_hot() if hot_rows < count else None
    mapped_rows = hot_rows if limit is None else min(hot_rows, limit)     # VRAM rows during this export
    for matrix in MATRICES:
        shape = tuple(loader._sources[f"{prefixes[0]}.{matrix}.trellis"].shape)
        local = xmoe.shard_matrix(matrix, {"trellis": torch.empty(shape, dtype=torch.int16, device="meta"),
                                           "suh": torch.empty(shape[0] * 16, device="meta"),
                                           "svh": torch.empty(shape[1] * 16, device="meta")}, rank, world)
        bank_shape = (count, *local["trellis"].shape)
        if hot_rows < count:
            name = f"{prefixes[0].rsplit('.experts.', 1)[0]}.exl3.{matrix}"
            region = mixed_allocate(name, bank_shape, torch.int16, device_rows=mapped_rows)
            if mapped_rows < hot_rows:
                declare_device_rows(region, hot_rows, math.prod(bank_shape[1:]) * 2)
            regions[matrix] = region
            trellis[matrix] = region.device_view
        else:
            trellis[matrix] = persistent_empty(bank_shape, torch.int16, device)
        suh[matrix] = persistent_empty((count, local["suh"].shape[0]), torch.float16, device)
        svh[matrix] = persistent_empty((count, local["svh"].shape[0]), torch.float16, device)
    # Direct transfers from the checkpoint's source views (``loader._copy``), as the MXFP4 loaders do. b12x's batch
    # executor writes only into CUDA device allocations (cuPointerGetAttribute MEMORY_TYPE == DEVICE for the whole
    # destination extent, _cuda_range.h), so host-backed rows of a parked bank (slots >= hot_rows) are read into a
    # VRAM staging chunk first and copied on the device, as TP3's MXFP4 parking does (_park_on_host).
    markers = set()
    chunk = STAGING_EXPERTS
    staging = {}
    for start in range(0, count, chunk):
        stop = min(count, start + chunk)
        hot_stop = min(stop, max(start, mapped_rows))    # [start, hot_stop) VRAM rows, [hot_stop, stop) host rows
        for slot in range(start, stop):
            prefix = prefixes[slot]
            for matrix in MATRICES:
                name = f"{prefix}.{matrix}"
                source = {part: loader._sources[f"{name}.{part}"] for part in ("trellis", "suh", "svh")}
                for marker in ("mul1", "mcg"):
                    if f"{name}.{marker}" in loader._sources:
                        markers.add(marker)
                local = xmoe.shard_matrix(matrix, source, rank, world)
                if tuple(local["trellis"].shape) != tuple(trellis[matrix].shape[1:]):
                    raise ValueError(f"{name}: EXL3 trellis geometry {tuple(source['trellis'].shape)} differs from the bank")
                if slot < hot_stop:
                    loader._copy(trellis[matrix][slot], local["trellis"])
                else:
                    if matrix not in staging:
                        staging[matrix] = torch.empty((chunk, *trellis[matrix].shape[1:]), dtype=torch.int16,
                                                      device=device)
                    loader._copy(staging[matrix][slot - hot_stop], local["trellis"])
                loader._copy(suh[matrix][slot], local["suh"])
                loader._copy(svh[matrix][slot], local["svh"])
        if hot_stop < stop:
            session = getattr(loader, "_session", None)
            if session is not None:
                session.flush()
            for matrix in MATRICES:
                trellis[matrix][hot_stop:stop].copy_(staging[matrix][:stop - hot_stop])
    session = getattr(loader, "_session", None)
    if session is not None:
        session.flush()
    staging.clear()
    if markers != {codebook}:
        raise ValueError(f"EXL3 experts carry codebook markers {sorted(markers)}, expected {codebook!r}")
    if bits != 3:
        raise ValueError(f"EXL3 experts at {bits} bits; the xmoe build has K=3 (and K=2 MCG) instances")
    for matrix in MATRICES:
        for part, tensor in (("trellis", trellis[matrix]), ("suh", suh[matrix]), ("svh", svh[matrix])):
            pointers[matrix, part] = _pointer_table(tensor, device)
    return Exl3Experts(trellis, suh, svh, pointers, bits, codebook, hot_rows,
                       mapped_rows if mapped_rows < hot_rows else None)


# ---- the routed-expert call -------------------------------------------------------------------------------------

PREFILL_MODE = prefill_mode()
PREFILL_GROUPS = prefill_groups()
PREFILL_PASS_ROUTES = prefill_pass_routes()
PREFILL_COLD_SIDE = prefill_cold_side()
COLD_WIDTH = cold_width()
FUSED_PREP = fused_prep()
FUSED_PREP_MAX_ROWS = fused_prep_max_rows()
BULK_MIN_ROWS = bulk_min_rows()
_BANKS: list = []
_SIDE: dict = {}       # device index -> (side stream, hot SMs, cold SMs) when FLARE_DS41_EXL3_COLD_SIDE is set


def _ensure_side(device, experts, rows, groups):
    """Create the cold side stream and grow its scratch before capture (parked banks, decode rows)."""
    k = cold_side_sms()
    if not k or not experts.parked or rows > 64:
        return
    if device.index not in _SIDE:
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        hot_sms, cold_sms = side_budgets(num_sms, k)
        _SIDE[device.index] = (torch.cuda.Stream(device=device), hot_sms, cold_sms)
    _scratch(device, rows, experts.local_intermediate, groups, side=True)


_PREFILL_SIDE: dict = {}   # device index -> side stream of the cold prefill launch (FLARE_DS41_EXL3_PREFILL_COLD_SIDE)


def _uses_prefill_side(experts, rows) -> bool:
    return PREFILL_COLD_SIDE and experts.parked and _decode_once(rows)


def _decode_once(rows) -> bool:
    return rows > 64 and rows >= DO_MIN_ROWS and PREFILL_MODE == "do"


def _ensure_prefill_side(device, experts, rows):
    """Create the cold prefill side stream before capture (parked banks, decode-once prefill graphs)."""
    if PREFILL_COLD_SIDE and PREFILL_MODE != "do":
        raise ValueError("FLARE_DS41_EXL3_PREFILL_COLD_SIDE=1 needs FLARE_DS41_EXL3_PREFILL=do (two group kernels "
                         "beside each other could hold each other's SMs in their group barriers)")
    if _uses_prefill_side(experts, rows) and device.index not in _PREFILL_SIDE:
        _PREFILL_SIDE[device.index] = torch.cuda.Stream(device=device)


_SCRATCH: dict = {}
_LOCKS: dict = {}


def _ensure_locks(device):
    """Point xmoe's lock/barrier/scheduler buffers at zeroed core-storage memory (once per device), so captured
    kernels reference package-owned memory instead of the extension's own cudaMalloc. Needs the lock-buffer setter
    (m3/xmoe-flare-locks.patch); an older build keeps its private buffers (fine for eager tests, not for export)."""
    if device.index in _LOCKS:
        return
    ext = xmoe.extension()
    if not hasattr(ext, "xmoe_set_lock_buffers"):
        _LOCKS[device.index] = None
        return
    from ignition.preparation.persistent import persistent_zeros

    nbytes = int(ext.xmoe_lock_buffer_bytes())
    buffers = persistent_zeros((2, (nbytes + 255) // 256 * 256), torch.uint8, device)
    ext.xmoe_set_lock_buffers(device.index, buffers[0].data_ptr(), buffers[1].data_ptr())
    _LOCKS[device.index] = buffers


def _scratch(device, rows, local, groups, *, decode_once=False, side=False, cold_side=False):
    """One shared scratch per device, grown to the largest need; graphs run one at a time per rank. ``side``: the cold
    side stream's own scratch (decode rows only). ``cold_side``: a prefill graph whose cold launch runs beside the
    decode-once passes (group views and one decode-once range side by side)."""
    need = scratch_halfs(rows, local, groups, decode_once=decode_once,
                         pass_routes=PREFILL_PASS_ROUTES if decode_once else 0, cold_side=cold_side)
    key = (device.index, local, side)
    buffer = _SCRATCH.get(key)
    if buffer is None or buffer.numel() < need:
        from ignition.preparation.persistent import persistent_empty

        buffer = persistent_empty((need,), torch.float16, device)
        _SCRATCH[key] = buffer
    return buffer


_DO_JOBS: dict = {}


def _do_jobs(device, capacity):
    """Per-device decode-once job table (int32 [3, J], int64 [3, J], overflow int32 [1]), grown before capture."""
    tables = _DO_JOBS.get(device.index)
    if tables is None or tables[0].shape[1] < capacity:
        from ignition.preparation.persistent import persistent_zeros

        tables = (persistent_zeros((3, capacity), torch.int32, device), persistent_zeros((3, capacity), torch.int64, device),
                  persistent_zeros((1,), torch.int32, device))
        _DO_JOBS[device.index] = tables
    return tables


def _do_views(buffer, routed, local):
    h, i = routed * xmoe.HIDDEN, routed * local
    return (buffer[:h].view(routed, xmoe.HIDDEN), buffer[h:2 * h].view(routed, xmoe.HIDDEN),
            buffer[2 * h:2 * h + i].view(routed, local), buffer[2 * h + i:2 * h + 2 * i].view(routed, local))


def _launch_do(ext, bank, hidden, out, counts, token_sorted, weight_sorted, row_exp, buffer, limit):
    """Decode-once MoE for the bank's VRAM experts (slot < hot_rows); rows with row_exp < 0 are left to the group kernel."""
    routed = token_sorted.shape[0]
    jobs = _do_jobs(hidden.device, do_job_capacity(routed, bank.num_experts))
    xg, xu, cg, cu = _do_views(buffer, routed, bank.local_intermediate)
    p = bank.pointers
    ext.xmoe_do_moe(hidden, out, token_sorted, weight_sorted, row_exp, counts, int(bank.hot_rows), *jobs,
                    p["w1", "trellis"], p["w1", "suh"], p["w1", "svh"],
                    p["w3", "trellis"], p["w3", "suh"], p["w3", "svh"],
                    p["w2", "trellis"], p["w2", "suh"], p["w2", "svh"],
                    xg, xu, cg, cu, bank.bits, 2 if bank.codebook == "mul1" else 1, float(limit), xmoe.ACT_SILU)


def _views(buffer, groups, rows, local):
    h = groups * rows * xmoe.HIDDEN
    i = groups * rows * local
    return (buffer[:h].view(groups, rows, xmoe.HIDDEN), buffer[h:2 * h].view(groups, rows, xmoe.HIDDEN),
            buffer[2 * h:2 * h + i].view(groups, rows, local), buffer[2 * h + i:2 * h + 2 * i].view(groups, rows, local))


def _launch(ext, bank, hidden, out, counts, token_sorted, weight_sorted, views, limit, *, force_group, width,
            sm_budget=0, lock_slot=0, bulk=False):
    mul1 = bank.codebook == "mul1"
    p = bank.pointers
    ext.xmoe_set_flat_bulk(int(bulk and not force_group))
    ext.xmoe_set_force_group(int(force_group))
    ext.xmoe_set_group_width(int(width))
    ext.xmoe_set_sm_budget(int(sm_budget))
    ext.xmoe_set_lock_slot(int(lock_slot))
    try:
        ext.exl3_moe(hidden, out, counts, token_sorted, weight_sorted, *views, xmoe.ACT_SILU,
                     bank.bits, bank.bits, bank.bits,
                     p["w1", "trellis"], p["w1", "suh"], p["w1", "svh"],
                     p["w3", "trellis"], p["w3", "suh"], p["w3", "svh"],
                     p["w2", "trellis"], p["w2", "suh"], p["w2", "svh"],
                     not mul1, mul1, not mul1, mul1, not mul1, mul1, float(limit), -1)
    finally:
        ext.xmoe_set_flat_bulk(0)
        ext.xmoe_set_force_group(0)
        ext.xmoe_set_group_width(0)
        ext.xmoe_set_sm_budget(0)
        ext.xmoe_set_lock_slot(0)


@torch.library.custom_op("flare::ds41_exl3_moe", mutates_args=("out",))
def ds41_exl3_moe(handle: int, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                  out: torch.Tensor, limit: float) -> None:
    _moe(handle, hidden, ids, weights, out, limit)


def _finish(acc, out, add):
    """The op's last step: ``out.copy_(acc)``, or with FLARE_DS41_EXL3_MERGE_EPILOGUE (``add`` = the shared expert's
    output) out = bf16(acc) + add in one kernel (merges.py)."""
    if add is None:
        out.copy_(acc)
    else:
        from . import merges

        merges.extension().xmoe_epilogue_add(acc, add.contiguous(), out)


def _moe(handle, hidden, ids, weights, out, limit, add=None):
    bank, groups = _BANKS[handle]
    ext = xmoe.extension()
    rows, top_k = ids.shape
    prefill = rows > 64
    if FUSED_PREP and not prefill and rows <= FUSED_PREP_MAX_ROWS:
        _fused_moe(ext, bank, groups, hidden, ids, weights, out, limit, add=add)
        return
    flat = ids.reshape(-1).to(torch.int64)
    order = torch.argsort(flat, stable=True)
    token_sorted = torch.div(order, top_k, rounding_mode="floor")
    weight_sorted = weights.reshape(-1).index_select(0, order).to(torch.float16)
    counts = torch.zeros(bank.num_experts + 1, dtype=torch.int64, device=hidden.device)
    counts.scatter_add_(0, flat, torch.ones_like(flat))
    decode_once = _decode_once(rows)
    g = PREFILL_GROUPS if prefill else groups
    cold_side = _uses_prefill_side(bank, rows)
    buffer = _scratch(hidden.device, rows, bank.local_intermediate, g, decode_once=decode_once, cold_side=cold_side)
    views = _views(buffer, g, rows, bank.local_intermediate)
    width = PREFILL_GROUP_WIDTH if prefill else 0
    cold_w = width if prefill else COLD_WIDTH
    bulk = not prefill and use_bulk(rows)
    acc = torch.zeros((rows, xmoe.HIDDEN), dtype=torch.float32, device=hidden.device)
    x = hidden.to(torch.float16).contiguous()
    if decode_once:
        sorted_ids = flat.index_select(0, order)
        row_exp = torch.where(sorted_ids < bank.hot_rows, sorted_ids, torch.full_like(sorted_ids, -1)).to(torch.int32)
        side = None
        if bank.parked:
            slots = torch.arange(bank.num_experts + 1, device=hidden.device)
            cold = torch.where(slots < bank.hot_rows, -counts, counts)
            if cold_side:
                side, main = _PREFILL_SIDE[hidden.device.index], torch.cuda.current_stream(hidden.device)
                side.wait_stream(main)              # x, acc (zeroed), cold counts, sorted routes ready
                with torch.cuda.stream(side):
                    _launch(ext, bank, x, acc, cold, token_sorted, weight_sorted, views, limit, force_group=True,
                            width=width, lock_slot=1)
        do_buffer = buffer[g * rows * (2 * xmoe.HIDDEN + 2 * bank.local_intermediate):] if cold_side else buffer
        routed = token_sorted.shape[0]
        for start, stop in do_passes(routed, PREFILL_PASS_ROUTES):
            part = counts if (start, stop) == (0, routed) else pass_counts(counts, start, stop)
            _launch_do(ext, bank, x, acc, part, token_sorted[start:stop], weight_sorted[start:stop],
                       row_exp[start:stop], do_buffer, limit)
        if side is not None:
            main.wait_stream(side)                  # joined before the BF16 copy and before x/acc/cold are reused
        elif bank.parked:
            _launch(ext, bank, x, acc, cold, token_sorted, weight_sorted, views, limit, force_group=True, width=width)
    elif not bank.parked:
        _launch(ext, bank, x, acc, counts, token_sorted, weight_sorted, views, limit, force_group=False, width=width,
                bulk=bulk)
    else:
        slots = torch.arange(bank.num_experts + 1, device=hidden.device)
        hot = torch.where(slots < bank.hot_rows, counts, -counts)
        cold = torch.where(slots < bank.hot_rows, -counts, counts)
        side = _SIDE.get(hidden.device.index) if not prefill else None
        if side is None:
            _launch(ext, bank, x, acc, hot, token_sorted, weight_sorted, views, limit, force_group=False, width=width,
                    bulk=bulk)
            _launch(ext, bank, x, acc, cold, token_sorted, weight_sorted, views, limit, force_group=True, width=cold_w)
        else:
            stream, hot_sms, cold_sms = side
            side_views = _views(_scratch(hidden.device, rows, bank.local_intermediate, g, side=True), g, rows,
                                bank.local_intermediate)
            main = torch.cuda.current_stream(hidden.device)
            stream.wait_stream(main)                    # inputs (x, acc zeroed, cold counts, sorted rows) ready
            with torch.cuda.stream(stream):
                _launch(ext, bank, x, acc, cold, token_sorted, weight_sorted, side_views, limit, force_group=True,
                        width=cold_w, sm_budget=cold_sms, lock_slot=1)
            _launch(ext, bank, x, acc, hot, token_sorted, weight_sorted, views, limit, force_group=False, width=width,
                    sm_budget=hot_sms, bulk=bulk)
            main.wait_stream(stream)    # joined before anything on main can reuse x/acc/cold's memory
    _finish(acc, out, add)


def _fused_moe(ext, bank, groups, hidden, ids, weights, out, limit, *, add=None, prepped=None):
    """Decode path with the fused prep: same launches as the torch path below, fed by xmoe_decode_prep's outputs
    (``prepped``: those outputs already written by flare::ds41_exl3_moe_prep, FLARE_DS41_EXL3_MERGE_TOPK)."""
    rows = hidden.shape[0]
    if prepped is None:
        hot_rows = bank.hot_rows if bank.parked else bank.num_experts + 1
        prepped = fused_prep_outputs(ext, ids, weights, hidden, bank.num_experts, hot_rows)
    token_sorted, weight_sorted, hot, cold, x, acc = prepped
    views = _views(_scratch(hidden.device, rows, bank.local_intermediate, groups), groups, rows, bank.local_intermediate)
    bulk = use_bulk(rows)
    if not bank.parked:
        _launch(ext, bank, x, acc, hot, token_sorted, weight_sorted, views, limit, force_group=False, width=0, bulk=bulk)
    else:
        side = _SIDE.get(hidden.device.index)
        if side is None:
            _launch(ext, bank, x, acc, hot, token_sorted, weight_sorted, views, limit, force_group=False, width=0,
                    bulk=bulk)
            _launch(ext, bank, x, acc, cold, token_sorted, weight_sorted, views, limit, force_group=True, width=COLD_WIDTH)
        else:
            stream, hot_sms, cold_sms = side
            side_views = _views(_scratch(hidden.device, rows, bank.local_intermediate, groups, side=True), groups, rows,
                                bank.local_intermediate)
            main = torch.cuda.current_stream(hidden.device)
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                _launch(ext, bank, x, acc, cold, token_sorted, weight_sorted, side_views, limit, force_group=True,
                        width=COLD_WIDTH, sm_budget=cold_sms, lock_slot=1)
            _launch(ext, bank, x, acc, hot, token_sorted, weight_sorted, views, limit, force_group=False, width=0,
                    sm_budget=hot_sms, bulk=bulk)
            main.wait_stream(stream)
    _finish(acc, out, add)


@ds41_exl3_moe.register_fake
def _ds41_exl3_moe_fake(handle, hidden, ids, weights, out, limit):
    pass


# FLARE_DS41_EXL3_SHARED_SIDE: the same op with an ordering operand. ``after`` is the shared expert's output, written by
# its side-stream join (flare::join_shared_expert mutates it), so reading it here orders every xmoe launch after the join:
# the shared expert overlaps only the router projection and top-k, never an xmoe grid barrier. The default path keeps
# flare::ds41_exl3_moe unchanged.
def shared_side() -> bool:
    value = os.environ.get("FLARE_DS41_EXL3_SHARED_SIDE", "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"FLARE_DS41_EXL3_SHARED_SIDE={value!r}: expected 0 or 1")
    return value == "1"


@torch.library.custom_op("flare::ds41_exl3_moe_after", mutates_args=("out",))
def ds41_exl3_moe_after(handle: int, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                        after: torch.Tensor, out: torch.Tensor, limit: float) -> None:
    ds41_exl3_moe(handle, hidden, ids, weights, out, limit)


@ds41_exl3_moe_after.register_fake
def _ds41_exl3_moe_after_fake(handle, hidden, ids, weights, after, out, limit):
    pass


# ---- exp/ds41-exl3-final-merges (merges.py): MERGE_TOPK prep op, prepped MoE op, MERGE_EPILOGUE --------------------

def merge_prep_rows(rows: int) -> bool:
    """FLARE_DS41_EXL3_MERGE_TOPK applies to graphs that take the fused prep (decode rows within its limit)."""
    from . import merges

    return merges.MERGE_TOPK and FUSED_PREP and rows <= min(64, FUSED_PREP_MAX_ROWS)


# ds41_exl3_moe_after with out = bf16(acc) + after (the graph's routed + shared folded into the op's last copy).
@torch.library.custom_op("flare::ds41_exl3_moe_after_sum", mutates_args=("out",))
def ds41_exl3_moe_after_sum(handle: int, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                            after: torch.Tensor, out: torch.Tensor, limit: float) -> None:
    _moe(handle, hidden, ids, weights, out, limit, add=after)


@ds41_exl3_moe_after_sum.register_fake
def _ds41_exl3_moe_after_sum_fake(handle, hidden, ids, weights, after, out, limit):
    pass


@torch.library.custom_op("flare::ds41_exl3_moe_prep",
                         mutates_args=("token_sorted", "weight_sorted", "hot", "cold", "x16", "acc"))
def ds41_exl3_moe_prep(handle: int, hidden: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
                       hot_map: torch.Tensor | None, token_sorted: torch.Tensor, weight_sorted: torch.Tensor,
                       hot: torch.Tensor, cold: torch.Tensor, x16: torch.Tensor, acc: torch.Tensor) -> None:
    """The fused prep on raw router IDs with the hot_map lookup folded in (xmoe_decode_prep_mapped)."""
    from . import merges

    bank, _ = _BANKS[handle]
    hot_rows = bank.hot_rows if bank.parked else bank.num_experts + 1
    table = hot_map if hot_map is not None else torch.empty(0, dtype=torch.int32, device=ids.device)
    merges.extension().xmoe_decode_prep_mapped(ids.contiguous(), table, weights.contiguous(), hidden.contiguous(),
                                               int(hot_rows), token_sorted, weight_sorted, hot, cold, x16, acc)


@ds41_exl3_moe_prep.register_fake
def _ds41_exl3_moe_prep_fake(handle, hidden, ids, weights, hot_map, token_sorted, weight_sorted, hot, cold, x16, acc):
    pass


@torch.library.custom_op("flare::ds41_exl3_moe_prepped", mutates_args=("acc", "out"))
def ds41_exl3_moe_prepped(handle: int, token_sorted: torch.Tensor, weight_sorted: torch.Tensor, hot: torch.Tensor,
                          cold: torch.Tensor, x16: torch.Tensor, acc: torch.Tensor, after: torch.Tensor | None,
                          add_after: bool, out: torch.Tensor, limit: float) -> None:
    """The MoE launches of the fused path on prep outputs written earlier; ``after`` orders them after the shared
    expert's join, ``add_after`` folds routed + after into the final copy."""
    bank, groups = _BANKS[handle]
    _fused_moe(xmoe.extension(), bank, groups, x16, None, None, out, limit,
               add=after if add_after else None, prepped=(token_sorted, weight_sorted, hot, cold, x16, acc))


@ds41_exl3_moe_prepped.register_fake
def _ds41_exl3_moe_prepped_fake(handle, token_sorted, weight_sorted, hot, cold, x16, acc, after, add_after, out, limit):
    pass


@dataclass(frozen=True)
class Exl3Prep:
    """Prep outputs allocated in the forward and written by flare::ds41_exl3_moe_prep."""

    token_sorted: torch.Tensor
    weight_sorted: torch.Tensor
    hot: torch.Tensor
    cold: torch.Tensor
    x16: torch.Tensor
    acc: torch.Tensor

    def tensors(self):
        return self.token_sorted, self.weight_sorted, self.hot, self.cold, self.x16, self.acc


class PreparedExl3Experts:
    """Drop-in for ``PreparedRoutedExperts`` in ``PreparedLayer`` when the layer's experts are ``Exl3Experts``."""

    plan = None

    def __init__(self, source, experts: Exl3Experts, ids, weights, output, *, session=None, name="", workspace=None,
                 router_plan=None, limit=xmoe.SWIGLU_LIMIT):
        if source.shape[-1] != xmoe.HIDDEN or output.shape != source.shape:
            raise ValueError("EXL3 experts take [rows, 5120] activations and write [rows, 5120]")
        self.experts, self.output, self.limit = experts, output, float(limit)
        groups = max(1, torch.cuda.get_device_properties(source.device).multi_processor_count // xmoe.MOE_SMS_PER_EXPERT)
        _ensure_locks(source.device)
        self.handle = len(_BANKS)
        _BANKS.append((experts, groups))
        rows = source.shape[0]
        g = PREFILL_GROUPS if rows > 64 else groups
        decode_once = _decode_once(rows)
        _scratch(source.device, rows, experts.local_intermediate, g, decode_once=decode_once,
                 cold_side=_uses_prefill_side(experts, rows))   # grow before capture
        _ensure_side(source.device, experts, rows, g)
        _ensure_prefill_side(source.device, experts, rows)
        if decode_once:
            _do_jobs(source.device, do_job_capacity(rows * ids.shape[-1], experts.num_experts))
        self.output_layout = tuple(output.shape), output.dtype
        # exp/ds41-exl3-final-merges (merges.py): decided per graph at preparation (static row count)
        self.merge_prep = merge_prep_rows(rows)
        self.top_k = ids.shape[-1]
        if self.merge_prep:
            from . import merges

            merges.extension()      # built/loaded eagerly, outside traced forwards

    def prep(self, normalized, ids, weights, hot_map=None) -> Exl3Prep:
        """FLARE_DS41_EXL3_MERGE_TOPK: the fused prep on raw router IDs (hot_map folded in), as its own op so it can
        run before the shared-expert join."""
        rows, dev = normalized.shape[0], normalized.device
        routes, bins = rows * self.top_k, self.experts.num_experts + 1
        prep = Exl3Prep(torch.empty(routes, dtype=torch.int64, device=dev),
                        torch.empty(routes, dtype=torch.float16, device=dev),
                        torch.empty(bins, dtype=torch.int64, device=dev), torch.empty(bins, dtype=torch.int64, device=dev),
                        torch.empty((rows, xmoe.HIDDEN), dtype=torch.float16, device=dev),
                        torch.empty((rows, xmoe.HIDDEN), dtype=torch.float32, device=dev))
        ds41_exl3_moe_prep(self.handle, normalized, ids, weights, hot_map, *prep.tensors())
        return prep

    def functional(self, *, session=None):
        self.output = None
        return self

    def __call__(self, normalized, ids, weights, *, router_logits=None, selected_logits=None, image_mask=None,
                 after=None, prepped: Exl3Prep | None = None, add_after=False):
        """``prepped``: the outputs of ``prep`` (MERGE_TOPK; ids/weights then unused); ``add_after``: return routed +
        after (MERGE_EPILOGUE; ``after`` = the shared expert's output)."""
        shape, dtype = self.output_layout
        out = torch.empty(shape, dtype=dtype, device=normalized.device)
        if add_after and after is None:
            raise ValueError("the merged epilogue adds the shared expert's output: pass it as after")
        if prepped is not None:
            ds41_exl3_moe_prepped(self.handle, *prepped.tensors(), after, bool(add_after), out, self.limit)
            return out
        if add_after:
            ds41_exl3_moe_after_sum(self.handle, normalized, ids, weights, after, out, self.limit)
            return out
        if after is not None:
            ds41_exl3_moe_after(self.handle, normalized, ids, weights, after, out, self.limit)
            return out
        ds41_exl3_moe(self.handle, normalized, ids, weights, out, self.limit)
        return out
