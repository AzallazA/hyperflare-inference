# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""cuBLAS's own device pool: recorded at export, but no graph uses it.

The export keeps every device allocation the module recorder saw that is still live and that a graph or
retained storage reaches. Reachability is conservative: any 8 bytes of an untyped kernel argument or of
retained storage whose value falls inside an allocation reach it. cuBLAS allocates a pool of its own
(64 MiB on GB202 with CUDA 13) at its first GEMM in a process. PyTorch hands every GEMM its own per-stream
workspace (``cublasSetWorkspace``; core storage, graph-referenced), so captured kernels never address
cuBLAS's pool; it is reached only by coincidence (DS41 160407: an unaligned window inside a 128-byte tensor
map on rank 0, a pair of int32 fields on rank 2; rank 1 and six later rank exports: not reached). Each
coincidence costs 64 MiB of KV cache per GPU on that rank.

``prime`` runs one GEMM of each kind PyTorch issues for a projection, with the recorder active and before
core storage exists, and records the device allocations cuBLAS made outside PyTorch. ``export_model`` passes
them to the exporter as inactive: the conservative scan ignores values inside them, so they are not
retained. Explicit references (relocations, execution buffers) are unaffected. ``FLARE_EXPORT_LIBRARY_SCRATCH=0``
turns this off.
"""

import os

MINIMUM_BYTES = 8 << 20


def enabled():
    return os.environ.get("FLARE_EXPORT_LIBRARY_SCRATCH", "1") != "0"


def new_library_allocations(before, after, torch_segments, minimum=MINIMUM_BYTES):
    """Device allocations in ``after`` but not ``before`` (address -> (nbytes, memory)), at least ``minimum``
    bytes, that do not overlap a live PyTorch segment ((address, nbytes) pairs): the library's own."""
    segments = sorted(torch_segments)
    found = []
    for address, (nbytes, memory) in sorted(after.items()):
        if address in before or memory != "device" or nbytes < minimum:
            continue
        end = address + nbytes
        if any(start < end and address < start + size for start, size in segments):
            continue
        found.append((address, nbytes))
    return found


def live_inactive(scratch, recorded):
    """(start, end) ranges of the recorded scratch allocations that are still live (still recorded)."""
    return [(address, address + nbytes) for address, nbytes in scratch
            if recorded.get(address, (None, None))[0] == nbytes]


def prime(images, device):
    """With the recorder active and no core storage pool entered: run the three GEMM forms of b12x's torch projection
    (bf16_gemv/_preparation.py ``_torch_projection``: bf16 x bf16 -> bf16, -> fp32 via ``out_dtype``, addmm with bias,
    all with ``out=``) on the rank's device on two streams, release PyTorch's own workspaces and blocks, and record what
    cuBLAS kept. Never fails the export: any error is logged and the export continues without exclusion."""
    if images is None or not enabled() or getattr(images, "library_scratch", None) is None:
        return []
    import sys

    try:
        found = _prime(images, device)
    except Exception as error:  # noqa: BLE001 - an optional exclusion must never cost an export
        print(f"library scratch priming failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return []
    images.library_scratch.extend(found)
    print("library scratch primed on cuda:%s: %s" % (device, ", ".join(f"{n / (1 << 20):.0f} MiB at {a:#x}" for a, n in found)
          if found else "nothing new (cuBLAS's pool predates the recorder or is absent): no exclusion"),
          file=sys.stderr, flush=True)
    return found


def _prime(images, device):
    import torch

    cuda = torch.device("cuda", device if isinstance(device, int) else torch.device(device).index)
    before = dict(images.allocations)
    with torch.cuda.device(cuda):
        main = torch.cuda.current_stream(cuda)
        side = torch.cuda.Stream(device=cuda)
        for stream in (main, side):
            stream.wait_stream(main)
            with torch.cuda.stream(stream):
                for rows in (16, 4096):
                    x = torch.zeros(rows, 1024, dtype=torch.bfloat16, device=cuda)
                    weight = torch.zeros(512, 1024, dtype=torch.bfloat16, device=cuda)
                    bias = torch.zeros(512, dtype=torch.bfloat16, device=cuda)
                    out = torch.empty(rows, 512, dtype=torch.bfloat16, device=cuda)
                    out32 = torch.empty(rows, 512, dtype=torch.float32, device=cuda)
                    torch.mm(x, weight.T, out_dtype=torch.float32, out=out32)
                    torch.mm(x, weight.T, out=out)
                    torch.addmm(bias, x, weight.T, out=out)
                    del x, weight, bias, out, out32
        torch.cuda.synchronize(cuda)
        del side, main
        # PyTorch's per-(handle, stream) workspaces of both streams (the map covers every handle and stream): later
        # GEMMs allocate their own where they run (core storage), exactly as without this step.
        torch._C._cuda_clearCublasWorkspaces()
        torch.cuda.empty_cache()
        segments = [(s["address"], s["total_size"]) for s in torch.cuda.memory_snapshot()
                    if s["device"] == cuda.index]
    return new_library_allocations(before, dict(images.allocations), segments)
