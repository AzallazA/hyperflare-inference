# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Run b12x's 16-head and 8-head MLA decode grids concurrently (TP3, BF16 MLA).

At TP3 a DeepSeek V4.1 rank holds 24 query heads. BF16 MLA decode tiles 16 heads per CTA, so b12x's
``run_unified_decode`` launches a full-block grid and then an 8-head remainder grid through the module-level
``_sparse_mla_decode_grid_flat_launch``. The kernel has no programmatic dependent launch, so the two serialize in the
graph although at small row counts each fills less than a third of the SMs (one CTA per SM).

``B12X_MLA_DECODE_REMAINDER_STREAM=1`` wraps that module attribute at runtime. For a full-block launch that a remainder
launch will follow (heads > grid blocks x valid heads), the wrapper forks a dedicated side stream from the current
stream and launches the full-block grid there; the remainder grid then launches on the current stream, which joins
the side stream right after it, so the split merge b12x enqueues next waits for both. Same compiled programs and
arguments, disjoint output heads, merge after the join: bit-identical. The graph capture records two parallel
branches, so the switch takes effect at export.

No b12x source file changes: b12x keys compiled programs and tuning races by a hash of the import closure of each
callable's source files (``b12x._lib.source_dependencies``), and a runtime attribute does not enter it. Unset/0 leaves
b12x untouched. The wrapper only reads the switch per call, so tests can toggle it after installation.
"""

import inspect
import os

import torch

ENV = "B12X_MLA_DECODE_REMAINDER_STREAM"
_TARGET = "_sparse_mla_decode_grid_flat_launch"
_ORIGINAL = "_flare_remainder_stream_original"


def enabled() -> bool:
    raw = os.environ.get(ENV)
    return raw is not None and raw.strip().lower() in {"1", "true", "on", "yes"}


def _kernel_module():
    from b12x.attention._shared.mla import kernel
    return kernel


class _RemainderStreamLaunch:
    """Replacement for kernel._sparse_mla_decode_grid_flat_launch; delegates every call."""

    def __init__(self, original, compile_only, stream_factory=None):
        self.original = original
        self.compile_only = compile_only
        self.signature = inspect.signature(original)
        self.stream_factory = stream_factory or _side_stream
        self.pending = {}  # device index -> side stream holding a full-block grid not yet joined

    def __call__(self, *args, **kwargs):
        if not enabled() or self.compile_only():
            return self.original(*args, **kwargs)
        bound = self.signature.bind(*args, **kwargs)
        q_all = bound.arguments["q_all"]
        if not q_all.is_cuda:
            return self.original(*args, **kwargs)
        device = q_all.device
        index = device.index if device.index is not None else torch.cuda.current_device()
        heads = int(q_all.shape[1])
        offset = int(bound.arguments["head_block_offset"])
        covered = int(bound.arguments["grid_h_blocks"]) * int(bound.arguments["valid_hpb"])
        main = torch.cuda.current_stream(device)
        if offset == 0 and covered < heads:
            # Full-block grid with a remainder to follow: fork, launch it on the side stream.
            side = self.stream_factory(index)
            side.wait_stream(main)
            self.pending[index] = side
            with torch.cuda.stream(side):
                return self.original(*args, **kwargs)
        side = self.pending.pop(index, None) if offset != 0 else None
        result = self.original(*args, **kwargs)
        if side is not None:
            main.wait_stream(side)  # join before b12x enqueues the split merge
        return result


_STREAMS = {}


def _side_stream(index):
    """One non-blocking stream per device, never from PyTorch's round-robin pool (it can alias a later capture stream;
    same reason as ScratchArena.auxiliary_stream)."""
    stream = _STREAMS.get(index)
    if stream is None:
        from cuda.bindings import driver as cuda

        with torch.cuda.device(index):
            status, handle = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING)
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cannot create MLA remainder stream: {status}")
            stream = torch.cuda.ExternalStream(int(handle), device=torch.device("cuda", index))
        _STREAMS[index] = stream
    return stream


def install(kernel=None):
    """Wrap the b12x launcher (idempotent). Returns the wrapper."""
    kernel = _kernel_module() if kernel is None else kernel
    if hasattr(kernel, "_launch_head_blocks"):
        raise RuntimeError("b12x carries the superseded remainder-stream overlay patch (cc3e8764); drop B12X_OVERLAY")
    current = getattr(kernel, _TARGET)
    if isinstance(current, _RemainderStreamLaunch):
        return current
    from b12x._lib.compile_plan import compile_only_launches_enabled

    wrapper = _RemainderStreamLaunch(current, compile_only_launches_enabled)
    setattr(kernel, _ORIGINAL, current)
    setattr(kernel, _TARGET, wrapper)
    return wrapper


def uninstall(kernel=None):
    kernel = _kernel_module() if kernel is None else kernel
    original = getattr(kernel, _ORIGINAL, None)
    if original is not None:
        setattr(kernel, _TARGET, original)
        delattr(kernel, _ORIGINAL)


def install_from_environment():
    """Install only when the switch is on, so the default leaves b12x exactly as shipped."""
    if enabled():
        install()
