# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 fork: export-time switches that change how long an export takes, never what it produces.

Environment (all unset by default, which keeps the upstream behaviour):

FLARE_EXPORT_FORWARD_WORKERS=N
    Forward compiler processes per host for "Compile model forwards", which runs after kernel
    tuning has finished (the GPUs are idle then; the workers trace on meta/fake tensors and run
    Inductor/Triton compilers on the CPU). Unset: the same budget as kernel tuning
    (--compile-workers / [preparation] compile_workers), as before. Host and device admission
    (compile_pool.compiler_resources / allocate_workers) still bound the admitted count.
FLARE_EXPORT_FORWARD_WORKER_GIB=G
    Host-memory admission estimate per forward compiler, in GiB (upstream: 32, sized for Kimi
    K3's 93-layer forwards). Applies with the switch above, after tuning only.
FLARE_EXPORT_TUNE_ONLY=1
    Stop after "Prepare and tune kernels": every rank flushes its tuning selections to the
    kernel cache, closes its session and exits the way a tuning replica does (TuningComplete);
    `ignition export` then returns 0 without compiling, capturing or writing a package. Use it
    for the warm-up before a cold-cache export (handbook D.2.2).

None of these values reaches a cache key, a tuning namespace or the package: the compiled
forward cache key is (namespace, Inductor options without compile_threads, torch, runtime
patches, device), see aot.ForwardCache.
"""

from __future__ import annotations

import os

FORWARD_WORKERS_ENV = "FLARE_EXPORT_FORWARD_WORKERS"
FORWARD_WORKER_GIB_ENV = "FLARE_EXPORT_FORWARD_WORKER_GIB"
TUNE_ONLY_ENV = "FLARE_EXPORT_TUNE_ONLY"


def _positive_int(name):
    text = os.environ.get(name, "").strip()
    if not text:
        return None
    if not text.isdecimal() or int(text) < 1:
        raise ValueError(f"{name} must be a positive integer, not {text!r}")
    return int(text)


def forward_workers():
    """Forward compiler processes per host requested for after tuning, or None."""
    return _positive_int(FORWARD_WORKERS_ENV)


def forward_worker_bytes():
    """Per-worker host-memory admission estimate requested for forward compilers, or None."""
    gib = _positive_int(FORWARD_WORKER_GIB_ENV)
    return None if gib is None else gib << 30


def tune_only():
    """Whether this export stops after kernel tuning."""
    text = os.environ.get(TUNE_ONLY_ENV, "").strip()
    if text not in ("", "0", "1"):
        raise ValueError(f"{TUNE_ONLY_ENV} must be 1 or 0, not {text!r}")
    return text == "1"


def forward_compile_settings(session):
    """(compile_workers, forward_worker_bytes) for a session's forward compilation.

    The overrides apply only once the session's kernel tuning is complete ("ready"): forward
    compiles that run inside tuning (DeepSeek V4.1 joint tuning's contextual trials) keep the
    tuning budget, and sessions without tuning state keep their own.
    """
    requested = getattr(session, "requested_compile_workers", None)
    if getattr(session, "_tune_status", None) != "ready":
        return requested, None
    workers = forward_workers()
    return (requested if workers is None else workers), forward_worker_bytes()
