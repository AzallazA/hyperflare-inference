# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Prefer lower-scratch kernels when autotuning measures them as near-equal.

The shared scratch arena holds the largest selected operator workspace, so a
prefill kernel that wins a race by noise with a bigger workspace costs KV
capacity on every rank. Among candidates within ``tolerance`` of the fastest
measured latency, the one with the least scratch wins if it saves at least
``min_saving_bytes``; otherwise the fastest wins. The savings floor keeps
operators with small workspaces (decode kernels) on their fastest candidate.

Environment: FLARE_TUNE_SCRATCH_TOLERANCE (fraction, default 0.02; 0 disables)
and FLARE_TUNE_SCRATCH_MIN_SAVING_MIB (default 16).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

TOLERANCE_ENV = "FLARE_TUNE_SCRATCH_TOLERANCE"
MIN_SAVING_ENV = "FLARE_TUNE_SCRATCH_MIN_SAVING_MIB"
DEFAULT_TOLERANCE = 0.02
DEFAULT_MIN_SAVING_MIB = 16


@dataclass(frozen=True)
class ScratchPreference:
    tolerance: float
    min_saving_bytes: int

    def __post_init__(self):
        if type(self.tolerance) not in (int, float) or not 0 < self.tolerance < 1:
            raise ValueError("scratch tolerance must be a fraction in (0, 1)")
        if type(self.min_saving_bytes) is not int or self.min_saving_bytes < 1:
            raise ValueError("minimum scratch saving must be a positive byte count")

    @property
    def identity(self):
        return {"version": 1, "tolerance": float(self.tolerance),
                "min_saving_bytes": self.min_saving_bytes}

    def choose(self, entries, reference=None):
        """Position in ``entries`` of ``(latency_us, candidate_index, scratch_nbytes)``.

        ``reference`` is a faster latency measured earlier in the same race, if any;
        bounding by it keeps successive near-ties from drifting away from the fastest.
        """
        entries = tuple(entries)
        if not entries:
            raise ValueError("scratch preference needs at least one timed candidate")
        fastest = min(range(len(entries)), key=lambda i: (entries[i][0], entries[i][1]))
        floor = entries[fastest][0] if reference is None else min(reference, entries[fastest][0])
        bound = floor * (1 + self.tolerance)
        eligible = [i for i, (latency, _, _) in enumerate(entries) if latency <= bound]
        if not eligible:
            return fastest
        lean = min(eligible, key=lambda i: (entries[i][2], entries[i][0], entries[i][1]))
        if entries[fastest][2] - entries[lean][2] >= self.min_saving_bytes:
            return lean
        return fastest


def _number(name, default):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative, got {raw!r}")
    return value


def from_environment():
    """The configured preference, or None when FLARE_TUNE_SCRATCH_TOLERANCE is 0."""
    tolerance = _number(TOLERANCE_ENV, DEFAULT_TOLERANCE)
    if tolerance == 0:
        return None
    if tolerance >= 1:
        raise ValueError(f"{TOLERANCE_ENV} is a fraction (0.02 = 2%), got {tolerance}")
    saving = _number(MIN_SAVING_ENV, DEFAULT_MIN_SAVING_MIB)
    return ScratchPreference(tolerance, max(1, int(saving * 2**20)))


def select_shard_winner(completed, preference):
    """Choose among shard winners of one ordinary race (fastest when no preference applies)."""
    fastest = min(completed, key=lambda item: (item.latency_us, item.candidate_index))
    if preference is None or any(getattr(item, "scratch_nbytes", None) is None for item in completed):
        return fastest
    reference = min(item.latency_us if getattr(item, "fastest_latency_us", None) is None
                    else item.fastest_latency_us for item in completed)
    entries = [(item.latency_us, item.candidate_index, item.scratch_nbytes) for item in completed]
    return completed[preference.choose(entries, reference)]
