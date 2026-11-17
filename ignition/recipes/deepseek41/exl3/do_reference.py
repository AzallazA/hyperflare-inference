# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""NumPy mirror of the decode-once prefill path (exp/ds41-exl3, ``quant/exl3_do.cu`` + ``experts._launch_do``).

It follows the kernel's bookkeeping step by step -- stable expert sort of the routed rows, the fixed-capacity job table
built from the routing counts (``xmoe_do_build_jobs``), the cold-row mask (``row_exp = -1`` for slots >= hot_rows), the
per-job gate/up GEMMs in the rotated domain, the fused SwiGLU (xmoe order), the down GEMM and the weighted scatter-add --
in float64, so a CPU test can compare it with ``reference.moe_reference`` restricted to the hot experts. The GPU
microtest compares the kernel itself with the same references.
"""

from __future__ import annotations

import numpy as np

from . import reference as ref

JOB_ROWS = 128


def build_jobs(counts, hot_rows: int, capacity: int, *, job_rows: int = JOB_ROWS):
    """The device job builder: (row0, rows, expert) int arrays of length ``capacity`` and the number of real jobs.
    ``counts`` may be negative (|count| rows, handled by another launch); only hot experts with count > 0 get jobs."""
    counts = np.asarray(counts, dtype=np.int64)
    row0 = np.zeros(capacity, np.int64)
    rows = np.zeros(capacity, np.int64)
    exp = np.zeros(capacity, np.int64)
    offset = job = 0
    for e, c in enumerate(counts):
        n = abs(int(c))
        if e < hot_rows and c > 0:
            for r in range(0, n, job_rows):
                if job < capacity:
                    row0[job], rows[job], exp[job] = offset + r, min(job_rows, n - r), e
                job += 1
        offset += n
    if job > capacity:
        raise ValueError(f"{job} jobs exceed the table capacity {capacity}")
    return row0, rows, exp, job


def do_moe_reference(x, topk_ids, topk_weights, mats: dict, *, hot_rows: int, limit: float = 10.0,
                     job_rows: int = JOB_ROWS) -> np.ndarray:
    """``out`` of the decode-once launch alone (hot experts only). ``mats[e] = {"w1"|"w3"|"w2": (w_hat, suh, svh)}``
    with ``w_hat [in, out]`` the decoded (rotated-domain) weight, as ``reference.apply`` takes it."""
    x = np.asarray(x, dtype=np.float64)
    rows_n, top_k = topk_ids.shape
    flat = np.asarray(topk_ids).reshape(-1)
    order = np.argsort(flat, kind="stable")
    token_sorted = order // top_k
    weight_sorted = np.asarray(topk_weights, dtype=np.float64).reshape(-1)[order]
    experts = len(mats)
    counts = np.bincount(flat, minlength=experts + 1)
    sorted_ids = flat[order]
    row_exp = np.where(sorted_ids < hot_rows, sorted_ids, -1)
    capacity = -(-flat.size // job_rows) + experts
    row0, nrows, exp, njobs = build_jobs(counts, hot_rows, capacity, job_rows=job_rows)
    out = np.zeros((rows_n, x.shape[1]))
    covered = np.zeros(flat.size, bool)
    for j in range(njobs):
        lo, hi, e = row0[j], row0[j] + nrows[j], exp[j]
        if not np.all(row_exp[lo:hi] == e):
            raise AssertionError(f"job {j} rows {lo}:{hi} are not all expert {e}")
        covered[lo:hi] = True
        xs = x[token_sorted[lo:hi]]
        m = mats[int(e)]
        gate = ref.apply(xs, *m["w1"])
        up = ref.apply(xs, *m["w3"])
        act = ref.swiglu(gate, up, limit, order="xmoe")
        down = ref.apply(act, *m["w2"])
        np.add.at(out, token_sorted[lo:hi], down * weight_sorted[lo:hi, None])
    if not np.array_equal(covered, row_exp >= 0):
        raise AssertionError("the jobs do not cover exactly the hot rows")
    return out
