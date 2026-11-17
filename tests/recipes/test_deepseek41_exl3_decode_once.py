# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the decode-once prefill path's bookkeeping (exp/ds41-exl3): the fixed-capacity job table, the cold-row
mask of parked banks, the scratch/job sizing and the FLARE_DS41_EXL3_PREFILL switch. The NumPy mirror
(``exl3.do_reference``) must equal ``reference.moe_reference`` over the hot experts."""

import numpy as np
import pytest

from ignition.recipes.deepseek41.exl3 import do_reference as dor
from ignition.recipes.deepseek41.exl3 import reference as ref

torch = pytest.importorskip("torch")
from ignition.recipes.deepseek41.exl3 import experts as ex  # noqa: E402


def brute_jobs(counts, hot_rows, job_rows=128):
    jobs, offset = [], 0
    for e, c in enumerate(counts):
        n = abs(int(c))
        if e < hot_rows and c > 0:
            jobs += [(offset + r, min(job_rows, n - r), e) for r in range(0, n, job_rows)]
        offset += n
    return jobs


@pytest.mark.parametrize("seed", range(6))
def test_job_table_matches_brute_force_and_fits_capacity(seed):
    rng = np.random.default_rng(seed)
    experts = 384
    rows = int(rng.choice([128, 512, 4096]))
    counts = rng.multinomial(rows * 6, rng.dirichlet(np.full(experts, 0.3)))
    counts = np.append(counts, 0)
    hot = int(rng.choice([experts, 308, 0]))
    signed = np.where(np.arange(experts + 1) < hot, counts, -counts) if seed % 2 else counts
    capacity = ex.do_job_capacity(rows * 6, experts)
    row0, n, e, njobs = dor.build_jobs(signed, hot, capacity)
    want = brute_jobs(signed, hot)
    assert njobs == len(want) <= capacity
    assert list(zip(row0[:njobs], n[:njobs], e[:njobs])) == want
    assert np.all(n[njobs:] == 0)


def test_capacity_bound_is_tight_enough_for_the_worst_case():
    # every expert gets 129 rows: 2 jobs each -> R / 128 + E bound holds with equality-ish
    experts, per = 384, 129
    counts = np.append(np.full(experts, per), 0)
    capacity = ex.do_job_capacity(experts * per, experts)
    _, _, _, njobs = dor.build_jobs(counts, experts, capacity)
    assert njobs == 2 * experts <= capacity


def _mats(rng, experts, hidden=256, inter=128):
    mats, eff = {}, {}
    for e in range(experts):
        m = {}
        for name, (i, o) in (("w1", (hidden, inter)), ("w3", (hidden, inter)), ("w2", (inter, hidden))):
            w_hat = rng.standard_normal((i, o)) / np.sqrt(i)
            suh = rng.choice([-1.0, 1.0], i) * rng.uniform(0.5, 1.5, i)
            svh = rng.uniform(0.5, 1.5, o)
            m[name] = (w_hat, suh, svh)
        mats[e] = m
        # effective [in, out] weight as the reference composes it: x @ W == apply(x, w_hat, suh, svh)
        eff[e] = {name: ref.apply(np.eye(m[name][0].shape[0]), *m[name]) for name in m}
    return mats, eff


@pytest.mark.parametrize("hot_rows", [12, 9, 0])
def test_decode_once_mirror_equals_the_reference_over_hot_experts(hot_rows):
    rng = np.random.default_rng(11 + hot_rows)
    experts, tokens, top_k = 12, 300, 6          # 1,800 routed rows: several experts need more than one 128-row job
    mats, eff = _mats(rng, experts)
    x = rng.standard_normal((tokens, 256))
    ids = np.stack([rng.choice(experts, top_k, replace=False, p=np.linspace(2, 1, experts) / np.linspace(2, 1, experts).sum())
                    for _ in range(tokens)])
    w = rng.dirichlet(np.ones(top_k), tokens)
    got = dor.do_moe_reference(x, ids, w, mats, hot_rows=hot_rows, limit=10.0)
    # reference over the hot experts only (cold slots are the group kernel's second launch)
    w_hot = np.where(ids < hot_rows, w, 0.0)
    want = ref.moe_reference(x, ids, w_hot, eff, limit=10.0, order="xmoe")
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-9)


def test_prefill_switch_and_scratch_sizing(monkeypatch):
    monkeypatch.setenv("FLARE_DS41_EXL3_PREFILL", "do")
    assert ex.prefill_mode() == "do"
    monkeypatch.setenv("FLARE_DS41_EXL3_PREFILL", "")
    assert ex.prefill_mode() == "group"
    monkeypatch.setenv("FLARE_DS41_EXL3_PREFILL", "fast")
    with pytest.raises(ValueError, match="FLARE_DS41_EXL3_PREFILL"):
        ex.prefill_mode()
    monkeypatch.delenv("FLARE_DS41_EXL3_PREFILL")
    assert ex.PREFILL_MODE in ("group", "do")
    local = 1152
    group = ex.scratch_halfs(4096, local, 4)
    do = ex.scratch_halfs(4096, local, 4, decode_once=True)
    assert group == 4 * 4096 * (2 * 5120 + 2 * local)
    assert do == 6 * 4096 * (2 * 5120 + 2 * local)      # 616 MB at TP2 vs 411 MB for the group views
    assert ex.do_job_capacity(4096 * 6, 384) == 192 + 384
