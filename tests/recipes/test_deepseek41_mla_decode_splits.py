# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for ignition/components/mla_decode_splits.py (FLARE_DS41_MLA_DECODE_SPLITS).

The hook wraps b12x's ``prepare_unified_decode_launch`` and passes ``forced_num_splits`` from a per-row-count policy
for DSV41 decode launches with an indexed cache. These tests check the policy parser, which launches it touches, that
the wrapper's signature matches b12x's, that b12x's own split planner honours the forced value (and that the
environment variable B12X_MLA_SM120_NUM_SPLITS does not, which is why the hook exists), and install/uninstall.
GPU proof: the attn-splits microbench (decode-levers-45-20261118/MICROBENCH-4.md).
"""

import inspect
import types

import pytest

from ignition.components import mla_decode_splits as ds


def test_parse_policy():
    assert ds.parse_policy(None) == () and ds.parse_policy(" ") == ()
    assert ds.parse_policy("9:10, 18:5,28:10,37:5") == ((9, 10), (18, 5), (28, 10), (37, 5))
    assert ds.format_policy(ds.parse_policy(" 9:10,18:5 ")) == "9:10,18:5"
    for bad in ("9", "9:0", "0:5", "18:5,9:10", "9:5,9:4", "a:b", "9:6", "9:9", "9:11"):
        with pytest.raises(ValueError):
            ds.parse_policy(bad)


def test_splits_for_picks_first_bound_and_none_above():
    policy = ds.parse_policy("9:10,18:5,37:5")
    assert [ds.splits_for(policy, rows) for rows in (1, 9, 10, 18, 19, 37, 38, 80)] == [10, 10, 5, 5, 5, 5, None, None]


def prepare(*, q_all, swa_k_cache, swa_indices, workspace, swa_page_size, sm_count, q_alignment=None,
            indexed_k_cache=None, indexed_indices=None, indexed_page_size=None, attn_sink=None,
            forced_num_splits=None, scale_format_override=None, model_type_override=None, fp8_rope_override=None,
            latent_scale_per_token=False, v41_compute_mode="fp8", v41_heads_per_block=None, traits_override=None,
            return_lse=False, controls=None):
    """Signature twin of b12x kernel.prepare_unified_decode_launch (checked against it below)."""
    return types.SimpleNamespace(rows=q_all.shape[0], forced=forced_num_splits,
                                 num_splits=None if forced_num_splits is None else min(forced_num_splits, 10))


class _Q:
    def __init__(self, rows):
        self.shape = (rows, 24, 512)


def _call(wrapper, rows, *, indexed=True, dsv41=True, forced=None):
    launch = wrapper(q_all=_Q(rows), swa_k_cache=None, swa_indices=None, workspace=None, swa_page_size=128,
                   sm_count=188, indexed_k_cache=object() if indexed else None, forced_num_splits=forced,
                   traits_override="dsv41" if dsv41 else "dsv4")
    return dict(rows=launch.rows, forced=launch.forced)


@pytest.fixture
def kernel():
    module = types.SimpleNamespace(prepare_unified_decode_launch=prepare)
    wrapper = ds.install(ds.parse_policy("9:10,18:5,37:5"), module, is_dsv41=lambda traits: traits == "dsv41")
    return module, wrapper


def test_policy_applies_to_dsv41_indexed_decode_only(kernel):
    module, wrapper = kernel
    assert module.prepare_unified_decode_launch is wrapper
    assert _call(wrapper, 5) == dict(rows=5, forced=10)
    assert _call(wrapper, 13) == dict(rows=13, forced=5)
    assert _call(wrapper, 32) == dict(rows=32, forced=5)
    assert _call(wrapper, 80) == dict(rows=80, forced=None)           # above the last bound: b12x's choice
    assert _call(wrapper, 13, indexed=False) == dict(rows=13, forced=None)  # SWA-only (drafts)
    assert _call(wrapper, 13, dsv41=False) == dict(rows=13, forced=None)    # other models
    assert _call(wrapper, 13, forced=3) == dict(rows=13, forced=3)          # an explicit caller value wins
    assert wrapper.applied == [(5, 10, 10), (13, 5, 5), (32, 5, 5)]  # (rows, requested, planned)
    assert ds.installed_policy(module) == ((9, 10), (18, 5), (37, 5))


def test_install_replaces_policy_and_uninstall_restores(kernel):
    module, first = kernel
    second = ds.install(((4, 2),), module, is_dsv41=lambda traits: traits == "dsv41")
    assert second.original is prepare and module.prepare_unified_decode_launch is second
    assert _call(second, 3) == dict(rows=3, forced=2)
    ds.uninstall(module)
    assert module.prepare_unified_decode_launch is prepare and not hasattr(module, ds._ORIGINAL)
    assert ds.installed_policy(module) == ()


def test_unset_environment_installs_nothing(monkeypatch):
    monkeypatch.delenv(ds.ENV, raising=False)
    assert ds.install_from_environment() is None


def test_signature_and_b12x_contract():
    from b12x.attention._shared.mla import kernel

    original = getattr(kernel, ds._ORIGINAL, kernel.prepare_unified_decode_launch)
    assert list(inspect.signature(original).parameters) == list(inspect.signature(prepare).parameters)
    source = inspect.getsource(original)
    # The hook relies on: DSV41 forcing the workspace split count only when no explicit value is given, and the
    # explicit value reaching plan_unified_decode_splits.
    assert "if traits.model_type == ModelType.DSV41 and forced_num_splits is None:" in source
    assert "forced_num_splits=forced_num_splits" in source
    # The compressed MLA preparation resolves the function from the module at call time (so the wrapper applies).
    from b12x.attention.compressed_sparse_mla import _preparation

    lower = inspect.getsource(_preparation._lower_launch)
    assert "prepare_unified_decode_launch,\n" in lower and "launch = prepare_unified_decode_launch(" in lower


def test_b12x_planner_honours_forced_splits_but_not_the_environment(monkeypatch):
    from b12x.attention._shared.mla.kernel import plan_unified_decode_splits

    monkeypatch.setenv("B12X_MLA_SM120_NUM_SPLITS", "5")
    common = dict(topk=128, extra_topk=512, max_chunks=10, num_tokens=13, h_blocks=2, sm_count=188)
    # DSV41 always passes the workspace count as forced_num_splits: the environment pin is ignored.
    assert plan_unified_decode_splits(forced_num_splits=10, **common) == (10, 10, 1)
    for forced, expected in ((5, (10, 5, 2)), (4, (10, 4, 3)), (3, (10, 3, 4)), (2, (10, 2, 5)), (1, (10, 1, 10)),
                             (12, (10, 10, 1))):
        assert plan_unified_decode_splits(forced_num_splits=forced, **common) == expected
    # Split capacity of the workspace caps the value (10 for the 64-wide geometry).
    assert plan_unified_decode_splits(forced_num_splits=5, **dict(common, max_chunks=1)) == (10, 1, 10)


def test_b12x_fingerprints_do_not_move_with_the_policy():
    from tests.recipes import test_deepseek41_mla_remainder_stream as rs_test

    plain = rs_test._fingerprints({"FP_PLAIN": "1"})
    on = rs_test._fingerprints({ds.ENV: "9:10,18:5,28:10,37:5"})
    assert len(plain["modules"]) >= 32
    for key in ("unified_decode", "sink_merge", "factory", "modules"):
        assert plain[key] == on[key], key


def _stub_worker_initializer():
    """initialize_b12x_worker with only b12x's own device setup stubbed (it needs a GPU)."""
    import b12x._lib.compile_pool as pool

    pool._initialize_worker = lambda *args: None
    from ignition.runtime_patches.triton import initialize_b12x_worker

    initialize_b12x_worker()


def _worker_probe(_):
    from b12x.attention._shared.mla import kernel

    return type(kernel.prepare_unified_decode_launch).__name__


_WORKER = r"""
import multiprocessing as mp
from tests.recipes.test_deepseek41_mla_decode_splits import _stub_worker_initializer, _worker_probe
if __name__ == "__main__":
    with mp.get_context("spawn").Pool(1, initializer=_stub_worker_initializer) as workers:  # b12x spawns workers
        print("WORKER", workers.map(_worker_probe, [0])[0])
"""


@pytest.mark.parametrize("policy,expected", [("9:10,18:5", "_DecodeSplitPolicy"), ("", "function")])
def test_spawned_b12x_compile_worker_installs_the_policy(policy, expected):
    """B1: a b12x compile worker is spawned and never imports compressed_mla; its initializer must install the hook
    from the inherited variable, or workers compile 10-split programs the main process does not ask for."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(root), ds.ENV: policy}
    result = subprocess.run([sys.executable, "-c", _WORKER], cwd=root, env=env, capture_output=True, text=True,
                            timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    assert f"WORKER {expected}" in result.stdout, result.stdout + result.stderr[-2000:]


def test_tp3_config_records_the_policy_and_checks_the_installed_hook():
    from ignition.recipes.deepseek41.tp3_config import TP3Config

    assert TP3Config().mla_decode_splits is None and "mla_decode_splits" not in TP3Config().to_metadata()
    config = TP3Config.from_environment({ds.ENV: " 9:10, 18:5"})
    assert config.mla_decode_splits == "9:10,18:5" and config.to_metadata()["mla_decode_splits"] == "9:10,18:5"
    assert TP3Config.from_environment({ds.ENV: ""}).mla_decode_splits is None
    config.check_mla_decode_splits(installed=((9, 10), (18, 5)))
    TP3Config().check_mla_decode_splits(installed=())
    with pytest.raises(ValueError, match="installed DSV41 decode split policy"):
        config.check_mla_decode_splits(installed=())
    with pytest.raises(ValueError, match="installed DSV41 decode split policy"):
        TP3Config().check_mla_decode_splits(installed=((9, 10),))
    with pytest.raises(ValueError):
        TP3Config.from_environment({ds.ENV: "9:7"})
