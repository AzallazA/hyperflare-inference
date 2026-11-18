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
    for bad in ("9", "9:0", "0:5", "18:5,9:10", "9:5,9:4", "a:b"):
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
    return dict(rows=q_all.shape[0], forced=forced_num_splits)


class _Q:
    def __init__(self, rows):
        self.shape = (rows, 24, 512)


def _call(wrapper, rows, *, indexed=True, dsv41=True, forced=None):
    return wrapper(q_all=_Q(rows), swa_k_cache=None, swa_indices=None, workspace=None, swa_page_size=128,
                   sm_count=188, indexed_k_cache=object() if indexed else None, forced_num_splits=forced,
                   traits_override="dsv41" if dsv41 else "dsv4")


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
    assert wrapper.applied == [(5, 10), (13, 5), (32, 5)]


def test_install_replaces_policy_and_uninstall_restores(kernel):
    module, first = kernel
    second = ds.install(((4, 2),), module, is_dsv41=lambda traits: traits == "dsv41")
    assert second.original is prepare and module.prepare_unified_decode_launch is second
    assert _call(second, 3) == dict(rows=3, forced=2)
    ds.uninstall(module)
    assert module.prepare_unified_decode_launch is prepare and not hasattr(module, ds._ORIGINAL)


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
