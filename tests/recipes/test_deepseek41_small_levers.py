# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DS41 TP3 decode small levers (perf/ds41-decode-small-levers, 2026-11-18), CPU checks of the switches that need no
kernel run: tp3.pin_small_qproj (pinned plan triple for <= 8-row attention declarations), tp3.draft_head_plans
(prepared draft-head overrides) and the shared ``_overridden`` plan helper. The kernel-level proofs are
tests/components/test_rotary_shared_angles_interpret.py and test_logprob_single_exchange_interpret.py."""

import json
from types import SimpleNamespace

import pytest

from ignition.recipes.deepseek41.tp3_config import PIN_SMALL_QPROJ, PIN_SMALL_ROWS, TP3Config


def test_pin_switch_is_off_by_default_parsed_and_recorded():
    assert TP3Config().pin_small_qproj is False and "pin_small_qproj" not in TP3Config().to_metadata()
    on = TP3Config.from_environment({"FLARE_DS41_PIN_SMALL_QPROJ": "1"})
    assert on.pin_small_qproj is True and on.to_metadata()["pin_small_qproj"] is True
    with pytest.raises(TypeError):
        TP3Config(pin_small_qproj="1")


def test_default_record_is_unchanged_by_the_new_switches():
    record = TP3Config().to_metadata()
    for name in ("rotary_once", "pin_small_qproj", "logprob_single_exchange", "no_generation_logprobs",
                 "draft_head_plans"):
        assert name not in record
    assert TP3Config().describe() == "TP3 config: defaults"


def _query(rows):
    """The fields b12x's dense GEMM knob domain reads, for an attention projection declaration (MXFP8 rows with RHS
    prefetch readiness, one matrix, provided output, unit alpha: QuantizedProjection's packed-MXFP8 path)."""
    return SimpleNamespace(recipe="mxfp8", rhs_prefetch_ready=True, batch=1, max_rows=rows, output_mode="provided",
                           alpha_mode="unit")


@pytest.mark.parametrize("rows", range(1, PIN_SMALL_ROWS + 1))
def test_pinned_gemm_configs_are_legal_for_every_pinned_row_count(rows):
    from b12x.gemm import _tuning

    domain = _tuning.knob_values(_query(rows))
    for role in ("qkv_a", "q_b"):
        config = _tuning.TUNING.decode_config(PIN_SMALL_QPROJ[role])
        for name, values in domain.items():
            value = getattr(config, name)
            assert any(type(value) is type(allowed) and value == allowed for allowed in values), (role, name, value)


def test_the_programmatic_variants_stop_at_eight_rows():
    """Why the pin covers <= 8 rows only: b12x offers programmatic launch / evict-first for MXFP8 GEMMs up to 8 rows
    (ready_small_mxfp8), so the pinned triple is illegal at 9 rows (FINDINGS' "<= 16 rows" needs a b12x change)."""
    from b12x.gemm import _tuning

    assert PIN_SMALL_ROWS == 8
    assert _tuning.knob_values(_query(9))["programmatic_serialization"] == (False,)
    assert _tuning.knob_values(_query(8))["programmatic_serialization"] == (False, True)


def test_pinned_norm_config_decodes():
    from b12x.quantization.mla_norm_mxfp8 import _tuning

    assert _tuning.TUNING.decode_config(PIN_SMALL_QPROJ["norm"]).programmatic_serialization is True


def test_pinned_values_are_the_cache_selections_at_three_and_six_rows():
    """The 11-15 tuning cache (REVIEW-FABLE.md section 1; read 2026-11-18): qkv_a K4 32x64 K128 programmatic, q_b K1
    16x128 K128 programmatic + evict-first, norm programmatic, at the 3- and 6-row target declarations."""
    qkv, q_b = PIN_SMALL_QPROJ["qkv_a"], PIN_SMALL_QPROJ["q_b"]
    assert (qkv["split_k_slices"], qkv["tile_m"], qkv["tile_n"], qkv["tile_k"]) == (4, 32, 64, 128)
    assert qkv["programmatic_serialization"] and not qkv["evict_first"] and qkv["large_m_unroll"]
    assert (q_b["split_k_slices"], q_b["tile_m"], q_b["tile_n"], q_b["tile_k"]) == (1, 16, 128, 128)
    assert q_b["programmatic_serialization"] and q_b["evict_first"] and not q_b["large_m_unroll"]


class _Plan:
    def __init__(self, query, override=None):
        self.query, self.override = query, override
        self.contract = SimpleNamespace(decode_config=lambda payload: ("decoded", tuple(sorted(payload.items()))))

    def same_declaration_except_override(self, other):
        return self.query == other.query


def test_overridden_keeps_the_declaration_and_decodes_with_the_plans_contract():
    from ignition.components.linear import _overridden

    def factory(query, override=None):
        return _Plan(query, override)

    assert _overridden(factory, "q", None).override is None
    plan = _overridden(factory, "q", {"b": 2, "a": 1})
    assert plan.override == ("decoded", (("a", 1), ("b", 2))) and plan.query == "q"
    with pytest.raises(TypeError):
        _overridden(factory, "q", [("a", 1)])

    def drifting(query, override=None):
        return _Plan(query if override is None else "other", override)

    with pytest.raises(ValueError, match="must not change the operator declaration"):
        _overridden(drifting, "q", {"a": 1})


def test_draft_head_plans_parse_record_and_validate(tmp_path):
    config = {"split_k": 4, "tile_k": 64, "tile_m": 16, "tile_n": 128, "mode": "a16", "a16_backend": "auto",
              "dense_config": None, "fuse_swiglu": False, "mma_warps": None}
    path = tmp_path / "plans.json"
    path.write_text(json.dumps({"80": config, "40": config}))
    parsed = TP3Config.from_environment({"FLARE_DS41_DRAFT_HEAD_PLANS": str(path)})
    assert [rows for rows, _ in parsed.draft_head_plans] == [40, 80]
    record = parsed.to_metadata()["draft_head_plans"]
    assert record == {"40": config, "80": config} and str(tmp_path) not in json.dumps(record)  # never the path
    inline = TP3Config.from_environment({"FLARE_DS41_DRAFT_HEAD_PLANS": json.dumps({"40": config})})
    assert inline.draft_head_plans == ((40, json.dumps(config, sort_keys=True)),)
    assert TP3Config.from_environment({}).draft_head_plans is None
    for bad in ('{"x": {}}', "[]", '{"40": 3}'):
        with pytest.raises(ValueError):
            TP3Config.from_environment({"FLARE_DS41_DRAFT_HEAD_PLANS": bad})
    with pytest.raises(ValueError):
        TP3Config(draft_head_plans=((40, "{}"), (40, "{}")))
