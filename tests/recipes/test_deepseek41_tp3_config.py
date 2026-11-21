# SPDX-License-Identifier: Apache-2.0
"""TP3 fork configuration: environment parsing, defaults, validation and the metadata record (CPU only)."""

from dataclasses import replace
from pathlib import Path

import pytest

from ignition.recipes.deepseek41.tp3_config import (TP3Config, describe_checkpoint, describe_counts_dir,
                                                    format_layers, parse_layers)

# The release environment as tp3-tools/release/export.sh really exports it with its defaults (since the 20261121
# release: PARK 20-39, HOT=355, COUNTS = the blended order in expert-orders/order-mix-provisional-exl3S, the round-3
# switches and GATHER_PULL/ROTARY_PREFILL on; before: HOT=342, COUNTS=$HERE/expert-order), VISION_HOST/VISION_STAGE 1, ENGRAM_CHUNK_ROWS 1024, SHARE_ENGRAM_RESIDUAL 1, and since the
# 20261116 release PARKED_ROWS 0, 768-row prefill all-reduce parts, index/context side streams, uneven wo_a split,
# host-row embedding gather, VRAM reclaim, mHC PDL): the script's environment block runs in bash (everything before
# ``exec``) and its FLARE_DS41_* / HF_HUB_OFFLINE / B12X_* values are read back, so a change to the script's defaults
# reaches these tests. The default two-shot pull and gather pull need B12X_OVERLAY (a directory with ``b12x_tree`` and
# the gather-pull patch's marker): a stub is passed.
EXPORT_SH = Path(__file__).resolve().parents[2] / "tp3-tools" / "release" / "export.sh"
RELEASE_CKPT = "/models/ds41-1017-engram-fp8"


def _overlay_stub():
    import atexit
    import shutil
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="b12x-overlay-stub-"))
    (root / "b12x_tree" / "comm" / "pcie").mkdir(parents=True)
    # the gather pull default (GATHER_PULL=1) checks the overlay for the gather-pull patch's marker
    (root / "b12x_tree" / "comm" / "pcie" / "_dcp_a2a_cute.py").write_text("_gather_pull_enabled\n")
    atexit.register(shutil.rmtree, root, True)
    return str(root)


OVERLAY_STUB = _overlay_stub()


def export_environment(**variables):
    """The script's FLARE_DS41_* and HF_HUB_OFFLINE values plus B12X_MHC_PDL (TP3Config mirrors b12x's switch)."""
    return {key: value for key, value in export_variables(**variables).items()
            if key.startswith("FLARE_DS41_") or key in ("HF_HUB_OFFLINE", "B12X_MHC_PDL")}


def export_variables(**variables):
    import os
    import subprocess

    script = EXPORT_SH.read_text()
    body = script[:script.index("\nexec ")]
    dump = '\nfor name in ${!FLARE_DS41_@} ${!B12X_@} HF_HUB_OFFLINE; do printf "%s=%s\\0" "$name" "${!name}"; done'
    output = subprocess.run(["bash", "-c", body + dump, str(EXPORT_SH)], check=True, capture_output=True,
                            env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "CKPT": RELEASE_CKPT,
                                 "B12X_OVERLAY": OVERLAY_STUB, **variables})
    pairs = (item.split("=", 1) for item in output.stdout.decode().split("\0") if "=" in item)
    return {key: value for key, value in pairs
            if key.startswith(("FLARE_DS41_", "B12X_")) or key == "HF_HUB_OFFLINE"}


RELEASE_ENV = export_environment()

RELEASE_COUNTS = EXPORT_SH.parent / "expert-orders" / "order-mix-provisional-exl3S"
# switches the 20261121 release export turns on beyond the 20261116 set (recorded when true / when set)
RELEASE_ON = dict(markov_embed_host=True, markov_stage_rows=True, vision_arena_host=True, glue_pass2=True,
                  rotary_once=True, pin_small_qproj=True, logprob_single_exchange=True, rotary_once_prefill=True,
                  mma_pad_once=True, mla_decode_splits="18:10,37:5,47:4,62:3,94:2,96:3", mla_h24="9:0,96:1",
                  prefill_skip_suffix=(4096,), index_row_split=(4096,), index_row_split_min_context=32768)

RELEASE_CONFIG = TP3Config(
    host_expert_layers=frozenset(range(20, 40)), hot_experts=355, merged_bank=True,
    expert_counts_dir=RELEASE_COUNTS, embed_host=True, embed_rows=True, vision_host=True,
    vision_stage=True, ar_rows=1024, ar_part_rows=768, engram_part_rows=768, mhc_pdl=True, tp3_graph_gib=1.0,
    tp3_vocab_parallel=True, index_side_stream=True, context_side_stream=True,
    share_engram_token_map=True, trim_vocab_gather=True, skip_draft_vocab_gather=True, share_vocab_columns=True,
    brrr_layers=frozenset(range(40)), brrr_parked_max_rows=0, brrr_ckpt=Path(RELEASE_CKPT),
    brrr_woa_uneven=1, brrr_woa_small=0, engram_chunk_rows=1024, share_engram_residual=True, **RELEASE_ON)


def test_export_script_sets_the_b12x_switches_of_the_release():
    variables = export_variables()
    assert {key: variables.get(key) for key in ("B12X_PCIE_TWOSHOT_PULL", "B12X_MLA_DECODE_REMAINDER_STREAM",
                                                 "B12X_MHC_PDL", "B12X_PCIE_DCP_GATHER_PULL")} == \
        {"B12X_PCIE_TWOSHOT_PULL": "1", "B12X_MLA_DECODE_REMAINDER_STREAM": "1", "B12X_MHC_PDL": "1",
         "B12X_PCIE_DCP_GATHER_PULL": "auto"}
    off = export_variables(TWOSHOT="0", ATTN="0", MHC_PDL="0", GATHER_PULL="0", B12X_OVERLAY="")
    assert (off["B12X_PCIE_TWOSHOT_PULL"], off["B12X_MLA_DECODE_REMAINDER_STREAM"], off["B12X_MHC_PDL"]) == ("0",) * 3
    assert "B12X_PCIE_DCP_GATHER_PULL" not in off


def test_export_script_environment_is_parsed_not_copied():
    assert RELEASE_ENV["FLARE_DS41_ENGRAM_CHUNK_ROWS"] == "1024" and RELEASE_ENV["FLARE_DS41_HOT_EXPERTS"] == "355"
    assert RELEASE_ENV["FLARE_DS41_EXPERT_COUNTS_DIR"] == str(RELEASE_COUNTS)
    # The script's overrides reach the parsed environment (it is the script, not a copy).
    overridden = export_environment(HOT="300", PARKED_ROWS="12")
    assert overridden["FLARE_DS41_HOT_EXPERTS"] == "300" and overridden["FLARE_DS41_BRRR_PARKED_MAX_ROWS"] == "12"


def test_release_environment_parses_to_the_expected_config():
    assert TP3Config.from_environment(RELEASE_ENV) == RELEASE_CONFIG


def test_release_environment_does_not_touch_the_unset_switches():
    config = TP3Config.from_environment(RELEASE_ENV)
    assert config.vision_stage_copy == "memcpy"
    assert config.brrr_dense_native and config.fused_attn_per_chunk
    assert not (config.brrr_native_scales or config.brrr_csf_check or config.brrr_log
                or config.mini or config.no_engram or config.probe)
    assert config.brrr_break is None


def test_defaults_are_the_bare_upstream_behaviour():
    config = TP3Config.from_environment({})
    assert config == TP3Config()
    assert config.host_expert_layers == frozenset() and config.brrr_layers == frozenset()
    assert config.hot_experts == 0 and config.ar_rows == 0 and config.tp3_graph_gib is None
    assert config.expert_counts_dir is None and config.brrr_ckpt is None
    assert config.brrr_dense_native is True and config.fused_attn_per_chunk is True
    assert config.describe() == "TP3 config: defaults"


def test_empty_values_count_as_unset_and_or_zero_integers():
    env = {key: "" for key in RELEASE_ENV if key != "B12X_MHC_PDL"}  # b12x reads a set-but-empty B12X_MHC_PDL as on
    assert TP3Config.from_environment(env) == TP3Config()
    assert TP3Config.from_environment({"FLARE_DS41_HOT_EXPERTS": "", "FLARE_DS41_AR_ROWS": ""}).hot_experts == 0
    # Two switches keep the base parsers' set-but-empty meaning instead (documented on the fields).
    assert TP3Config.from_environment({"FLARE_DS41_BRRR_DENSE_NATIVE": "", "FLARE_DS41_BRRR_LAYERS": "1",
                                       "FLARE_DS41_BRRR_CKPT": "/c"}).brrr_dense_native is False
    assert TP3Config.from_environment({"FLARE_DS41_FUSED_ATTN_PER_CHUNK": ""}).fused_attn_per_chunk is False
    assert TP3Config.from_environment({"FLARE_DS41_VISION_STAGE_COPY": ""}).vision_stage_copy == "memcpy"


def test_layer_specs_accept_ranges_and_comma_lists():
    assert parse_layers("20-22") == frozenset({20, 21, 22})
    assert parse_layers("20,21") == frozenset({20, 21})
    assert parse_layers(" 0-1, 7 ,9-9") == frozenset({0, 1, 7, 9})
    assert parse_layers("") == frozenset()
    assert format_layers({0, 1, 2, 7, 9, 10}) == "0-2,7,9-10"
    assert parse_layers(format_layers(range(20, 34))) == frozenset(range(20, 34))
    config = TP3Config.from_environment({"FLARE_DS41_HOST_EXPERT_LAYERS": "20,22-23"})
    assert config.host_expert_layers == frozenset({20, 22, 23})


def test_flag_semantics_match_the_historical_parsers():
    # "1" flags: only "1" switches them on.
    assert TP3Config.from_environment({"FLARE_DS41_MERGED_BANK": "1"}).merged_bank
    assert not TP3Config.from_environment({"FLARE_DS41_MERGED_BANK": "yes"}).merged_bank
    assert not TP3Config.from_environment({"FLARE_DS41_EMBED_HOST": "0"}).embed_host
    # "not in ('', '0')" switches: any other value is on.
    assert TP3Config.from_environment({"FLARE_DS41_TP3_VOCAB_PARALLEL": "yes"}).tp3_vocab_parallel
    assert not TP3Config.from_environment({"FLARE_DS41_TP3_VOCAB_PARALLEL": "0"}).tp3_vocab_parallel
    assert TP3Config.from_environment({"FLARE_DS41_NO_ENGRAM": "2"}).no_engram
    assert TP3Config.from_environment({"FLARE_DS41_PROBE": "1"}).probe
    # Default-on switches.
    assert not TP3Config.from_environment({"FLARE_DS41_BRRR_DENSE_NATIVE": "0", "FLARE_DS41_BRRR_LAYERS": "1",
                                           "FLARE_DS41_BRRR_CKPT": "/c"}).brrr_dense_native
    assert not TP3Config.from_environment({"FLARE_DS41_FUSED_ATTN_PER_CHUNK": "0"}).fused_attn_per_chunk
    assert not TP3Config.from_environment({"FLARE_DS41_FUSED_ATTN_PER_CHUNK": "x"}).fused_attn_per_chunk  # atoi
    assert TP3Config.from_environment({"FLARE_DS41_FUSED_ATTN_PER_CHUNK": "2"}).fused_attn_per_chunk
    assert TP3Config.from_environment({"FLARE_DS41_FUSED_ATTN_PER_CHUNK": " 1x"}).fused_attn_per_chunk
    assert not TP3Config.from_environment({"FLARE_DS41_BRRR_DENSE_NATIVE": "yes", "FLARE_DS41_BRRR_LAYERS": "1",
                                           "FLARE_DS41_BRRR_CKPT": "/c"}).brrr_dense_native
    # Numbers.
    assert TP3Config.from_environment({"FLARE_DS41_TP3_GRAPH_GIB": "1.5"}).tp3_graph_gib == 1.5
    assert TP3Config.from_environment({"FLARE_DS41_VISION_HOST": "1", "FLARE_DS41_VISION_STAGE": "1",
                                       "FLARE_DS41_VISION_STAGE_COPY": "kernel"}).vision_stage_copy == "kernel"
    assert TP3Config.from_environment({"FLARE_DS41_BRRR_LAYERS": "3", "FLARE_DS41_BRRR_CKPT": "/c",
                                       "FLARE_DS41_BRRR_BREAK": "rope"}).brrr_break == "rope"


@pytest.mark.parametrize("env,message", [
    ({"FLARE_DS41_BRRR_PARKED_MAX_ROWS": "12"}, "brrr_parked_max_rows needs brrr_layers"),
    ({"FLARE_DS41_BRRR_LAYERS": "0-39"}, "brrr_layers needs brrr_ckpt"),
    ({"FLARE_DS41_VISION_HOST": "1", "FLARE_DS41_VISION_STAGE_COPY": "kernel"}, "vision_stage_copy=kernel needs vision_stage"),
    ({"FLARE_DS41_VISION_HOST": "1", "FLARE_DS41_VISION_STAGE": "1", "FLARE_DS41_VISION_STAGE_COPY": "dma"},
     "vision_stage_copy must be 'memcpy' or 'kernel'"),
    ({"FLARE_DS41_HOST_EXPERT_LAYERS": "20-33", "FLARE_DS41_HOT_EXPERTS": "342", "FLARE_DS41_EXPERT_COUNTS_DIR": "/c"},
     "hot_experts needs merged_bank"),
    ({"FLARE_DS41_HOST_EXPERT_LAYERS": "20-33", "FLARE_DS41_HOT_EXPERTS": "342"}, "hot_experts needs expert_counts_dir"),
    ({"FLARE_DS41_BRRR_CSF_CHECK": "1"}, "brrr_csf_check needs brrr_layers"),
    ({"FLARE_DS41_BRRR_NATIVE_SCALES": "1"}, "brrr_native_scales needs brrr_layers"),
    ({"FLARE_DS41_BRRR_BREAK": "rope"}, "brrr_break needs brrr_layers"),
    ({"FLARE_DS41_BRRR_SPIN_NS": "50"}, "brrr_spin_ns needs brrr_layers"),
    ({"FLARE_DS41_BRRR_SPIN_NS": "0"}, "brrr_spin_ns needs brrr_layers"),
    ({"FLARE_DS41_BRRR_TWO_SHOT_MIN": "13"}, "brrr_two_shot_min needs brrr_layers"),
    ({"FLARE_DS41_BRRR_LAYERS": "3", "FLARE_DS41_BRRR_CKPT": "/c", "FLARE_DS41_BRRR_SPIN_NS": "-1"},
     r"brrr_spin_ns must be unset or an integer in \[0, 1000000\]"),
    ({"FLARE_DS41_BRRR_LAYERS": "3", "FLARE_DS41_BRRR_CKPT": "/c", "FLARE_DS41_BRRR_TWO_SHOT_MIN": "0"},
     r"brrr_two_shot_min must be unset or an integer in \[1, 129\]"),
    ({"FLARE_DS41_BRRR_LAYERS": "3", "FLARE_DS41_BRRR_CKPT": "/c", "FLARE_DS41_BRRR_TWO_SHOT_MIN": "130"},
     r"brrr_two_shot_min must be unset or an integer in \[1, 129\]"),
    ({"FLARE_DS41_BRRR_LAYERS": "3", "FLARE_DS41_BRRR_CKPT": "/c", "FLARE_DS41_BRRR_BREAK": "norm"},
     "brrr_break must be unset or 'rope'"),
    ({"FLARE_DS41_BRRR_DENSE_NATIVE": "0"}, "brrr_dense_native=False needs brrr_layers"),
    ({"FLARE_DS41_TP3_GRAPH_GIB": "0"}, "tp3_graph_gib must be positive"),
    ({"FLARE_DS41_HOT_EXPERTS": "-1"}, "hot_experts must be a non-negative integer"),
    ({"FLARE_DS41_BRRR_PROLOGUE": "1"}, "FLARE_DS41_BRRR_PROLOGUE is no longer supported"),
    ({"FLARE_DS41_QB_ATOMIC": "1"}, "FLARE_DS41_QB_ATOMIC is no longer supported"),
    ({"DS41_FUSED_ATTN_PER_CHUNK": "0"}, "DS41_FUSED_ATTN_PER_CHUNK is no longer supported"),
    # Default-off experiments not carried into the 2026-11-14 upstream port.
    *(({name: "1"}, f"{name} is no longer supported") for name in (
        "FLARE_DS41_GLOBAL_COLD", "FLARE_DS41_JOINT_HEADROOM_MIB", "FLARE_DS41_BRRR_SPARSE",
        "FLARE_DS41_BRRR_SPLIT_TAIL", "FLARE_DS41_BRRR_FAST_ROUTE", "FLARE_DS41_BRRR_COUNTER_PARITY",
        "FLARE_DS41_FUSED_QKV_NORM")),
])
def test_validation_errors_are_explicit(env, message):
    with pytest.raises(ValueError, match=message):
        TP3Config.from_environment(env)


def test_vision_stage_implies_vision_host_as_the_base_recipe_did():
    # The release export.sh defaults FLARE_DS41_VISION_STAGE=1; FLARE_DS41_VISION_HOST=0 must not kill the export.
    config = TP3Config.from_environment({"FLARE_DS41_VISION_STAGE": "1", "FLARE_DS41_VISION_HOST": "0"})
    assert config.vision_host and config.vision_stage
    assert TP3Config(vision_stage=True).vision_host is True
    config = TP3Config.from_environment({"FLARE_DS41_VISION_HOST": "0"})
    assert not config.vision_host and not config.vision_stage
    assert TP3Config.from_environment({"FLARE_DS41_VISION_STAGE": "1", "FLARE_DS41_VISION_STAGE_COPY": "kernel"}).vision_stage_copy == "kernel"


def test_planner_peeks_read_one_variable_without_validating_the_rest():
    stale = {"FLARE_DS41_BRRR_PARKED_MAX_ROWS": "12", "DS41_FUSED_ATTN_PER_CHUNK": "0",  # both invalid for from_environment
             "FLARE_DS41_TP3_GRAPH_GIB": "1.5", "FLARE_DS41_TP3_VOCAB_PARALLEL": "1"}
    with pytest.raises(ValueError):
        TP3Config.from_environment(stale)
    assert TP3Config.graph_gib_from_environment(stale) == 1.5
    assert TP3Config.vocab_parallel_from_environment(stale) is True
    assert TP3Config.graph_gib_from_environment({}) is None and TP3Config.graph_gib_from_environment({"FLARE_DS41_TP3_GRAPH_GIB": ""}) is None
    assert TP3Config.graph_gib_from_environment({"FLARE_DS41_TP3_GRAPH_GIB": "lots"}) is None
    assert TP3Config.vocab_parallel_from_environment({"FLARE_DS41_TP3_VOCAB_PARALLEL": "0"}) is False


def test_recipe_import_and_planner_helpers_survive_a_stale_environment(monkeypatch):
    import importlib
    import sys

    from ignition.recipes.deepseek41 import recipe

    monkeypatch.setenv("FLARE_DS41_BRRR_PARKED_MAX_ROWS", "12")  # invalid alone: parked rows without BRRR layers
    monkeypatch.setenv("FLARE_DS41_TP3_GRAPH_GIB", "2")
    monkeypatch.setenv("FLARE_DS41_TP3_VOCAB_PARALLEL", "1")
    assert recipe._graph_bytes(3) == 2 << 30 and recipe._graph_bytes(4) == 8 << 30
    from ignition import planning
    assert planning._tp3_vocabulary_parallel() is True
    module = importlib.reload(sys.modules[recipe.__name__])  # the declaration is rebuilt at import time
    assert module.RECIPE.ranks[3].graph_bytes == 2 << 30
    with pytest.raises(ValueError, match="brrr_parked_max_rows needs brrr_layers"):
        TP3Config.from_environment()


def test_metadata_never_records_absolute_paths(tmp_path):
    counts = tmp_path / "expert-order"
    counts.mkdir()
    for layer in (20, 21):
        (counts / f"expert-routes-rank0-layer{layer}.json").write_text('{"counts": [1, 2]}')
    view = tmp_path / "ds41-view"
    view.mkdir()
    (view / "config.json").write_text('{"model_type": "deepseek_v41"}')
    config = TP3Config(host_expert_layers=frozenset({20}), hot_experts=1, merged_bank=True, expert_counts_dir=counts,
                       brrr_layers=frozenset({0}), brrr_ckpt=view)
    record = config.to_metadata()
    assert record["expert_counts_dir"].startswith("expert-order#") and record["brrr_ckpt"].startswith("ds41-view#")
    assert str(tmp_path) not in json_text(record) and str(tmp_path) not in config.describe()
    assert len(record["expert_counts_dir"].split("#")[1]) == 12 and len(record["brrr_ckpt"].split("#")[1]) == 12
    # The fingerprint follows the content: a changed counts file or config.json changes it.
    (counts / "expert-routes-rank0-layer21.json").write_text('{"counts": [1, 2, 3]}')
    assert config.to_metadata()["expert_counts_dir"] != record["expert_counts_dir"]
    (view / "config.json").write_text('{"model_type": "other"}')
    assert config.to_metadata()["brrr_ckpt"] != record["brrr_ckpt"]
    assert describe_counts_dir(tmp_path / "absent") == "absent#missing"
    assert describe_checkpoint(tmp_path / "absent") == "absent#missing"
    assert describe_counts_dir(None) is None and describe_checkpoint(None) is None


def json_text(record):
    import json
    return json.dumps(record)


def test_malformed_numbers_and_layers_raise():
    with pytest.raises(ValueError):
        TP3Config.from_environment({"FLARE_DS41_HOT_EXPERTS": "many"})
    with pytest.raises(ValueError):
        TP3Config.from_environment({"FLARE_DS41_HOST_EXPERT_LAYERS": "20-x"})
    with pytest.raises(TypeError):
        TP3Config(merged_bank=1)


def test_metadata_record_is_json_ready_and_complete():
    import json
    from dataclasses import fields

    record = TP3Config.from_environment(RELEASE_ENV).to_metadata()
    from ignition.recipes.deepseek41.tp3_config import RECORDED_WHEN_SET, RECORDED_WHEN_TRUE

    # the release export sets the two wo_a switches (recorded when set), the other tuning switches stay unset
    assert set(record) == ({item.name for item in fields(TP3Config)} - set(RECORDED_WHEN_SET) - set(RECORDED_WHEN_TRUE)
                           | {"brrr_woa_small", "brrr_woa_uneven"} | set(RELEASE_ON))
    assert (record["brrr_woa_small"], record["brrr_woa_uneven"]) == (0, 1)
    assert record["host_expert_layers"] == "20-39" and record["brrr_layers"] == "0-39"
    counts, counts_digest = record["expert_counts_dir"].split("#")
    assert counts == "order-mix-provisional-exl3S" and (counts_digest == "missing" or len(counts_digest) == 12)
    assert record["tp3_graph_gib"] == 1.0
    # The release view may exist on the build host, so only the shape is fixed: basename#12-hex-or-missing.
    name, digest = record["brrr_ckpt"].split("#")
    assert name == "ds41-1017-engram-fp8" and (digest == "missing" or len(digest) == 12)
    assert "/" not in json_text(record)
    assert record["brrr_parked_max_rows"] == 0 and record["vision_stage"] is True and record["brrr_break"] is None
    json.dumps(record)  # every value serializes
    assert TP3Config().to_metadata()["host_expert_layers"] == ""
    described = TP3Config.from_environment(RELEASE_ENV).describe()
    assert described.startswith("TP3 config: ") and "hot_experts=355" in described and "brrr_log" not in described


def test_config_is_frozen_and_hashable():
    config = TP3Config.from_environment(RELEASE_ENV)
    with pytest.raises(AttributeError):
        config.hot_experts = 300
    assert hash(config) == hash(TP3Config.from_environment(dict(RELEASE_ENV)))


# The 2026-11-14 base port: the release config minus the features the port stages brought back (BRRR = S6, Engram
# chunk rows and the shared residual = S5, vision staging = S7). The tower stays host-parked (vision_host).
# The round-2 and reclaim switches of the 20261116 release came later and are off as well.
BASE_PORT_CONFIG = replace(RELEASE_CONFIG, brrr_layers=frozenset(), brrr_parked_max_rows=0, brrr_ckpt=None,
                           vision_stage=False, engram_chunk_rows=0, share_engram_residual=False, embed_rows=False,
                           ar_part_rows=0, engram_part_rows=0, mhc_pdl=False, index_side_stream=False,
                           context_side_stream=False, share_engram_token_map=False, trim_vocab_gather=False,
                           skip_draft_vocab_gather=False, share_vocab_columns=False, brrr_woa_uneven=None,
                           brrr_woa_small=None)


def test_preparation_config_carries_the_tp3_config():
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    assert PreparationConfig(max_active=4).tp3 == TP3Config()
    config = PreparationConfig(max_active=4, tp3=BASE_PORT_CONFIG)
    assert config.tp3 is BASE_PORT_CONFIG
    with pytest.raises(TypeError):
        PreparationConfig(max_active=4, tp3={"hot_experts": 342})


def test_every_switch_is_ported_and_the_full_release_environment_prepares():
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    # S5 (Engram chunk rows, shared Engram residual), S6 (BRRR) and S7 (vision staging) are ported: the real
    # export.sh environment, both S5 switches included, passes require_ported() and builds a PreparationConfig.
    full = TP3Config.from_environment(RELEASE_ENV)
    assert full.engram_chunk_rows == 1024 and full.share_engram_residual is True
    assert full.unported() == {} and full.require_ported() is full
    assert PreparationConfig(max_active=4, tp3=full).tp3 is full
    assert BASE_PORT_CONFIG.unported() == {} and BASE_PORT_CONFIG.require_ported() is BASE_PORT_CONFIG
    for off in (dict(FLARE_DS41_ENGRAM_CHUNK_ROWS="0"), dict(FLARE_DS41_SHARE_ENGRAM_RESIDUAL="0")):
        config = TP3Config.from_environment({**RELEASE_ENV, **off})
        assert config.require_ported() is config and PreparationConfig(max_active=4, tp3=config).tp3 is config


def test_brrr_is_ported_but_refuses_joint_tuning():
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    brrr = replace(BASE_PORT_CONFIG, brrr_layers=frozenset(range(40)), brrr_parked_max_rows=12, brrr_ckpt=Path("/c"))
    assert brrr.unported() == {}
    assert PreparationConfig(max_active=4, tp3=brrr).tp3 is brrr
    with pytest.raises(ValueError, match="brrr_layers cannot be combined with --joint-tuning"):
        PreparationConfig(max_active=4, tp3=brrr, joint_tuning=True)
    assert PreparationConfig(max_active=4, tp3=BASE_PORT_CONFIG, joint_tuning=True).joint_tuning


def test_engram_table_layout_switch():
    assert TP3Config().engram_tables == "auto"
    assert [TP3Config().shared_engram_tables(world) for world in (1, 3, 4, 6)] == [True, True, False, False]
    shared = TP3Config.from_environment({"FLARE_DS41_ENGRAM_TABLES": "shared"})
    assert [shared.shared_engram_tables(world) for world in (1, 3, 4, 6)] == [True] * 4
    partitioned = TP3Config(engram_tables="partitioned")
    assert [partitioned.shared_engram_tables(world) for world in (1, 3, 4)] == [True, False, False]
    assert TP3Config.from_environment({"FLARE_DS41_ENGRAM_TABLES": ""}).engram_tables == "auto"
    with pytest.raises(ValueError, match="engram_tables must be one of auto, shared, partitioned"):
        TP3Config.from_environment({"FLARE_DS41_ENGRAM_TABLES": "per-rank"})
    assert "engram_tables='shared'" in shared.describe() and "engram_tables" not in TP3Config().describe()


def test_sidecars_are_written_into_the_cache_entry_directory(tmp_path):
    from types import SimpleNamespace

    from ignition.preparation.model_core import _write_sidecars

    entry = tmp_path / "models" / "ds41" / "gb202-tp3-1"
    entry.mkdir(parents=True)
    model = SimpleNamespace(package_sidecars={"tp3-config.json": {"hot_experts": 342}})
    notes = []
    _write_sidecars(model, SimpleNamespace(entry_path=str(entry)), notes.append)
    assert (entry / "tp3-config.json").read_text() == '{\n  "hot_experts": 342\n}\n'
    assert not (entry.parent / "tp3-config.json").exists()  # a file there would read as a fake package version
    assert notes == [f"wrote {entry / 'tp3-config.json'}"]
    notes.clear()
    _write_sidecars(model, SimpleNamespace(entry_path=None), notes.append)  # no-cache export
    assert notes == ["no cache entry for this export; sidecars not written: tp3-config.json"]
    _write_sidecars(SimpleNamespace(), SimpleNamespace(entry_path=str(entry)), notes.append)  # recipes without sidecars
    assert len(notes) == 1


def test_planner_helpers_read_the_environment_through_the_config(monkeypatch, tmp_path):
    from ignition.recipes.deepseek41.recipe import _graph_bytes, _parked_bytes

    for key in RELEASE_ENV:
        monkeypatch.delenv(key, raising=False)
    assert _graph_bytes(3) == 8 << 30 and _parked_bytes(3) == 0
    assert _graph_bytes(3, RELEASE_CONFIG) == 1 << 30 and _graph_bytes(4, RELEASE_CONFIG) == 8 << 30
    local = -(-2304 // 3)
    per_expert = 2 * local * 5120 // 2 + 5120 * local // 2
    assert _parked_bytes(3, RELEASE_CONFIG) == 20 * (384 - 355) * per_expert
    for key, value in RELEASE_ENV.items():
        monkeypatch.setenv(key, value)
    assert _graph_bytes(3) == 1 << 30 and _parked_bytes(3) == _parked_bytes(3, RELEASE_CONFIG)


@pytest.mark.parametrize("changes", [
    dict(host_expert_layers=frozenset()),                                            # no parked layer at all
    dict(brrr_layers=frozenset(range(20))),                                           # no BRRR layer is parked
    dict(hot_experts=0, merged_bank=False),                                           # whole bank in host memory: no hot_map
    dict(hot_experts=384),                                                            # fully resident: nothing parked
])
def test_parked_row_gate_is_refused_where_it_would_gate_nothing(changes):
    assert RELEASE_CONFIG.brrr_parked_max_rows == 0  # the 20261116 release runs BRRR at every width on parked layers
    gated = replace(RELEASE_CONFIG, brrr_parked_max_rows=12)  # the 20261111 gate (BRRR 0-39, H342 merged) is accepted
    with pytest.raises(ValueError, match="brrr_parked_max_rows gates BRRR layers parked with a merged bank"):
        replace(gated, **changes)
    replace(gated, brrr_parked_max_rows=0, **changes)  # ungated: accepted


def test_layer_sets_are_checked_against_the_model():
    assert RELEASE_CONFIG.check_layers(40) is RELEASE_CONFIG
    with pytest.raises(ValueError, match="brrr_layers names layers 8-39 but the model has 8 layers"):
        RELEASE_CONFIG.check_layers(8)  # e.g. the release BRRR set on a mini8 view
    replace(RELEASE_CONFIG, brrr_layers=frozenset(range(8)), host_expert_layers=frozenset({3}),
            brrr_parked_max_rows=12).check_layers(8)
    TP3Config().check_layers(6)


LAYOUT_DEFAULTS = {"KS_WOB": 16, "WOA_SMALL": 16, "WOA_UNEVEN": 0, "PER_SM": 2}  # job-layout switches (deep dive 11-15)


def test_brrr_layout_switches_default_to_the_serving_build():
    """Deep dive 2026-11-15: the job-layout switches (wo_b slices per job, wo_a 4-stage row limit, uneven wo_a split,
    CTAs per SM) emit no define and no record key when unset or at the kernel default; a value is its own build."""
    from ignition.recipes.deepseek41.brrr import kernel_defines

    base = {"FLARE_DS41_BRRR_LAYERS": "0-39", "FLARE_DS41_BRRR_CKPT": "/c"}
    plain = TP3Config.from_environment(base)
    names = ("brrr_ks_wob", "brrr_woa_small", "brrr_woa_uneven", "brrr_per_sm")
    assert all(getattr(plain, name) is None for name in names) and kernel_defines(plain) == ()
    assert not set(names) & set(plain.to_metadata())
    explicit = replace(plain, brrr_ks_wob=16, brrr_woa_small=16, brrr_woa_uneven=0, brrr_per_sm=2)
    assert kernel_defines(explicit) == ()
    env = TP3Config.from_environment({**base, "FLARE_DS41_BRRR_KS_WOB": "24", "FLARE_DS41_BRRR_WOA_UNEVEN": "1",
                                      "FLARE_DS41_BRRR_WOA_SMALL": "0"})
    assert kernel_defines(env) == ("BRRR_KS_WOB=24", "BRRR_WOA_SMALL=0", "BRRR_WOA_UNEVEN=1")
    assert env.to_metadata()["brrr_ks_wob"] == 24 and env.to_metadata()["brrr_woa_uneven"] == 1
    assert kernel_defines(replace(plain, brrr_per_sm=1)) == ("BRRR_PER_SM=1",)
    for bad in (dict(brrr_ks_wob=20), dict(brrr_woa_small=8), dict(brrr_woa_uneven=2), dict(brrr_per_sm=3)):
        with pytest.raises(ValueError):
            replace(plain, **bad)
    with pytest.raises(ValueError):
        TP3Config(brrr_ks_wob=24)  # needs brrr_layers


def test_brrr_tuning_switches_select_a_separate_build_only_when_not_default():
    """ROUND2 #6: unset or the kernel default (SPIN_NS 200, TWO_SHOT_MIN 17) = no define = the plain serving build;
    any other value = defines (spin first, then two-shot) = its own extension name."""
    from ignition.recipes.deepseek41.brrr import _tuning, kernel_defines

    base = {"FLARE_DS41_BRRR_LAYERS": "0-39", "FLARE_DS41_BRRR_CKPT": "/c"}
    plain = TP3Config.from_environment(base)
    assert plain.brrr_spin_ns is None and plain.brrr_two_shot_min is None
    assert kernel_defines(plain) == ()
    # the release export pins the wo_a split (WOA_UNEVEN 1, WOA_SMALL 0): its own build, as package 160815 recorded
    assert kernel_defines(TP3Config.from_environment(RELEASE_ENV)) == ("BRRR_WOA_SMALL=0", "BRRR_WOA_UNEVEN=1")
    assert _tuning(plain) == {"SPIN_NS": 200, "TWO_SHOT_MIN": 17, **LAYOUT_DEFAULTS}
    explicit = TP3Config.from_environment({**base, "FLARE_DS41_BRRR_SPIN_NS": "200", "FLARE_DS41_BRRR_TWO_SHOT_MIN": "17"})
    assert explicit.brrr_spin_ns == 200 and kernel_defines(explicit) == ()
    empty = TP3Config.from_environment({**base, "FLARE_DS41_BRRR_SPIN_NS": "", "FLARE_DS41_BRRR_TWO_SHOT_MIN": ""})
    assert empty == plain
    both = TP3Config.from_environment({**base, "FLARE_DS41_BRRR_SPIN_NS": "0", "FLARE_DS41_BRRR_TWO_SHOT_MIN": "129"})
    assert kernel_defines(both) == ("BRRR_SPIN_NS=0", "BRRR_TWO_SHOT_MIN=129")
    assert kernel_defines(replace(plain, brrr_spin_trap=True, brrr_two_shot_min=9)) == ("BRRR_SPIN_TRAP=1", "BRRR_TWO_SHOT_MIN=9")
    # recorded only when set: a default export's record is the 151744 reference's, key for key
    assert both.to_metadata()["brrr_spin_ns"] == 0 and both.to_metadata()["brrr_two_shot_min"] == 129
    assert "brrr_spin_ns" not in plain.to_metadata() and "brrr_two_shot_min" not in plain.to_metadata()
    assert "brrr_spin_ns=0" in both.describe() and "brrr_spin_ns" not in plain.describe()
    with pytest.raises(ValueError):
        TP3Config.from_environment({**base, "FLARE_DS41_BRRR_SPIN_NS": "fast"})
    with pytest.raises(ValueError):
        replace(plain, brrr_spin_ns=True)  # a bool is not an integer value here


def test_brrr_check_switches_compares_the_tuning_values():
    from types import SimpleNamespace

    from ignition.recipes.deepseek41.brrr import check_switches

    config = TP3Config(brrr_layers=frozenset({3}), brrr_ckpt=Path("/c"), brrr_spin_ns=50)
    check_switches(SimpleNamespace(SPIN_TRAP=0, SPIN_NS=50, TWO_SHOT_MIN=17, **LAYOUT_DEFAULTS), config)
    for wrong in (dict(SPIN_NS=200, TWO_SHOT_MIN=17), dict(SPIN_NS=50, TWO_SHOT_MIN=9), dict(TWO_SHOT_MIN=17)):
        with pytest.raises(RuntimeError, match="do not match"):
            check_switches(SimpleNamespace(SPIN_TRAP=0, **{**LAYOUT_DEFAULTS, **wrong}), config)
    with pytest.raises(RuntimeError, match="do not match"):  # a build without the layout attributes (older extension)
        check_switches(SimpleNamespace(SPIN_TRAP=0, SPIN_NS=50, TWO_SHOT_MIN=17), config)
    with pytest.raises(RuntimeError, match="do not match"):
        check_switches(SimpleNamespace(SPIN_TRAP=0, SPIN_NS=50, TWO_SHOT_MIN=17, **{**LAYOUT_DEFAULTS, "KS_WOB": 24}), config)


def test_prefill_graph_subset_switch():
    """FLARE_DS41_PREFILL_GRAPHS (exp/ds41-exl3-final-bulk): unset = the full ladder and no record entry; set = that
    subset of the ladder, which must keep 128 and the chunk."""
    from ignition.recipes.deepseek41.bootstrap import PreparationConfig

    plain = TP3Config.from_environment({})
    assert plain.prefill_graphs is None and "prefill_graphs" not in plain.to_metadata()
    assert PreparationConfig(max_active=4, tp3=plain).prefill_capacities == (
        128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)
    kept = TP3Config.from_environment({"FLARE_DS41_PREFILL_GRAPHS": "128,256,512,1024,2048,3072,4096"})
    assert kept.prefill_graphs == (128, 256, 512, 1024, 2048, 3072, 4096)
    assert kept.to_metadata()["prefill_graphs"] == (128, 256, 512, 1024, 2048, 3072, 4096)
    assert PreparationConfig(max_active=4, tp3=kept).prefill_capacities == (128, 256, 512, 1024, 2048, 3072, 4096)
    for bad in ("256,4096", "128,128,4096", "4096,128", "128,100,4096", "x"):
        with pytest.raises(ValueError, match="PREFILL_GRAPHS|prefill_graphs"):
            TP3Config.from_environment({"FLARE_DS41_PREFILL_GRAPHS": bad})
    for off_ladder in ("128,768,4096", "128,2048"):   # not on the ladder / does not end at the chunk
        config = TP3Config.from_environment({"FLARE_DS41_PREFILL_GRAPHS": off_ladder})
        with pytest.raises(ValueError, match="PREFILL_GRAPHS"):
            PreparationConfig(max_active=4, tp3=config)
