# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DS41 TP3 E3c-a exact levers on CPU (perf/ds41-e3c-a = G3 24ef125f + F + d, 2026-11-19): F = dcp_gather_pull (record
of b12x's B12X_PCIE_DCP_GATHER_PULL, overlay patch flashinfer-dcp-gather-pull-world3), d = rotary_once_prefill (prefill
graphs read the shared RoPE table, inverse rotation included). Switches default off with an unchanged default record,
parsing / record rules, validation, the b12x record check (skipped without the patched overlay), the mode rule, and
export.sh's knobs resolved through a copy of export.sh whose exec line prints the environment. Rotation identity is
tests/components/test_rotary_shared_angles_interpret.py (prefill shapes + inverse); GPU identity: seq358 M2 / M4."""

import os
import subprocess

import pytest

from ignition.recipes.deepseek41 import decoder as decoder_module
from ignition.recipes.deepseek41.tp3_config import E3C_A, TP3Config
from tests.recipes.test_deepseek41_tp3_config import EXPORT_SH


def test_switches_default_off_and_default_record_unchanged():
    record = TP3Config().to_metadata()
    for name in E3C_A:
        assert name not in record
    assert TP3Config.from_environment({}).to_metadata() == record
    # export.sh's default with an overlay (gather pull forced 0) records nothing either
    assert TP3Config.from_environment({"B12X_PCIE_DCP_GATHER_PULL": "0"}).to_metadata() == record


def test_switches_parse_and_record():
    on = TP3Config.from_environment({"B12X_PCIE_DCP_GATHER_PULL": "auto", "FLARE_DS41_ROTARY_ONCE_PREFILL": "1"})
    record = on.to_metadata()
    assert record["rotary_once_prefill"] is True and record["dcp_gather_pull"] is True
    assert set(record) - set(TP3Config().to_metadata()) == {"rotary_once_prefill", "dcp_gather_pull"}


@pytest.mark.parametrize("value,expected", [("", False), ("0", False), ("off", False), ("push", False),
                                            ("auto", True), ("1", True), ("pull", True), ("ON", True)])
def test_gather_pull_record_rule(value, expected):
    assert TP3Config.gather_pull_from_environment({"B12X_PCIE_DCP_GATHER_PULL": value}) is expected


@pytest.mark.parametrize("name", E3C_A)
def test_switches_must_be_booleans(name):
    with pytest.raises(TypeError, match="must be a boolean"):
        TP3Config(**{name: 1})


def test_b12x_record_must_match_the_imported_overlay(monkeypatch):
    dcp = pytest.importorskip("b12x.comm.pcie._dcp_a2a_cute")
    if not hasattr(dcp, "_gather_pull_enabled"):
        with pytest.raises(ValueError, match="needs the b12x overlay patch"):
            TP3Config(dcp_gather_pull=True).check_b12x_switches(world=3)
        TP3Config().check_b12x_switches(world=3)
        return
    monkeypatch.setenv("B12X_PCIE_DCP_GATHER_PULL", "auto")
    TP3Config(dcp_gather_pull=True).check_b12x_switches(world=3)
    with pytest.raises(ValueError, match="dcp_gather_pull differs"):
        TP3Config().check_b12x_switches(world=3)                 # b12x pulls, the record says push
    monkeypatch.setenv("B12X_PCIE_DCP_GATHER_PULL", "0")
    TP3Config().check_b12x_switches(world=3)
    with pytest.raises(ValueError, match="dcp_gather_pull differs"):
        TP3Config(dcp_gather_pull=True).check_b12x_switches(world=3)
    TP3Config(dcp_gather_pull=True).check_b12x_switches(world=1)  # single GPU: no DCP gather


@pytest.mark.parametrize("once,prefill,expected", [
    (False, False, set()), (True, False, {"decode", "verify"}), (False, True, {"prefill"}),
    (True, True, {"decode", "verify", "prefill"})])
def test_shared_rotary_modes(once, prefill, expected):
    tp3 = TP3Config(rotary_once=once, rotary_once_prefill=prefill)
    assert {mode for mode in ("prefill", "decode", "verify", "draft")
            if decoder_module.shared_rotary_angles(tp3, mode)} == expected


def _resolve(tmp_path, overlay_patched, **knobs):
    """export.sh with its exec line replaced: returns (rc, environment lines)."""
    copy = tmp_path / "export-copy.sh"
    text = EXPORT_SH.read_text()
    assert text.count("exec .venv/bin/ignition export ") == 1
    copy.write_text(text.replace("exec .venv/bin/ignition export ", "/usr/bin/env | sort; exit 0; : "))
    overlay = tmp_path / "overlay"
    (overlay / "b12x_tree/comm/pcie").mkdir(parents=True)
    (overlay / "b12x_tree/comm/pcie/_dcp_a2a_cute.py").write_text(
        "def _gather_pull_enabled(world):\n    return True\n" if overlay_patched else "x = 1\n")
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "B12X_OVERLAY": str(overlay),
           "CKPT": str(tmp_path), **knobs}
    run = subprocess.run(["bash", str(copy)], env=env, capture_output=True, text=True, cwd=tmp_path, timeout=60)
    return run.returncode, run.stdout.splitlines(), run.stderr


def test_export_sh_resolves_the_e3c_a_knobs(tmp_path):
    rc, lines, err = _resolve(tmp_path, True, GATHER_PULL="1", ROTARY_PREFILL="1")
    assert rc == 0, err
    assert "B12X_PCIE_DCP_GATHER_PULL=auto" in lines and "FLARE_DS41_ROTARY_ONCE_PREFILL=1" in lines
    (tmp_path / "base").mkdir()
    rc, lines, err = _resolve(tmp_path / "base", True)   # since 20261121 both are export.sh defaults
    assert rc == 0, err
    assert "B12X_PCIE_DCP_GATHER_PULL=auto" in lines and "FLARE_DS41_ROTARY_ONCE_PREFILL=1" in lines
    (tmp_path / "off").mkdir()
    rc, lines, err = _resolve(tmp_path / "off", True, GATHER_PULL="0", ROTARY_PREFILL="0")
    assert rc == 0, err
    assert "B12X_PCIE_DCP_GATHER_PULL=0" in lines
    assert not any(line.startswith("FLARE_DS41_ROTARY_ONCE_PREFILL=") for line in lines)


def test_export_sh_refuses_gather_pull_without_the_patch(tmp_path):
    rc, _, err = _resolve(tmp_path, False, GATHER_PULL="1")
    assert rc != 0 and "flashinfer-dcp-gather-pull-world3" in err
