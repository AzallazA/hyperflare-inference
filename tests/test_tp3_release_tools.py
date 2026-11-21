# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import struct
import subprocess

import pytest


TOOLS = Path(__file__).resolve().parents[1] / "tp3-tools"


def run_script(script, cwd, *args, **settings):
    return subprocess.run(
        [TOOLS / script, *args],
        cwd=cwd,
        env={"PATH": os.environ["PATH"], **settings},
        capture_output=True,
        text=True,
    )


def make_snapshot(root):
    root.mkdir()
    weight_map = {}
    for layer, number in ((1, 47), (14, 48)):
        shard = f"model-{number:05d}-of-00048.safetensors"
        header = {}
        for field, dtype, width, start in (("weight", "F8_E4M3", 256, 0), ("scale", "F8_E8M0", 8, 256)):
            key = f"layers.{layer}.engram.embed.{field}"
            header[key] = {"dtype": dtype, "shape": [1, width], "data_offsets": [start, start + width]}
            weight_map[key] = shard
        encoded = json.dumps(header).encode()
        (root / shard).write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(264))
    index = root / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": weight_map, "metadata": {"total_size": 528}}))
    return index


def test_checkpoint_view_publishes_only_after_validation(tmp_path):
    index = make_snapshot(tmp_path / "snapshot")
    script = "release/make_checkpoint_view.sh"
    output = tmp_path / "releases" / "test view"
    result = run_script(script, tmp_path, "snapshot", "releases/test view")
    assert result.returncode == 0, result.stderr
    assert (output / index.name).readlink() == index
    assert "missing 0" in result.stdout

    index.write_text("{}")
    result = run_script(script, tmp_path, "snapshot", "releases/invalid")
    assert result.returncode != 0
    assert list(output.parent.iterdir()) == [output]

    broken = output.parent / "broken"
    broken.symlink_to("missing")
    result = run_script(script, tmp_path, "snapshot", "releases/broken//")
    assert result.returncode != 0
    assert broken.readlink() == Path("missing")


def test_checkpoint_view_omits_nested_output_ancestry(tmp_path):
    index = make_snapshot(tmp_path / "snapshot")
    snapshot = index.parent
    archive = snapshot / "archive"
    archive.mkdir()
    original = archive / "original.txt"
    original.write_text("keep")
    (snapshot / "archive alias").symlink_to(archive)
    (tmp_path / "snapshot alias").symlink_to(snapshot)

    script = "release/make_checkpoint_view.sh"
    result = run_script(script, tmp_path, "snapshot", "snapshot alias/archive/view")
    assert result.returncode == 0, result.stderr
    view = archive / "view"
    assert (view / index.name).readlink() == index
    assert not (view / "archive").is_symlink()
    assert not (view / "archive alias").is_symlink()
    assert original.read_text() == "keep"

    external = tmp_path / "external"
    external.mkdir()
    (archive / "external").symlink_to(external)
    result = run_script(script, tmp_path, "snapshot", "snapshot alias/archive/external/view")
    assert result.returncode == 0, result.stderr
    external_view = external / "view"
    assert (external_view / index.name).readlink() == index
    assert not (external_view / "archive").is_symlink()
    assert not (external_view / "archive alias").is_symlink()

    fresh_view = snapshot / "new parent" / "nested" / "view"
    result = run_script(script, tmp_path, "snapshot", fresh_view)
    assert result.returncode == 0, result.stderr
    assert (fresh_view / index.name).readlink() == index
    assert not (fresh_view / "new parent").is_symlink()
    assert not list(snapshot.rglob(".checkpoint-view.*"))
    assert original.read_text() == "keep"


def make_overlay_source(tmp_path):
    purelib = tmp_path / "site-packages"
    tree = purelib / "flashinfer" / "experimental" / "b12x"
    tree.mkdir(parents=True)
    (tree / "example.py").write_text("original\n")
    shim = purelib / "b12x" / "__init__.py"
    shim.parent.mkdir()
    shim.write_text('_root = Path(_flashinfer.origin).parent / "experimental" / "b12x"\n')
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/bin/sh\nprintf "%s\\n" "$PURELIB"\n')
    python.chmod(0o755)
    return {"VENV": str(python.parents[1]), "PURELIB": str(purelib)}


def test_overlay_replacement_preserves_destination_on_failure(tmp_path):
    settings = make_overlay_source(tmp_path)
    patch = tmp_path / "example.patch"
    patch.write_text(
        "--- a/flashinfer/experimental/b12x/example.py\n"
        "+++ b/flashinfer/experimental/b12x/example.py\n"
        "@@ -1 +1 @@\n-original\n+patched\n"
    )
    script = "make-b12x-overlay.sh"
    output = tmp_path / "overlay"
    result = run_script(script, tmp_path, "overlay", B12X_OVERLAY_PATCHES=patch.name, **settings)
    assert result.returncode == 0, result.stderr
    copied = output / "b12x_tree" / "example.py"
    assert copied.read_text() == "patched\n"
    (output / "b12x" / "__pycache__").mkdir()

    patch.write_text(patch.read_text().replace("-original", "-wrong"))
    result = run_script(script, tmp_path, "overlay", B12X_OVERLAY_PATCHES=patch.name, **settings)
    assert result.returncode != 0
    assert copied.read_text() == "patched\n"
    result = run_script(script, tmp_path, "absent", B12X_OVERLAY_PATCHES=patch.name, **settings)
    assert result.returncode != 0
    assert not (tmp_path / "absent").exists()
    assert not list(tmp_path.glob(".b12x-overlay.*"))

    result = run_script(script, tmp_path, "overlay", B12X_OVERLAY_PATCHES="", **settings)
    assert result.returncode == 0, result.stderr
    assert copied.read_text() == "original\n"
    unrelated = output / "notes"
    unrelated.write_text("keep\n")
    link = tmp_path / "link"
    link.symlink_to(output)
    for dest in ("overlay", "link//"):
        result = run_script(script, tmp_path, dest, B12X_OVERLAY_PATCHES="", **settings)
        assert result.returncode != 0
    assert unrelated.read_text() == "keep\n"
    assert link.is_symlink()


def make_export_command(tmp_path):
    command = tmp_path / ".venv" / "bin" / "ignition"
    command.parent.mkdir(parents=True)
    command.write_text("#!/usr/bin/python3\nimport json, os, sys\nprint(json.dumps([sys.argv[1:], dict(os.environ)]))\n")
    command.chmod(0o755)


def run_export(tmp_path, **settings):
    return run_script("release/export.sh", tmp_path, CKPT="checkpoint view", **settings)


def test_export_checks_both_overlay_capabilities(tmp_path):
    make_export_command(tmp_path)
    overlay = tmp_path / "overlay"
    comm = overlay / "b12x_tree" / "comm" / "pcie"
    comm.mkdir(parents=True)
    shim = overlay / "b12x" / "__init__.py"
    shim.parent.mkdir()
    shim.touch()
    settings = {"B12X_OVERLAY": str(overlay)}
    result = run_export(tmp_path, **settings)
    assert result.returncode != 0 and "flashinfer-pcie-twoshot-pull-world3.patch" in result.stderr
    (comm / "_collective_tuning.py").write_text("def twoshot_pull_enabled(): ...\n")
    result = run_export(tmp_path, **settings)
    assert result.returncode != 0 and "flashinfer-pcie-twoshot-pull-world3.patch" in result.stderr
    (comm / "pcie_allreduce.py").write_text("if twoshot_pull_enabled(): ...\n")
    result = run_export(tmp_path, **settings)
    assert result.returncode != 0 and "flashinfer-dcp-gather-pull-world3.patch" in result.stderr
    (comm / "_dcp_a2a_cute.py").write_text(
        "def _gather_pull_enabled(world_size): ...\ngather_pull = _gather_pull_enabled(world_size)\n"
    )
    result = run_export(tmp_path, **settings, PYTHONPATH="other")
    assert result.returncode == 0, result.stderr
    argv, env = json.loads(result.stdout)
    assert argv[:2] == ["export", "checkpoint view"]
    assert env["B12X_PCIE_TWOSHOT_PULL"] == "1"
    assert env["B12X_PCIE_DCP_GATHER_PULL"] == "auto"
    assert env["PYTHONPATH"] == f"{overlay}:other"


@pytest.mark.parametrize("index_split", ["", "0"])
def test_export_explicit_switches_override_inherited_settings(tmp_path, index_split):
    make_export_command(tmp_path)
    settings = {
        "TWOSHOT": "0", "GATHER_PULL": "0", "B12X_PCIE_DCP_GATHER_PULL": "auto",
        "INDEX_SIDE": "0", "FLARE_DS41_INDEX_SIDE_STREAM": "1",
        "MLA_SPLITS": "", "FLARE_DS41_MLA_DECODE_SPLITS": "18:10",
        "ENGRAM_SIDE": "", "FLARE_DS41_ENGRAM_SIDE_LAYERS": "8",
        "IDX_SPLIT": index_split, "FLARE_DS41_INDEX_ROW_SPLIT": "4096",
        "FLARE_DS41_INDEX_ROW_SPLIT_MIN_CTX": "32768", "FLARE_DS41_INDEX_ROW_SPLIT_VARIANTS": "prefill",
        "STEERING": "sidecar with spaces.safetensors", "STEERING_PLAIN_ID": "0",
        "FLARE_DS41_STEERING_PLAIN_IDENTITY": "1",
        "NO_GEN_LOGPROBS": "0", "FLARE_DS41_NO_GENERATION_LOGPROBS": "1",
        "RECLAIM": "0", "FLARE_DS41_SHARE_VOCAB_COLUMNS": "1",
        "MARKOV_HOST": "0", "MARKOV_STAGE": "0", "FLARE_DS41_MARKOV_STAGE_ROWS": "1",
        "FLARE_EXPORT_LIBRARY_SCRATCH": "0", "FLARE_DS41_MARKOV_TILE_ROWS": "4",
    }
    result = run_export(tmp_path, **settings)
    assert result.returncode == 0, result.stderr
    argv, env = json.loads(result.stdout)
    assert argv[-2:] == ["--steering-sidecar", settings["STEERING"]]
    assert env["FLARE_DS41_INDEX_ROW_SPLIT"] == index_split
    assert "FLARE_DS41_INDEX_ROW_SPLIT_MIN_CTX" not in env
    assert "FLARE_DS41_INDEX_ROW_SPLIT_VARIANTS" not in env
    for name in (
        "B12X_PCIE_DCP_GATHER_PULL", "FLARE_DS41_INDEX_SIDE_STREAM",
        "FLARE_DS41_STEERING_PLAIN_IDENTITY", "FLARE_DS41_NO_GENERATION_LOGPROBS",
        "FLARE_DS41_SHARE_VOCAB_COLUMNS", "FLARE_DS41_MARKOV_STAGE_ROWS", "FLARE_EXPORT_LIBRARY_SCRATCH",
    ):
        assert env[name] == "0"
    assert env["FLARE_DS41_MARKOV_TILE_ROWS"] == "4"
    assert env["FLARE_DS41_MLA_DECODE_SPLITS"] == env["FLARE_DS41_ENGRAM_SIDE_LAYERS"] == ""


@pytest.mark.parametrize("mode", ["1", "layer1", "0"])
def test_serve_resolves_storage_and_explicit_off_switches(tmp_path, mode):
    probe = tmp_path / "flare"
    probe.write_text("#!/usr/bin/python3\nimport json, os, sys\nprint(json.dumps([sys.argv[1:], dict(os.environ)]))\n")
    probe.chmod(0o755)
    inherited = {
        "FLARE_ENGRAM_RAM_TABLES": "0", "FLARE_ENGRAM_RAM_LAYERS": "99",
        "FLARE_ENGRAM_PRIVATE_TABLES": "1", "FLARE_FAST_EXIT": "1",
        "FLARE_CUDA_STACK_LIMIT": "256", "FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS": "512",
        "FLARE_ENGRAM_READ_AHEAD": "1",
    }
    result = run_script("release/serve.sh", tmp_path, FLARE=str(probe), ENGRAM_RAM=mode,
                        FAST="0", STACK_LIMIT="", PREFILL_MERGE="", READ_AHEAD="", **inherited)
    assert result.returncode == 0, result.stderr
    argv, env = json.loads(result.stdout)
    selectors = {"1": {"FLARE_ENGRAM_RAM_TABLES": "0"}, "layer1": {"FLARE_ENGRAM_RAM_LAYERS": "0"}, "0": {}}
    assert {key: env[key] for key in inherited if key in env} == selectors[mode]
    if mode == "1":
        assert "--embedding-table-cache-size" not in argv
    else:
        assert argv[argv.index("--embedding-table-storage") + 1] == "disk"
        assert argv[argv.index("--embedding-table-cache-size") + 1] == "32GiB"
