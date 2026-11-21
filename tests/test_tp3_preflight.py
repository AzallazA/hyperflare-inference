# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tp3-tools/release/check_env.sh"
RELEASE_TAG = "hyperflare-tp3-20261121.1"
SOURCE_PATCH = "--- a/release.txt\n+++ b/release.txt\n@@ -1 +1 @@\n-upstream\n+fork\n"


def git(checkout, *args):
    return subprocess.check_output(["git", "-C", checkout, *args], text=True).strip()


def copy_script(directory):
    directory.mkdir(parents=True)
    return Path(shutil.copy2(SCRIPT, directory))


@pytest.fixture
def checkout(tmp_path):
    checkout = tmp_path / "upstream"
    checkout.mkdir()
    git(checkout, "init", "-q")
    git(checkout, "config", "user.name", "TP3 Test")
    git(checkout, "config", "user.email", "test@example.test")
    git(checkout, "config", "commit.gpgsign", "false")
    (checkout / "release.txt").write_text("upstream\n")
    git(checkout, "add", "release.txt")
    git(checkout, "commit", "-qm", "Upstream base")
    return checkout


@pytest.fixture
def run_preflight(tmp_path):
    commands = tmp_path / "bin"
    commands.mkdir()

    def stub(name, body):
        command = commands / name
        command.write_text(f"#!/bin/sh\n{body}\n")
        command.chmod(0o755)

    stub("nvcc", 'case "$1" in --version) echo "release 13.1";; *) echo sm_120a;; esac')
    for tool, version in (("rustc", "1.88"), ("cargo", "1.88"), ("uv", "0.11")):
        stub(tool, f'echo "{tool} {version}"')
    stub("nvidia-smi", '''case "$1" in
        --query-gpu=driver_version) echo 615.71.09;;
        *) for i in $VISIBLE_GPUS; do echo "$i, Blackwell, 12.0, 0, 450, 3000, 2000"; done;;
    esac''')
    bash_env = tmp_path / "bash_env"
    bash_env.write_text("ulimit() { echo unlimited; }\n")

    def run(script, checkout, base, **settings):
        return subprocess.run(
            [script, checkout],
            cwd=tmp_path,
            env={
                "PATH": f"{commands}:{os.environ['PATH']}",
                "BASH_ENV": str(bash_env),
                "CUDA_HOME": str(commands.parent),
                "FLARE_CACHE": str(tmp_path),
                "MIN_FREE_GB": "0",
                "BASE": base,
                "VISIBLE_GPUS": "0 1 2 3",
                **settings,
            },
            capture_output=True,
            text=True,
            check=False,
        )

    return run


def test_fork_preflight_accepts_tagged_shallow_checkout(checkout, tmp_path, run_preflight):
    base = git(checkout, "rev-parse", "HEAD")
    script = copy_script(checkout / "tp3-tools/release")
    patch = checkout / "tp3-patches/flashinfer-tuning-world3.patch"
    patch.parent.mkdir()
    patch.touch()
    git(checkout, "add", "tp3-tools", "tp3-patches")
    git(checkout, "commit", "-qm", "TP3 release")
    git(checkout, "tag", "-am", "TP3 release", RELEASE_TAG)

    result = run_preflight(script, checkout, base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "contains upstream base" in result.stdout

    shallow = tmp_path / "shallow"
    git(checkout, "clone", "-q", "--depth=1", "--branch", RELEASE_TAG, checkout.as_uri(), shallow)
    script = shallow / "tp3-tools/release/check_env.sh"
    result = run_preflight(script, shallow, base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "has a release tag at HEAD" in result.stdout

    git(shallow, "tag", "-d", RELEASE_TAG)
    result = run_preflight(script, shallow, base)
    assert result.returncode == 1
    assert "cannot verify TP3 fork checkout" in result.stdout


def test_fork_preflight_requires_committed_release_files(checkout, run_preflight):
    base = git(checkout, "rev-parse", "HEAD")
    script = copy_script(checkout / "tp3-tools/release")
    result = run_preflight(script, checkout, base)
    assert result.returncode == 1
    assert "lacks the committed TP3 release tools or FlashInfer patch" in result.stdout


def test_recipe_preflight_checks_adjacent_patch(checkout, tmp_path, run_preflight):
    base = git(checkout, "rev-parse", "HEAD")
    script = copy_script(tmp_path / "recipe")
    result = run_preflight(script, checkout, base)
    assert result.returncode == 1
    assert "recipe source patch missing or unreadable" in result.stdout

    patch = script.with_name("source.patch")
    patch.write_text(SOURCE_PATCH)
    result = run_preflight(script, checkout, base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "WARN  source.patch applies cleanly" in result.stdout
    assert (checkout / "release.txt").read_text() == "upstream\n"

    git(checkout, "apply", patch)
    result = run_preflight(script, checkout, base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS  source.patch applied" in result.stdout

    (checkout / "release.txt").write_text("local change\n")
    result = run_preflight(script, checkout, base)
    assert result.returncode == 1
    assert "source.patch neither applied nor applicable" in result.stdout


@pytest.mark.parametrize(
    ("gpus", "visible", "error"),
    [
        ("2,0,3", "0 1 2 3", None),
        ("0,0,1", "0 1 2 3", "exactly 3 distinct"),
        ("0,1,2,9", "0 1 2 3", "exactly 3 distinct"),
        ("0,1,2,", "0 1 2 3", "exactly 3 distinct"),
        ("0,,2", "0 1 2 3", "exactly 3 distinct"),
        ("0,1,3", "0 0 1", "2 of the 3 GPUs"),
    ],
)
def test_preflight_requires_three_distinct_visible_indices(checkout, tmp_path, run_preflight, gpus, visible, error):
    base = git(checkout, "rev-parse", "HEAD")
    script = copy_script(tmp_path / "recipe")
    script.with_name("source.patch").write_text(SOURCE_PATCH)
    result = run_preflight(script, checkout, base, GPUS=gpus, VISIBLE_GPUS=visible)
    if error:
        assert result.returncode == 1
        assert error in result.stdout
        assert "visible and idle" not in result.stdout
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"PASS  3 SM120 GPUs ({gpus}) visible and idle" in result.stdout
