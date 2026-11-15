# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the b12x overlay patch b12x-mla-decode-remainder-stream.

At TP3 a DS41 rank holds 24 query heads. BF16 MLA decode (the recipe default) tiles 16 heads per CTA, so b12x launches a
16-head grid and an 8-head remainder grid that serialize in the graph. With B12X_MLA_DECODE_REMAINDER_STREAM=1 the
patched b12x forks the remainder grid onto a side stream and joins it before the split merge. These tests check the
patch against the installed b12x, the switch, and the launch/fork/join order with fake streams. The GPU proof (same
bytes, concurrency in the captured graph) is in the attn1 test plan.
"""

import ast
import contextlib
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "tp3-patches" / "overlay" / "b12x-mla-decode-remainder-stream.patch"
ENV = "B12X_MLA_DECODE_REMAINDER_STREAM"

pytestmark = pytest.mark.skipif(shutil.which("patch") is None, reason="needs the patch tool")


def _installed_kernel():
    from b12x.attention._shared.mla import kernel
    return kernel


def _patch(args, target_dir):
    return subprocess.run(["patch", "-d", str(target_dir), "-p4", "--batch", *args], stdin=PATCH.open("rb"),
                          capture_output=True)


@pytest.fixture(scope="module")
def patched_source(tmp_path_factory):
    """kernel.py with the patch: the installed file when an overlay is active, else a patched private copy."""
    installed = Path(_installed_kernel().__file__)
    text = installed.read_text()
    if "_launch_head_blocks" in text:
        return text
    tree = tmp_path_factory.mktemp("b12x_tree")
    target = tree / "attention" / "_shared" / "mla"
    target.mkdir(parents=True)
    shutil.copy(installed, target / "kernel.py")
    result = _patch(["--forward"], tree)
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    return (target / "kernel.py").read_text()


@pytest.fixture(scope="module")
def kernel(patched_source):
    """The patch's helpers, compiled from the patched source without importing the module (importing a second copy
    would register b12x's decode custom op twice)."""
    import torch

    wanted = {"_MLA_SM120_DECODE_REMAINDER_STREAM_ENV", "_env_decode_remainder_stream", "_launch_head_blocks"}
    tree = ast.parse(patched_source)
    body = [node for node in tree.body
            if (isinstance(node, ast.FunctionDef) and node.name in wanted)
            or (isinstance(node, ast.Assign) and any(getattr(t, "id", None) in wanted for t in node.targets))]
    assert len(body) == len(wanted)
    namespace = {"os": os, "torch": torch}
    exec(compile(ast.Module(body=body, type_ignores=[]), "kernel.py", "exec"), namespace)
    return types.SimpleNamespace(**{name: namespace[name] for name in wanted}, source=patched_source)


def test_patch_matches_installed_b12x(tmp_path):
    installed = Path(_installed_kernel().__file__)
    target = tmp_path / "attention" / "_shared" / "mla"
    target.mkdir(parents=True)
    shutil.copy(installed, target / "kernel.py")
    # Installed copy already patched (overlay on PYTHONPATH) -> the patch must reverse cleanly; else apply cleanly.
    mode = "--reverse" if "_launch_head_blocks" in installed.read_text() else "--forward"
    result = _patch([mode, "--dry-run"], tmp_path)
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    assert b"FAILED" not in result.stdout and b"offset" not in result.stdout


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("", False), ("off", False),
                                             ("1", True), ("true", True), (" ON ", True), ("yes", True)])
def test_switch_defaults_off(kernel, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert kernel._env_decode_remainder_stream() is expected


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(f"{self.name}.wait({other.name})")


def _launchers(log, current):
    return (lambda: log.append(f"full@{current[0]}"), lambda: log.append(f"remainder@{current[0]}"))


@pytest.fixture
def streams(monkeypatch):
    """Fake torch.cuda.stream: records the current stream name while a launcher runs."""
    import torch

    log, current = [], ["main"]

    @contextlib.contextmanager
    def fake_stream(stream):
        previous, current[0] = current[0], stream.name
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.cuda, "stream", fake_stream)
    return log, current, _Stream("main", log), _Stream("side", log)


def test_serial_without_side_stream(kernel, streams):
    log, current, main, _ = streams
    full, remainder = _launchers(log, current)
    kernel._launch_head_blocks(full, remainder, side_stream=None, current_stream=main)
    assert log == ["full@main", "remainder@main"]


@pytest.mark.parametrize("which", ["full_only", "remainder_only"])
def test_single_grid_never_forks(kernel, streams, which):
    log, current, main, side = streams
    full, remainder = _launchers(log, current)
    if which == "full_only":
        kernel._launch_head_blocks(full, None, side_stream=side, current_stream=main)
        assert log == ["full@main"]
    else:
        kernel._launch_head_blocks(None, remainder, side_stream=side, current_stream=main)
        assert log == ["remainder@main"]


def test_concurrent_fork_launch_join_order(kernel, streams):
    log, current, main, side = streams
    full, remainder = _launchers(log, current)
    kernel._launch_head_blocks(full, remainder, side_stream=side, current_stream=main)
    # Fork before either grid, full grid on the caller's stream, remainder on the side stream, join before return
    # (the split merge is enqueued after the join). Call order full -> remainder keeps prepared program indices.
    assert log == ["side.wait(main)", "full@main", "remainder@side", "main.wait(side)"]
    assert current == ["main"]


def test_call_site_gates_concurrency(kernel):
    tree = ast.parse(kernel.source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_unified_decode")
    source = ast.get_source_segment(kernel.source, function)
    call = source[source.index("concurrent = bool("):source.index("from .merge import")]
    for term in ("h_blocks_full > 0", "rem_heads", "q_all.is_cuda", "not compile_only_launches_enabled()",
                 "_env_decode_remainder_stream()"):
        assert term in call
    assert call.index("_launch_grid(h_blocks_full, hpb, 0)") < call.index("_launch_grid(1, rem_heads, h_blocks_full)")
    assert "side_stream=_decode_remainder_stream(q_all.device) if concurrent else None" in call


def test_overlay_script_applies_selected_patch(tmp_path):
    out = tmp_path / "overlay"
    env = dict(os.environ, VENV=sys.prefix, B12X_OVERLAY_PATCHES=str(PATCH))
    result = subprocess.run(["bash", str(ROOT / "tp3-tools" / "make-b12x-overlay.sh"), str(out)], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    patched = (out / "b12x_tree" / "attention" / "_shared" / "mla" / "kernel.py").read_text()
    assert "_launch_head_blocks" in patched and ENV in patched
    # Only the selected patch: the file the GLM pull-gather patch edits is the venv's copy.
    gather = Path("comm") / "pcie" / "_dcp_a2a_cute.py"
    venv_tree = Path(_installed_kernel().__file__).parents[3]
    if (venv_tree / gather).exists() and "_launch_head_blocks" not in Path(_installed_kernel().__file__).read_text():
        assert (out / "b12x_tree" / gather).read_bytes() == (venv_tree / gather).read_bytes()
    assert "b12x_tree" in (out / "b12x" / "__init__.py").read_text()
