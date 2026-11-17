# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the debug-serve route counters (routecount.py) and the export script's debug-serve switch."""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
import torch

from ignition.recipes.deepseek41 import routecount

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tp3-tools" / "ds41-exl3-export-tp2.sh"
PACK = Path(os.environ.get("FLARE_TEST_DS41_EXL3_PACK") or "/nonexistent-exl3-pack")  # the EXL3 pack directory


@pytest.fixture
def counts_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(routecount.ENV, str(tmp_path))
    routecount._STATE.clear()
    routecount.MISMATCHES.clear()
    yield tmp_path
    routecount._STATE.clear()
    routecount.MISMATCHES.clear()


def test_off_is_a_no_op(monkeypatch):
    monkeypatch.delenv(routecount.ENV, raising=False)
    routecount._STATE.clear()
    counter = routecount.RouteCounter()
    counter.allocate("cpu", 384)
    counter.record(torch.zeros((4, 6), dtype=torch.int64), torch.zeros(4, dtype=torch.int64), 3, "decode", 384)
    assert not counter.enabled and not routecount._STATE


def test_live_rows_modes_and_snapshot(counts_dir):
    counter = routecount.RouteCounter()
    counter.allocate("cpu", 384)
    ids = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 7, 8, 9, 10, 383], [1, 1, 1, 1, 1, 1]], dtype=torch.int32)
    positions = torch.tensor([5, 6, -1])                       # third row = graph padding
    counter.record(ids, positions, 2, "prefill", 384)
    counter.record(ids, positions, 2, "verify", 384)
    counter.record(ids, positions, 2, "decode", 384)
    counter.record(ids, positions, 2, "draft", 384)              # DSpark layers are not counted
    counter.record(ids, positions, 41, "decode", 384)            # outside the target stack: ignored
    table = routecount._STATE[None]
    assert table[2].tolist()[:3] == [2, 1, 1] and table[2, 383] == 1 and table[2].sum() == 12
    assert table[routecount.LAYERS + 2].sum() == 24 and table[routecount.LAYERS + 2, 0] == 4
    assert table.sum() == 36                                     # nothing else counted (no padding, no draft)
    (counts_dir / "request-t1").touch()
    procs = []
    for _ in range(50):
        procs = [p for p in counts_dir.iterdir() if p.is_dir() and (p / "done-t1").exists()]
        if procs:
            break
        time.sleep(0.1)
    assert len(procs) == 1 and procs[0].name.startswith("dev")
    snap = json.loads((procs[0] / "snap-t1.json").read_text())
    assert snap["tag"] == "t1" and snap["experts"] == 384 and snap["unmasked"] == []
    assert snap["groups"]["prefill"][2][:3] == [2, 1, 1] and sum(snap["groups"]["step"][2]) == 24


def test_row_mismatch_counts_unmasked_and_is_reported(counts_dir):
    counter = routecount.RouteCounter()
    counter.allocate("cpu", 384)
    counter.record(torch.zeros((2, 6), dtype=torch.int64), torch.tensor([-1, -1, -1]), 0, "decode", 384)
    assert routecount._STATE[None][routecount.LAYERS + 0, 0] == 12
    assert routecount.MISMATCHES == {"layer 0 decode: 2 rows vs 3 positions"}


def _script(args, **env):
    full = {k: v for k, v in os.environ.items() if not k.startswith(("FLARE_DS41_", "DEBUG_SERVE", "HELIOPAUSE_PATH"))}
    full.update(env, CKPT=str(PACK), CUDA_VISIBLE_DEVICES="")
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=full, timeout=300)


@pytest.mark.skipif(not (PACK / "config.json").is_file(), reason="EXL3 pack not present")
def test_export_refuses_route_counts():
    out = _script(["--dry-run"], FLARE_DS41_ROUTE_COUNTS="/tmp/x")
    assert out.returncode == 2 and "never export with it" in out.stderr


@pytest.mark.skipif(not (PACK / "config.json").is_file(), reason="EXL3 pack not present")
def test_debug_serve_argv_matches_export_argv():
    export = _script(["--dry-run"], PARK="2-4,6-39", HOT="320")
    serve = _script(["--dry-run"], PARK="2-4,6-39", HOT="320", DEBUG_SERVE="1", PORT="15015",
                    FLARE_DS41_ROUTE_COUNTS="/tmp/x", HELIOPAUSE_PATH="/nonexistent-qual")
    assert export.returncode == 0 and serve.returncode == 0, (export.stderr, serve.stderr)
    a = next(line for line in export.stdout.splitlines() if line.startswith("export argv:")).split()
    b = next(line for line in serve.stdout.splitlines() if line.startswith("export argv:")).split()
    assert a[3] == "export" and b[3] == "debug-serve"
    assert b[:3] + b[4:-4] == a[:3] + a[4:] and b[-4:] == ["--host", "127.0.0.1", "--port", "15015"]
    env_a = [line for line in export.stdout.splitlines() if line.startswith(("FLARE_DS41_", "FLARE_TUNE_", "B12X_"))]
    env_b = [line for line in serve.stdout.splitlines() if line.startswith(("FLARE_DS41_", "FLARE_TUNE_", "B12X_"))]
    assert [line for line in env_b if not line.startswith("FLARE_DS41_ROUTE_COUNTS=")] == env_a
