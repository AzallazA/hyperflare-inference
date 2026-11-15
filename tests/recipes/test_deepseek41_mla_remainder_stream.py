# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for ignition/components/mla_remainder_stream.py (B12X_MLA_DECODE_REMAINDER_STREAM).

At TP3 a DS41 rank holds 24 query heads; BF16 MLA decode runs a 16-head grid and an 8-head remainder grid that
serialize in the graph. The switch wraps b12x's module-level grid launcher at runtime so the full-block grid runs on
a forked side stream and the remainder on the caller's stream, joined before the split merge. These tests check the
launch/fork/join order with fake streams, the default-off path, the installation, and that no b12x source fingerprint
moves (b12x keys compiled programs and tuning races by source fingerprints). GPU proof: the attn1 test plan.
"""

import contextlib
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from ignition.components import mla_remainder_stream as rs


ENV = rs.ENV
ROOT = Path(__file__).resolve().parents[2]


def flat_launch(q_all, kv_flat, swa_indices, mid_out, mid_lse, swa_len_t, extra_kv_flat, extra_indices_t,
                extra_len_t, sm_scale, latent_scale, model_type, compute_mode, scale_format, fp8_rope,
                swa_page_size, topk, extra_topk, num_main_chunks, num_splits, chunks_per_split, stride_kv_block,
                pbs_extra, stride_extra_kv_block, grid_h_blocks, valid_hpb, head_block_offset, has_extra,
                per_token_len, latent_scale_per_token=False, v41_fp8_internal=False, v41_heads_per_block=16,
                native_glm_h8_override=None, native_dsv4_h8_override=None, native_dsv4_h16_override=None,
                native_dsv41_fp8_override=None, vector_q_override=None, indexed_page_table=None,
                indexed_num_pages=0, _prepared=None):
    """Signature twin of b12x kernel._sparse_mla_decode_grid_flat_launch (checked against it below)."""
    LOG.append(f"grid(off={head_block_offset},valid={valid_hpb})@{CURRENT[0]}")
    return "launcher"


LOG, CURRENT = [], ["main"]


class _Stream:
    def __init__(self, name):
        self.name = name

    def wait_stream(self, other):
        LOG.append(f"{self.name}.wait({other.name})")


class _Query:
    def __init__(self, heads, cuda=True):
        self.shape = (6, heads, 512)
        self.is_cuda = cuda
        self.device = types.SimpleNamespace(index=0)


def _args(q, grid_h_blocks, valid_hpb, offset):
    return (q,) + (None,) * 23 + (grid_h_blocks, valid_hpb, offset, True, True)


@pytest.fixture
def fake(monkeypatch):
    LOG.clear()
    CURRENT[0] = "main"

    @contextlib.contextmanager
    def stream(s):
        previous, CURRENT[0] = CURRENT[0], s.name
        try:
            yield
        finally:
            CURRENT[0] = previous

    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _Stream("main"))
    compile_only = [False]
    wrapper = rs._RemainderStreamLaunch(flat_launch, lambda: compile_only[0], stream_factory=lambda index: _Stream("side"))
    return wrapper, compile_only


def _decode(wrapper, heads=24, cuda=True):
    q = _Query(heads, cuda)
    full, rem = heads // 16, heads % 16
    if full:
        assert wrapper(*_args(q, full, 16, 0), _prepared="p0") == "launcher"
    if rem:
        assert wrapper(*_args(q, 1, rem, full), _prepared="p1") == "launcher"


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("", False), ("off", False),
                                             ("1", True), ("true", True), (" ON ", True), ("yes", True)])
def test_switch_defaults_off(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    assert rs.enabled() is expected


def test_switch_off_is_serial_passthrough(fake, monkeypatch):
    wrapper, _ = fake
    monkeypatch.delenv(ENV, raising=False)
    _decode(wrapper)
    assert LOG == ["grid(off=0,valid=16)@main", "grid(off=1,valid=8)@main"]
    assert not wrapper.pending


def test_switch_on_forks_full_grid_and_joins_after_remainder(fake, monkeypatch):
    wrapper, _ = fake
    monkeypatch.setenv(ENV, "1")
    _decode(wrapper)
    # Fork before the full grid (side stream), remainder on the caller's stream, join before b12x's merge.
    assert LOG == ["side.wait(main)", "grid(off=0,valid=16)@side", "grid(off=1,valid=8)@main", "main.wait(side)"]
    assert not wrapper.pending and CURRENT == ["main"]
    LOG.clear()
    _decode(wrapper)  # every layer forks and joins again
    assert LOG == ["side.wait(main)", "grid(off=0,valid=16)@side", "grid(off=1,valid=8)@main", "main.wait(side)"]


@pytest.mark.parametrize("heads", [16, 32, 8])
def test_single_grid_never_forks(fake, monkeypatch, heads):
    wrapper, _ = fake
    monkeypatch.setenv(ENV, "1")
    _decode(wrapper, heads=heads)
    assert all("@main" in entry for entry in LOG) and not any("wait" in entry for entry in LOG)


@pytest.mark.parametrize("case", ["compile_only", "cpu"])
def test_no_fork_when_compiling_or_on_cpu(fake, monkeypatch, case):
    wrapper, compile_only = fake
    monkeypatch.setenv(ENV, "1")
    compile_only[0] = case == "compile_only"
    _decode(wrapper, cuda=case != "cpu")
    assert LOG == ["grid(off=0,valid=16)@main", "grid(off=1,valid=8)@main"]


def test_signature_matches_b12x_launcher():
    import inspect
    from b12x.attention._shared.mla import kernel

    original = getattr(kernel, "_flare_remainder_stream_original", kernel._sparse_mla_decode_grid_flat_launch)
    assert list(inspect.signature(original).parameters) == list(inspect.signature(flat_launch).parameters)
    source = inspect.getsource(kernel.run_unified_decode)
    # The hook relies on: full-block launch first (offset 0), remainder second (offset = full blocks), merge after,
    # and the launcher looked up as a module global at call time.
    full = source.index("_launch_grid(h_blocks_full, hpb, 0)")
    remainder = source.index("_launch_grid(1, rem_heads, h_blocks_full)")
    merge = source.index("run_sparse_mla_split_decode_merge(")
    assert full < remainder < merge
    assert "launched = _sparse_mla_decode_grid_flat_launch(" in source


def test_install_is_idempotent_and_reversible():
    kernel = types.SimpleNamespace(_sparse_mla_decode_grid_flat_launch=flat_launch)
    wrapper = rs.install(kernel)
    assert isinstance(kernel._sparse_mla_decode_grid_flat_launch, rs._RemainderStreamLaunch)
    assert rs.install(kernel) is wrapper and wrapper.original is flat_launch
    rs.uninstall(kernel)
    assert kernel._sparse_mla_decode_grid_flat_launch is flat_launch
    assert not hasattr(kernel, "_flare_remainder_stream_original")


def test_install_refuses_superseded_overlay():
    kernel = types.SimpleNamespace(_sparse_mla_decode_grid_flat_launch=flat_launch, _launch_head_blocks=object())
    with pytest.raises(RuntimeError, match="superseded"):
        rs.install(kernel)


_FINGERPRINTS = r"""
import json, os, sys
if os.environ.get("FP_PLAIN") != "1":
    import ignition.components.compressed_mla  # installs the hook when the switch is on
import b12x.attention.compressed_sparse_mla
from b12x.attention._shared.mla import kernel, merge
from b12x._lib.compiler import _target_source_fingerprint
from b12x.preparation._discovery import _factory_sources
from b12x._lib.source_dependencies import source_dependencies, SourceDependencies
from pathlib import Path
root = Path(kernel.__file__).resolve().parents[3]
fresh = SourceDependencies(root)
modules = {}
for path in sorted(root.rglob("*.py")):
    rel = path.relative_to(root).with_suffix("").parts
    name = "b12x" + "".join("." + part for part in (rel[:-1] if rel[-1] == "__init__" else rel))
    if fresh.module_path(name) != path:
        continue
    try:
        closure = fresh.closure(name, path)
    except ValueError:
        continue
    if any(m == "b12x.attention._shared.mla.kernel" for m, _ in closure):
        modules[name] = fresh.fingerprint(((name, path),))
print(json.dumps(dict(
    installed=type(kernel._sparse_mla_decode_grid_flat_launch).__name__,
    unified_decode=_target_source_fingerprint(kernel.UnifiedDecodeKernel),
    sink_merge=_target_source_fingerprint(merge.SparseMLASplitDecodeSinkMergeKernel),
    factory=_factory_sources(("b12x.attention.compressed_sparse_mla._preparation:compile_compressed_sparse_mla",)),
    modules=modules)))
"""


def _fingerprints(env):
    result = subprocess.run([sys.executable, "-c", _FINGERPRINTS], cwd=ROOT, capture_output=True, text=True,
                            env={**{k: v for k, v in __import__("os").environ.items() if k != ENV}, **env,
                                 "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)})
    assert result.returncode == 0, result.stderr[-3000:]
    import json
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_b12x_fingerprints_do_not_move_with_the_switch():
    plain, off, on = _fingerprints({"FP_PLAIN": "1"}), _fingerprints({}), _fingerprints({ENV: "1"})
    assert plain["installed"] == off["installed"] == "function" and on["installed"] == "_RemainderStreamLaunch"
    assert len(plain["modules"]) >= 32
    for key in ("unified_decode", "sink_merge", "factory", "modules"):
        assert plain[key] == off[key] == on[key], key
