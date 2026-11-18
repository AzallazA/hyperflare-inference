# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the one-launch 24-head DSV41 decode (ignition/components/mla_h24_decode.py + mla_h24_kernel.py,
FLARE_DS41_MLA_H24).

The hook wraps b12x's module-level grid launcher: for a selected 24-head DSV41 BF16 fused-mapping decode the 16-head
grid call launches our H24 kernel and the remainder call becomes a launcher that launches nothing. These tests check
the policy parser, the routing (selected / not selected / prepared replay / composition with the remainder-stream
hook), the b12x contracts the hook relies on, the shared-memory layout and traits, the TP3 record, that spawned b12x
compile workers install the hook, and (offline, fake sm_120 target, no CUDA) that the kernel lowers and compiles with
the expected program set, no spills, two MMAs per decoded fragment and one CTA per SM. GPU proof:
tests/components/test_mla_h24_decode_cuda.py (bit identity vs 16 + 8 at equal split counts) and
tp3-tools/attn_split_bench.py --h24.
"""

import inspect
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from ignition.components import mla_h24_decode as h24
from tests.recipes import test_deepseek41_mla_remainder_stream as rs_test

ROOT = Path(__file__).resolve().parents[2]
CUOBJDUMP = "/usr/local/cuda-13.3/bin/cuobjdump"
NVDISASM = "/usr/local/cuda-13.3/bin/nvdisasm"


def test_parse_policy():
    assert h24.parse_policy(None) == () and h24.parse_policy(" ") == ()
    assert h24.parse_policy("9:0, 80:1") == ((9, False), (80, True))
    assert h24.format_policy(h24.parse_policy(" 9:0,37:1,80:1 ")) == "9:0,37:1,80:1"
    for bad in ("9", "9:2", "0:1", "18:1,9:1", "9:1,9:0", "a:1", "9:0", "9:0,80:0", "9:yes"):
        with pytest.raises(ValueError):
            h24.parse_policy(bad)


def test_selected_first_bound_wins_and_off_above():
    policy = h24.parse_policy("9:0,37:1,40:0,80:1")
    assert [h24.selected(policy, rows) for rows in (1, 9, 10, 37, 38, 40, 41, 80, 81)] == [
        False, False, True, True, False, False, True, True, False]


# ---------------------------------------------------------------------------------------------------------------
# Routing with a fake b12x launcher (signature twin of kernel._sparse_mla_decode_grid_flat_launch)
# ---------------------------------------------------------------------------------------------------------------
class _Tensor:
    def __init__(self, shape, pointer):
        self.shape = shape
        self.pointer = pointer
        self.is_cuda = True
        self.device = types.SimpleNamespace(index=0)

    def data_ptr(self):
        return self.pointer


class _Program:
    def __init__(self, name):
        self.__b12x_programs__ = (types.SimpleNamespace(name=name, key="k"),)


def _args(q, mid_out, blocks, valid, offset, *, model_type=3, compute_mode=1, scale_format=2, has_extra=True,
          per_token_len=True, topk=128, extra_topk=512, splits=10):
    a = [None] * 32
    a[0], a[3] = q, mid_out
    a[11], a[12], a[13], a[14] = model_type, compute_mode, scale_format, False
    a[15], a[16], a[17], a[18], a[19], a[20] = 128, topk, extra_topk, 2, splits, 1
    a[24], a[25], a[26], a[27], a[28] = blocks, valid, offset, has_extra, per_token_len
    a[9], a[10] = 512 ** -.5, 1.0
    a[29], a[30], a[31] = False, False, 16
    return tuple(a)


@pytest.fixture
def routing():
    log = []

    def inner(*args, **kwargs):
        log.append(("inner", args[26], kwargs.get("_prepared")))
        return "inner-launcher"

    def real(*args, **kwargs):
        log.append(("real", args[26], kwargs.get("_prepared")))
        return "real-launcher"

    real.__signature__ = inspect.signature(rs_test.flat_launch)
    inner.__signature__ = inspect.signature(rs_test.flat_launch)

    def launcher(q_all, *rest):
        log.append(("h24", int(q_all.shape[0]), rest[14]))  # rows, num_splits
        return _Program(h24.KERNEL_ID)

    module = types.SimpleNamespace(_sparse_mla_decode_grid_flat_launch=inner, _flare_remainder_stream_original=real)
    wrapper = h24.install(h24.parse_policy("9:0,80:1"), module, launcher=launcher)
    return module, wrapper, log


def _decode(wrapper, rows, *, heads=24, prepared=(None, None), **kw):
    q, mid = _Tensor((rows, heads, 512), 1000 + rows), _Tensor((rows, heads, 10, 512), 5000 + rows)
    first = wrapper(*_args(q, mid, 1, 16, 0, **kw), _prepared=prepared[0])
    second = wrapper(*_args(q, mid, 1, heads - 16, 1, **kw), _prepared=prepared[1]) if heads > 16 else None
    return first, second


def test_selected_rows_launch_h24_and_skip_the_remainder(routing):
    module, wrapper, log = routing
    assert module._sparse_mla_decode_grid_flat_launch is wrapper and h24.installed_policy(module) == ((9, False), (80, True))
    first, second = _decode(wrapper, 13, splits=5)
    assert h24._is_h24_program(first) and second is h24.REMAINDER_DONE
    assert log == [("h24", 13, 5)] and wrapper.applied == [(13, 5)] and not wrapper.pending


def test_unselected_rows_and_other_decodes_use_b12x(routing):
    _, wrapper, log = routing
    for rows, kw in ((5, {}), (81, {}), (13, dict(compute_mode=0)), (13, dict(model_type=0)),
                     (13, dict(per_token_len=False)), (13, dict(has_extra=False)), (13, dict(scale_format=1))):
        log.clear()
        assert _decode(wrapper, rows, **kw) == ("inner-launcher", "inner-launcher")
        assert [entry[0] for entry in log] == ["inner", "inner"], (rows, kw)
    log.clear()
    _decode(wrapper, 13, heads=16)  # a single grid: nothing to fuse
    assert log == [("inner", 0, None)]
    assert wrapper.applied == []


def test_prepared_replay_runs_the_program_through_b12x_and_skips_the_remainder(routing):
    _, wrapper, log = routing
    program = _Program(h24.KERNEL_ID)
    assert _decode(wrapper, 13, prepared=(program, h24.REMAINDER_DONE)) == ("real-launcher", None)
    assert log == [("real", 0, program)]
    # A prepared 16 + 8 pair (rows not selected, or selected only now) still goes through b12x's own path.
    log.clear()
    _decode(wrapper, 13, prepared=(_Program("attention.mla.sm120.decode"), _Program("attention.mla.sm120.decode")))
    assert [entry[0] for entry in log] == ["inner", "inner"] and wrapper.applied == []
    log.clear()
    _decode(wrapper, 5, prepared=(_Program("attention.mla.sm120.decode"), _Program("attention.mla.sm120.decode")))
    assert [entry[0] for entry in log] == ["inner", "inner"]


def test_remainder_without_its_h24_launch_is_refused(routing):
    _, wrapper, _ = routing
    q, mid = _Tensor((13, 24, 512), 1), _Tensor((13, 24, 10, 512), 2)
    with pytest.raises(RuntimeError, match="unexpected grid call"):
        wrapper(*_args(q, mid, 1, 8, 1))


def test_remainder_done_is_an_inert_launcher():
    from b12x._lib.compile_plan import attach_programs, load_programs, program_keys

    assert program_keys(h24.REMAINDER_DONE) == () and h24.REMAINDER_DONE(1, 2, x=3) is None
    carrier = attach_programs(lambda: None, h24.REMAINDER_DONE)
    assert carrier.__b12x_programs__ == ()
    load_programs((h24.REMAINDER_DONE,))  # b12x's executable loader accepts it (explicit, empty dependencies)


def test_install_composes_with_the_remainder_stream_hook_and_uninstall_restores():
    from ignition.components import mla_remainder_stream as rs

    def original(*args, **kwargs):
        return "b12x"

    original.__signature__ = inspect.signature(rs_test.flat_launch)
    module = types.SimpleNamespace(_sparse_mla_decode_grid_flat_launch=original)
    stream = rs.install(module)
    first = h24.install(((80, True),), module, launcher=lambda *a: None)
    assert first.inner is stream and first.real is original
    second = h24.install(((40, True),), module, launcher=lambda *a: None)  # replaces, never stacks
    assert second.inner is stream and module._sparse_mla_decode_grid_flat_launch is second
    h24.uninstall(module)
    assert module._sparse_mla_decode_grid_flat_launch is stream and h24.installed_policy(module) == ()


def test_unset_environment_installs_nothing(monkeypatch):
    monkeypatch.delenv(h24.ENV, raising=False)
    assert h24.install_from_environment() is None


# ---------------------------------------------------------------------------------------------------------------
# b12x contracts
# ---------------------------------------------------------------------------------------------------------------
def test_b12x_contracts():
    from b12x.attention._shared.mla import kernel

    original = getattr(kernel, "_flare_remainder_stream_original", None) or getattr(kernel, h24._ORIGINAL, None) \
        or kernel._sparse_mla_decode_grid_flat_launch
    assert list(inspect.signature(original).parameters) == list(inspect.signature(rs_test.flat_launch).parameters)
    run = inspect.getsource(kernel.run_unified_decode)
    # Full grid then remainder, the prepared program picked by head offset, the launcher looked up at call time.
    assert run.index("_launch_grid(h_blocks_full, hpb, 0)") < run.index("_launch_grid(1, rem_heads, h_blocks_full)")
    assert "program_index = 1 if head_block_offset else 0" in run
    assert "launched = _sparse_mla_decode_grid_flat_launch(" in run
    # compile_unified_decode_launch counts one launcher per grid call (REMAINDER_DONE fills the remainder slot).
    assert "grid_count = int(prepared.h_blocks_full > 0) + int(prepared.rem_heads > 0)" in \
        inspect.getsource(kernel.compile_unified_decode_launch)
    flat = inspect.getsource(original)
    # A prepared program is run with b12x's own runtime tuple, which our entry point mirrors for the fused mapping.
    assert "if _prepared is not None:\n        return run_compiled(_prepared, args)" in flat
    assert "args[:-2]" in flat and "Int64(indexed_num_pages)," in flat
    entry = inspect.signature(kernel.UnifiedDecodeKernel.call_mapped_extra_pertok)
    from ignition.components.mla_h24_kernel import H24DecodeKernel
    ours = inspect.signature(H24DecodeKernel.call_mapped_extra_pertok)
    assert len(entry.parameters) == len(ours.parameters) == 21


def test_traits_and_shared_memory_layout():
    from b12x.attention._shared.mla.smem import SM120_SMEM_CARVEOUT_BYTES
    from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits
    from ignition.components.mla_h24_kernel import storage_cls

    h24.check_traits(make_unified_traits(ModelType.DSV41, ComputeMode.BF16, ScaleFormat.NVFP4_E4M3, fp8_rope=False))
    with pytest.raises(ValueError):
        h24.check_traits(make_unified_traits(ModelType.DSV4, ComputeMode.BF16, ScaleFormat.UE8M0_BYTE))
    regions, total = h24.smem_offsets()
    storage = storage_cls()
    assert int(storage.size_in_bytes()) == total == 99712
    assert total + 1024 <= SM120_SMEM_CARVEOUT_BYTES  # plus the DSL's 1 KB static reservation
    spans = sorted((offset, offset + size) for _, offset, size in regions)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))  # no overlap


def test_source_identity_covers_both_files():
    own, base = h24.source_identity()
    assert len(own) == len(base) == 64 and own != base


def test_tp3_config_records_the_selection_and_checks_the_installed_hook():
    from ignition.recipes.deepseek41.tp3_config import TP3Config

    assert TP3Config().mla_h24 is None and "mla_h24" not in TP3Config().to_metadata()
    config = TP3Config.from_environment({h24.ENV: " 9:0, 80:1"})
    assert config.mla_h24 == "9:0,80:1" and config.to_metadata()["mla_h24"] == "9:0,80:1"
    assert TP3Config.from_environment({h24.ENV: ""}).mla_h24 is None
    config.check_mla_h24(installed=((9, False), (80, True)))
    TP3Config().check_mla_h24(installed=())
    with pytest.raises(ValueError, match="24-head decode selection"):
        config.check_mla_h24(installed=())
    with pytest.raises(ValueError, match="24-head decode selection"):
        TP3Config().check_mla_h24(installed=((80, True),))
    with pytest.raises(ValueError):
        TP3Config.from_environment({h24.ENV: "9:0"})


def test_b12x_fingerprints_do_not_move_with_the_switch():
    plain = rs_test._fingerprints({"FP_PLAIN": "1"})
    on = rs_test._fingerprints({h24.ENV: "9:0,80:1"})
    for key in ("unified_decode", "sink_merge", "factory", "modules"):
        assert plain[key] == on[key], key


def _worker_probe(_):
    from b12x.attention._shared.mla import kernel

    return type(kernel._sparse_mla_decode_grid_flat_launch).__name__


_WORKER = r"""
import multiprocessing as mp
from tests.recipes.test_deepseek41_mla_decode_splits import _stub_worker_initializer
from tests.recipes.test_deepseek41_mla_h24_decode import _worker_probe
if __name__ == "__main__":
    with mp.get_context("spawn").Pool(1, initializer=_stub_worker_initializer) as workers:
        print("WORKER", workers.map(_worker_probe, [0])[0])
"""


@pytest.mark.parametrize("policy,expected", [("9:0,80:1", "_H24Launch"), ("", "function")])
def test_spawned_b12x_compile_worker_installs_the_hook(policy, expected):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT), h24.ENV: policy}
    result = subprocess.run([sys.executable, "-c", _WORKER], cwd=ROOT, env=env, capture_output=True, text=True,
                            timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    assert f"WORKER {expected}" in result.stdout, result.stdout + result.stderr[-2000:]


# ---------------------------------------------------------------------------------------------------------------
# Offline compile (fake sm_120 device, as a b12x compile worker; no CUDA)
# ---------------------------------------------------------------------------------------------------------------
def _offline_run(rows, splits, policy):
    """Spawned b12x-worker-like process: plan + compile the decode job offline; programs and hook record."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import multiprocessing, torch
    from b12x._lib import compile_pool
    compile_pool._initialize_worker(0, (12, 0), "GPU-offline", "offline", 188, 101376, 102400,
                                    multiprocessing.get_context("spawn").Array("q", (0, 0)))
    from b12x._lib.compile_plan import observe_programs, serialized_compilations, program_keys
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention.compressed_sparse_mla import _preparation as prep
    from b12x.attention.compressed_sparse_mla._tuning import SparseMlaConfig
    from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference
    from ignition.components import mla_decode_splits, mla_h24_decode, mla_remainder_stream
    os.environ[mla_remainder_stream.ENV] = "1"
    mla_remainder_stream.install()
    mla_decode_splits.install(((rows, splits),))
    hook = mla_h24_decode.install(mla_h24_decode.parse_policy(policy)) if policy else None
    values = torch.zeros((128, 512), dtype=torch.bfloat16)
    swa_w = int(pack_deepseek_v41_cache_reference(values, page_size=128, cache_kind="swa").shape[1])
    idx_w = int(pack_deepseek_v41_cache_reference(values[:64], page_size=64, cache_kind="indexed").shape[1])
    d = lambda shape, dtype: dict(shape=tuple(shape), stride=tuple(torch.empty(shape, dtype=torch.uint8).stride()),
                                  alignment=64, dtype=dtype)
    invocation = prep.invocation_from_descriptors(q=d((rows, 24, 512), "bfloat16"),
        swa_cache=d((2 * rows + 2, swa_w), "uint8"), indexed_cache=d((65536, idx_w), "uint8"),
        attn_sink_present=True, output_mode="provided")
    caps = mla.Caps(device=torch.device("cuda", 0), num_q_heads=24, max_q_rows=rows, max_width=640, swa_width=128,
                    indexed_width=512, max_page_table_width=2048, swa_page_size=128, indexed_page_size=64,
                    cache_format="deepseek_v41", mode="decode", decode_row_capacity=rows, use_cuda_graph=True,
                    compute_mode="bf16")
    plan = mla.plan(caps, invocation=invocation)
    config = SparseMlaConfig(max_chunks_per_row=54, split_chunk_size=12, single_pass=False,
                             v41_compute_mode="bf16", v41_heads_per_block=16)
    (job,) = plan._compile_jobs(config, None)
    with observe_programs(), serialized_compilations():
        keys = program_keys(job.invoke())
    return dict(programs=sorted({k.name for k in keys}), keys=sorted({k.key for k in keys}),
                applied=list(hook.applied) if hook else None, cuda=torch.cuda.is_initialized())


_COMPILE = r"""
import json, multiprocessing, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tests.recipes.test_deepseek41_mla_h24_decode import _offline_run
if __name__ == "__main__":
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        print("RESULT", json.dumps(pool.apply(_offline_run, (int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]))))
"""


def _offline_compile(tmp_path, rows, splits, policy):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT),
           "B12X_COMPILE_CACHE_DIR": str(tmp_path / "b12x"), "TRITON_CACHE_DIR": str(tmp_path / "triton")}
    env.pop(h24.ENV, None)
    result = subprocess.run([sys.executable, "-c", _COMPILE, str(rows), str(splits), policy], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-4000:]
    import json
    line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


def _device_elfs(path):
    import struct
    data = path.read_bytes()
    start, out = 1, []
    while (i := data.find(b"\x7fELF", start)) >= 0:
        start = i + 4
        if struct.unpack_from("<H", data, i + 18)[0] != 190:
            continue
        shoff, = struct.unpack_from("<Q", data, i + 0x28)
        phoff, = struct.unpack_from("<Q", data, i + 0x20)
        phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHH", data, i + 0x36)
        end = max(shoff + shnum * shentsize, phoff + phnum * phentsize)
        for k in range(shnum):
            o = i + shoff + k * shentsize
            if struct.unpack_from("<I", data, o + 4)[0] != 8:
                off, size = struct.unpack_from("<QQ", data, o + 0x18)
                end = max(end, off + size)
        out.append(data[i:i + end])
    return out


@pytest.mark.parametrize("rows,splits", [(13, 10), (32, 5), (80, 2)])
def test_offline_compile_programs_registers_and_mma_sharing(tmp_path, rows, splits):
    on = _offline_compile(tmp_path, rows, splits, f"{rows}:1")
    assert not on["cuda"] and on["applied"] == [[rows, splits]]
    assert on["programs"] == ["attention.mla.sink_merge", h24.KERNEL_ID]  # no 16-head or remainder decode program
    off = _offline_compile(tmp_path, rows, splits, "")
    assert off["programs"] == ["attention.mla.sink_merge", "attention.mla.sm120.decode"] and len(off["keys"]) == 3
    assert set(off["keys"]) & set(on["keys"])  # the merge program is shared (same split count)
    if not (Path(CUOBJDUMP).exists() and Path(NVDISASM).exists()):
        pytest.skip("CUDA 13.3 cuobjdump/nvdisasm not installed: resource checks skipped")
    stats = {}
    for obj in (tmp_path / "b12x").rglob("*.o"):
        for n, blob in enumerate(_device_elfs(obj)):
            cubin = tmp_path / f"{obj.stem[:16]}-{n}.cubin"
            cubin.write_bytes(blob)
            usage = subprocess.run([CUOBJDUMP, "--dump-resource-usage", str(cubin)], capture_output=True,
                                   text=True).stdout
            sass = subprocess.run([NVDISASM, "-c", str(cubin)], capture_output=True, text=True).stdout
            name = "h24" if "H24DecodeKernel" in usage else ("merge" if "Merge" in usage else "b12x")
            regs = int(re.search(r"REG:(\d+)", usage).group(1))
            stack = int(re.search(r"STACK:(\d+)", usage).group(1))
            local = int(re.search(r"LOCAL:(\d+)", usage).group(1))
            hmma = len(re.findall(r"\bHMMA\.", sass))
            lds = len(re.findall(r"\bLDS(?:\.\w+)*\s", sass))
            stats.setdefault(name, []).append(dict(regs=regs, stack=stack, local=local, hmma=hmma, lds=lds))
    (h,) = stats["h24"]
    remainder, full = sorted(stats["b12x"], key=lambda s: s["regs"])  # 16-head program 168 regs, remainder 150
    assert h["stack"] == h["local"] == 0 and h["regs"] <= 224  # no spills; 288 threads fit one CTA per SM
    assert h["hmma"] == full["hmma"] + remainder["hmma"]          # both head tiles' MMAs
    assert h["lds"] < 1.15 * max(full["lds"], remainder["lds"])    # K/V decode loads once, not twice
    shutil.rmtree(tmp_path / "b12x", ignore_errors=True)
