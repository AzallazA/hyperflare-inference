"""CPU test: the H24 program key is the same in a planning process and in a compile worker that already compiled.

E2 export seq348 failed with "factory program keys changed between planning and compilation" for the H24 program:
b12x fingerprints a compiled function outside its own tree by ``inspect.getsource``, and CuTe's first real compile of a
``@cute.jit`` method rewrites that function's ``__code__``, so a worker that had compiled one H24 program keyed every
later one differently from the planning process (components/mla_h24_decode.py: _StableFingerprint).

Both processes here are spawned offline b12x compile workers initialised by ignition's real ``initialize_b12x_worker``
with the E2 environment (FLARE_DS41_MLA_DECODE_SPLITS + FLARE_DS41_MLA_H24). One plans every case; the other first
compiles one H24 job for real through b12x's own worker routine (``compile_pool._run_job``, which checks the planned
keys), then keys every case on the launch path."""

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ignition.components import mla_decode_splits, mla_h24_decode  # noqa: E402

SPLITS = "18:10,37:5,47:4,62:3,94:2,96:3"
H24 = "9:0,96:1"
ROWS = (9, 10, 13, 18, 23, 32, 37, 47, 62, 80, 94, 96)
# (rows, swa_page_size, indexed_page_size, max_page_table_width, use_cuda_graph): the decode graphs' shape plus the
# E2 seq348 verification shape (rows 96, pages 134/128, page table 8192, no CUDA graph).
SHAPES = tuple((rows, 128, 64, 2048, True) for rows in ROWS) + ((96, 134, 128, 8192, False),)
FIRST = (13, 128, 64, 2048, True)


def _plan(shape):
    import torch
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention.compressed_sparse_mla import _preparation as prep, _tuning as tn
    from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference
    rows, swa_page, idx_page, table_width, graph = shape
    values = torch.zeros((max(swa_page, idx_page), 512), dtype=torch.bfloat16)
    swa_w = int(pack_deepseek_v41_cache_reference(values[:swa_page], page_size=swa_page, cache_kind="swa").shape[1])
    idx_w = int(pack_deepseek_v41_cache_reference(values[:idx_page], page_size=idx_page, cache_kind="indexed").shape[1])
    d = lambda s: dict(shape=tuple(s), stride=tuple(torch.empty(s, dtype=torch.uint8).stride()), alignment=16,
                       dtype="uint8")
    invocation = prep.invocation_from_descriptors(
        q=dict(shape=(rows, 24, 512), stride=(12288, 512, 1), alignment=16, dtype="bfloat16"),
        swa_cache=d((2 * rows + 2, swa_w)), indexed_cache=d((4096, idx_w)), attn_sink_present=True,
        output_mode="provided")
    caps = mla.Caps(device=torch.device("cuda", 1), num_q_heads=24, max_q_rows=rows, max_width=640, swa_width=128,
                    indexed_width=512, max_page_table_width=table_width, swa_page_size=swa_page,
                    indexed_page_size=idx_page, cache_format="deepseek_v41", mode="decode", decode_row_capacity=rows,
                    use_cuda_graph=graph, compute_mode="bf16")
    plan = mla.plan(caps, invocation=invocation)
    geometries = sorted({(s.num_chunks, s.chunk_size) for limit in (1, 2, 4, 8, 16, 32, 64, 128, 256)
                         for s in (tn._split_config(plan.query, limit),)})
    return plan, [(f"{n}x{c}", tn.SparseMlaConfig(max_chunks_per_row=n, split_chunk_size=c, single_pass=False,
                                                  v41_compute_mode="bf16", v41_heads_per_block=16))
                  for n, c in geometries]


def _cases():
    for shape in SHAPES:
        plan, configs = _plan(shape)
        for geometry, config in configs:
            (job,) = plan._compile_jobs(config, None)
            yield ",".join(map(str, shape)) + ":" + geometry, job


def _worker(phase):
    """Runs inside a spawned b12x compile worker initialised exactly like the export's (ignition's initializer)."""
    import multiprocessing
    import pickle
    from ignition.runtime_patches.triton import initialize_b12x_worker
    initialize_b12x_worker(1, (12, 0), "GPU-offline", "offline", 188, 101376, 102400,
                           multiprocessing.get_context("spawn").Array("q", (0, 0)))
    from b12x._lib import compile_pool, compiler
    from b12x._lib.compile_plan import observe_programs, program_keys
    cute = lambda programs: sorted({(p.name, p.key) for p in programs if p.dialect == "cute"})
    if phase == "plan":
        return {name: cute(compile_pool.describe_compilation(job).programs) for name, job in _cases()}
    out = {}
    plan, configs = _plan(FIRST)
    (job,) = plan._compile_jobs(dict(configs)["54x12"], None)
    expected = tuple(compile_pool.describe_compilation(job).programs)
    from b12x._lib.compile_plan import evict_planning_artifacts
    evict_planning_artifacts(expected)
    compile_pool._run_job(pickle.dumps(job, protocol=5), expected)  # real compile; raises on a key change
    out["compiled_first"] = cute(expected)
    saved = compiler._call_cute_compile, compiler._store_cute_compile_to_disk
    compiler._call_cute_compile = lambda *a, **k: types.SimpleNamespace()  # key only: compile-only launches never run
    compiler._store_cute_compile_to_disk = lambda *a, **k: None
    try:
        for name, job in _cases():
            with observe_programs() as seen:
                program_keys(job.invoke())
            out[name] = cute(seen)
    finally:
        compiler._call_cute_compile, compiler._store_cute_compile_to_disk = saved
    return out


_SCRIPT = r"""
import json, multiprocessing, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tests.recipes.test_deepseek41_mla_h24_key_stability import _worker
if __name__ == "__main__":
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        print("RESULT", json.dumps(pool.apply(_worker, (sys.argv[1],))))
"""


def _spawn(tmp_path, phase):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT),
           mla_decode_splits.ENV: SPLITS, mla_h24_decode.ENV: H24,
           "B12X_COMPILE_CACHE_DIR": str(tmp_path / f"b12x-{phase}"), "TRITON_CACHE_DIR": str(tmp_path / "triton")}
    result = subprocess.run([sys.executable, "-c", _SCRIPT, phase], cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=2400)
    assert result.returncode == 0, result.stderr[-4000:]
    line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


@pytest.mark.skipif(os.environ.get("FLARE_SKIP_OFFLINE_COMPILE") == "1", reason="offline compile disabled")
def test_h24_keys_identical_between_planning_and_a_compiling_worker(tmp_path):
    planned = _spawn(tmp_path, "plan")
    launched = _spawn(tmp_path, "launch")
    assert any(kernel == mla_h24_decode.KERNEL_ID for kernel, _ in launched.pop("compiled_first"))
    assert planned.keys() == launched.keys()
    assert any(name.endswith(":1x1024") for name in planned) and any(name.startswith("96,134,") for name in planned)
    h24_cases = 0
    for name, programs in planned.items():
        assert launched[name] == programs, name
        rows = int(name.split(",")[0])
        has_h24 = any(kernel == mla_h24_decode.KERNEL_ID for kernel, _ in programs)
        assert has_h24 == mla_h24_decode.selected(mla_h24_decode.parse_policy(H24), rows), name
        h24_cases += has_h24
    assert h24_cases >= 30


def test_fingerprint_wrapper_only_touches_the_h24_kernel_module():
    from b12x._lib import compiler
    import ignition.components.mla_h24_kernel as kernel_module
    wrapper = mla_h24_decode.install_stable_fingerprint()
    assert mla_h24_decode.install_stable_fingerprint() is wrapper and compiler._function_fingerprint is wrapper
    target = kernel_module.H24DecodeKernel.call_mapped_extra_pertok
    module, qualname, digest = wrapper(target)
    assert (module, qualname) == ("ignition.components.mla_h24_kernel", "H24DecodeKernel.call_mapped_extra_pertok")
    import hashlib
    assert digest == hashlib.sha256(Path(kernel_module.__file__).read_bytes()).hexdigest()
    # A rewritten __code__ (what CuTe's first compile does) no longer moves the fingerprint.
    import inspect
    function = inspect.unwrap(target)
    original = function.__code__
    before = wrapper.original(target)
    try:
        function.__code__ = original.replace(co_firstlineno=original.co_firstlineno + 1)
        assert wrapper.original(target) != before  # b12x's own fingerprint moves (the E2 seq348 failure) ...
        assert wrapper(target) == (module, qualname, digest)  # ... ours does not
    finally:
        function.__code__ = original

    def plain():
        return 1
    assert wrapper(plain) == wrapper.original(plain)
