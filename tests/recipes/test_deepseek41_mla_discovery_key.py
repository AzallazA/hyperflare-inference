"""CPU tests: b12x's persisted program discovery is keyed to the DSV41 decode hooks (components/mla_discovery_key.py).

E2 export seq346 failed because the rank's progress database restored compressed-MLA discovery records written by a
run without ``FLARE_DS41_MLA_H24``: the planned program set had no H24 program, and materialisation under b12x's
no-compilation phase then launched one (``SparseMlaConfig(max_chunks_per_row=1)``, rows 96). The spawned tests
rebuild that: one offline b12x worker plans every raced split geometry x row count like the run that wrote the record,
another plans with the E2 policy, a third materialises with the E2 policy under no-compilation, and the discovery keys
must separate the two policies while the E2 plan covers every program the E2 launch path compiles."""

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ignition.components import mla_decode_splits, mla_discovery_key, mla_h24_decode  # noqa: E402

SPLITS = "18:10,37:5,47:4,62:3,94:2,96:3"
H24 = "9:0,96:1"
ROWS = (1, 9, 10, 13, 18, 19, 23, 32, 37, 38, 47, 48, 62, 63, 80, 94, 95, 96)
H24_ID = mla_h24_decode.KERNEL_ID
MLA_FACTORY = "b12x.attention.compressed_sparse_mla._preparation:compile_compressed_sparse_mla"


# ------------------------------------------------------------------------------------------------ key wrapper (unit)
def _candidates(factory):
    return [({"x": 1}, ("payload",), (types.SimpleNamespace(factory=factory),))]


@pytest.fixture
def clean_hooks():
    mla_h24_decode.uninstall()
    mla_decode_splits.uninstall()
    mla_discovery_key.uninstall()
    yield
    mla_h24_decode.uninstall()
    mla_decode_splits.uninstall()
    mla_discovery_key.uninstall()


def test_key_follows_installed_hooks_for_mla_candidates_only(clean_hooks):
    session = types.SimpleNamespace(discovery_key=lambda candidates, device: "b12x-key")
    wrapper = mla_discovery_key.install(session)
    assert mla_discovery_key.install(session) is wrapper  # idempotent
    mla, other = _candidates(MLA_FACTORY), _candidates("b12x.moe.fused_moe._preparation:compile_moe")
    assert wrapper(mla, None) == "b12x-key"  # no hook installed: b12x's key unchanged
    mla_decode_splits.install(mla_decode_splits.parse_policy(SPLITS))
    splits_only = wrapper(mla, None)
    assert splits_only != "b12x-key"
    mla_h24_decode.install(mla_h24_decode.parse_policy(H24))
    e2 = wrapper(mla, None)
    assert e2 not in ("b12x-key", splits_only)
    assert wrapper(other, None) == "b12x-key"  # other operations keep their records
    mla_h24_decode.install(mla_h24_decode.parse_policy("9:0,80:1"))
    assert wrapper(mla, None) not in (e2, splits_only, "b12x-key")
    mla_h24_decode.uninstall()
    assert wrapper(mla, None) == splits_only
    mla_decode_splits.uninstall()
    assert wrapper(mla, None) == "b12x-key"
    session.discovery_key = wrapper
    mla_discovery_key.uninstall(session)
    assert not isinstance(session.discovery_key, mla_discovery_key._HookedDiscoveryKey)


def test_key_none_stays_none(clean_hooks):
    session = types.SimpleNamespace(discovery_key=lambda candidates, device: None)
    wrapper = mla_discovery_key.install(session)
    mla_h24_decode.install(mla_h24_decode.parse_policy(H24))
    assert wrapper(_candidates(MLA_FACTORY), None) is None  # uncached discovery stays uncached


def test_hook_installs_wrap_b12x_session(clean_hooks):
    from b12x.preparation import session
    mla_h24_decode.install(mla_h24_decode.parse_policy(H24))
    assert isinstance(session.discovery_key, mla_discovery_key._HookedDiscoveryKey)
    mla_discovery_key.uninstall()
    mla_decode_splits.install(mla_decode_splits.parse_policy(SPLITS))
    assert isinstance(session.discovery_key, mla_discovery_key._HookedDiscoveryKey)


def test_identity_covers_h24_source():
    identity = None
    try:
        mla_h24_decode.install(mla_h24_decode.parse_policy(H24))
        identity = mla_discovery_key.hook_identity()
    finally:
        mla_h24_decode.uninstall()
        mla_discovery_key.uninstall()
    own, base = mla_h24_decode.source_identity()
    assert identity == {"mla_h24": [H24, mla_h24_decode.KERNEL_VERSION, own, base]}


# ------------------------------------------------------------------------------- offline plan vs launch (spawned)
def _device():
    return types.SimpleNamespace(identity=types.SimpleNamespace(vendor="nvidia", compute_capability=(12, 0),
                                                                sm_count=188),
                                 max_shared_memory_per_block=101376, max_shared_memory_per_multiprocessor=102400)


def _plan_for(rows):
    import torch
    from b12x.attention import compressed_sparse_mla as mla
    from b12x.attention.compressed_sparse_mla import _preparation as prep, _tuning as tn
    from b12x.attention._shared.mla.compressed_reference import pack_deepseek_v41_cache_reference
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
    # Every split geometry the race can try (tuning's own split ladder), the one-chunk S1 among them.
    geometries = sorted({(s.num_chunks, s.chunk_size) for limit in (1, 2, 4, 8, 16, 32, 64, 128, 256)
                         for s in (tn._split_config(plan.query, limit),)})
    configs = [tn.SparseMlaConfig(max_chunks_per_row=n, split_chunk_size=c, single_pass=False,
                                  v41_compute_mode="bf16", v41_heads_per_block=16) for n, c in geometries]
    return plan, configs


def _offline_phase(phase, splits, h24, rows_list):
    """One offline b12x compile worker. ``plan``: discovery keys + planned program sets (describe_compilation, as the
    session does before restoring/saving a record). ``launch``: the materialise path (the factory again, under
    no-compilation) recording every CuTe program key it compiles, without lowering anything."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import multiprocessing
    from b12x._lib import compile_pool, compiler
    compile_pool._initialize_worker(0, (12, 0), "GPU-offline", "offline", 188, 101376, 102400,
                                    multiprocessing.get_context("spawn").Array("q", (0, 0)))
    from b12x._lib.compile_plan import observe_programs, program_keys
    from b12x.preparation import session
    from ignition.components import mla_decode_splits as sp, mla_h24_decode as hh, mla_remainder_stream
    os.environ[mla_remainder_stream.ENV] = "1"
    mla_remainder_stream.install()
    if splits:
        sp.install(sp.parse_policy(splits))
    if h24:
        hh.install(hh.parse_policy(h24))
    out = {}
    for rows in rows_list:
        plan, configs = _plan_for(rows)
        for config in configs:
            name = f"{rows}:{config.max_chunks_per_row}x{config.split_chunk_size}"
            jobs = tuple(plan._compile_jobs(config, None))
            if phase == "plan":
                payload = plan.contract.config_payload(config)
                key = session.discovery_key([({}, payload, jobs)], _device())
                programs = sorted({(p.name, p.key) for job in jobs
                                   for p in compile_pool.describe_compilation(job).programs if p.dialect == "cute"})
                out[name] = dict(key=key, programs=programs)
                continue
            def stub(*_args, **_kwargs):
                return types.SimpleNamespace()  # stands in for the executable; compile-only launches never run it

            saved = compiler._call_cute_compile, compiler._store_cute_compile_to_disk
            compiler._call_cute_compile, compiler._store_cute_compile_to_disk = stub, lambda *a, **k: None
            try:
                with observe_programs() as seen:
                    for job in jobs:
                        program_keys(job.invoke())
            finally:
                compiler._call_cute_compile, compiler._store_cute_compile_to_disk = saved
            launched = [(p.name, p.key) for p in seen if p.dialect == "cute"]
            out[name] = dict(programs=sorted(set(launched)))
    return out


_SCRIPT = r"""
import json, multiprocessing, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from tests.recipes.test_deepseek41_mla_discovery_key import _offline_phase
if __name__ == "__main__":
    phase, splits, h24 = sys.argv[1], sys.argv[2], sys.argv[3]
    rows = tuple(int(r) for r in sys.argv[4].split(","))
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        print("RESULT", json.dumps(pool.apply(_offline_phase, (phase, splits, h24, rows))))
"""


def _spawn(tmp_path, phase, splits, h24, rows=ROWS):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT),
           "B12X_COMPILE_CACHE_DIR": str(tmp_path / f"b12x-{phase}"), "TRITON_CACHE_DIR": str(tmp_path / "triton")}
    for name in (mla_h24_decode.ENV, mla_decode_splits.ENV):
        env.pop(name, None)
    result = subprocess.run([sys.executable, "-c", _SCRIPT, phase, splits, h24, ",".join(map(str, rows))], cwd=ROOT,
                            env=env, capture_output=True, text=True, timeout=1800)
    assert result.returncode == 0, result.stderr[-4000:]
    line = next(line for line in result.stdout.splitlines() if line.startswith("RESULT "))
    return json.loads(line[len("RESULT "):])


@pytest.mark.skipif(os.environ.get("FLARE_SKIP_OFFLINE_COMPILE") == "1", reason="offline planning disabled")
def test_e2_plan_covers_launch_and_old_records_are_not_restored(tmp_path):
    e1 = _spawn(tmp_path, "plan", SPLITS, "")      # the run that wrote the restored records (no H24)
    e2 = _spawn(tmp_path, "plan", SPLITS, H24)     # E2's planning
    launch = _spawn(tmp_path, "launch", SPLITS, H24)  # E2's materialisation
    assert e1.keys() == e2.keys() == launch.keys()
    assert any(name.endswith(":1x1024") for name in e2) and any(name.startswith("96:") for name in e2)
    h24_cases = 0
    for name, planned in e2.items():
        rows = int(name.split(":")[0])
        planned_keys = {key for _, key in planned["programs"]}
        launched = launch[name]["programs"]
        assert launched, name
        assert {key for _, key in launched} <= planned_keys, (name, launched, planned["programs"])
        has_h24 = any(kernel == H24_ID for kernel, _ in launched)
        assert has_h24 == mla_h24_decode.selected(mla_h24_decode.parse_policy(H24), rows), name
        if has_h24:
            h24_cases += 1
            # The E2 failure: a record from the no-H24 run lacks the program E2 launches ...
            assert not {key for _, key in launched} <= {key for _, key in e1[name]["programs"]}, name
            # ... so it must live under another discovery key.
            assert e1[name]["key"] is not None and e2[name]["key"] != e1[name]["key"], name
    assert h24_cases > 0
