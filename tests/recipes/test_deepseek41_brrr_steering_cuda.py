# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3: steering inside the BRRR fused tail (BRRR_STEER build), layer-level, on three GPUs.

Run (three free GPUs, a truncated real-weight checkpoint):

    cd ~/flare-up-steer && CUDA_VISIBLE_DEVICES=0,1,2 FLARE_RUN_CUDA_TESTS=1 \\
        FLARE_DS41_SNAPSHOT=/m2-2/flare-ckpt/ds41-mini24-fp8-e1 PYTHONPATH=$PWD <venv>/bin/pytest -x -s \\
        tests/recipes/test_deepseek41_brrr_steering_cuda.py

Optional: FLARE_DS41_BRRR_STEER_TEST_LAYER (default 4), FLARE_DS41_STEERING_SIDECAR (a real sidecar; layer in 4..36;
default: a seeded synthetic direction / gain / alpha 3.3), FLARE_DS41_BRRR_TEST_ROWS (default 1,6,8,12,16,17,24,48),
FLARE_DS41_BRRR_STEER_TIMING=0 skips D. The tuning switches come from the environment (TP3Config.from_environment).

v2 (2026-11-20, after seq373): the plain kernel is not run-to-run deterministic (its FP32 red.add accumulations have no
fixed order: 26 of 48 plain repeats differed on the GPU), so one equal plain repeat proved nothing and v1's A / C
"failures" were that noise. This version builds four engines in one 3-rank world for one layer: plain and steering
(the serving builds) and their BRRR_ORDERED=1 twins (test-only builds whose FP32 accumulations run in a fixed order, so
a launch is a pure function of its inputs). Per row count x input profile:

A. (gate) ordered plain twice -> bitwise equal (the twins really are deterministic), and ordered steering at scale 0
   -> bitwise equal to ordered plain (the steering code changes nothing at scale 0), and in a mixed batch (scales 0 / 1 /
   0.5 / ...) every scale-0 row bitwise equal to ordered plain's row. (info) serving builds: K plain and K steering@0
   launches, how many distinct results each and how many steering@0 results equal some plain result.
B. attention site, serving steering build vs Flare's reference + upstream steer_attention (FFN mHC outputs, no routing
   in them): (gate) cosine >= TOLERANCE as the main layer test (this is steered-vs-steered-reference; the plain kernel
   sits at the same ~0.9994-0.9997 against its reference: FP8 activation quantization of the o-projections); (gate) the
   steering EFFECT, ordered steering - ordered plain vs reference(s) - reference(0), cosine >= DELTA_COS on the
   steered rows' ffn_residual (and its relative norm error printed); the reference must move.
C. shared site, ordered builds, zero attention input (attention site and routing independent of the scale):
   delta = mlp_output(s) - mlp_output(0); (gate) scale-0 rows: delta == 0 bitwise; (gate) the per-row coefficient
   delta . d against want = -s * alpha * sum_r (sh_r . d) (the reference's per-rank BF16 shared partials): least-squares
   slope in [1 - C_SLOPE, 1 + C_SLOPE] and Pearson >= C_CORR over the steered rows (rows with tiny dots are dominated by
   BF16 rounding of mlp_output, so v1's per-row max relative error was not a usable gate); off-direction share printed.
D. (info) kernel time per launch from CUDA-graph replays (20 launches per graph, 5 interleaved rounds, median), plain /
   steering@0 / steering@1, serving builds; eager fallback (labelled) if capture fails.
"""

import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import test_deepseek41_brrr_tail_cuda as tail_test  # sibling module; pytest puts tests/recipes on sys.path
except ImportError:  # pragma: no cover
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_deepseek41_brrr_tail_cuda as tail_test

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires three explicitly assigned CUDA devices")

WORLD, COMPARED, INPUTS = tail_test.WORLD, tail_test.COMPARED, tail_test.INPUTS
MHC = ("ffn_residual", "ffn_post", "ffn_comb", "pre_mix")
MIXED = (0.0, 1.0, 0.5, 0.0, 0.25, 1.0, 0.0, 0.75)
ORDERED = ("BRRR_ORDERED=1",)
SERVING_REPEATS = 4
DELTA_COS = 0.98
C_SLOPE, C_CORR = 0.10, 0.95


def _scales(kind, m, device):
    values = {"zero": [0.0] * m, "one": [1.0] * m, "half": [0.5] * m,
              "mixed": [MIXED[i % len(MIXED)] for i in range(m)]}[kind]
    if kind == "mixed" and m == 1:
        values = [1.0]
    return torch.tensor(values, dtype=torch.float32, device=device)[:, None]


def _bits(t):
    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _mismatch(a, b):
    return int((_bits(a) != _bits(b)).sum())


def _mismatches(a, b, names=COMPARED, rows=None):
    pick = (lambda t: t) if rows is None else (lambda t: t[rows])
    return {name: _mismatch(pick(a[name]), pick(b[name])) for name in names}


def _key(out):
    return tuple(_bits(out[name]).cpu().numpy().tobytes() for name in COMPARED)


def _steering(layer, device):
    from ignition.preparation.persistent import persistent_copy
    from ignition.recipes.deepseek41.steering import LayerSteering, SteeringSidecar

    path = os.environ.get("FLARE_DS41_STEERING_SIDECAR")
    if path:
        return SteeringSidecar.load(path, device).layers[layer]
    g = torch.Generator().manual_seed(4100 + layer)
    raw = torch.randn(5120, generator=g)
    gain = (1.0 + 0.02 * torch.randn(5120, generator=g)).to(torch.bfloat16)
    return LayerSteering(alpha=3.3, direction=persistent_copy((raw / raw.norm()).to(torch.bfloat16).to(device)),
                         attention_gain_delta=persistent_copy((gain.float() - 1.0).to(torch.bfloat16).to(device)))


def _run(tail, inputs, positions, scale=None):
    args = [inputs[name].clone() for name in INPUTS]
    kwargs = {} if scale is None else {"token_scale": scale}
    mlp_output, (residual, post, comb), pre_mix = tail(*args, positions, **kwargs)
    return dict(mlp_output=mlp_output, ffn_residual=residual, ffn_post=post, ffn_comb=comb, pre_mix=pre_mix)


def _cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


class _SteeredReference(tail_test._Reference):
    """Flare's reference tail with upstream's hooks: steer_attention after the attention all-reduce (the shared-site
    hook is not needed: C reads the per-rank shared partials directly)."""

    steering = scale = None

    def __call__(self, inputs, positions):
        from ignition.recipes.deepseek41.steering import steer_attention

        m = self.m
        attended = inputs["attended"].clone()
        self.rotary.apply(attended, positions, compressed=self.compressed, inverse=True)
        attention_output = self._all_reduce(self.output_projection(attended.view(m, -1, 4096)))
        if self.steering is not None:
            attention_output = steer_attention(attention_output, self.steering, self.scale)
        next_mix = torch.zeros_like(inputs["attn_mix"])
        residual, post, comb, normalized = self.ffn_hc(attention_output, inputs["residual"], inputs["post"],
            inputs["comb"], pre_mix=inputs["attn_mix"], pre_out=next_mix)
        self.normalized = normalized
        self.shared_partial = self.shared(normalized)  # this rank's BF16 partial (sh_r)
        return dict(ffn_residual=residual.clone(), ffn_post=post.clone(), ffn_comb=comb.clone(), pre_mix=next_mix.clone())


def _graph_time(tail, inputs, positions, scale, launches=20):
    """us per launch from one CUDA graph of `launches` tail launches (static inputs; outputs graph-local)."""
    args = [inputs[name].clone() for name in INPUTS]
    kwargs = {} if scale is None else {"token_scale": scale}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            tail(*args, positions, **kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(launches):
            tail(*args, positions, **kwargs)
    torch.cuda.synchronize()
    return graph, launches


def _rank(rank, rendezvous, snapshot, env, layer):
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=WORLD, timeout=timedelta(minutes=30))
    group = dist.group.WORLD

    from ignition.components.parallel import TensorParallelShard
    from ignition.preparation.checkpoint import inspect_indexed_checkpoint
    from ignition.preparation.core_storage import CoreStorage
    from ignition.preparation.scratch import ScratchArena
    from ignition.preparation.tuning import RecordedPreparationSession
    from ignition.recipes.deepseek41 import brrr
    from ignition.recipes.deepseek41.attention import Rotary
    from ignition.recipes.deepseek41.decoder import Scratch
    from ignition.recipes.deepseek41.tp3_config import TP3Config
    from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

    tp3 = TP3Config.from_environment()
    rows_sweep = tail_test._rows()
    capacity = max(rows_sweep)
    failures = []
    try:
        with CoreStorage(rank=rank, world=WORLD, device=rank) as storage:
            manifest = inspect_indexed_checkpoint(Path(snapshot))
            with DeepSeekWeightLoader(manifest, device, shard=TensorParallelShard(rank, WORLD), prefixes=(f"layers.{layer}.",),
                                      expert_scale_compression="csf", tp3=tp3) as loader:
                weights, config = loader.load_layer(layer), loader.config
            steering = _steering(layer, device)
            ipc = storage.peer_storage
            engines = {"plain": brrr.Engine(group, device, ipc, tp3, steering=False),
                       "steer": brrr.Engine(group, device, ipc, tp3, steering=True),
                       "plain_ord": brrr.Engine(group, device, ipc, tp3, steering=False, test_defines=ORDERED),
                       "steer_ord": brrr.Engine(group, device, ipc, tp3, steering=True, test_defines=ORDERED)}
            assert [e.ext.STEER for e in engines.values()] == [0, 1, 0, 1]
            assert [e.ext.ORDERED for e in engines.values()] == [0, 0, 1, 1]
            assert len({id(e.ext) for e in engines.values()}) == 4
            compressed = bool(config.text["compress_ratios"][layer])
            rotary = Rotary(config, device)
            frequencies = rotary.frequencies[compressed]
            declarations = Scratch.allocate(capacity, capacity, WORLD, device)
            heads = declarations.attention.attended.shape[1]
            arena = ScratchArena(device, overlap=False)
            kwargs = dict(config=config, tp3=tp3)
            with RecordedPreparationSession(device=device, checkpoint="ds41-brrr-steer-test", workload_schema="tp3-brrr-steer",
                                            numerical_policy="checkpoint", compile_workers=8) as session:
                session.register_scratch(arena)
                refs = {m: _SteeredReference(m, weights, config, declarations.rows(m, m), rotary, compressed,
                                             session=session, workspace=arena, name=f"layer{layer}.m{m}") for m in rows_sweep}
                session.tune(show_progress=False)
                tails = {name: brrr.PreparedBRRRTail(layer, weights, torch.int64, frequencies, engine_=engine,
                                                     **({"steering": steering} if engine.steering else {}), **kwargs)
                         for name, engine in engines.items()}

                def note(text):
                    if rank == 0:
                        print(f"[brrr-steer] layer {layer} {text}", flush=True)

                def sync():
                    torch.cuda.synchronize(device)
                    dist.barrier(group=group)

                def fail(*what):
                    failures.append((rank, *what))

                seed = 7000 + 100 * layer
                for m in rows_sweep:
                    ref = refs[m]
                    for profile, resid_scale in tail_test.PROFILES:
                        seed += 1
                        inputs, positions = tail_test._make_inputs(m, heads, seed=seed, rank=rank, profile=profile,
                                                                   resid_scale=resid_scale, device=device)
                        zero = _scales("zero", m, device)
                        # A. ordered twins: determinism, then scale 0 == plain bit for bit.
                        o1 = _run(tails["plain_ord"], inputs, positions)
                        o2 = _run(tails["plain_ord"], inputs, positions)
                        s0 = _run(tails["steer_ord"], inputs, positions, zero)
                        mixed = _scales("mixed", m, device)
                        sm = _run(tails["steer_ord"], inputs, positions, mixed)
                        sync()
                        repeat, zero_mm = _mismatches(o1, o2), _mismatches(s0, o1)
                        idle = (mixed[:, 0] == 0)
                        mixed_mm = _mismatches(sm, o1, rows=idle) if bool(idle.any()) else {name: 0 for name in COMPARED}
                        bad = []
                        if any(repeat.values()):
                            bad.append("ordered plain not deterministic")
                        if any(zero_mm.values()):
                            bad.append("scale 0 differs from plain")
                        if any(mixed_mm.values()):
                            bad.append("scale-0 rows of a mixed batch differ from plain")
                        # Serving builds (info): K plain and K steering@0 results.
                        plain_keys = [_key(_run(tails["plain"], inputs, positions)) for _ in range(SERVING_REPEATS)]
                        steer_keys = [_key(_run(tails["steer"], inputs, positions, zero)) for _ in range(SERVING_REPEATS)]
                        sync()
                        info = (f"serving: plain distinct {len(set(plain_keys))}/{SERVING_REPEATS}, steer@0 distinct "
                                f"{len(set(steer_keys))}/{SERVING_REPEATS}, steer@0 results equal to a plain result "
                                f"{sum(k in set(plain_keys) for k in steer_keys)}/{SERVING_REPEATS}")
                        note(f"m={m} {profile} A ordered: scale-0 mismatches {zero_mm}, mixed scale-0 rows {mixed_mm}, "
                             f"plain repeat {repeat} | {info}" + (f" FAIL {bad}" if bad else ""))
                        if bad:
                            fail(m, profile, "A", bad, zero_mm, mixed_mm, repeat)
                        # B. attention site against the reference.
                        ref.steering = None
                        unsteered = ref(inputs, positions)
                        for kind in ("one", "half", "mixed"):
                            scale = _scales(kind, m, device)
                            got = _run(tails["steer"], inputs, positions, scale)
                            effect = _run(tails["steer_ord"], inputs, positions, scale)
                            ref.steering, ref.scale = steering, scale
                            want = ref(inputs, positions)
                            sync()
                            stats = {name: tail_test._stats(got[name], want[name]) for name in MHC}
                            bad = [name for name, st in stats.items() if not st["finite"] or st["cosine"] < tail_test.TOLERANCE]
                            live = scale[:, 0] != 0
                            dk = (effect["ffn_residual"].float() - o1["ffn_residual"].float())[live]
                            dr = (want["ffn_residual"].float() - unsteered["ffn_residual"].float())[live]
                            if not bool(dr.ne(0).any()):
                                bad.append("reference did not move (steering had no effect)")
                            dcos = _cos(dk, dr)
                            drel = float((dk - dr).norm() / dr.norm().clamp_min(1e-30))
                            if not dcos >= DELTA_COS:
                                bad.append(f"steering effect cosine {dcos:.4f} < {DELTA_COS}")
                            note(f"m={m} {profile} B {kind}: " + " ".join(
                                f"{n} cos={st['cosine']:.6f} rowmin={st['row_min_cosine']:.6f}" for n, st in stats.items())
                                + f" effect ffn_residual cos={dcos:.6f} relerr={drel:.4f}" + (f" FAIL {bad}" if bad else ""))
                            if bad:
                                fail(m, profile, "B", kind, bad)
                # C. shared site, ordered builds, zero attention input.
                m, ref = capacity, refs[capacity]
                seed += 1
                inputs, positions = tail_test._make_inputs(m, heads, seed=seed, rank=rank, profile="random",
                                                           resid_scale=1.0, device=device)
                inputs["attended"].zero_()
                base = _run(tails["steer_ord"], inputs, positions, _scales("zero", m, device))
                ref.steering = None
                ref(inputs, positions)
                d = steering.direction.float()
                partial_dot = (ref.shared_partial.float() @ d).cpu()
                dist.all_reduce(partial_dot)  # sum_r sh_r . d
                for kind in ("one", "mixed"):
                    scale = _scales(kind, m, device)
                    got = _run(tails["steer_ord"], inputs, positions, scale)
                    sync()
                    delta = got["mlp_output"].float() - base["mlp_output"].float()
                    live = (scale[:, 0] != 0)
                    coefficient = (delta @ d).cpu()[live.cpu()]
                    want = -(scale[:, 0].cpu() * steering.alpha * partial_dot)[live.cpu()]
                    slope = float((coefficient * want).sum() / (want * want).sum().clamp_min(1e-30))
                    corr = float(torch.corrcoef(torch.stack([coefficient, want]))[0, 1]) if int(live.sum()) > 2 else 1.0
                    off = (delta - (delta @ d)[:, None] * d[None, :])[live].norm() / delta[live].norm().clamp_min(1e-30)
                    bad = []
                    if _mismatch(got["mlp_output"][~live], base["mlp_output"][~live]):
                        bad.append("scale-0 rows changed")
                    if not abs(slope - 1.0) <= C_SLOPE:
                        bad.append(f"slope {slope:.4f}")
                    if not corr >= C_CORR:
                        bad.append(f"pearson {corr:.4f}")
                    note(f"m={m} C {kind}: slope {slope:.4f} pearson {corr:.4f} off-direction share {float(off):.4f}"
                         + (f" FAIL {bad}" if bad else ""))
                    if bad:
                        fail(m, "C", kind, bad)
                # D. timing (information): CUDA-graph replays, interleaved rounds.
                if os.environ.get("FLARE_DS41_BRRR_STEER_TIMING", "1") != "0":
                    for m in rows_sweep:
                        inputs, positions = tail_test._make_inputs(m, heads, seed=99, rank=rank, profile="random",
                                                                   resid_scale=1.0, device=device)
                        variants = (("plain", tails["plain"], None), ("steer0", tails["steer"], _scales("zero", m, device)),
                                    ("steer1", tails["steer"], _scales("one", m, device)))
                        mode, graphs = "graph", {}
                        try:
                            for name, tail, scale in variants:
                                graphs[name] = _graph_time(tail, inputs, positions, scale)
                                sync()
                        except Exception as error:  # noqa: BLE001 - informational timing only
                            mode, graphs = f"eager ({type(error).__name__})", {}
                            sync()
                        samples = {name: [] for name, _, _ in variants}
                        for _ in range(5):
                            for name, tail, scale in variants:
                                sync()
                                start = time.perf_counter()
                                if graphs:
                                    graph, launches = graphs[name]
                                    graph.replay()
                                else:
                                    launches = 20
                                    for _ in range(launches):
                                        _run(tail, inputs, positions, scale)
                                torch.cuda.synchronize(device)
                                samples[name].append((time.perf_counter() - start) / launches * 1e6)
                        med = {name: sorted(v)[len(v) // 2] for name, v in samples.items()}
                        note(f"m={m} D us/launch ({mode}) " + " ".join(f"{k}={v:.1f}" for k, v in med.items()))
                        del graphs
                gathered = [None] * WORLD
                dist.all_gather_object(gathered, failures, group=group)
                every = [f for part in gathered for f in part]
                assert not every, f"BRRR steering checks failed: {every}"
    finally:
        torch.cuda.synchronize(device)
        dist.destroy_process_group()


CASES = {"perchunk1": "1", "perchunk0": "0"}


@pytest.mark.parametrize("case", list(CASES))
def test_brrr_steering_layer(tmp_path, case):
    if torch.cuda.device_count() < WORLD:
        pytest.skip("requires three explicitly assigned CUDA devices")
    snapshot = tail_test._snapshot()
    layer = int(os.environ.get("FLARE_DS41_BRRR_STEER_TEST_LAYER", "4"))
    env = tail_test._base_env(snapshot, layer)
    env["FLARE_DS41_FUSED_ATTN_PER_CHUNK"] = CASES[case]
    context = mp.spawn(_rank, args=((tmp_path / f"rendezvous-{case}").as_uri(), snapshot, env, layer),
                       nprocs=WORLD, join=False)
    deadline = time.monotonic() + float(os.environ.get("FLARE_DS41_BRRR_TEST_TIMEOUT", "3600"))
    while not context.join(timeout=10):
        if time.monotonic() > deadline:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            pytest.fail(f"BRRR steering case {case} did not finish in time (protocol hang?)")
