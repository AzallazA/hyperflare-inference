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
   steering EFFECT, ordered steering - ordered plain vs reference(s) - reference(0) on the steered rows' ffn_residual:
   |dk - dr| <= EFFECT_REL |dr| + EFFECT_NOISE x the BF16 output rounding noise (v3: seq375's cosine >= 0.98 gate failed
   where the effect is close to that noise: scale 0.5, 'uniform' rows); cosine, relative error and noise / |dr| printed.
   v3 runs B for both attention-dot modes (set_steer_dot: post = from the reduced BF16 chunk, pre = before the
   all-reduce).
C. shared site, ordered builds, zero attention input (attention site and routing independent of the scale):
   delta = mlp_output(s) - mlp_output(0); (gate) scale-0 rows: delta == 0 bitwise; (gate) the per-row coefficient
   delta . d against want = -s * alpha * sum_r (sh_r . d) (the reference's per-rank BF16 shared partials): least-squares
   slope in [1 - C_SLOPE, 1 + C_SLOPE] and Pearson >= C_CORR over the steered rows (rows with tiny dots are dominated by
   BF16 rounding of mlp_output, so v1's per-row max relative error was not a usable gate); off-direction share printed.
D. (info) kernel time per launch from CUDA-graph replays (20 launches per graph, 5 interleaved rounds, median), plain /
   steering@0 / steering@1 (post) / steering@1 (pre), serving builds; eager fallback (labelled) if capture fails.
E. FLARE_DS41_BRRR_STEER_PROF=1 (with DS41_FUSED_PROF=1; replaces A-D): per row count and variant the kernel span and per
   job type 'first run start - last run end / mean run / mean dependency wait' in us (eager launches, median of 5).
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
EFFECT_REL, EFFECT_NOISE = 0.05, 4.0
C_SLOPE, C_CORR = 0.10, 0.95
DOT_MODES = ("post", "pre")
JOBS = ("WOA", "AMIX", "WOB", "XA", "OWNA", "XNORM", "GATE", "ROUTE", "SH1", "SH2", "MIX", "R1", "R2", "XF")


def _ulp(t):
    """BF16 spacing at each value of a BF16 tensor (as FP64)."""
    a = t.double().abs().clamp_min(2.0 ** -126)
    return torch.exp2(torch.floor(torch.log2(a)) - 7)


def _effect(k_s, k_0, r_s, r_0, live, directions):
    """Steering effect of the kernel (dk = k_s - k_0) against the reference's (dr = r_s - r_0) on the steered rows of a
    [m, HC, HID] output. The four are BF16 with steps comparable to the effect itself (the projection moves a row by
    s alpha (y . d) d, the gain by s (g - 1) y: a few BF16 steps of the residual), so an element-wise comparison measures
    mostly output rounding: seq375's cosine >= 0.98 gate failed on small effects (scale 0.5, 'uniform' rows) while
    the kernel was right. The gate instead compares the effect's coefficients along the two directions steering
    moves a row, per row and hc stream: c[row, h] = delta[row, h, :] . v[row, :] for v = d (projection) and
    v = (g - 1) * y_ref (gain); a 5120-long dot averages the rounding away (noise = its standard deviation, from the
    four outputs' ulp^2 / 12, an upper bound). Returns {direction: (|ck - cr|, |cr|, noise)} and the element-wise
    cosine (information)."""
    dk = (k_s.double() - k_0.double())[live]
    dr = (r_s.double() - r_0.double())[live]
    var = sum(_ulp(t)[live] ** 2 for t in (k_s, k_0, r_s, r_0)) / 12.0
    out = {}
    for name, v in directions.items():
        v = v.double()[live][:, None, :] if v.dim() == 2 else v.double()[None, None, :]
        ck, cr = (dk * v).sum(-1), (dr * v).sum(-1)
        noise = float((var * v * v).sum(-1).sum().sqrt())
        out[name] = (float((ck - cr).norm()), float(cr.norm()), noise)
    return out, _cos(dk, dr)


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
        self.attention_output = attention_output.clone()  # before steering (the gain direction of the effect check)
        if self.steering is not None:
            attention_output = steer_attention(attention_output, self.steering, self.scale)
        next_mix = torch.zeros_like(inputs["attn_mix"])
        residual, post, comb, normalized = self.ffn_hc(attention_output, inputs["residual"], inputs["post"],
            inputs["comb"], pre_mix=inputs["attn_mix"], pre_out=next_mix)
        self.normalized = normalized
        self.shared_partial = self.shared(normalized)  # this rank's BF16 partial (sh_r)
        return dict(ffn_residual=residual.clone(), ffn_post=post.clone(), ffn_comb=comb.clone(), pre_mix=next_mix.clone())


def _variants(tails, m, device):
    """Timing / profile variants: name, tail, scale, attention-dot mode, null constants. 'steer1null' steers every row
    with zero constants (every steering code path runs, the values and so the routing stay those of scale 0):
    steer1 - steer1null is the data-dependent part of steer1's cost, steer1null - steer0 the code paths'."""
    zero, one = _scales("zero", m, device), _scales("one", m, device)
    return (("plain", tails["plain"], None, "post", False), ("steer0", tails["steer"], zero, "post", False),
            ("steer1", tails["steer"], one, "post", False), ("steer1pre", tails["steer"], one, "pre", False),
            ("steer1null", tails["steer"], one, "post", True))


def _set_constants(engine, layer, null):
    """Re-register layer's steering constants: the real ones, or a no-op steering (zero direction, u, v and gain delta;
    set_steering requires alpha > 0, so alpha stays): every steering code path runs and the values stay those of scale 0
    (up to the sign of a zero)."""
    steering, u, v = engine.steered[layer]
    if null:
        consts = [torch.zeros_like(t) for t in (steering.direction, steering.attention_gain_delta, u, v)]
        engine.keep += consts
    else:
        consts = [steering.direction, steering.attention_gain_delta, u, v]
    engine.ext.set_steering(engine.local_handle, engine.slots[layer], *consts, float(steering.alpha))


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

                if os.environ.get("FLARE_DS41_BRRR_STEER_PROF") == "1":
                    # E. per-job timeline (DS41_FUSED_PROF=1 in the environment: the kernel records per job type the first run start
                    # and last run end); only E runs in this mode (the phase marks cost time, so A-D run in another process).
                    for m in rows_sweep:
                        inputs, positions = tail_test._make_inputs(m, heads, seed=99, rank=rank, profile="random", resid_scale=1.0, device=device)
                        for name, tail, scale, dot, null in _variants(tails, m, device):
                            engines["steer"].ext.set_steer_dot(dot)
                            _set_constants(engines["steer"], layer, null)
                            ext = engines["plain" if name == "plain" else "steer"].ext
                            runs = []
                            for i in range(8):
                                sync()
                                ext.prof_read()
                                _run(tail, inputs, positions, scale)
                                sync()
                                runs.append(ext.prof_read().double())
                            pr = torch.stack(runs[3:]).median(0).values  # [32, 4]
                            t0 = pr[16 + 15, 0]
                            span = (pr[16 + 15, 3] - t0) / 1e3
                            parts = []
                            for j, job in enumerate(JOBS):
                                if pr[j, 3] > 0:
                                    parts.append(f"{job}={(pr[16 + j, 1] - t0) / 1e3:.1f}-{(pr[16 + j, 2] - t0) / 1e3:.1f}"
                                                 f"/{pr[j, 2] / pr[j, 3] / 1e3:.2f}/{pr[j, 1] / pr[j, 3] / 1e3:.2f}")
                            note(f"m={m} E prof {name}: span={float(span):.1f} " + " ".join(parts))
                        engines["steer"].ext.set_steer_dot("post")
                        _set_constants(engines["steer"], layer, False)
                else:
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
                            # B. attention site against the reference, both dot modes.
                            ref.steering = None
                            unsteered = ref(inputs, positions)
                            directions = {"projection": steering.direction,
                                          "gain": steering.attention_gain_delta.float()[None, :] * ref.attention_output.float()}
                            for kind in ("one", "half", "mixed"):
                                scale = _scales(kind, m, device)
                                ref.steering, ref.scale = steering, scale
                                want = ref(inputs, positions)
                                live = scale[:, 0] != 0
                                for mode in DOT_MODES:
                                    for name in ("steer", "steer_ord"):
                                        engines[name].ext.set_steer_dot(mode)
                                    got = _run(tails["steer"], inputs, positions, scale)
                                    effect = _run(tails["steer_ord"], inputs, positions, scale)
                                    sync()
                                    stats = {name: tail_test._stats(got[name], want[name]) for name in MHC}
                                    bad = [name for name, st in stats.items() if not st["finite"] or st["cosine"] < tail_test.TOLERANCE]
                                    coeff, dcos = _effect(effect["ffn_residual"], o1["ffn_residual"], want["ffn_residual"],
                                                          unsteered["ffn_residual"], live, directions)
                                    text = []
                                    for dname, (err, norm, noise) in coeff.items():
                                        text.append(f"{dname} relerr={err / max(norm, 1e-30):.4f} noise={noise / max(norm, 1e-30):.4f}")
                                        if not norm > 0:
                                            bad.append(f"reference did not move along {dname}")
                                        elif not err <= EFFECT_REL * norm + EFFECT_NOISE * noise:
                                            bad.append(f"effect along {dname}: error {err / norm:.4f} of the reference's > {EFFECT_REL} + "
                                                       f"{EFFECT_NOISE} x noise {noise / norm:.4f}")
                                    note(f"m={m} {profile} B {kind} {mode}: " + " ".join(
                                        f"{n} cos={st['cosine']:.6f} rowmin={st['row_min_cosine']:.6f}" for n, st in stats.items())
                                        + f" effect ffn_residual cos={dcos:.6f} " + " ".join(text) + (f" FAIL {bad}" if bad else ""))
                                    if bad:
                                        fail(m, profile, "B", kind, mode, bad)
                            for name in ("steer", "steer_ord"):
                                engines[name].ext.set_steer_dot("post")
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
                            variants = _variants(tails, m, device)
                            mode, graphs = "graph", {}
                            try:
                                for name, tail, scale, dot, null in variants:
                                    engines["steer"].ext.set_steer_dot(dot)
                                    _set_constants(engines["steer"], layer, null)
                                    graphs[name] = _graph_time(tail, inputs, positions, scale)
                                    sync()
                            except Exception as error:  # noqa: BLE001 - informational timing only
                                mode, graphs = f"eager ({type(error).__name__})", {}
                                sync()
                            samples = {name: [] for name, *_ in variants}
                            for _ in range(5):
                                for name, tail, scale, dot, null in variants:
                                    if not graphs:  # a graph keeps its capture's mode and constants
                                        engines["steer"].ext.set_steer_dot(dot)
                                        _set_constants(engines["steer"], layer, null)
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
                            engines["steer"].ext.set_steer_dot("post")
                            _set_constants(engines["steer"], layer, False)
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
