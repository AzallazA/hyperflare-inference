# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3: steering inside the BRRR fused tail (BRRR_STEER build), layer-level, on three GPUs.

Run (three free GPUs, truncated real-weight checkpoint; NOT run yet, written with the implementation 2026-11-19):

    cd ~/flare-up-steer && CUDA_VISIBLE_DEVICES=0,1,2 FLARE_RUN_CUDA_TESTS=1 \\
        FLARE_DS41_SNAPSHOT=/m2-2/flare-ckpt/ds41-mini8 PYTHONPATH=$PWD <venv>/bin/pytest -x -s \\
        tests/recipes/test_deepseek41_brrr_steering_cuda.py

Optional: FLARE_DS41_BRRR_STEER_TEST_LAYER (default 4), FLARE_DS41_STEERING_SIDECAR (a real sidecar; the layer must
be in 4..36; default: a seeded synthetic direction / gain / alpha 3.3), FLARE_DS41_BRRR_TEST_ROWS (default
1,6,8,12,16,17,24,48), FLARE_DS41_BRRR_STEER_TIMING=0 to skip the timing lines.

One 3-process world per case builds, for one layer, a plain engine (the serving build, no BRRR_STEER) and a steering
engine (BRRR_STEER=1, same tuning switches) and checks:

A. scale 0 = the plain kernel bit for bit: every output of the steered tail with an all-zero scale equals the plain
   tail's on the same inputs (rows sweep x profiles). The kernel's float atomics (zacc / yacc / yracc) are unordered,
   so each pair is first checked for run-to-run identity of the plain kernel itself (two plain launches): A is
   asserted bitwise whenever the plain kernel repeats bitwise, and otherwise within the plain kernel's own
   run-to-run difference (both counts printed).
B. attention site: with scales 1 / 0.5 / mixed (0, 1, 0.5, 0, 0.25, ...), the kernel's FFN mHC outputs (ffn_residual,
   post, comb, pre_mix: no routing in them) against Flare's reference path with upstream's steer_attention on the
   all-reduced attention output (cosine >= TOLERANCE per tensor, as the main layer test); steered rows must move vs the
   unsteered reference; scale-0 rows of a mixed batch equal the plain kernel's rows bit for bit (same determinism
   rule as A).
C. shared site: with a zero attention input (yattn = 0, so the attention site and the routing are unchanged by the
   scale), delta = mlp_output(s) - mlp_output(0) must lie along d with coefficient -s * alpha * sum_r (sh_r . d)
   (the reference's per-rank BF16 shared partials sh_r from Flare's MLP): relative coefficient error <= 5% (two
   independent FP8 activation quantizers) and an off-direction part <= 5% of |delta|; scale-0 rows: delta == 0.
D. timing (informational): plain / steered-at-0 / steered-at-1 launches per row count, us per launch.
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


class _SteeredReference(tail_test._Reference):
    """Flare's reference tail with upstream's hooks: steer_attention after the attention all-reduce, steer_shared_expert
    on this rank's shared partial before the MLP all-reduce."""

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
        return dict(ffn_residual=residual, ffn_post=post, ffn_comb=comb, pre_mix=next_mix)


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
    failures, lines = [], []
    try:
        with CoreStorage(rank=rank, world=WORLD, device=rank) as storage:
            manifest = inspect_indexed_checkpoint(Path(snapshot))
            with DeepSeekWeightLoader(manifest, device, shard=TensorParallelShard(rank, WORLD), prefixes=(f"layers.{layer}.",),
                                      expert_scale_compression="csf", tp3=tp3) as loader:
                weights, config = loader.load_layer(layer), loader.config
            steering = _steering(layer, device)
            plain = brrr.Engine(group, device, storage.peer_storage, tp3, steering=False)
            steered = brrr.Engine(group, device, storage.peer_storage, tp3, steering=True)
            assert plain.ext.STEER == 0 and steered.ext.STEER == 1 and plain.ext is not steered.ext
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
                plain_tail = brrr.PreparedBRRRTail(layer, weights, torch.int64, frequencies, engine_=plain, **kwargs)
                steer_tail = brrr.PreparedBRRRTail(layer, weights, torch.int64, frequencies, engine_=steered,
                                                   steering=steering, **kwargs)

                def note(text):
                    if rank == 0:
                        print(f"[brrr-steer] layer {layer} {text}", flush=True)

                def sync():
                    torch.cuda.synchronize(device)
                    dist.barrier(group=group)

                seed = 7000 + 100 * layer
                for m in rows_sweep:
                    ref = refs[m]
                    for profile, resid_scale in tail_test.PROFILES:
                        seed += 1
                        inputs, positions = tail_test._make_inputs(m, heads, seed=seed, rank=rank, profile=profile,
                                                                   resid_scale=resid_scale, device=device)
                        # A. scale 0 vs the plain kernel, with the plain kernel's own repeat as the determinism control.
                        p1 = _run(plain_tail, inputs, positions)
                        p2 = _run(plain_tail, inputs, positions)
                        s0 = _run(steer_tail, inputs, positions, _scales("zero", m, device))
                        sync()
                        repeat = {name: _mismatch(p1[name], p2[name]) for name in COMPARED}
                        zero = {name: _mismatch(s0[name], p1[name]) for name in COMPARED}
                        deterministic = not any(repeat.values())
                        ok = (not any(zero.values())) if deterministic else all(
                            float((s0[n].float() - p1[n].float()).abs().max()) <= float((p2[n].float() - p1[n].float()).abs().max())
                            for n in COMPARED)
                        note(f"m={m} {profile} A scale-0 mismatches {zero} (plain repeat {repeat})" + ("" if ok else " FAIL"))
                        if not ok:
                            failures.append((rank, m, profile, "A", zero, repeat))
                        # B. attention site against the reference with steer_attention.
                        ref.steering = None
                        unsteered = {name: value.clone() for name, value in ref(inputs, positions).items()}
                        for kind in ("one", "half", "mixed"):
                            scale = _scales(kind, m, device)
                            got = _run(steer_tail, inputs, positions, scale)
                            ref.steering, ref.scale = steering, scale
                            want = ref(inputs, positions)
                            sync()
                            stats = {name: tail_test._stats(got[name], want[name]) for name in MHC}
                            bad = [name for name, st in stats.items() if not st["finite"] or st["cosine"] < tail_test.TOLERANCE]
                            live = (scale[:, 0] != 0).cpu()
                            moved = not torch.equal(want["ffn_residual"][live.to(device)], unsteered["ffn_residual"][live.to(device)])
                            if not moved:
                                bad.append("reference did not move (steering had no effect)")
                            if kind == "mixed" and deterministic:
                                idle = (~live).to(device)
                                same = {name: _mismatch(got[name][idle], p1[name][idle]) for name in MHC}
                                if any(same.values()):
                                    bad.append(f"scale-0 rows differ from the plain kernel {same}")
                            note(f"m={m} {profile} B {kind}: " + " ".join(
                                f"{n} cos={st['cosine']:.6f} rowmin={st['row_min_cosine']:.6f}" for n, st in stats.items())
                                + (f" FAIL {bad}" if bad else ""))
                            if bad:
                                failures.append((rank, m, profile, "B", kind, bad))
                # C. shared site with a zero attention input (routing and attention site unchanged by the scale).
                m, ref = capacity, refs[capacity]
                seed += 1
                inputs, positions = tail_test._make_inputs(m, heads, seed=seed, rank=rank, profile="random",
                                                           resid_scale=1.0, device=device)
                inputs["attended"].zero_()
                base = _run(steer_tail, inputs, positions, _scales("zero", m, device))
                ref.steering = None
                ref(inputs, positions)
                d = steering.direction.float()
                partial_dot = (ref.shared_partial.float() @ d).cpu()
                dist.all_reduce(partial_dot)  # sum_r sh_r . d
                for kind in ("one", "mixed"):
                    scale = _scales(kind, m, device)
                    got = _run(steer_tail, inputs, positions, scale)
                    sync()
                    delta = got["mlp_output"].float() - base["mlp_output"].float()
                    coefficient = (delta @ d).cpu()
                    want = -(scale[:, 0].cpu() * steering.alpha * partial_dot)
                    off = (delta - (delta @ d)[:, None] * d[None, :]).norm(dim=1).cpu()
                    live = scale[:, 0].cpu() != 0
                    bad = []
                    if bool((delta[~live.to(device)] != 0).any()):
                        bad.append("scale-0 rows changed")
                    rel = ((coefficient - want).abs() / want.abs().clamp_min(1e-6))[live]
                    if live.any() and float(rel.max()) > 0.05:
                        bad.append(f"coefficient rel error {float(rel.max()):.3e}")
                    if live.any() and float((off[live] / delta[live.to(device)].norm(dim=1).cpu().clamp_min(1e-12)).max()) > 0.05:
                        bad.append("delta not along the direction")
                    note(f"m={m} C {kind}: coefficient rel err max {float(rel.max()) if live.any() else 0:.3e}"
                         + (f" FAIL {bad}" if bad else ""))
                    if bad:
                        failures.append((rank, m, "C", kind, bad))
                # D. timing (informational).
                if os.environ.get("FLARE_DS41_BRRR_STEER_TIMING", "1") != "0":
                    for m in rows_sweep:
                        inputs, positions = tail_test._make_inputs(m, heads, seed=99, rank=rank, profile="random",
                                                                   resid_scale=1.0, device=device)
                        timings = {}
                        for name, tail, scale in (("plain", plain_tail, None), ("steer0", steer_tail, _scales("zero", m, device)),
                                                  ("steer1", steer_tail, _scales("one", m, device))):
                            for _ in range(20):
                                _run(tail, inputs, positions, scale)
                            sync()
                            start = time.perf_counter()
                            for _ in range(200):
                                _run(tail, inputs, positions, scale)
                            sync()
                            timings[name] = (time.perf_counter() - start) / 200 * 1e6
                        note(f"m={m} D us/launch (incl. input clones) " + " ".join(f"{k}={v:.1f}" for k, v in timings.items()))
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
