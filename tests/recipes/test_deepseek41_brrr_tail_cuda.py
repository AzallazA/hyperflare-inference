# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 BRRR fused layer tail vs. Flare's stock layer path, layer-level, on three GPUs.

Run (three free GPUs, truncated real-weight checkpoint):

    cd ~/flare-up-port && CUDA_VISIBLE_DEVICES=0,1,2 FLARE_RUN_CUDA_TESTS=1 \\
        FLARE_DS41_SNAPSHOT=/m2-2/flare-ckpt/ds41-mini8 .venv/bin/pytest -x -s \\
        tests/recipes/test_deepseek41_brrr_tail_cuda.py

Optional: FLARE_DS41_BRRR_TEST_LAYER (default 2, resident), FLARE_DS41_BRRR_TEST_PARKED_LAYER (default 3, merged
hot/cold bank), FLARE_DS41_BRRR_TEST_ROWS="1,6,8,12,16,17,24,48", FLARE_DS41_BRRR_TEST_HOT (default 64 hot experts
of 384 in the parked case). FLARE_DS41_BRRR_* variables already in the environment (CSF_CHECK, LOG, ...) pass through.

2026-11-14 upstream port: the tail is functional (``PreparedBRRRTail.__call__(attended, residual, post, comb, attn_mix,
positions, out=...)`` returns its outputs), and the stock reference is the port's own functional layer tail
(decoder.py ``PreparedLayer._run`` from the attention output on: components made ``.functional()``, values passed and
returned). The item-6 cases (split tail, fast route, counter parity, their equivalence and A/B passes) were removed
with the switches; the rest is tp3-main's test.

For one layer each rank builds, on identical realistic inputs, (a) Flare's stock path from the attention output to
the MLP all-reduce (inverse RoPE, grouped o-projection, all-reduce, FFN mHC post/pre, router + hot_map remap,
routed experts, shared expert, all-reduce) and (b) the BRRR tail (one flare::brrr_layer_tail launch), then compares
mlp_output, the FFN mHC residual/post/comb and pre_mix (cosine > 0.999, stats printed). Rows sweep 1..48 across the
kernel's WOA_SMALL=16 / XNORM_MIN=16 / TWO_SHOT_MIN=17 regimes; the "uniform" profile routes every row to the same
six experts so one expert takes > 16 routes (two or three 16-route chunks). Each configuration runs in its own
3-process world because FLARE_DS41_FUSED_ATTN_PER_CHUNK is an engine-creation setting of the kernel. Each rank
builds one TP3Config from its environment (TP3Config.from_environment) and passes it to the loader, the engine and
the tails. The rope negative control (FLARE_DS41_BRRR_BREAK=rope) must fail the comparison.

The stock path's two all-reduces are replaced by an exact FP32 Gloo all-reduce (rounded to BF16) so the reference is
independent of b12x's PCIe runtime.

Routing: both paths compute the router from their own bf16 FFN input (two mHC/RMSnorm implementations), so the fp32 gate
logits differ by ~1e-2 and the top-6 selection flips whenever the 6th/7th key margin is of that order (frequent on random
inputs: ~1 row in 4). A flipped row is a different function (20-35% off), so the kernel's routing is observed per row
(each row is re-launched alone through a 1-row probe tail with add_layer(route_counts=...) counters): every dropped /
added expert pair must be a near-tie (key margin < TIE_MARGIN), then the reference is re-routed with the kernel's
choice and its experts / MLP all-reduce rerun.
mlp_output then differs only by the FP8 (e4m3, 32-group) activation quantization each implementation applies to the
expert inputs and intermediates: two independent quantizers give ~5% relative norm (cosine ~0.9985), so mlp_output uses
MLP_TOLERANCE while the mHC outputs keep TOLERANCE. A dropped route, missing rank partial or wrong expert fails both.

Primary check (brrr_tail_reference.py, after NVIDIA's persistent-decoder qualification): every observable kernel output
is compared elementwise with an FP64 recomputation of its stage from that stage's own inputs, within a per-operator
bound of gamma(n) FP32 accumulation error (n = reduction length) plus half an ulp of each declared BF16/E4M3 seam
(the kernel's activation quantizers are emulated in FP64; elements a 32-sigma FP32 error model could move across a
rounding boundary get one grid step of slack, see brrr_tail_reference.py). ffn_hc.residual is checked from the shared
inputs (inverse RoPE, WO-A, WO-B, attention all-reduce, residual mix); pre_mix/post/comb from the kernel's residual
(rtol 2e-5, atol 4e-5); the kernel's routing must equal the FP64 top-6 of its own FFN input except for swaps inside
the propagated key error bound (such rows are counted and excluded from the mlp_output bound); mlp_output from the
FP64 routed/shared experts (checkpoint MXFP4 and MXFP8 bytes decoded in FP64) with the kernel's routing and the MLP
all-reduce. The cosine statistics above remain as the secondary report. After the width sweep one held-out random
input runs through a freshly constructed PreparedBRRRTail (new outputs, positions and object) under the same bounds;
it also runs the tail with forward-local outputs (out=None), the path serving graphs take.
The rope negative control fails on the ffn_residual bound.
"""

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import brrr_tail_reference as reference64  # sibling module; pytest puts tests/recipes on sys.path
except ImportError:  # pragma: no cover - imported from another working directory
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import brrr_tail_reference as reference64

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires three explicitly assigned CUDA devices")

WORLD = 3
TOLERANCE = 0.999
MLP_TOLERANCE = 0.995  # mlp_output: two independent e4m3 activation quantizations (~5% relative norm, cosine ~0.9985)
MLP_ROW_TOLERANCE = 0.99
TIE_MARGIN = 0.05  # kernel/Flare top-k disagreements must be near-ties in key = sqrt(softplus(logit)) + bias
DEFAULT_ROWS = (1, 6, 8, 12, 16, 17, 24, 48)
# (profile, residual scale): rows independent / every row identical (one expert gets m routes) / small residual so
# the attention branch dominates the FFN input (sensitive to o-projection and RoPE errors).
PROFILES = (("random", 1.0), ("uniform", 1.0), ("branch", 0.05))
COMPARED = ("mlp_output", "ffn_residual", "ffn_post", "ffn_comb", "pre_mix")
INPUTS = ("attended", "residual", "post", "comb", "attn_mix")


def _observe_routes(ext, route_counts):
    """Make every add_layer of this extension build pass route_counts (idempotent per module). False = unsupported."""
    if getattr(ext, "_route_counts_patched", False):
        return True
    add_layer = ext.add_layer
    if "route_counts" not in (add_layer.__doc__ or ""):
        return False
    ext.add_layer = lambda handle, **kwargs: add_layer(handle, route_counts=route_counts, **kwargs)
    ext._route_counts_patched = True
    return True


def _rows():
    spec = os.environ.get("FLARE_DS41_BRRR_TEST_ROWS", "")
    return tuple(int(p) for p in spec.split(",") if p.strip()) or DEFAULT_ROWS


def _stats(got, want):
    a, b = got.detach().double().reshape(got.shape[0], -1), want.detach().double().reshape(want.shape[0], -1)
    diff = a - b
    cosine = float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
    row_cos = ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-30))
    return dict(cosine=cosine, row_min_cosine=float(row_cos.min()), max_abs=float(diff.abs().max()),
                max_rel=float(diff.abs().max() / b.abs().max().clamp_min(1e-30)),
                rel_norm=float(diff.norm() / b.norm().clamp_min(1e-30)), finite=bool(torch.isfinite(a).all()))


def _make_inputs(m, heads, *, seed, rank, profile, resid_scale, device):
    """Replicated hyperconnection state (same on every rank), this rank's attention-head shard, and positions."""
    shared = torch.Generator(device="cpu").manual_seed(seed)
    local = torch.Generator(device="cpu").manual_seed(seed * 7919 + rank + 1)
    rows = 1 if profile == "uniform" else m

    def expand(t):
        return t.expand(m, *t.shape[1:]).contiguous() if rows == 1 else t

    residual = expand(torch.randn((rows, 4, 5120), generator=shared) * resid_scale)
    post = expand(1 + 0.1 * torch.randn((rows, 4), generator=shared))
    comb = expand(torch.eye(4).expand(rows, 4, 4) + 0.05 * torch.randn((rows, 4, 4), generator=shared))
    attn_mix = expand(0.25 + 0.05 * torch.randn((rows, 4), generator=shared))
    positions = expand(torch.randint(1, 4096, (rows,), generator=shared, dtype=torch.int64))
    attended = expand(torch.randn((rows, heads, 512), generator=local) * 0.3)
    values = dict(attended=attended.to(torch.bfloat16), residual=residual.to(torch.bfloat16), post=post, comb=comb,
                  attn_mix=attn_mix)
    return {name: value.contiguous().to(device) for name, value in values.items()}, positions.to(device)


def _poisoned(m, device):
    """Tail output destinations filled with NaN, so an unwritten tensor cannot pass by luck."""
    shapes = dict(mlp_output=((m, 5120), torch.bfloat16), ffn_residual=((m, 4, 5120), torch.bfloat16),
                  ffn_post=((m, 4), torch.float32), ffn_comb=((m, 4, 4), torch.float32), pre_mix=((m, 4), torch.float32))
    return {name: torch.full(shape, float("nan"), dtype=dtype, device=device) for name, (shape, dtype) in shapes.items()}


def _run_tail(tail, inputs, positions, out=None):
    """One launch on copies of the inputs (the kernel writes the attention mHC post/comb/pre-mix); outputs by name."""
    args = [inputs[name].clone() for name in INPUTS]
    destinations = None if out is None else tuple(out[name] for name in COMPARED)
    mlp_output, (residual, post, comb), pre_mix = tail(*args, positions, out=destinations)
    return dict(mlp_output=mlp_output, ffn_residual=residual, ffn_post=post, ffn_comb=comb, pre_mix=pre_mix)


class _Reference:
    """Flare's stock functional layer tail after attention (decoder.py PreparedLayer._run from attention_reduce on)."""

    def __init__(self, m, weights, config, scratch, rotary, compressed, *, session, workspace, name):
        from ignition.components.attention_output import PreparedGroupedOutputProjection
        from ignition.components.linear import UnquantizedProjection
        from ignition.components.mhc import PreparedMHC
        from ignition.components.mlp import PreparedMLP
        from ignition.components.moe import PreparedRouter, PreparedRoutedExperts
        from ignition.recipes.deepseek41.decoder import _mhc_weights

        s, text = scratch, config.text
        self.m, self.rotary, self.compressed = m, rotary, compressed
        options = dict(rms_eps=text["rms_norm_eps"], norm_eps=text["rms_norm_eps"], hc_eps=text["hc_eps"],
                       sinkhorn_iters=text["hc_sinkhorn_iters"])
        self.output_projection = PreparedGroupedOutputProjection(s.attention.attended.view(m, -1, 4096),
            weights.attention["output"], s.attention.output, session=session, name=name + ".output", workspace=workspace)
        self.ffn_hc = PreparedMHC((s.attention_output, s.attention_hc.residual, s.attention_hc.post, s.attention_hc.comb),
            _mhc_weights(weights, "ffn", "ffn_norm"), s.ffn_hc, session=session, name=name + ".ffn_hc", workspace=workspace,
            pre_mix=s.attn_mix, pre_out=s.pre_mix, **options)
        self.router_projection = UnquantizedProjection(s.ffn_hc.normalized, weights.experts.router, s.router_logits,
            session=session, name=name + ".router")
        self.router = PreparedRouter(s.router_logits, s.router, session=session, name=name + ".topk",
            score_func=text["scoring_func"], renormalize=text["norm_topk_prob"],
            routed_scaling_factor=text["routed_scaling_factor"], correction_bias=weights.experts.correction)
        self.hot_map = weights.experts.hot_map
        self.hot_rows = getattr(weights.experts, "hot_rows", None)
        self.correction = weights.experts.correction
        self.routed_scaling_factor = float(text["routed_scaling_factor"])
        self.renormalize = bool(text["norm_topk_prob"])
        if text["scoring_func"] != "sqrtsoftplus" or not self.renormalize:
            raise ValueError("re-routing assumes sqrtsoftplus scores with renormalized top-k weights")
        self.experts = PreparedRoutedExperts(s.ffn_hc.normalized, weights.experts.experts, s.router.ids, s.router.weights,
            s.routed, session=session, name=name + ".experts", workspace=workspace, router_plan=self.router.plan)
        self.shared = PreparedMLP(s.ffn_hc.normalized, weights.experts.shared, s.shared.gate_up, s.shared.intermediate,
            s.shared.output, session=session, name=name + ".shared", workspace=workspace,
            swiglu_limit=text["swiglu_limit"], round_silu=False, activation_mode="auto")
        # As PreparedLayer.__init__: every operator turned functional (values in, values out).
        self.output_projection.functional(session=session)
        self.ffn_hc.functional()
        self.router_projection.functional(session=session)
        self.router.functional(session=session)
        self.experts.functional(session=session)
        self.shared.functional(session=session)

    @staticmethod
    def _all_reduce(source):
        # Exact FP32 sum over Gloo, rounded once to BF16 (stands in for PreparedChunkedAllReduce).
        value = source.float().cpu()
        dist.all_reduce(value)
        return value.to(source.dtype).to(source.device)

    def _remap(self, ids):
        return ids if self.hot_map is None else torch.index_select(self.hot_map, 0, ids.reshape(-1)).view(ids.shape)

    def __call__(self, inputs, positions):
        m = self.m
        attended = inputs["attended"].clone()
        self.rotary.apply(attended, positions, compressed=self.compressed, inverse=True)
        attention_output = self._all_reduce(self.output_projection(attended.view(m, -1, 4096)))
        next_mix = torch.zeros_like(inputs["attn_mix"])
        residual, post, comb, normalized = self.ffn_hc(attention_output, inputs["residual"], inputs["post"],
            inputs["comb"], pre_mix=inputs["attn_mix"], pre_out=next_mix)
        logits = self.router_projection(normalized)
        route = self.router(logits)
        self.normalized, self.logits, self.ids, self.weights = normalized, logits, route.ids, route.weights
        self.selected_logits = route.selected_logits
        self.shared_output = self.shared(normalized)
        self.outputs = dict(ffn_residual=residual, ffn_post=post, ffn_comb=comb, pre_mix=next_mix)
        self._experts()
        return self.outputs

    def _experts(self):
        routed = self.experts(self.normalized, self._remap(self.ids), self.weights, router_logits=self.logits,
                              selected_logits=self.selected_logits)
        self.outputs["mlp_output"] = self._all_reduce(routed + self.shared_output)

    def reconcile(self, kernel_rows, launch_counts):
        """Adopt the kernel's top-k where it differs from Flare's by a near-tie, then rerun experts + MLP all-reduce.

        kernel_rows[row, expert]: the kernel's route counts for each row launched alone (probe); launch_counts: the
        counts of the batched launch (must equal the probe sum). Per row, every expert the kernel chose instead of
        Flare's must be a near-tie (key margin < TIE_MARGIN, strongest dropped paired with strongest added), so a
        real routing bug (wrong expert at a clear margin, wrong route count) is reported as "routing_error". Runs
        collectives: call on all ranks.
        """
        m = self.m
        ids = self.ids[:m].cpu().long()
        logits = self.logits[:m].float().cpu()
        score = torch.sqrt(torch.nn.functional.softplus(logits))
        key = score + self.correction.float().cpu()
        weights = self.weights[:m].float().cpu()
        want = self.routed_scaling_factor * score.gather(1, ids) / score.gather(1, ids).sum(1, keepdim=True)
        formula_err = float((want - weights).abs().max())
        kernel_rows, launch_counts = kernel_rows.cpu().long(), launch_counts.cpu().long()
        swaps, margins, errors = 0, [], []
        if formula_err > 5e-3:
            errors.append(f"Flare's router weights do not follow scale * score / sum(score): max err {formula_err}")
        if not torch.equal(kernel_rows.sum(0), launch_counts):
            errors.append("per-row probe launches route differently from the batched launch: experts "
                          f"{(kernel_rows.sum(0) != launch_counts).nonzero().flatten().tolist()}")
        for row in range(m):
            kernel_set = kernel_rows[row].nonzero().flatten().tolist()
            if int(kernel_rows[row].sum()) != ids.shape[1] or len(kernel_set) != ids.shape[1]:
                errors.append(f"row {row}: kernel routed {kernel_rows[row].sum()} routes to {len(kernel_set)} experts")
                continue
            flare_set = ids[row].tolist()
            dropped = sorted((e for e in flare_set if e not in kernel_set), key=lambda e: -float(key[row, e]))
            added = sorted((e for e in kernel_set if e not in flare_set), key=lambda e: -float(key[row, e]))
            for old, new in zip(dropped, added):
                margin = float(key[row, old] - key[row, new])
                margins.append(margin)
                if margin >= TIE_MARGIN:
                    errors.append(f"row {row}: kernel routed expert {new} instead of {old} at key margin {margin:.4f}")
                ids[row, flare_set.index(old)] = new
                swaps += 1
        report = dict(swapped_routes=swaps, max_swap_margin=max(margins, default=0.0), weight_formula_err=formula_err)
        if errors:
            report["routing_error"] = "; ".join(errors)
            swaps = 0  # keep Flare's routing; the comparison then reports the disagreement as a failure
        if swaps:
            new_weights = self.routed_scaling_factor * score.gather(1, ids) / score.gather(1, ids).sum(1, keepdim=True)
            self.ids = ids.to(self.ids.dtype).to(self.ids.device)
            self.weights = new_weights.to(self.weights.dtype).to(self.weights.device)
        # Always rerun (collective) so every rank takes the same path whatever its own swap count.
        self._experts()
        return report

    def route_report(self):
        ids = self.ids.cpu()
        counts = torch.bincount(ids.reshape(-1).long(), minlength=384)
        report = dict(max_routes_per_expert=int(counts.max()), distinct_experts=int((counts > 0).sum()),
                      weight_sum_min=float(self.weights.sum(1).min()),
                      weight_sum_max=float(self.weights.sum(1).max()))
        if self.hot_map is not None and self.hot_rows is not None:
            report["cold_routes"] = int((self._remap(self.ids).cpu() >= self.hot_rows).sum())
        return report


def _rank(rank, rendezvous, snapshot, env, layer, expect_failure):
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
    from ignition.recipes.deepseek41.attention import Rotary
    from ignition.recipes.deepseek41.brrr import PreparedBRRRTail, create_engine, extension, kernel_defines
    from ignition.recipes.deepseek41.decoder import Scratch
    from ignition.recipes.deepseek41.tp3_config import TP3Config
    from ignition.recipes.deepseek41.weights import DeepSeekWeightLoader

    tp3 = TP3Config.from_environment()
    assert layer in tp3.brrr_layers, (layer, os.environ.get("FLARE_DS41_BRRR_LAYERS"))
    parked = layer in tp3.host_expert_layers
    rows_sweep = _rows()
    capacity = max(rows_sweep)
    failures, reports = [], []
    try:
        with CoreStorage(rank=rank, world=WORLD, device=rank) as storage:
            manifest = inspect_indexed_checkpoint(Path(snapshot))
            with DeepSeekWeightLoader(manifest, device, shard=TensorParallelShard(rank, WORLD), prefixes=(f"layers.{layer}.",),
                                      expert_scale_compression="csf",
                                      host_allocator=storage.host_allocate if parked else None, tp3=tp3) as loader:
                weights, config = loader.load_layer(layer), loader.config
            torch.cuda.synchronize(device)
            if parked:
                assert weights.experts.hot_map is not None and weights.experts.hot_rows is not None, "parked layer did not load a merged bank"
            else:
                assert weights.experts.hot_map is None, "resident layer unexpectedly remapped"
            engine = create_engine(group, device, storage.peer_storage, tp3)
            assert engine is not None
            ext = engine.ext
            assert ext is extension(kernel_defines(tp3))
            assert ext.RANKS == WORLD and capacity <= ext.MAXM
            # Observe the kernel's routing: per-global-expert route counters (optional add_layer argument).
            route_counts = torch.zeros((ext.EXPERTS,), dtype=torch.int32, device=device)
            if not _observe_routes(ext, route_counts):
                route_counts = None
            tail_kwargs = {"config": config, "tp3": tp3}
            compressed = bool(config.text["compress_ratios"][layer])
            rotary = Rotary(config, device)
            frequencies = rotary.frequencies[compressed]
            declarations = Scratch.allocate(capacity, capacity, WORLD, device)  # declaration examples of the reference
            heads = declarations.attention.attended.shape[1]
            arena = ScratchArena(device, overlap=False)
            with RecordedPreparationSession(device=device, checkpoint="ds41-brrr-tail-test", workload_schema="tp3-brrr-tail",
                                            numerical_policy="checkpoint", compile_workers=8) as session:
                session.register_scratch(arena)
                paths = {}
                for m in rows_sweep:
                    reference = _Reference(m, weights, config, declarations.rows(m, m), rotary, compressed, session=session,
                                           workspace=arena, name=f"layer{layer}.m{m}")
                    paths[m] = (reference, PreparedBRRRTail(layer, weights, torch.int64, frequencies, **tail_kwargs))
                # One-row probe: re-launches each row alone to read the kernel's routing for that row.
                probe = PreparedBRRRTail(layer, weights, torch.int64, frequencies, **tail_kwargs)

                def kernel_routing(inputs, positions, m):
                    rows = torch.zeros((m, route_counts.numel()), dtype=torch.int32)
                    for r in range(m):
                        route_counts.zero_()
                        torch.cuda.synchronize(device)
                        _run_tail(probe, {name: value[r:r + 1] for name, value in inputs.items()}, positions[r:r + 1])
                        torch.cuda.synchronize(device)
                        rows[r] = route_counts.cpu()
                    return rows
                session.tune(show_progress=False)
                # FP64 stage references with per-operator bounds (primary check); weights come from the checkpoint
                # bytes the kernel's planes were packed from, routing/mHC parameters from Flare's loaded tensors.
                ref64 = reference64.TailReference(
                    layer=layer, rank=rank, world=WORLD, group=group, device=device, checkpoint=engine.checkpoint,
                    text=config.text, frequencies=frequencies, oa=ext.OA, inter=ext.INTER,
                    gate=weights.experts.router, gate_bias=weights.experts.correction,
                    fn=weights.residual["hc_ffn_fn"], hc_scale=weights.residual["hc_ffn_scale"],
                    hc_base=weights.residual["hc_ffn_base"], norm=weights.residual["ffn_norm.weight"])

                def launch(m, profile, resid_scale, seed, reference, tail, *, local=False):
                    inputs, positions = _make_inputs(m, heads, seed=seed, rank=rank, profile=profile,
                                                     resid_scale=resid_scale, device=device)
                    if route_counts is not None:
                        route_counts.zero_()
                    torch.cuda.synchronize(device)
                    want = reference(inputs, positions)
                    # Same launch sequence on every rank: the kernel all-reduces internally.
                    got = _run_tail(tail, inputs, positions, out=None if local else _poisoned(m, device))
                    torch.cuda.synchronize(device)
                    dist.barrier(group=group)
                    routes = reference.route_report()
                    kernel_rows = None
                    if route_counts is not None:
                        launch_counts = route_counts.clone()
                        kernel_rows = kernel_routing(inputs, positions, m)
                        routes.update(reference.reconcile(kernel_rows, launch_counts))
                        want = reference.outputs
                    stats = {name: _stats(got[name], want[name]) for name in COMPARED}
                    # Collective (two FP64 all-reduces): every rank evaluates every launch.
                    bounds = ref64.evaluate(inputs={**inputs, "positions": positions}, outputs=got, kernel_rows=kernel_rows)
                    bad = [f"bound:{name}" for name, report in bounds.items() if not report["ok"]]
                    bad += [name for name, st in stats.items() if not st["finite"]
                            or st["cosine"] < (MLP_TOLERANCE if name == "mlp_output" else TOLERANCE)
                            or (name == "mlp_output" and st["row_min_cosine"] < MLP_ROW_TOLERANCE)]
                    if "routing_error" in routes:
                        bad.append("routing")
                    reports.append((m, profile, routes, bounds, stats, bad))
                    if rank == 0:
                        line = " ".join(f"{name}: cos={st['cosine']:.6f} rowmin={st['row_min_cosine']:.6f} "
                                        f"maxrel={st['max_rel']:.3e} maxabs={st['max_abs']:.3e}" for name, st in stats.items())
                        print(f"[brrr-tail] layer {layer} m={m} {profile}: {routes} | bounds: {reference64.summarize(bounds)} | {line}"
                              + (f" | FAIL {bad}" if bad else ""), flush=True)
                    if bad:
                        details = {name: stats[name] for name in bad if name in stats}
                        details.update({name: bounds[name.removeprefix("bound:")] for name in bad if name.startswith("bound:")})
                        if "routing_error" in routes:
                            details["routing"] = routes["routing_error"]
                        failures.append((rank, m, profile, bad, details))
                    if profile == "uniform" and m > 16:
                        assert routes["max_routes_per_expert"] == m > 16, routes  # > one 16-route chunk per expert
                    if parked and profile == "random" and m == capacity:
                        assert routes["cold_routes"] > 0, f"no cold route hit with hot_rows={weights.experts.hot_rows}: {routes}"

                seed = 1000 * layer
                for m in rows_sweep:
                    reference, tail = paths[m]
                    for profile, resid_scale in PROFILES:
                        seed += 1
                        launch(m, profile, resid_scale, seed, reference, tail)
                # Held-out input on a freshly constructed PreparedBRRRTail with forward-local outputs (out=None, as in
                # serving graphs), same bounds.
                heldout = PreparedBRRRTail(layer, weights, torch.int64, frequencies, **tail_kwargs)
                launch(capacity, "heldout", 1.0, 1000 * layer + 999, paths[capacity][0], heldout, local=True)
                dist.barrier(group=group)
                gathered = [None] * WORLD
                dist.all_gather_object(gathered, failures, group=group)
                every = [f for part in gathered for f in part]
                if expect_failure:
                    assert every, "negative control (FLARE_DS41_BRRR_BREAK=rope) passed the comparison; the test is not sensitive"
                else:
                    assert not every, f"BRRR tail differs from Flare's path: {every}"
            del paths
    finally:
        torch.cuda.synchronize(device)
        dist.destroy_process_group()


def _snapshot():
    snapshot = os.environ.get("FLARE_DS41_SNAPSHOT")
    if snapshot is None or not (Path(snapshot) / "model.safetensors.index.json").is_file():
        pytest.skip("FLARE_DS41_SNAPSHOT must identify a DeepSeek V4.1 Flash checkpoint (ds41-mini8 is fast)")
    return str(Path(snapshot).resolve())


def _base_env(snapshot, layer):
    text = json.loads((Path(snapshot) / "config.json").read_text())["text_config"]
    return {
        "FLARE_DS41_MINI": "1" if text["num_hidden_layers"] != 40 else None,
        "FLARE_DS41_BRRR_LAYERS": str(layer),
        "FLARE_DS41_BRRR_CKPT": snapshot,
        "FLARE_DS41_BRRR_BREAK": None,
        "FLARE_DS41_HOST_EXPERT_LAYERS": None,
        "FLARE_DS41_HOT_EXPERTS": None,
        "FLARE_DS41_MERGED_BANK": None,
        "FLARE_DS41_BRRR_PARKED_MAX_ROWS": None,
        "FLARE_DS41_BRRR_SPIN_TRAP": None,
        "DS41_FUSED_TEST": None,
    }


def _parked_env(tmp_path, layer):
    hot = int(os.environ.get("FLARE_DS41_BRRR_TEST_HOT", "64"))
    counts = tmp_path / "counts"
    counts.mkdir(exist_ok=True)
    # Experts 0..hot-1 are the "hottest"; the rest live in pinned host memory behind the merged bank.
    (counts / f"expert-routes-rank0-layer{layer}.json").write_text(json.dumps({"counts": [384 - e for e in range(384)]}))
    return {"FLARE_DS41_HOST_EXPERT_LAYERS": str(layer), "FLARE_DS41_HOT_EXPERTS": str(hot), "FLARE_DS41_MERGED_BANK": "1",
            "FLARE_DS41_EXPERT_COUNTS_DIR": str(counts)}


CASES = {
    "resident-perchunk1": dict(parked=False, per_chunk="1", broken=False),
    "resident-perchunk0": dict(parked=False, per_chunk="0", broken=False),
    "parked-perchunk1": dict(parked=True, per_chunk="1", broken=False),
    "parked-perchunk0": dict(parked=True, per_chunk="0", broken=False),
    "resident-break-rope": dict(parked=False, per_chunk="1", broken=True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_brrr_tail_matches_flare_layer_path(tmp_path, case):
    if torch.cuda.device_count() < WORLD:
        pytest.skip("requires three explicitly assigned CUDA devices")
    options = CASES[case]
    snapshot = _snapshot()
    layer = int(os.environ.get("FLARE_DS41_BRRR_TEST_PARKED_LAYER" if options["parked"] else "FLARE_DS41_BRRR_TEST_LAYER",
                               "3" if options["parked"] else "2"))
    env = _base_env(snapshot, layer)
    env["FLARE_DS41_FUSED_ATTN_PER_CHUNK"] = options["per_chunk"]
    if options["parked"]:
        env.update(_parked_env(tmp_path, layer))
    if options["broken"]:
        env["FLARE_DS41_BRRR_BREAK"] = "rope"
    context = mp.spawn(_rank, args=((tmp_path / f"rendezvous-{case}").as_uri(), snapshot, env, layer, options["broken"]),
                       nprocs=WORLD, join=False)
    # A protocol bug hangs the kernel rather than failing it: bound the case (FLARE_DS41_BRRR_TEST_TIMEOUT seconds,
    # default 3600) and kill the ranks so the GPUs come free.
    import time

    deadline = time.monotonic() + float(os.environ.get("FLARE_DS41_BRRR_TEST_TIMEOUT", "3600"))
    while not context.join(timeout=10):
        if time.monotonic() > deadline:
            for process in context.processes:
                if process.is_alive():
                    process.kill()
            pytest.fail(f"BRRR tail case {case} did not finish in time (protocol hang?)")
