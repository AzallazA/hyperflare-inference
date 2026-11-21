# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3: steering (upstream PR #404) inside the BRRR fused tail, on CPU (no kernel, no GPU).

The CUDA layer test checks the kernel itself (GPU plan).
This file checks, without a GPU:

* the arithmetic the kernel implements (brrr/kernels/decoder.cu, ``#if BRRR_STEER``), mirrored here operation by
  operation in PyTorch: the attention site (fixed-point per-chunk dot of the replicated row, projection, row gain), the
  shared site (dot through u = W2_rank^T d from the E4M3 activation and the dequantized FP8 shard) and the per-row
  select. Against upstream's eager reference (steering.steer_attention / steer_shared_expert): scale-0 rows are the
  input bit for bit (signed zeros included), steered rows agree within one BF16 step and almost always exactly;
* the pieces with an exact contract: the E4M3 decode the kernel uses (all 256 bytes against torch's float8_e4m3fn),
  ``shared_down_direction`` against a dense FP64 reference, the order independence of the fixed-point dot, the gain at
  scale 0;
* wiring under ``make_fx``: a steered BRRR layer issues ``flare::brrr_layer_tail_steered`` once with the forward's
  per-token scale and runs no Flare-path steering operator; an unsteered layer keeps the plain op; the build switch
  and the engine contract (BRRR_STEER define, STEER attribute, refusal of steering on a plain engine).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.deepseek41 import brrr, decoder
from ignition.recipes.deepseek41.steering import LayerSteering, steer_attention, steer_shared_expert
from ignition.recipes.deepseek41.tp3_config import TP3Config

try:
    import test_deepseek41_brrr_wiring as wiring  # sibling module; pytest puts tests/recipes on sys.path
except ImportError:  # pragma: no cover - imported from another working directory
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import test_deepseek41_brrr_wiring as wiring

HID, INTER, CHUNK, SH_BLOCK = 5120, 768, 128, 32
FIXED = 2.0 ** 32
OP = torch.ops.flare.brrr_layer_tail.default
OP_STEERED = torch.ops.flare.brrr_layer_tail_steered.default


# --------------------------------------------------------------------------------------------------------------------
# Mirror of the kernel's steering arithmetic (decoder.cu #if BRRR_STEER), one PyTorch operation per kernel operation.


def bf(x):
    return x.to(torch.bfloat16).float()


def fixed_sum(parts):
    """steer_dot_add + steer_dot: FP32 partials -> __float2ll_rn(part * 2^32) (round half to even) -> integer sum ->
    __ll2float_rn(sum) / 2^32. ``parts`` [m, n] FP32; returns [m] FP32."""
    ints = torch.round(parts.double() * FIXED).to(torch.int64)  # x 2^32 is exact; torch.round = half to even
    return ints.sum(-1).to(torch.float32) / FIXED


def kernel_attention(y, scale, steering, dot=None):
    """owna_tile's steered ya: y [m, HID] BF16 (the reduced attention output), scale [m] FP32, dot [m] FP32 (adot;
    default: y . d from y's 40 chunks, as v1)."""
    yf, d, g = y.float(), steering.direction.float(), steering.attention_gain_delta.float()
    if dot is None:
        dot = fixed_sum((yf.view(-1, HID // CHUNK, CHUNK) * d.view(HID // CHUNK, CHUNK)).sum(-1))
    alpha = torch.tensor(steering.alpha, dtype=torch.float32)
    t = (scale * alpha) * dot                                   # __fmul_rn(__fmul_rn(s, alpha), dot)
    v = yf - t[:, None] * d                                     # steer_project: __fsub_rn(x, __fmul_rn(t, d))
    v = v * (1.0 + scale[:, None] * g)                          # __fmul_rn(.., __fadd_rn(1, __fmul_rn(s, g)))
    return torch.where((scale != 0)[:, None], v.to(torch.bfloat16), y)  # rows with s == 0: the loaded value


def kernel_shared(sh, h, u, scale, steering):
    """job_xf's steered shared partial: sh [m, HID] BF16 (bf(shacc)), h [m, INTER] FP32 (the dequantized E4M3
    activation finish_sh1 writes), u [INTER] FP32 (W2_rank^T d)."""
    parts = (h.view(-1, INTER // SH_BLOCK, SH_BLOCK) * u.view(INTER // SH_BLOCK, SH_BLOCK)).sum(-1)  # one per sh1 block
    dot = fixed_sum(parts)
    alpha = torch.tensor(steering.alpha, dtype=torch.float32)
    t = (scale * alpha) * dot
    v = sh.float() - t[:, None] * steering.direction.float()
    return torch.where((scale != 0)[:, None], v.to(torch.bfloat16), sh)


def e4m3_value(byte):
    """decoder.cu e4m3_value, bit for bit in FP32."""
    e, m = (byte >> 3) & 15, byte & 7
    if e:
        v = torch.tensor([((e + 120) << 23) | (m << 20)], dtype=torch.int32).view(torch.float32).item()
    else:
        v = float(m) * 2.0 ** -9
    return -v if byte & 0x80 else v


def quant_h(a):
    """finish_sh1 (and finish_woa's zq): per-32-column E4M3 quantization (quant_exp with AMAX_FLOOR 1e-4, satfinite) and its dequantized
    value e4m3_value(q) * 2^e, which J_SH2 multiplies and the kernel's dot uses."""
    blocks = a.view(a.shape[0], -1, SH_BLOCK)
    amax = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-4) * (1.0 / 448.0)
    bits = amax.view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    q = (blocks * torch.exp2(-e.float())).to(torch.float8_e4m3fn)
    return (q.float() * torch.exp2(e.float())).view_as(a)


def _steering(seed, *, gain_spread=0.02, alpha=3.3):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(HID, generator=g)
    gain = (1.0 + gain_spread * torch.randn(HID, generator=g)).to(torch.bfloat16)
    return LayerSteering(alpha=alpha, direction=(raw / raw.norm()).to(torch.bfloat16),
                         attention_gain_delta=(gain.float() - 1.0).to(torch.bfloat16))


def _rows(seed, m, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    y = (torch.randn(m, HID, generator=g) * scale).to(torch.bfloat16)
    y[0, :7] = -0.0  # signed zeros must survive the scale-0 select
    return y


def _ulps(a, b):
    """Distance in BF16 steps between two BF16 tensors (same sign assumed for the bound)."""
    ai, bi = a.view(torch.int16).int(), b.view(torch.int16).int()
    ai = torch.where(ai < 0, -(ai & 0x7FFF), ai)
    bi = torch.where(bi < 0, -(bi & 0x7FFF), bi)
    return (ai - bi).abs()


SCALES = [0.0, 1.0, 0.5, 0.0, 0.25, 1.0, -0.0, 0.75]


# --------------------------------------------------------------------------------------------------------------------
# Exact pieces


def test_e4m3_value_matches_torch_for_every_byte():
    for byte in range(256):
        if byte & 0x7F == 0x7F:  # NaN encodings: quant_byte is satfinite and never writes them
            continue
        want = torch.tensor([byte], dtype=torch.uint8).view(torch.float8_e4m3fn).float().item()
        got = e4m3_value(byte)
        assert got == want and (str(got) == str(want)), (byte, got, want)


def test_shared_down_direction_matches_dense_reference():
    g = torch.Generator().manual_seed(7)
    w = (torch.randn(HID, INTER, generator=g) * 0.5).to(torch.float8_e4m3fn)
    s = torch.randint(118, 126, (HID // 32, INTER // 32), generator=g, dtype=torch.uint8)
    d = _steering(1).direction
    dense = w.double() * torch.exp2(s.double() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)
    want = (d.double() @ dense).float()
    for weight in (w, w.view(torch.uint8)):  # E4M3 values or their bytes
        got = brrr.shared_down_direction(weight, s, d)
        assert got.dtype == torch.float32 and got.shape == (INTER,) and torch.equal(got, want)
    # 128 x 128 blocks are inferred from the shapes as well.
    s128 = s[::4, ::4].contiguous()
    dense = w.double() * torch.exp2(s128.double() - 127).repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert torch.equal(brrr.shared_down_direction(w, s128, d), (d.double() @ dense).float())


@pytest.mark.parametrize("bad", ["scale_shape", "direction", "nan_scale", "dtype"])
def test_shared_down_direction_refuses_bad_inputs(bad):
    w = torch.zeros(64, 64, dtype=torch.float8_e4m3fn)
    s = torch.full((2, 2), 127, dtype=torch.uint8)
    d = torch.zeros(64, dtype=torch.bfloat16)
    if bad == "scale_shape":
        s = torch.full((3, 2), 127, dtype=torch.uint8)
    elif bad == "direction":
        d = torch.zeros(65, dtype=torch.bfloat16)
    elif bad == "nan_scale":
        s[1, 1] = 255
    else:
        w = torch.zeros(64, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        brrr.shared_down_direction(w, s, d)


def test_fixed_point_dot_is_independent_of_the_partial_order():
    g = torch.Generator().manual_seed(3)
    parts = torch.randn(16, HID // CHUNK, generator=g) * 40
    want = fixed_sum(parts)
    for _ in range(8):
        assert torch.equal(fixed_sum(parts[:, torch.randperm(parts.shape[1], generator=g)]), want)
    # Range: |dot| < 2^31 keeps the int64 sum exact; realistic attention rows are many orders below.
    assert float(want.abs().max()) < 2.0 ** 31


def test_gain_at_scale_zero_is_exactly_one_and_never_applied():
    gd = torch.arange(-0x7F7F, 0x7F7F, 7, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float()
    gd = gd[torch.isfinite(gd)]
    for zero in (0.0, -0.0):
        assert torch.equal(1.0 + torch.tensor(zero) * gd, torch.ones_like(gd))
    # The kernel does not rely on it: a scale-0 row never enters the projection (kernel_attention's select).
    steering = _steering(2, gain_spread=5.0)  # gains far from 1: any leak into a scale-0 row would show
    y = _rows(2, 4)
    out = kernel_attention(y, torch.zeros(4), steering)
    assert torch.equal(out.view(torch.int16), y.view(torch.int16))


# --------------------------------------------------------------------------------------------------------------------
# The kernel's arithmetic against upstream's reference


@pytest.mark.parametrize("m,magnitude", [(1, 1.0), (8, 1.0), (16, 30.0), (48, 0.05), (128, 4.0)])
def test_attention_site_matches_reference(m, magnitude):
    steering = _steering(10 + m)
    y = _rows(100 + m, m, magnitude)
    scale = torch.tensor([SCALES[i % len(SCALES)] for i in range(m)], dtype=torch.float32)
    if m == 1:
        scale[0] = 1.0
    got = kernel_attention(y, scale, steering)
    want = steer_attention(y, steering, scale[:, None])
    zero = scale == 0
    # Scale-0 rows: the input, bit for bit (including -0.0); also equal to the reference's where().
    assert torch.equal(got[zero].view(torch.int16), y[zero].view(torch.int16))
    assert torch.equal(got[zero].view(torch.int16), want[zero].view(torch.int16))
    # Steered rows: the dot is summed in another order (fixed point over 40 chunk partials vs one FP32 GEMV), which
    # moves a result across a BF16 rounding boundary rarely and by one step at most.
    steps = _ulps(got[~zero], want[~zero])
    assert int(steps.max()) <= 1 and float((steps > 0).float().mean()) < 2e-3, (int(steps.max()), float((steps > 0).float().mean()))
    assert not torch.equal(got[~zero], y[~zero])  # the projection did something


@pytest.mark.parametrize("m", [1, 8, 17, 64])
def test_shared_site_matches_reference(m):
    g = torch.Generator().manual_seed(500 + m)
    steering = _steering(20 + m)
    a = (torch.randn(m, INTER, generator=g) * 0.7).to(torch.bfloat16).float()
    h = quant_h(a)
    w = (torch.randn(HID, INTER, generator=g) * 0.4).to(torch.float8_e4m3fn)
    s = torch.randint(117, 123, (HID // 32, INTER // 32), generator=g, dtype=torch.uint8)
    dense = w.double() * torch.exp2(s.double() - 127).repeat_interleave(32, 0).repeat_interleave(32, 1)
    sh = (h.double() @ dense.T).float().to(torch.bfloat16)  # bf(shacc): this rank's BF16 shared partial
    u = brrr.shared_down_direction(w, s, steering.direction)
    scale = torch.tensor([SCALES[(i + 1) % len(SCALES)] for i in range(m)], dtype=torch.float32)
    got = kernel_shared(sh, h, u, scale, steering)
    want = steer_shared_expert(sh, steering, scale[:, None])
    zero = scale == 0
    assert torch.equal(got[zero].view(torch.int16), sh[zero].view(torch.int16))
    # The kernel's dot is h . u (FP32 sh before its BF16 rounding, other summation order) instead of bf16(sh) . d.
    dot_ref = sh.float() @ steering.direction.float()
    dot_u = fixed_sum((h.view(m, -1, SH_BLOCK) * u.view(-1, SH_BLOCK)).sum(-1))
    bound = 2.0 ** -8 * (sh.float().abs() @ steering.direction.float().abs()) + 1e-6
    assert bool(((dot_u - dot_ref).abs() <= bound).all())
    # Per element the steered values then differ by at most one BF16 step plus the projection of the dot difference
    # (s * alpha * |dot_u - dot_ref| * |d_i|): a few percent of a typical element's BF16 step, visible in steps only
    # where the projection cancels the element to near zero. Normwise the difference is far below BF16 rounding.
    t_err = (scale * steering.alpha * (dot_u - dot_ref).abs())[:, None] * steering.direction.float().abs()
    rows = ~zero
    w = want[rows].float()
    step = lambda x: torch.exp2(torch.floor(torch.log2(x.abs().clamp_min(2.0 ** -126))) - 7)  # BF16 spacing at |x|
    ulp = torch.maximum(step(w), step(got[rows].float()))
    assert bool(((got[rows].float() - w).abs() <= ulp + 1.01 * t_err[rows] + 1e-30).all())
    assert float((got[rows].float() - w).norm() / w.norm()) < 2.0 ** -9
    assert float((got[rows] == want[rows]).float().mean()) > 0.85


def test_rank_partials_project_like_their_sum():
    """TP3: the shared site projects each rank's partial (linear map), the all-reduce sums them: equal to projecting
    the summed shared output up to the BF16 rounding of the partials (as upstream's TP4)."""
    g = torch.Generator().manual_seed(9)
    steering = _steering(9)
    parts = [(torch.randn(8, HID, generator=g) * 0.3).to(torch.bfloat16) for _ in range(3)]
    scale = torch.tensor(SCALES, dtype=torch.float32)[:, None]
    summed = sum(steer_shared_expert(p, steering, scale).float() for p in parts)
    whole = steer_shared_expert(sum(p.float() for p in parts).to(torch.bfloat16), steering, scale).float()
    assert float((summed - whole).abs().max()) <= 4 * 2.0 ** -8 * float(whole.abs().max())


# --------------------------------------------------------------------------------------------------------------------
# Build switch and engine contract


def test_kernel_defines_add_the_steering_build_last():
    plain = TP3Config()
    assert brrr.kernel_defines(plain) == () and brrr.kernel_defines(plain, steering=True) == ("BRRR_STEER=1",)
    tuned = TP3Config(brrr_layers=frozenset({4}), brrr_ckpt="/c", brrr_woa_small=0, brrr_woa_uneven=1)
    assert brrr.kernel_defines(tuned, True) == brrr.kernel_defines(tuned) + ("BRRR_STEER=1",)
    assert brrr.engine_key(tuned) != brrr.engine_key(tuned, True)


def test_check_switches_compares_the_steer_attribute():
    tp3 = TP3Config()
    values = {attr: default for _, _, attr, default in brrr.TUNING}
    ext = SimpleNamespace(SPIN_TRAP=0, STEER=1, **values)
    brrr.check_switches(ext, tp3, steering=True)
    with pytest.raises(RuntimeError, match="STEER"):
        brrr.check_switches(ext, tp3)
    with pytest.raises(RuntimeError, match="STEER"):  # a build from before the switch has no attribute
        brrr.check_switches(SimpleNamespace(SPIN_TRAP=0, **values), tp3, steering=True)


def test_plain_engine_refuses_steering_constants():
    engine = object.__new__(brrr.Engine)
    engine.steering, engine.steered, engine.slots = False, {}, {4: 0}
    with pytest.raises(RuntimeError, match="without steering"):
        engine.register_steering(4, _steering(4))
    engine.steering = True
    with pytest.raises(RuntimeError, match="register the layer"):
        engine.register_steering(5, _steering(5))


def test_tail_requires_a_scale_exactly_when_steered():
    tail = wiring._tail()
    rows = torch.zeros(2, 4, HID, dtype=torch.bfloat16)
    args = (torch.zeros(2, 24, 512, dtype=torch.bfloat16), rows, torch.zeros(2, 4), torch.zeros(2, 4, 4),
            torch.zeros(2, 4), torch.zeros(2, dtype=torch.int64))
    with pytest.raises(ValueError, match="takes none"):
        tail(*args, token_scale=torch.zeros(2, 1))
    tail.steered = True
    with pytest.raises(ValueError, match="needs the per-token steering scale"):
        tail(*args)
    with pytest.raises(ValueError, match="one FP32 value per row"):
        tail(*args, token_scale=torch.zeros(3, 1))
    with pytest.raises(ValueError, match="one FP32 value per row"):
        tail(*args, token_scale=torch.zeros(2, 1, dtype=torch.bfloat16))


# --------------------------------------------------------------------------------------------------------------------
# Wiring under make_fx


@pytest.fixture
def steer_stubs(monkeypatch):
    built = []

    def make_tail(index, weights, positions_dtype, frequencies, *, config=None, tp3=None, steering=None):
        assert positions_dtype == torch.int64 and config is not None and tp3 is not None
        built.append((index, steering))
        tail = wiring._tail(slot=len(built) - 1)
        tail.steered = steering is not None
        return tail

    for name in ("PreparedMHC", "PreparedChunkedAllReduce", "UnquantizedProjection", "PreparedRouter",
                 "PreparedRoutedExperts", "PreparedMLP", "PreparedMHCPost", "ConcurrentSharedExpert"):
        monkeypatch.setattr(decoder, name, wiring.Component)
    monkeypatch.setattr(decoder, "PreparedAttention", wiring.Attention)
    monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)
    monkeypatch.setattr(brrr, "PreparedBRRRTail", make_tail)
    monkeypatch.setattr(decoder, "_BRRR_DECISIONS", [])
    monkeypatch.setattr(decoder, "_BRRR_CHECKED", 0)
    for name in ("steer_attention", "steer_shared_expert"):  # the Flare-path hooks must not run in a BRRR layer
        monkeypatch.setattr(decoder, name, lambda *args, **kwargs: pytest.fail("Flare-path steering ran in BRRR"))
    return built


def _steered_layer(index, steering, *, rows=6, mode="decode"):
    scratch = decoder.Scratch.allocate(rows, 1, wiring.WORLD, "cpu")
    experts = SimpleNamespace(hot_map=None, router=None, correction=None, vision_correction=None, experts=None,
                              shared=None)
    weights = SimpleNamespace(engram=None, attention=None, experts=experts)
    metadata = SimpleNamespace(token_ids=torch.zeros(rows, dtype=torch.int64), positions=torch.zeros(rows, dtype=torch.int64))
    rotary = SimpleNamespace(frequencies={False: torch.ones(32), True: torch.ones(32) * 2})
    return decoder.PreparedLayer(index, weights, None, scratch, metadata,
        SimpleNamespace(config=SimpleNamespace(text=wiring.TEXT)), rotary, shard=TensorParallelShard(0, wiring.WORLD),
        mode=mode, runtime=None, coordinator=None, session=object(), name=f"{mode}{rows}.layer{index}",
        workspace=SimpleNamespace(overlap=False), carry=False, defer_post=True, tp3=wiring._tp3(), steering=steering)


def _forward(layer):
    def forward(residual, pre_mix, positions, scale):
        indices = SimpleNamespace(position=positions)
        metadata = SimpleNamespace(token_ids=torch.zeros(residual.shape[0], dtype=torch.int64))
        incoming, pre_mix, _, _ = layer((residual,), pre_mix, metadata, indices, (None, None, None), token_scale=scale)
        return (*incoming, pre_mix)
    return forward


def _calls(graph, target):
    return [node for node in graph.nodes if node.op == "call_function" and node.target is target]


def test_steered_brrr_layer_launches_the_steered_tail_with_the_scale(steer_stubs):
    steering = _steering(4)
    layer = _steered_layer(wiring.LAYER, steering)
    assert layer.brrr is not None and layer.brrr.steered and steer_stubs == [(wiring.LAYER, steering)]
    wiring._forbid(layer)
    wiring._drive(layer)
    graph = make_fx(_forward(layer), tracing_mode="fake", _allow_non_fake_inputs=True)(
        *wiring._inputs(6), torch.zeros(6, 1)).graph
    (launch,) = _calls(graph, OP_STEERED)
    assert _calls(graph, OP) == []
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    handle, slot, attended, positions, frequencies, residual, post, comb, attn_mix, scale, *written = launch.args
    assert scale is placeholders[3] and positions is placeholders[2] and len(written) == 5
    assert all(str(node.target).startswith("aten.empty") for node in written)
    # No Flare-path steering arithmetic (matmul / where) is traced around the launch.
    names = {str(node.target) for node in graph.nodes if node.op == "call_function"}
    assert not any(name.startswith(("aten.mm", "aten.matmul", "aten.where", "aten.mv")) for name in names), names


def test_steered_layer_needs_the_scale(steer_stubs):
    layer = _steered_layer(wiring.LAYER, _steering(4))
    wiring._forbid(layer)
    wiring._drive(layer)
    with pytest.raises(ValueError, match="requires per-token scale"):
        layer((torch.zeros(6, 4, HID, dtype=torch.bfloat16),), torch.zeros(6, 4), SimpleNamespace(),
              SimpleNamespace(position=torch.zeros(6, dtype=torch.int64)), (None, None, None))


def test_unsteered_brrr_layer_keeps_the_plain_tail(steer_stubs):
    layer = _steered_layer(wiring.LAYER, None)
    assert layer.brrr is not None and not layer.brrr.steered and steer_stubs == [(wiring.LAYER, None)]
    wiring._forbid(layer)
    wiring._drive(layer)

    def forward(residual, pre_mix, positions):
        incoming, pre_mix, _, _ = layer((residual,), pre_mix, SimpleNamespace(), SimpleNamespace(position=positions),
                                        (None, None, None))
        return (*incoming, pre_mix)
    graph = make_fx(forward, tracing_mode="fake", _allow_non_fake_inputs=True)(*wiring._inputs(6)).graph
    assert len(_calls(graph, OP)) == 1 and _calls(graph, OP_STEERED) == []


def test_steered_prefill_layer_stays_on_flares_path(steer_stubs):
    """Prefill (and parked layers gated off BRRR) keep PR #404's Flare-path hooks: no BRRR tail is built."""
    layer = _steered_layer(wiring.LAYER, _steering(4), rows=64, mode="prefill")
    assert layer.brrr is None and layer.steering is not None and steer_stubs == []


def test_fake_steered_op_accepts_the_tail_arguments():
    rows = 3
    args = (0, 0, torch.zeros(rows, 24, 512, dtype=torch.bfloat16), torch.zeros(rows, dtype=torch.int64),
            torch.ones(32), torch.zeros(rows, 4, HID, dtype=torch.bfloat16), torch.zeros(rows, 4),
            torch.zeros(rows, 4, 4), torch.zeros(rows, 4), torch.zeros(rows, 1),
            torch.empty(rows, HID, dtype=torch.bfloat16), torch.empty(rows, 4, HID, dtype=torch.bfloat16),
            torch.empty(rows, 4), torch.empty(rows, 4, 4), torch.empty(rows, 4))
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode(allow_non_fake_inputs=True):
        OP_STEERED(*args)


# --------------------------------------------------------------------------------------------------------------------
# The GPU layer test's own-y expectation (v4 B gate)


def test_own_y_residual_is_the_hc_mix_of_the_steered_rows():
    import test_deepseek41_brrr_steering_cuda as layer_test

    g = torch.Generator().manual_seed(17)
    m, hc = 6, 4
    steering = _steering(17)
    y = (torch.randn(m, HID, generator=g) * 0.5).to(torch.bfloat16)
    inputs = {"residual": (torch.randn(m, hc, HID, generator=g)).to(torch.bfloat16),
              "post": 1 + 0.1 * torch.randn(m, hc, generator=g), "comb": 0.25 + 0.05 * torch.randn(m, hc, hc, generator=g)}
    scale = torch.tensor(SCALES[:m], dtype=torch.float32)[:, None]
    got = layer_test._own_y_residual(y, inputs, scale, steering)
    ya = steer_attention(y, steering, scale).float()
    terms = [inputs["post"][:, :, None] * ya[:, None, :]] + [inputs["comb"][:, k, :, None] * inputs["residual"][:, k, None, :].float()
                                                              for k in range(hc)]
    want = sum(terms)
    bound = 2.0 ** -8 * sum(t.abs() for t in terms) + 1e-30  # BF16 output rounding + FP32 sums, against their magnitudes
    assert bool(((got.float() - want).abs() <= bound).all())
    assert float((layer_test._bf16_steps(got, want.to(torch.bfloat16)) > 0).float().mean()) < 0.02
    zero = scale[:, 0] == 0
    plain = layer_test._own_y_residual(y, inputs, torch.zeros_like(scale), steering)
    assert torch.equal(got[zero].view(torch.int16), plain[zero].view(torch.int16))
    assert not torch.equal(got[~zero], plain[~zero])
    assert int(layer_test._bf16_steps(torch.tensor([1.0], dtype=torch.bfloat16), torch.tensor([1.0078125], dtype=torch.bfloat16)).max()) == 1


def test_own_y_check_passes_rounding_and_catches_row_and_element_errors():
    """v5 B gate: an exact kernel passes; a kernel whose hc mix runs in another FP32 order (here: the terms summed in
    reverse) passes; a row steered with a 10 % wrong dot (a missed chunk / wrong row's scale) fails the projection check;
    one element off by many BF16 steps where the terms do not cancel fails the element bound."""
    import test_deepseek41_brrr_steering_cuda as layer_test

    g = torch.Generator().manual_seed(23)
    m, hc = 8, 4
    steering = _steering(23)
    y = (torch.randn(m, HID, generator=g) * 0.5).to(torch.bfloat16)
    inputs = {"residual": torch.randn(m, hc, HID, generator=g).to(torch.bfloat16),
              "post": 1 + 0.1 * torch.randn(m, hc, generator=g), "comb": 0.25 + 0.05 * torch.randn(m, hc, hc, generator=g)}
    scale = torch.tensor(SCALES[:m], dtype=torch.float32)[:, None]
    exact = layer_test._own_y_residual(y, inputs, scale, steering)

    def ok(got):
        r = layer_test._own_y_check(got, y, inputs, scale, steering)
        return all(v["out"] == 0 for k, v in r.items() if k != "proj") and r["proj"][2] == 0, r

    good, r = ok(exact)
    assert good and r["proj"][0] >= 3, r
    ya = steer_attention(y, steering, scale).float().double()
    post, comb, rk = inputs["post"].double(), inputs["comb"].double(), inputs["residual"].double()
    rev = torch.stack([(sum(comb[:, k, j, None] * rk[:, k] for k in reversed(range(hc))) + post[:, j, None] * ya).float()
                       for j in range(hc)], 1).to(torch.bfloat16)
    assert ok(rev)[0], ok(rev)[1]
    wrong = steering.alpha * 1.1
    bad_ya = ya.clone()
    row = int(torch.nonzero(scale[:, 0] == 1.0)[0])
    yd = (y[row].float().double() @ steering.direction.double())
    bad_ya[row] = ((y[row].float().double() - wrong * yd * steering.direction.double())
                   * (1 + steering.attention_gain_delta.double())).to(torch.bfloat16).double()
    bad = torch.stack([(post[:, j, None] * bad_ya + sum(comb[:, k, j, None] * rk[:, k] for k in range(hc))).float()
                       for j in range(hc)], 1).to(torch.bfloat16)
    good, r = ok(bad)
    assert not good and r["proj"][2] >= 1, r
    poke = exact.clone()
    poke[1, 2, 100] = (poke[1, 2, 100].float() * 1.25 + 0.5).to(torch.bfloat16)
    good, r = ok(poke)
    assert not good, r
