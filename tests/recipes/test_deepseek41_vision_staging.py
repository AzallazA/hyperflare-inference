# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 vision weight staging (TP3Config.vision_stage) on upstream's functional forwards, on CPU with stub kernels.

components/deepseek_vision.py ``PreparedVision(stage_weights=True)`` copies the parked tower's weights into
forward-local tensors (``vision_staging.stage_copies``) and rebinds its prepared kernels to them per call. The stubs
below compute the tower's math on CPU from whatever weight tensor they are called with, so:

* a staged forward equals the resident forward bit for bit (memcpy and kernel copies), and breaking the copies
  changes the result (negative control);
* every staged kernel is declared on VRAM declaration examples shared by all blocks and capacities, never on the
  parked tensors, and drops them after tuning (the persistent arena must not keep them);
* under ``make_fx`` the forward issues 34 ``flare::stage_parked_weights`` copies (patch embedding + final norm, 32
  blocks, aligner) into graph-local destinations, each block's copy ordered after the previous block's output, and
  every staged kernel reads its own block's copies;
* weights.py ``vision_host_parked`` recognizes a tower inside the mapped host regions.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx

from ignition.components import deepseek_vision as vision, vision_staging
from ignition.recipes.deepseek41.weights import vision_host_parked

HIDDEN, WIDTH, OUT = 1024, 64, 32
STAGE = torch.ops.flare.stage_parked_weights.default


def _weights(seed=0):
    g = torch.Generator().manual_seed(seed)

    def rand(*shape, scale=0.02):
        return (torch.randn(shape, generator=g) * scale).to(torch.bfloat16)

    weights = {"vision.patch_embed.proj.weight": rand(HIDDEN, 588), "vision.patch_embed.proj.bias": rand(HIDDEN),
               "vision.norm.weight": 1 + rand(HIDDEN), "aligner.w1.weight": rand(OUT, 9 * HIDDEN),
               "aligner.w1.bias": rand(OUT), "aligner.w2.weight": rand(OUT, OUT, scale=0.2), "aligner.w2.bias": rand(OUT),
               "image_start": rand(OUT)}
    for index in range(vision.VISION_BLOCKS):
        p = f"vision.blocks.{index}."
        weights.update({p + "norm1.weight": 1 + rand(HIDDEN), p + "attn.wqkv.weight": rand(3 * HIDDEN, HIDDEN),
                        p + "attn.wqkv.bias": rand(3 * HIDDEN), p + "attn.wo.weight": rand(HIDDEN, HIDDEN),
                        p + "attn.wo.bias": rand(HIDDEN), p + "norm2.weight": 1 + rand(HIDDEN),
                        p + "mlp.w1.weight": rand(2 * WIDTH, HIDDEN), p + "mlp.w2.weight": rand(HIDDEN, WIDTH)})
    return weights


@dataclass(frozen=True)
class _FunctionalNorm:
    weight: torch.Tensor

    def __call__(self, source):
        value = source.float()
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6) * self.weight.float()).bfloat16()


@pytest.fixture
def stubs(monkeypatch):
    declared = []

    class Projection:
        def __init__(self, source, weight, out, *, session, name, bias=None):
            declared.append((name, weight, bias))
            self.weight, self.bias = weight, bias

        def functional(self, *, session):
            return self

        def __call__(self, source):
            value = source.float() @ self.weight.float().T
            return (value if self.bias is None else value + self.bias.float()).bfloat16()

    class Norm:
        def __init__(self, source, weight, output, *, eps, session, name):
            declared.append((name, weight, None))
            self.weight = weight

        def functional(self):
            return _FunctionalNorm(self.weight)

    class MLP:
        def __init__(self, source, weights, gate_up, intermediate, output, *, session, name, workspace, round_silu):
            self.up_projection = Projection(source, weights.gate_up, gate_up, session=session, name=name + ".gate_up")
            self.down_projection = Projection(source, weights.down, output, session=session, name=name + ".down")

        def functional(self, *, session):
            return self

        def __call__(self, source):
            gate, up = self.up_projection(source).float().chunk(2, dim=-1)
            return self.down_projection((F.silu(gate) * up).bfloat16())

    class Attention:
        def __init__(self, *args, **kwargs):
            pass

        def functional(self, *, session):
            return self

        def __call__(self, query, key, value, cu):
            return (query.float() + 0.5 * key.float() + value.float()).bfloat16()

    monkeypatch.setattr(vision, "UnquantizedProjection", Projection)
    monkeypatch.setattr(vision, "PreparedRMSNorm", Norm)
    monkeypatch.setattr(vision, "PreparedMLP", MLP)
    monkeypatch.setattr(vision, "PreparedVarlenAttention", Attention)
    return declared


def _encoder(weights, *, staged, capacity=18, examples=None, kernel_copy=False):
    scratch = vision.VisionScratch.allocate(capacity, "cpu", hidden=OUT)
    return vision.PreparedVision(weights, scratch, None, session=object(), workspace=None, name=f"vision{capacity}",
                                 stage_weights=staged, kernel_copy=kernel_copy, examples=examples)


def _inputs(capacity=18, seed=1):
    g = torch.Generator().manual_seed(seed)
    pixels = torch.randn((capacity, 588), generator=g).to(torch.bfloat16)
    positions = torch.randint(0, 16, (capacity, 2), generator=g)
    indices = torch.arange(capacity, dtype=torch.int32).view(capacity // 9, 9)
    indices[-1, -2:] = -1
    return pixels, positions, torch.tensor([0, capacity], dtype=torch.int32), indices


@pytest.mark.parametrize("kernel_copy", [False, True])
def test_staged_forward_matches_the_resident_forward(stubs, kernel_copy):
    weights = _weights()
    resident = _encoder(weights, staged=False)(*_inputs())
    staged = _encoder(weights, staged=True, kernel_copy=kernel_copy)(*_inputs())
    assert staged.shape == (2, OUT) and torch.isfinite(staged.float()).all()
    assert torch.equal(staged, resident)


def test_negative_control_broken_copies_change_the_result(stubs, monkeypatch):
    weights = _weights()
    resident = _encoder(weights, staged=False)(*_inputs())
    encoder = _encoder(weights, staged=True)
    real = vision.stage_copies
    monkeypatch.setattr(vision, "stage_copies",
                        lambda token, sources, kernel_copy=False: {name: value * 0.5 for name, value in
                                                                   real(token, sources, kernel_copy=kernel_copy).items()})
    assert not torch.equal(encoder(*_inputs()), resident)


def test_staged_kernels_are_declared_on_shared_examples_and_release_them(stubs):
    weights, examples = _weights(), {}
    first = _encoder(weights, staged=True, capacity=18, examples=examples)
    count = len(stubs)
    second = _encoder(weights, staged=True, capacity=36, examples=examples)
    assert len(stubs) == 2 * count
    parked = {id(tensor) for tensor in weights.values()}
    declared = [tensor for _, weight, bias in stubs for tensor in (weight, bias) if tensor is not None]
    assert declared and not any(id(tensor) in parked for tensor in declared)
    assert {id(tensor) for tensor in declared} == {id(tensor) for tensor in examples.values()}
    # One example per block weight name (shared by the 32 blocks and both capacities) plus the resident/aligner sets.
    assert len(examples) == len(vision.BLOCK_WEIGHTS) + len(vision.RESIDENT_WEIGHTS) + len(vision.ALIGNER_WEIGHTS)
    for encoder in (first, second):  # the finalizer ran (session is not a recorded session): no example kept
        held = [encoder.patch.weight, encoder.norm.weight, encoder.align1.weight, encoder.align2.weight]
        for norm1, qkv, output, norm2, mlp in encoder.blocks:
            held += [norm1.weight, qkv.weight, output.weight, norm2.weight, mlp.up_projection.weight,
                     mlp.down_projection.weight]
        assert all(tensor is None for tensor in held)
    block, resident, aligner = first.staging_bytes
    assert block == sum(weights[f"vision.blocks.0.{suffix}"].nbytes for suffix in vision.BLOCK_WEIGHTS)
    assert resident == sum(weights[name].nbytes for name in vision.RESIDENT_WEIGHTS)
    assert aligner == sum(weights[name].nbytes for name in vision.ALIGNER_WEIGHTS)
    assert _encoder(weights, staged=False).staging_bytes is None


def test_staging_requires_one_block_geometry(stubs):
    weights = _weights()
    weights["vision.blocks.7.mlp.w1.weight"] = weights["vision.blocks.7.mlp.w1.weight"][:-2]
    with pytest.raises(ValueError, match="share one weight geometry"):
        _encoder(weights, staged=True)


def _ancestors(node):
    seen, stack = set(), [node]
    while stack:
        for parent in stack.pop().all_input_nodes:
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def test_functionalized_forward_copies_into_graph_local_slots_in_block_order(stubs):
    """The graph AOTAutograd hands Inductor: 34 functionalized copies whose destinations are graph-local empties,
    each block's copy ordered after the previous copy (through the previous block's output), and every matmul
    reading staged copies."""
    from torch._functorch.aot_autograd import aot_function

    weights = _weights()
    encoder = _encoder(weights, staged=True)
    sets = [encoder.resident_sources, *encoder.block_sources, encoder.aligner_sources]
    names = [list(sources) for sources in sets]
    flat = [sources[name] for sources, keys in zip(sets, names) for name in keys]
    captured = []

    class Captured(Exception):
        pass

    def compiler(graph, inputs):
        captured.append(graph)
        raise Captured

    def forward(pixels, positions, cu, indices, frequencies, *parked):
        # Constants enter as graph inputs here (fake mode refuses real constants).
        encoder.frequencies = frequencies
        values = iter(parked)
        rebuilt = [{name: next(values) for name in keys} for keys in names]
        encoder.resident_sources, encoder.aligner_sources = rebuilt[0], rebuilt[-1]
        encoder.block_sources = tuple(rebuilt[1:-1])
        return encoder(pixels, positions, cu, indices)

    with pytest.raises(Captured):
        aot_function(forward, fw_compiler=compiler)(*_inputs(), encoder.frequencies, *flat)
    graph = captured[0].graph
    copies = [node for node in graph.nodes if node.op == "call_function"
              and "auto_functionalized" in str(node.target) and node.args[0] is STAGE]
    assert len(copies) == vision.VISION_BLOCKS + 2
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    for node in copies:
        assert node.kwargs["stream"] == 0 and node.kwargs["kernel_copy"] is False
        assert all(base.op == "call_function" and str(base.target).startswith("aten.empty")
                   for base in node.kwargs["_all_bases"])
    assert copies[0].kwargs["token"] is placeholders[0]  # the patch set follows the pixels
    for previous, node in zip(copies, copies[1:]):
        assert previous in _ancestors(node), "a block's copy was not ordered after the previous one"
    mms = [node for node in graph.nodes if node.op == "call_function" and "mm" in str(node.target)]
    assert len(mms) == 1 + 4 * vision.VISION_BLOCKS + 2  # patch, 4 per block, aligner
    # Each matmul's weight operand comes from its own set's copy: the latest copy in the weight's ancestry is the
    # patch set (patch GEMM), block i's copy (its 4 GEMMs), the aligner copy (2 GEMMs).
    expected = [0] + [1 + block for block in range(vision.VISION_BLOCKS) for _ in range(4)] + [len(copies) - 1] * 2
    latest = [max(i for i, copy in enumerate(copies) if copy in _ancestors(node.args[1])) for node in mms]
    assert latest == expected
    assert not [node for node in graph.nodes if node.op == "call_function"
                and str(node.target).startswith("aten.copy_") and node.args[0] in placeholders]


def test_stage_copies_runs_on_the_current_stream_on_cpu():
    sources = {"a": torch.arange(6, dtype=torch.bfloat16), "b": torch.arange(4, dtype=torch.int32).view(2, 2)}
    for kernel_copy in (False, True):
        copies = vision_staging.stage_copies(torch.zeros(1), sources, kernel_copy=kernel_copy)
        assert all(torch.equal(copies[name], sources[name]) and copies[name].data_ptr() != sources[name].data_ptr()
                   for name in sources)


def test_vision_host_parked_needs_every_tower_tensor_in_a_mapped_region():
    tower = {"vision.a": torch.zeros(4), "aligner.b": torch.zeros(4), "image_start": torch.zeros(2)}
    spans = [SimpleNamespace(address=t.data_ptr(), extent=t.nbytes, closed=False) for t in tower.values()]
    assert vision_host_parked(tower, spans)
    assert not vision_host_parked(tower, spans[1:])            # negative control: one tensor outside
    assert not vision_host_parked(tower, [SimpleNamespace(**{**vars(s), "closed": True}) for s in spans])
    assert not vision_host_parked({"image_start": torch.zeros(2)}, spans)


def test_rebinding_reaches_the_real_components_low_level_calls(monkeypatch):
    """The staging rebinds (vision._projection_with / _mlp_with / dataclasses.replace) against the real component
    classes: the weight a functional UnquantizedProjection, PreparedMLP and RMSNorm hand their kernel is the rebound
    one, and the prepared object keeps its own (the stubs above cannot show this)."""
    from dataclasses import replace

    from ignition.components import linear, mlp as mlp_module
    from ignition.components.rms_norm import _FunctionalRMSNorm

    seen = []
    monkeypatch.setattr(linear.bf16_gemv, "mm", lambda source, weight, *, out, plan, bias: seen.append((weight, bias)) or out)
    monkeypatch.setattr(mlp_module.hyperconnection, "run_swiglu", lambda gate_up, *, limit, round_silu, out, plan: out)
    monkeypatch.setattr(torch.ops.b12x, "hyperconnection_grouped_rmsnorm",
                        lambda source, weight, output, eps, handle, zero_centered: seen.append((weight, None)),
                        raising=False)

    def projection(columns, weight, bias):
        prepared = object.__new__(linear.UnquantizedProjection)
        prepared.dense, prepared.output, prepared.plan = False, None, object()
        prepared.output_layout = ((4, columns), (columns, 1), torch.bfloat16)
        prepared.weight, prepared.bias = weight, bias
        return prepared

    declared, staged = (torch.zeros(8, 8), torch.zeros(8)), (torch.ones(8, 8), torch.ones(8))
    prepared = projection(8, *declared)
    vision._projection_with(prepared, *staged)(torch.zeros(4, 8))
    assert seen.pop() == staged and (prepared.weight, prepared.bias) == declared

    mlp = object.__new__(mlp_module.PreparedMLP)
    mlp.swiglu_limit, mlp.round_silu, mlp.activation_plan = 10., True, object()
    mlp._functional_layout = (None, None, None, (4, 8), (8, 1), torch.bfloat16, None, None, None)
    mlp.up_projection, mlp.down_projection = projection(16, torch.zeros(16, 8), None), projection(8, torch.zeros(8, 8), None)
    up, down = torch.ones(16, 8), torch.ones(8, 8)
    vision._mlp_with(mlp, up, down)(torch.zeros(4, 8))
    assert [weight for weight, _ in seen] == [up, down] and mlp.up_projection.weight is not up
    seen.clear()

    norm = _FunctionalRMSNorm(torch.zeros(8), SimpleNamespace(handle=0), 1e-6, False, False, False)
    weight = torch.ones(8)
    replace(norm, weight=weight)(torch.zeros(4, 8))
    assert seen == [(weight, None)] and norm.weight is not weight
