# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 BRRR fused layer tail: wiring into upstream's functional forwards, on CPU (no kernel, no GPU).

The CUDA layer test (test_deepseek41_brrr_tail_cuda.py) checks the kernel's numerics against Flare's path; this file
checks what the 2026-11-14 port changed around it, with stubs:

* decoder.py ``PreparedLayer._run`` under ``make_fx`` (fake tensors): a BRRR layer issues ``flare::brrr_layer_tail``
  exactly once, every tensor the op writes is created inside the traced forward (graph-local, so Inductor places it in
  the graph's combined pool), its outputs become the layer's outgoing values (deferred or posted mHC), and none of
  Flare's tail operators run. Two consecutive BRRR layers give two launches;
* the decision rule: decode/verify only, ``brrr_parked_max_rows`` gates wide graphs of parked layers, decisions are
  recorded for every layer and graph, and ``check_brrr_decisions`` raises when ranks disagree;
* attention.py ``PreparedAttention._run`` with ``external_tail`` returns the attention output before inverse RoPE and
  the o-projection;
* brrr.py ``Engine.register`` hands the kernel persistent copies of what it reads from the checkpoint, keeps Flare's
  own dense weights and b12x's planes as views, and gives CSF layers a placeholder instead of b12x's expanded scale
  planes (preparation temporaries since FlashInfer aa6c4221).

Negative controls: a graph-wider-than-the-gate parked layer takes Flare's path; a rank with a different decision list
fails the check; a layer that would run Flare's tail (no BRRR) issues no launch.
"""

from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.deepseek41 import attention as attention_module, brrr, decoder
from ignition.recipes.deepseek41.tp3_config import TP3Config

WORLD, HIDDEN, HEADS = 3, 5120, 24
LAYER, PARKED_LAYER = 2, 21
TEXT = dict(rms_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20, scoring_func="sqrtsoftplus", norm_topk_prob=True,
            routed_scaling_factor=1.5, swiglu_limit=10.)
OP = torch.ops.flare.brrr_layer_tail.default
TAIL = brrr.PreparedBRRRTail  # the real class (the fixture replaces the module attribute with a recording factory)


class Component:
    """Construction stub with a unique plan; ``functional`` keeps it as is."""

    def __init__(self, *args, **kwargs):
        self.plan = object()

    def functional(self, **kwargs):
        return self


class Attention(Component):
    ratio = 0
    owns_window_metadata = False
    external_tail = False


def _tp3(**changes):
    base = dict(brrr_layers=frozenset({LAYER, PARKED_LAYER}), brrr_ckpt="/c")
    return TP3Config(**{**base, **changes})


def _tail(handle=0, slot=0):
    tail = object.__new__(TAIL)
    tail.handle, tail.slot, tail.frequencies = handle, slot, torch.ones(32)
    return tail


@pytest.fixture
def stubs(monkeypatch):
    built = []

    def make_tail(index, weights, positions_dtype, frequencies, *, config=None, tp3=None):
        assert positions_dtype == torch.int64 and config is not None and tp3 is not None
        built.append(index)
        return _tail(slot=len(built) - 1)

    for name in ("PreparedMHC", "PreparedChunkedAllReduce", "UnquantizedProjection", "PreparedRouter",
                 "PreparedRoutedExperts", "PreparedMLP", "PreparedMHCPost", "ConcurrentSharedExpert"):
        monkeypatch.setattr(decoder, name, Component)
    monkeypatch.setattr(decoder, "PreparedAttention", Attention)
    monkeypatch.setattr(decoder, "_mhc_weights", lambda *args: None)
    monkeypatch.setattr(brrr, "PreparedBRRRTail", make_tail)
    monkeypatch.setattr(decoder, "_BRRR_DECISIONS", [])
    monkeypatch.setattr(decoder, "_BRRR_CHECKED", 0)
    return built


def _layer(index, tp3, *, rows=6, mode="decode", hot_map=None, defer_post=True, carry=False):
    scratch = decoder.Scratch.allocate(rows, 1, WORLD, "cpu")
    experts = SimpleNamespace(hot_map=hot_map, router=None, correction=None, vision_correction=None,
                              experts=None, shared=None)
    weights = SimpleNamespace(engram=None, attention=None, experts=experts)
    metadata = SimpleNamespace(token_ids=torch.zeros(rows, dtype=torch.int64), positions=torch.zeros(rows, dtype=torch.int64))
    rotary = SimpleNamespace(frequencies={False: torch.ones(32), True: torch.ones(32) * 2})
    return decoder.PreparedLayer(index, weights, None, scratch, metadata,
        SimpleNamespace(config=SimpleNamespace(text=TEXT)), rotary, shard=TensorParallelShard(0, WORLD), mode=mode,
        runtime=None, coordinator=None, session=object(), name=f"{mode}{rows}.layer{index}",
        workspace=SimpleNamespace(overlap=False), carry=carry, defer_post=defer_post, tp3=tp3)


def _forbid(layer):
    """Flare's tail operators must not run in a BRRR layer."""
    def fail(*args, **kwargs):
        raise AssertionError("Flare's layer tail ran in a BRRR layer")
    for name in ("attention_reduce", "ffn_hc", "router_projection", "router", "experts", "shared", "mlp_reduce"):
        setattr(layer, name, fail)


def _drive(layer, *, post=None):
    """Give the layer pass-through attention stubs; ``post`` replaces PreparedMHCPost when not deferred."""
    def attention_hc(*incoming, pre_mix, pre_out):
        # (residual,) or a carried (mlp_output, residual, post, comb): the next residual depends on every value.
        residual = incoming[0] * 2 if len(incoming) == 1 else incoming[1] + incoming[0][:, None, :]
        rows = residual.shape[0]
        return residual, torch.ones(rows, 4), torch.ones(rows, 4, 4), residual[:, 0, :]
    layer.attention_hc = attention_hc
    layer.attention._run = lambda normalized, metadata, selection, **kwargs: (
        torch.ones(normalized.shape[0], HEADS, 512, dtype=torch.bfloat16), "records", selection, {"attended": None})
    if layer.post is not None:
        layer.post = post


def _forward(layers):
    """Run consecutive layers like PreparedDecoder.__call__ (incoming, pre_mix and positions as values)."""
    def forward(residual, pre_mix, positions):
        incoming, indices = (residual,), SimpleNamespace(position=positions)
        metadata = SimpleNamespace(token_ids=torch.zeros(residual.shape[0], dtype=torch.int64))
        for layer in layers:
            incoming, pre_mix, _, _ = layer(incoming, pre_mix, metadata, indices, (None, None, None))
        return (*incoming, pre_mix)
    return forward


def _inputs(rows):
    return (torch.zeros(rows, 4, HIDDEN, dtype=torch.bfloat16), torch.zeros(rows, 4),
            torch.arange(rows, dtype=torch.int64))


def _ancestors(node):
    seen, stack = set(), [node]
    while stack:
        for parent in stack.pop().all_input_nodes:
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def _launches(graph):
    return [node for node in graph.nodes if node.op == "call_function" and node.target is OP]


# --------------------------------------------------------------------------------------------------------------------
# PreparedLayer._run under make_fx


@pytest.mark.parametrize("defer_post", [True, False])
def test_brrr_layer_issues_one_launch_with_graph_local_outputs(stubs, defer_post):
    layer = _layer(LAYER, _tp3(), defer_post=defer_post)
    assert layer.brrr is not None and layer.attention.external_tail and stubs == [LAYER]
    _forbid(layer)
    posted = []
    _drive(layer, post=lambda output, residual, post, comb: posted.append(1)
           or residual * post[:, :, None] + output[:, None, :] + comb.sum())
    graph = make_fx(_forward([layer]), tracing_mode="fake", _allow_non_fake_inputs=True)(*_inputs(6)).graph
    (launch,) = _launches(graph)
    placeholders = [node for node in graph.nodes if node.op == "placeholder"]
    # Arguments: handle, slot, o, positions, frequencies, resid0, attn_post, attn_comb, attn_pre, then the outputs.
    handle, slot, attended, positions, frequencies, residual, post, comb, attn_mix, *written = launch.args
    assert (handle, slot) == (0, 0) and positions is placeholders[2]
    assert len(written) == 5 and all(node.op == "call_function" and node not in placeholders for node in written)
    assert all(str(node.target).startswith("aten.empty") for node in written), [node.target for node in written]
    (output,) = [node for node in graph.nodes if node.op == "output"]
    returned = output.args[0]
    if defer_post:  # (mlp_output, ffn residual, post, comb) carried to the next layer, then pre_mix
        assert list(returned) == written
    else:
        assert posted == [1] and returned[-1] is written[-1] and len(returned) == 2
        assert all(any(user is not launch for user in node.users) for node in written[:4])


def test_two_brrr_layers_launch_twice_and_chain_their_values(stubs):
    first, second = _layer(LAYER, _tp3(), carry=False), _layer(PARKED_LAYER, _tp3(), carry=True)
    for layer in (first, second):
        _forbid(layer)
        _drive(layer)
    graph = make_fx(_forward([first, second]), tracing_mode="fake", _allow_non_fake_inputs=True)(*_inputs(6)).graph
    a, b = _launches(graph)
    assert (a.args[1], b.args[1]) == (0, 1)
    # The second launch consumes the first one's values (carried mHC: its attention mHC stub reads the first
    # mlp_output; its attn_mix is a fresh zeros_like of the first pre_mix), so the order is a data dependence.
    assert {a.args[9], a.args[10]} <= _ancestors(b.args[5]) and a.args[13] in _ancestors(b.args[8])
    assert a not in _ancestors(a) and b not in _ancestors(a.args[9])


def test_flare_layer_issues_no_launch(stubs):
    tp3 = _tp3(brrr_layers=frozenset({PARKED_LAYER}))
    layer = _layer(LAYER, tp3)
    assert layer.brrr is None and not layer.attention.external_tail
    assert decoder._BRRR_DECISIONS == [("decode6.layer2", LAYER, "decode", 6, False, False)]


# --------------------------------------------------------------------------------------------------------------------
# Decision rule and the cross-rank check


@pytest.mark.parametrize("mode,rows,parked,expect,gated", [
    ("decode", 6, False, True, False),
    ("verify", 48, False, True, False),
    ("prefill", 128, False, False, False),
    ("draft", 6, False, False, False),
    ("verify", 12, True, True, False),
    ("verify", 13, True, False, True),     # negative control: wider than the gate on a parked layer
])
def test_decision_rule_and_records(stubs, mode, rows, parked, expect, gated):
    tp3 = _tp3(host_expert_layers=frozenset({PARKED_LAYER}), brrr_parked_max_rows=12, hot_experts=342,
               merged_bank=True, expert_counts_dir="/counts")
    index = PARKED_LAYER if parked else LAYER
    hot_map = torch.arange(384, dtype=torch.int32) if parked else None
    layer = _layer(index, tp3, rows=rows, mode=mode, hot_map=hot_map)
    assert (layer.brrr is not None) == expect
    assert decoder._BRRR_DECISIONS == [(f"{mode}{rows}.layer{index}", index, mode, rows, expect, gated)]


@pytest.mark.parametrize("mode", ["decode", "verify"])
def test_graph_wider_than_the_kernel_fails_at_construction(stubs, mode):
    assert _layer(LAYER, _tp3(), rows=brrr.MAX_ROWS, mode=mode).brrr is not None
    with pytest.raises(ValueError, match=f"has {brrr.MAX_ROWS + 1} rows, the fused kernel takes at most 128"):
        _layer(LAYER, _tp3(), rows=brrr.MAX_ROWS + 1, mode=mode)
    # A parked layer the row gate keeps on Flare's path is not refused (it never reaches the kernel).
    gated = _tp3(host_expert_layers=frozenset({PARKED_LAYER}), brrr_parked_max_rows=12, hot_experts=342,
                 merged_bank=True, expert_counts_dir="/counts")
    layer = _layer(PARKED_LAYER, gated, rows=brrr.MAX_ROWS + 1, mode=mode, hot_map=torch.arange(384, dtype=torch.int32))
    assert layer.brrr is None and decoder._BRRR_DECISIONS[-1][-2:] == (False, True)


def test_no_decisions_without_brrr_layers(stubs):
    _layer(LAYER, TP3Config())
    assert decoder._BRRR_DECISIONS == []


def test_check_brrr_decisions_compares_ranks(monkeypatch, stubs):
    import torch.distributed as dist

    _layer(LAYER, _tp3())
    _layer(LAYER, _tp3(), rows=12, mode="verify")
    monkeypatch.setattr(brrr, "_ENGINE", [object()])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 3)
    peers = {}

    def gather(out, value, group=None):
        out[:] = [value, *peers["others"]]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    mine = list(decoder._BRRR_DECISIONS)
    peers["others"] = [mine, mine]
    assert decoder.check_brrr_decisions(object()) == 2
    peers["others"] = [[], []]
    assert decoder.check_brrr_decisions(object()) == 0  # only decisions recorded since the previous check
    _layer(LAYER, _tp3(), rows=24, mode="verify")
    differing = [(*decoder._BRRR_DECISIONS[-1][:4], False, True)]
    peers["others"] = [differing, list(decoder._BRRR_DECISIONS[-1:])]
    with pytest.raises(RuntimeError, match="BRRR layer decisions differ between ranks"):
        decoder.check_brrr_decisions(object())
    monkeypatch.setattr(brrr, "_ENGINE", [])
    assert decoder.check_brrr_decisions(object()) == 0  # no engine, no kernel to deadlock


# --------------------------------------------------------------------------------------------------------------------
# PreparedAttention._run with external_tail


def test_attention_external_tail_skips_inverse_rope_and_output_projection():
    attention = object.__new__(attention_module.PreparedAttention)
    calls = []
    rows = 4
    attended = torch.full((rows, HEADS, 512), 3.)
    attention.mode, attention.ratio, attention.owns_window_metadata = "decode", 0, False
    attention.window = torch.zeros(rows, 528)
    attention.projections = {"qkv_a": SimpleNamespace(encoded_layout=None, __call__=None)}
    attention.projections["qkv_a"] = type("Q", (), {"encoded_layout": None,
        "__call__": lambda self, source, encoded_bases=None: torch.zeros(rows, 1792)})()
    attention.projections["q_b"] = lambda query_low, encoded_bases=None: torch.zeros(rows, HEADS * 512)
    attention.norm_quant, attention.concurrent_query = None, None
    attention.query_norm = attention.window_norm = lambda x: x
    attention.query_shape = (rows, HEADS, 512)
    attention.rotary = SimpleNamespace(apply=lambda tensor, position, compressed, inverse=False: calls.append(inverse))
    attention.window_writer = lambda *args, **kwargs: None
    attention.attention = lambda *args, **kwargs: attended
    attention.output_projection = lambda *args: calls.append("output") or torch.zeros(rows, HIDDEN)
    indices = SimpleNamespace(position=torch.zeros(rows, dtype=torch.int64), window_slots=None,
                              window_indices=None, window_lengths=None)
    metadata = SimpleNamespace(state_rows=None)
    attention.external_tail = True
    output, _, _, observations = attention._run(torch.zeros(rows, HIDDEN), metadata, (None, None, None), indices=indices)
    assert output is attended and calls == [False, False] and "output" not in observations
    calls.clear()
    attention.external_tail = False  # negative control: Flare's path rotates back and projects
    output, _, _, _ = attention._run(torch.zeros(rows, HIDDEN), metadata, (None, None, None), indices=indices)
    assert calls == [False, False, True, "output"] and output.shape == (rows, HIDDEN)


# --------------------------------------------------------------------------------------------------------------------
# Engine.register: persistent copies, views, CSF placeholder


class _Ext:
    INTER, OA, EXPERTS, SH2_LD = 256, 1024, 384, 16

    def __init__(self):
        self.layers = []

    def add_layer(self, handle, **kwargs):
        self.layers.append(kwargs)
        return len(self.layers) - 1


class _Checkpoint:
    def __init__(self):
        self.reads = []

    def shape(self, name):
        return (4096, 4096)

    def get(self, name, rows=slice(None), cols=slice(None)):
        self.reads.append(name)
        if name.endswith(("gate.bias", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base")):
            return torch.ones(8)
        if name.endswith("norm.weight"):
            return torch.ones(8, dtype=torch.bfloat16)
        return torch.ones(4, 32, dtype=torch.uint8)


def _engine(monkeypatch, *, csf):
    engine = object.__new__(brrr.Engine)
    engine.ext, engine.rank, engine.tp3, engine.device = _Ext(), 0, _tp3(), torch.device("cpu")
    engine.local_handle, engine.slots, engine.keep = 0, {}, []
    engine.checkpoint = _Checkpoint()
    engine.unused_scales = torch.zeros(1, 16, dtype=torch.uint8)
    copies = []

    def persistent(tensor):
        copies.append(tensor.clone())
        return copies[-1]

    monkeypatch.setattr(brrr, "persistent_copy", persistent)
    monkeypatch.setattr(brrr, "_rows2d", lambda weight: weight)
    monkeypatch.setattr(brrr, "_csf_args", lambda scales: dict(storage=scales, gdesc=torch.zeros(4, dtype=torch.int32),
                                                               meta=[0] * 9))
    rep = SimpleNamespace(n64_repack=False, intermediate_size=256, num_experts=384, w13_layout="w31",
                          w13_rp=torch.zeros(4, 8), w13_sfb=torch.zeros(4, 2), w2_rp=torch.zeros(4, 8), w2_sfb=torch.zeros(4, 2))
    decoder_planes = SimpleNamespace(first=SimpleNamespace(scales=torch.zeros(8, dtype=torch.uint8)),
                                     second=SimpleNamespace(scales=torch.zeros(8, dtype=torch.uint8))) if csf else None
    impl = SimpleNamespace(representation_for=lambda mode: rep, nvfp4_csf=decoder_planes)
    dense = {name: torch.zeros(4, 32, dtype=torch.uint8) for name in ("wo_a", "wo_b", "router", "gate_up", "down")}
    weights = SimpleNamespace(
        experts=SimpleNamespace(experts=SimpleNamespace(_impl=impl), hot_map=torch.arange(384, dtype=torch.int32),
                                hot_rows=64, router=dense["router"],
                                shared=SimpleNamespace(gate_up=dense["gate_up"], down=dense["down"])),
        attention={"output": SimpleNamespace(wo_a=dense["wo_a"], wo_b=dense["wo_b"])})
    return engine, weights, rep, dense, copies


@pytest.mark.parametrize("csf", [False, True])
def test_register_packs_checkpoint_copies_and_keeps_weight_views(monkeypatch, csf):
    engine, weights, rep, dense, copies = _engine(monkeypatch, csf=csf)
    assert engine.register(LAYER, weights) == 0 and engine.register(LAYER, weights) == 0  # registered once
    (kwargs,) = engine.ext.layers
    # Flare's own dense weights and b12x's weight planes are read in place (dense-native default).
    for name, source in (("woa", "wo_a"), ("wob", "wo_b"), ("gate", "router"), ("sh13", "gate_up"), ("sh2", "down")):
        assert kwargs[name] is dense[source]
    assert kwargs["w13"] is rep.w13_rp and kwargs["w2"] is rep.w2_rp
    # Everything built from checkpoint reads went through persistent_copy; the hot map is already a persistent
    # int32 forward constant and is passed as is (no second copy).
    assert kwargs["hot_map"] is weights.experts.hot_map
    copied = {"woa_s", "wob_s", "gate_b", "sh13_s", "sh2_s", "fn", "hc_scale", "hc_base", "norm"}
    if csf:
        copied |= {"csf13_gdesc", "csf2_gdesc"}
    assert len(copies) == len(copied) and {id(kwargs[name]) for name in copied} == {id(t) for t in copies}
    if csf:  # the kernel reads the payload; b12x's expanded planes are not held
        assert kwargs["s13"] is engine.unused_scales and kwargs["s2"] is engine.unused_scales
        # ... in particular not the loader's shared scale scratch b12x's planes view (rep.w13_sfb / w2_sfb).
        assert all(kwargs[name].data_ptr() != plane.data_ptr()
                   for name in ("s13", "s2") for plane in (rep.w13_sfb, rep.w2_sfb))
        assert kwargs["csf13_storage"] is weights.experts.experts._impl.nvfp4_csf.first.scales
    else:
        assert kwargs["s13"] is rep.w13_sfb and kwargs["s2"] is rep.w2_sfb
    assert engine.keep == [kwargs]


def test_register_refuses_a_hot_map_that_is_not_int32(monkeypatch):
    engine, weights, _, _, _ = _engine(monkeypatch, csf=True)
    weights.experts.hot_map = torch.arange(384, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="hot_map must be contiguous int32"):
        engine.register(LAYER, weights)


def test_register_refuses_a_foreign_expert_plane(monkeypatch):
    engine, weights, rep, _, _ = _engine(monkeypatch, csf=True)
    rep.w13_rp = torch.zeros(8, 4).t()  # non-contiguous: register's .contiguous() would hand the kernel a copy
    with pytest.raises(RuntimeError, match="w13 is not b12x's resident plane"):
        engine.register(LAYER, weights)
    assert engine.ext.layers == []


@torch.library.custom_op("flare_test::brrr_layer_tail_old_schema",
                         mutates_args=("attn_post", "attn_comb", "attn_pre", "y", "resid1", "post1", "comb1", "pre1"))
def _old_schema(handle: int, slot: int, o: torch.Tensor, positions: torch.Tensor, frequencies: torch.Tensor,
                resid0: torch.Tensor, attn_post: torch.Tensor, attn_comb: torch.Tensor, attn_pre: torch.Tensor,
                y: torch.Tensor, resid1: torch.Tensor, post1: torch.Tensor, comb1: torch.Tensor,
                pre1: torch.Tensor) -> None:
    raise AssertionError("never run")


@_old_schema.register_fake
def _(*args):
    pass


def _functionalized(first, second):
    from torch._functorch.aot_autograd import aot_function

    captured = []

    class Captured(Exception):
        pass

    def compiler(graph, inputs):
        captured.append(graph)
        raise Captured  # never run: the real op needs the kernel extension

    def forward(residual, pre_mix, positions, frequencies):
        for layer in (first, second):
            layer.brrr.frequencies = frequencies  # a graph input here (fake mode refuses real constants)
        return _forward([first, second])(residual, pre_mix, positions)

    with pytest.raises(Captured):
        aot_function(forward, fw_compiler=compiler)(*_inputs(6), torch.ones(32))
    return captured[0].graph


def _two_layers():
    first, second = _layer(LAYER, _tp3()), _layer(PARKED_LAYER, _tp3(), carry=True)
    for layer in (first, second):
        _forbid(layer)
        _drive(layer)
    return first, second


def test_brrr_launch_functionalizes_once_per_layer(stubs):
    """What AOTAutograd hands Inductor (aot_function, the functionalized inference graph): the launch as one
    auto_functionalized node per layer whose only mutated bases are the five graph-local outputs. The attention mHC
    inputs (post, comb, attn_mix) are read-only arguments, so nothing clones or copies them around the launch."""
    schema = {argument.name: argument for argument in OP._schema.arguments}
    assert [name for name, argument in schema.items() if argument.alias_info is not None
            and argument.alias_info.is_write] == ["y", "resid1", "post1", "comb1", "pre1"]
    graph = _functionalized(*_two_layers())
    functional = [node for node in graph.nodes if node.op == "call_function"
                  and "auto_functionalized" in str(node.target)]
    assert len(functional) == 2 and all(node.args[0] is OP for node in functional), \
        [str(node.target) for node in graph.nodes if node.op == "call_function"]
    for node in functional:
        bases = node.kwargs["_all_bases"]
        assert len(bases) == 5 and all(str(base.target).startswith("aten.empty") for base in bases)
        assert all(node.kwargs[name] not in bases for name in ("attn_post", "attn_comb", "attn_pre"))
    assert not [node for node in graph.nodes if node.op == "call_function"
                and any(word in str(node.target) for word in ("clone", "copy"))]


def test_negative_control_old_schema_makes_the_attention_inputs_mutated_bases(stubs, monkeypatch):
    monkeypatch.setattr(brrr, "brrr_layer_tail", _old_schema)
    graph = _functionalized(*_two_layers())
    functional = [node for node in graph.nodes if node.op == "call_function"
                  and "auto_functionalized" in str(node.target)]
    assert len(functional) == 2 and all(len(node.kwargs["_all_bases"]) == 8 for node in functional)
