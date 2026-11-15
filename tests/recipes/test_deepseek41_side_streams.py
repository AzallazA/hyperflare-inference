# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DS41 TP3 side streams (index chain, Engram, DSpark context): fork / launch / join order, switch-off equivalence,
operand lifetimes and the TP3 switches, on CPU with fake CUDA streams (components/side_streams.py)."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import dspark as dspark_module
from ignition.components import side_streams
from ignition.recipes.deepseek41 import attention as attention_module
from ignition.recipes.deepseek41 import decoder as decoder_module
from ignition.recipes.deepseek41 import engram as engram_module
from ignition.recipes.deepseek41.tp3_config import TP3Config


# --------------------------------------------------------------------------------------------------------------------
# Fake CUDA streams: every kernel stand-in logs (stream, name); waits log too.


class _Clock:
    def __init__(self):
        self.log = []
        self.current = "main"


class _Stream:
    def __init__(self, clock, name):
        self.clock, self.name = clock, name

    def wait_stream(self, other):
        self.clock.log.append((self.name, f"wait-stream:{other.name}"))

    def wait_event(self, event):
        self.clock.log.append((self.name, f"wait-event:{event.name}"))


@pytest.fixture
def clock(monkeypatch):
    clock = _Clock()
    streams = {}
    events = []

    def external(handle, device=None):
        return streams.setdefault(handle, _Stream(clock, f"side{handle}"))

    class Context:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.previous, clock.current = clock.current, self.stream.name

        def __exit__(self, *exc):
            clock.current = self.previous

    class Event:
        def __init__(self):
            self.name = f"e{len(events)}"
            events.append(self)

        def record(self, stream):
            clock.log.append((stream.name, f"record:{self.name}"))

    monkeypatch.setattr(torch.cuda, "ExternalStream", external)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: _Stream(clock, clock.current))
    monkeypatch.setattr(torch.cuda, "stream", Context)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    # one stable fake stream handle per purpose
    fake_handles = {}
    monkeypatch.setattr(side_streams.Task, "stream_handle",
                        lambda self, device: fake_handles.setdefault(self.purpose, 100 + len(fake_handles)))
    return clock


def kernel(clock, name):
    """A stand-in launch: logs the stream it runs on."""
    clock.log.append((clock.current, name))


def names(clock, stream=None):
    return [name for where, name in clock.log if stream is None or where == stream]


# --------------------------------------------------------------------------------------------------------------------
# side_streams: launch / join


def test_launch_forks_runs_the_task_on_the_side_stream_and_join_waits_for_its_event(clock):
    seen = []
    task = side_streams.register(lambda inputs, outputs: (kernel(clock, "task"), seen.append((inputs, outputs))),
                                 "unit")
    anchor, source, out = torch.zeros(2), torch.ones(2), torch.empty(2)
    task.launch(anchor, [source], [out])
    kernel(clock, "main-work")
    task.join(source, [source], [out])
    side = "side100"
    assert clock.log == [(side, "wait-stream:main"), (side, "task"), (side, "record:e0"),
                         ("main", "main-work"), ("main", "wait-event:e0")]
    (inputs, outputs), = seen
    assert inputs[0] is source and outputs[0] is out


def test_join_waits_only_for_its_own_task_and_needs_a_launch(clock):
    first = side_streams.register(lambda i, o: kernel(clock, "first"), "unit")
    second = side_streams.register(lambda i, o: kernel(clock, "second"), "unit")
    a, b = torch.empty(1), torch.empty(1)
    first.launch(a, [], [torch.empty(1)])
    second.launch(b, [], [torch.empty(1)])
    first.join(a, [], [torch.empty(1)])
    assert clock.log[-1] == ("main", "wait-event:e0")  # not the stream tail (e1)
    with pytest.raises(RuntimeError, match="without a preceding launch"):
        first.join(a, [], [torch.empty(1)])  # each launch is joined once


def test_outputs_may_not_alias_other_operands():
    task = side_streams.register(lambda i, o: None, "unit")
    x = torch.empty(2)
    with pytest.raises(ValueError, match="output may not"):
        task.launch(torch.empty(1), [x], [x])
    with pytest.raises(ValueError, match="output may not"):
        task.join(x, [], [x])


def test_ops_are_registered_with_fake_kernels():
    assert hasattr(torch.ops.flare, "side_stream_launch") and hasattr(torch.ops.flare, "side_stream_join")
    # Inductor traces them through their fake implementations (no launch, no allocation).
    with torch._subclasses.FakeTensorMode():
        side_streams._launch(torch.empty(1), [torch.empty(1)], [torch.empty(1)], 1, 1)
        side_streams._join(torch.empty(1), [torch.empty(1)], [torch.empty(1)], 1)


# --------------------------------------------------------------------------------------------------------------------
# TP3 switches


def test_switches_default_off_and_parse():
    base = TP3Config()
    assert base.index_side_stream is False and base.engram_side_layers == frozenset()
    assert base.context_side_stream is False
    assert TP3Config.from_environment({}) == base
    config = TP3Config.from_environment({"FLARE_DS41_INDEX_SIDE_STREAM": "1",
                                         "FLARE_DS41_ENGRAM_SIDE_LAYERS": "14",
                                         "FLARE_DS41_CONTEXT_SIDE_STREAM": "1"})
    assert config.index_side_stream and config.context_side_stream and config.engram_side_layers == {14}
    record = config.to_metadata()
    assert record["index_side_stream"] is True and record["engram_side_layers"] == "14"
    assert record["context_side_stream"] is True
    assert TP3Config().to_metadata()["engram_side_layers"] == ""
    assert "index_side_stream=True" in config.describe() and "engram_side_layers='14'" in config.describe()
    # flags: only "1" is on (release convention)
    assert not TP3Config.from_environment({"FLARE_DS41_INDEX_SIDE_STREAM": "0"}).index_side_stream


def test_switch_validation():
    with pytest.raises(TypeError):
        TP3Config(index_side_stream=1)
    with pytest.raises(ValueError):
        TP3Config(engram_side_layers={-1})
    with pytest.raises(ValueError, match="engram_side_layers names layers 40"):
        TP3Config(engram_side_layers={40}).check_layers(40)


# --------------------------------------------------------------------------------------------------------------------
# #4: index chain + gates on the side stream (PreparedAttention._run, PreparedCompressor gates hook)

ROWS = 3


class _Projection:
    """Fake functional projection: __call__ (main path) and allocate/run_into (side path) log the same kernel."""

    def __init__(self, clock, name, width, dtype=torch.bfloat16, bases=False):
        self.clock, self.name, self.width, self.dtype = clock, name, width, dtype
        self.allocates_bases = bases

    def __call__(self, source, encoded_bases=None):
        out = torch.empty(source.shape[0], self.width, dtype=self.dtype)
        self.run_into(source, out, None)
        return out

    def allocate(self, device):
        out = torch.empty(ROWS, self.width, dtype=self.dtype)
        return out, torch.empty(16, dtype=torch.uint8), (torch.empty(4), torch.empty(4), torch.zeros(4)) \
            if self.allocates_bases else None

    def run_into(self, source, output, workspace=None, *, encoded_bases=None):
        kernel(self.clock, self.name)
        output.fill_(float(len(self.name)))
        return output


class _UnquantizedProjection(_Projection):
    def allocate(self, device):
        return torch.empty(ROWS, self.width, dtype=self.dtype)

    def __call__(self, source):
        out = torch.empty(source.shape[0], self.width, dtype=self.dtype)
        return self.run_into(source, out)

    def run_into(self, source, output):
        return super().run_into(source, output)


def _fake_attention(clock, *, kv_source, ratio=2):
    attention = object.__new__(attention_module.PreparedAttention)
    attention.mode, attention.ratio, attention.owns_window_metadata = "verify", ratio, False
    attention.layer, attention.candidate_source = 2, 20  # at or below the candidate source: no candidates read
    attention.window = torch.zeros(ROWS, 528)
    qkv = type("Q", (), {"encoded_layout": None,
                         "__call__": lambda self, source, encoded_bases=None: (kernel(clock, "qkv_a"),
                                                                               torch.zeros(ROWS, 1792))[1]})()
    attention.projections = {"qkv_a": qkv,
                             "q_b": lambda q, encoded_bases=None: (kernel(clock, "q_b"), torch.zeros(ROWS, 8 * 512))[1]}
    attention.norm_quant, attention.concurrent_query = None, None
    attention.query_norm = lambda x: (kernel(clock, "q_norm"), x.clone())[1]
    attention.window_norm = lambda x: (kernel(clock, "kv_norm"), x.clone())[1]
    attention.query_shape = (ROWS, 8, 512)
    attention.rotary = SimpleNamespace(apply=lambda tensor, position, compressed, inverse=False: kernel(
        clock, f"rope{tuple(tensor.shape)}"))
    attention.window_writer = lambda *args, **kwargs: kernel(clock, "window_write")
    attention.attention = lambda *args, **kwargs: (kernel(clock, "attend"), torch.zeros(ROWS, 8, 512))[1]
    attention.output_projection = lambda *args: torch.zeros(ROWS, 5120)
    attention.index_query = _Projection(clock, "index_q", 4096)
    attention.index_weight = _UnquantizedProjection(clock, "index_weights", 32)
    attention.external_tail = True

    class Selector:
        def __call__(self, index_query, index_weights, *args, **kwargs):
            kernel(clock, "selector")
            assert index_query.shape == (ROWS, 32, 128) and index_weights.shape == (ROWS, 32)
            return torch.zeros(ROWS, 512, dtype=torch.int32)
    attention.selectors = [Selector()]
    attention.global_kv = None
    if kv_source:
        compressor = SimpleNamespace(ratio=ratio, project_gates=_UnquantizedProjection(clock, "gates", 512, torch.float32))

        def global_kv(source, indexer, metadata, geometry, *, gates=None):
            if gates is None:
                kernel(clock, "gates")
                kernel(clock, "values")
            else:
                values = (kernel(clock, "values"), torch.zeros(ROWS, 512))[1]
                assert gates(values).dtype == torch.float32
            kernel(clock, "compress")
            kernel(clock, "index_key_write")
            return "records"
        attention.global_kv = SimpleNamespace(compressor=compressor)
        attention.global_kv_call = global_kv
        # PreparedAttention calls self.global_kv(...): give the namespace a __call__ through a small class
        attention.global_kv = type("G", (), {"compressor": compressor,
                                             "__call__": lambda self, *a, **k: global_kv(*a, **k)})()
    return attention


def _run(attention):
    geometry = SimpleNamespace(lengths=torch.zeros(ROWS, dtype=torch.int32), active_width=torch.zeros(1),
                               index_lengths=torch.zeros(ROWS, dtype=torch.int32))
    indices = SimpleNamespace(position=torch.zeros(ROWS, dtype=torch.int64), window_slots=None, window_indices=None,
                              window_lengths=None, compressed=lambda ratio: geometry)
    metadata = SimpleNamespace(state_rows=None, token_page_table=torch.zeros(ROWS, 1, dtype=torch.int32))
    source = torch.randn(ROWS, 5120).bfloat16()
    return attention._run(source, metadata, (None, None, None), indices=indices, window_cache=torch.zeros(ROWS, 528))


@pytest.mark.parametrize("kv_source", [True, False])
def test_index_side_off_keeps_the_serial_order(clock, kv_source):
    attention = _fake_attention(clock, kv_source=kv_source)
    assert attention.index_side is None  # class default: no side chain
    _run(attention)
    serial = ["qkv_a", "q_norm", "kv_norm", "q_b", "rope(3, 8, 512)", "rope(3, 512)", "window_write",
              *(["gates", "values", "compress", "index_key_write"] if kv_source else []),
              "index_q", "rope(3, 32, 128)", "index_weights", "selector", "attend"]
    # (q_b is serial here: no concurrent query chain in the fake)
    got = names(clock)
    assert all(stream == "main" for stream, _ in clock.log)
    assert sorted(got) == sorted(serial) and got.index("index_q") > got.index("window_write")


@pytest.mark.parametrize("kv_source", [True, False])
def test_index_side_runs_the_query_chain_and_gates_on_the_side_stream(clock, kv_source):
    attention = _fake_attention(clock, kv_source=kv_source)
    attention.index_side = attention_module.IndexSideChain(attention)
    assert (attention.index_side.gates_task is not None) is kv_source
    _run(attention)
    side = "side100"
    on_side = names(clock, side)
    expected_side = ["wait-stream:main", *(["gates", "record:e0", "wait-stream:main"] if kv_source else []),
                     "index_q", "rope(3, 32, 128)", "index_weights", f"record:e{1 if kv_source else 0}"]
    assert on_side == expected_side
    main = names(clock, "main")
    # the fork happens after the q/KV norm (query_low exists), before the KV path
    log = clock.log
    first_fork = log.index((side, "wait-stream:main"))
    assert log.index(("main", "q_norm")) < first_fork < log.index(("main", "window_write"))
    query_join = f"wait-event:e{1 if kv_source else 0}"
    assert main.index(query_join) < main.index("selector") < main.index("attend")
    assert main.index("window_write") < main.index(query_join)
    if kv_source:
        # gates join: after the values projection, before the compression kernel; the query join after the KV path
        assert main.index("values") < main.index("wait-event:e0") < main.index("compress")
        assert main.index("index_key_write") < main.index(query_join)
    assert "index_q" not in main and "gates" not in main


def test_index_side_join_keeps_every_side_input_alive_and_outputs_are_fresh(clock, monkeypatch):
    attention = _fake_attention(clock, kv_source=True)
    attention.index_side = chain = attention_module.IndexSideChain(attention)
    launches, joins = [], []
    real_launch, real_join = side_streams.Task.launch, side_streams.Task.join
    monkeypatch.setattr(side_streams.Task, "launch",
                        lambda self, a, i, o: (launches.append((self, a, list(i), list(o))), real_launch(self, a, i, o))[1])
    monkeypatch.setattr(side_streams.Task, "join",
                        lambda self, d, i, o: (joins.append((self, d, list(i), list(o))), real_join(self, d, i, o))[1])
    _run(attention)
    assert [task for task, *_ in launches] == [chain.gates_task, chain.query_task]
    assert [task for task, *_ in joins] == [chain.gates_task, chain.query_task]
    for (task, _, inputs, outputs), (joined, _, join_inputs, join_outputs) in zip(launches, joins):
        assert task is joined
        # lifetime: every tensor the side stream reads or writes is an operand of the join as well
        assert all(a is b for a, b in zip(inputs, join_inputs)) and len(inputs) == len(join_inputs)
        assert all(a is b for a, b in zip(outputs, join_outputs)) and len(outputs) == len(join_outputs)
        # poison: nothing the side stream writes is also one of its inputs
        assert not {id(t) for t in outputs} & {id(t) for t in inputs}
    query_inputs = launches[1][2]
    assert len(query_inputs) == 3  # query_low, source, position: all produced before the fork


def test_compressor_gates_hook_projects_values_first_then_joins(monkeypatch):
    calls = []
    compressor = object.__new__(attention_module.PreparedCompressor)
    compressor.ratio, compressor.layer, compressor.deferred = 2, 2, True
    compressor.pool = SimpleNamespace(region=lambda name: name)
    compressor.weight, compressor.plan = None, SimpleNamespace(handle=0)
    compressor.project_gates = lambda source: calls.append("gates") or torch.zeros(ROWS, 512)
    compressor.project_values = lambda source: calls.append("values") or torch.zeros(ROWS, 512)
    monkeypatch.setattr(attention_module, "gather_rows_three", lambda *a: calls.append("gather"))
    import b12x.attention.mla_compress._impl as impl
    monkeypatch.setattr(impl, "_run_ratio2", lambda *a: calls.append("compress"))
    metadata = SimpleNamespace(starts=torch.zeros(1, dtype=torch.int64), state_rows=None, query_start_loc=None,
                               live_counts=None, lengths=torch.zeros(1))
    source = torch.zeros(ROWS, 5120)
    compressor(source, metadata, None)
    assert calls == ["gather", "gates", "values", "compress"]  # base order unchanged
    calls.clear()
    compressor(source, metadata, None, gates=lambda values: calls.append("join") or torch.zeros(ROWS, 512))
    assert calls == ["gather", "values", "join", "compress"]
    compressor.ratio = 1
    with pytest.raises(ValueError, match="ratio-2"):
        compressor(source, metadata, None, gates=lambda values: None)


# --------------------------------------------------------------------------------------------------------------------
# #5: Engram on the side stream, started one layer early


class _Part:
    def __init__(self, clock):
        self.clock = clock

    def __call__(self, source, output):
        kernel(self.clock, f"all_reduce{tuple(source.shape)}")
        output.copy_(source)


def _fake_engram(clock, monkeypatch, tokens=ROWS):
    engram = object.__new__(engram_module.PreparedEngram)
    engram.chunked, engram.bounds, engram.sharded = False, ((0, tokens),), True
    engram.columns = torch.arange(0, 6144, 3)
    engram.projection = _Projection(clock, "engram_gemm", 25600, bases=True)
    engram.reduce = SimpleNamespace(bounds=[(0, tokens * 5)], parts=[_Part(clock)])
    engram.lookup = SimpleNamespace(output_layout=((tokens, 6144), torch.bfloat16), weights=None, scales=None,
                                    hashes=None, plan=SimpleNamespace(handle=7), handle=8,
                                    wait=SimpleNamespace(operands=lambda anchor: (torch.zeros(1, 8), torch.empty(1, 8)),
                                                         timings=SimpleNamespace(timestamps=None, frame=None),
                                                         interval=0, handle=0))
    engram.norm_weight, engram.eps, engram.mix_plan = None, 1e-6, None
    from ignition.components import engram as engram_component
    monkeypatch.setattr(engram_component, "_lookup", lambda anchor, *args: kernel(clock, "readiness+lookup"))
    monkeypatch.setattr(engram_module.hyperconnection, "run_engram_mix",
                        lambda state, projected, weight, **kw: kernel(clock, "mix"))
    return engram.enable_side()


def test_engram_side_task_runs_the_one_chunk_kernels_in_order(clock, monkeypatch):
    engram = _fake_engram(clock, monkeypatch)
    anchor = torch.zeros(ROWS, 4, 5120, dtype=torch.bfloat16)
    metadata = SimpleNamespace(live_counts=torch.tensor([ROWS, 1], dtype=torch.int32),
                               positions=torch.zeros(ROWS, dtype=torch.int64), token_ids=torch.zeros(ROWS, dtype=torch.int64))
    pending = engram.start_side(anchor, metadata)
    kernel(clock, "layer13-attention")
    attended = torch.zeros(ROWS, 8, 512)
    projected = engram.finish_side(pending, attended)
    kernel(clock, "layer13-brrr")
    assert projected.shape == (ROWS, 25600)
    engram(anchor, metadata, projected=projected)
    side = "side100"
    assert names(clock, side) == ["wait-stream:main", "readiness+lookup", "engram_gemm",
                                  f"all_reduce{(ROWS * 5, 5120)}", "record:e0"]
    assert names(clock, "main") == ["layer13-attention", "wait-event:e0", "layer13-brrr", "mix"]
    inputs, outputs = pending
    assert inputs[0] is anchor and outputs[6] is projected


def test_engram_side_requires_one_chunk_and_tensor_parallel(clock, monkeypatch):
    engram = _fake_engram(clock, monkeypatch)
    engram.chunked, engram.bounds = True, ((0, 2), (2, 3))
    with pytest.raises(ValueError, match="one chunk"):
        engram.enable_side()
    engram.chunked, engram.bounds, engram.sharded = False, ((0, 3),), False
    with pytest.raises(ValueError, match="tensor-parallel"):
        engram.enable_side()
    plain = object.__new__(engram_module.PreparedEngram)
    plain.bounds = ((0, ROWS),)
    with pytest.raises(ValueError, match="needs the side stream"):
        plain(torch.zeros(ROWS, 4, 8), SimpleNamespace(positions=torch.zeros(ROWS), token_ids=torch.zeros(ROWS)),
              projected=torch.zeros(1))


class _Layer:
    def __init__(self, engram=None):
        self.engram, self.engram_side, self.boundary = engram, None, None


class _Enable:
    def __init__(self):
        self.enabled = 0

    def enable_side(self):
        self.enabled += 1
        return self


def test_configure_engram_side_marks_the_previous_layer_in_decode_graphs_only():
    engram = _Enable()
    layers = [_Layer(), _Layer(_Enable()), *(_Layer() for _ in range(12)), _Layer(engram), _Layer()]
    decoder_module.configure_engram_side(layers, frozenset({14}), "prefill")
    assert all(layer.engram_side is None for layer in layers) and engram.enabled == 0
    decoder_module.configure_engram_side(layers, frozenset(), "verify")
    assert all(layer.engram_side is None for layer in layers)
    decoder_module.configure_engram_side(layers, frozenset({14}), "verify")
    assert layers[13].engram_side is engram and engram.enabled == 1
    assert [i for i, layer in enumerate(layers) if layer.engram_side is not None] == [13]
    for bad in ({0}, {2}, {99}):
        with pytest.raises(ValueError, match="not an Engram layer"):
            decoder_module.configure_engram_side(layers, frozenset(bad), "decode")


def _fake_layer(clock, *, starts_side, has_side_engram):
    layer = object.__new__(decoder_module.PreparedLayer)
    layer.probe = SimpleNamespace(record=lambda *a: None, router=lambda *a: None, full=lambda *a: None)
    layer.probe_index, layer.probe_mode, layer.no_engram = 13, "verify", False
    layer.engram = None
    if has_side_engram:
        layer.engram = type("E", (), {"side_task": object(),
            "__call__": lambda self, source, metadata, projected=None: (kernel(clock, f"engram(projected={projected is not None})"), source)[1]})()
    if starts_side:
        class Side:
            def start_side(self, anchor, metadata):
                kernel(clock, "fork-engram14")
                return "pending"

            def finish_side(self, pending, attended):
                kernel(clock, "join-engram14")
                return "projected14"
        layer.engram_side = Side()
    layer.attention_hc = lambda *incoming, pre_mix, pre_out: (kernel(clock, "attn_hc"), (None, None, None, None))[1]
    layer.attention = SimpleNamespace(_run=lambda *a, **k: (kernel(clock, "attention"),
                                                             (torch.zeros(1), None, None, {}))[1])
    layer.brrr = lambda *a: (kernel(clock, "brrr"), (torch.zeros(1), (None, None, None), None))[1]
    layer.post = None
    return layer


_INDICES = SimpleNamespace(position=None)


def test_layer_forks_before_its_attention_and_joins_before_its_tail(clock):
    layer = _fake_layer(clock, starts_side=True, has_side_engram=False)
    handoff = []
    layer._run((torch.zeros(1),), torch.zeros(1), None, _INDICES, None, engram_handoff=handoff)
    assert names(clock) == ["fork-engram14", "attn_hc", "attention", "join-engram14", "brrr"]
    assert handoff == ["projected14"]
    clock.log.clear()
    nxt = _fake_layer(clock, starts_side=False, has_side_engram=True)
    nxt._run((torch.zeros(1),), torch.zeros(1), None, _INDICES, None, engram_projected="projected14")
    assert names(clock)[0] == "engram(projected=True)"
    with pytest.raises(ValueError, match="must receive its projection"):
        nxt._run((torch.zeros(1),), torch.zeros(1), None, _INDICES, None)
    with pytest.raises(ValueError, match="hands its projection over"):
        layer._run((torch.zeros(1),), torch.zeros(1), None, _INDICES, None)


def test_layer_without_side_engram_is_unchanged(clock):
    layer = _fake_layer(clock, starts_side=False, has_side_engram=False)
    layer._run((torch.zeros(1),), torch.zeros(1), None, _INDICES, None)
    assert names(clock) == ["attn_hc", "attention", "brrr"]


# --------------------------------------------------------------------------------------------------------------------
# #7: DSpark context projections on the side stream


def _fake_context(clock, monkeypatch, *, side):
    context = object.__new__(dspark_module.PreparedContext)
    context.compact = False
    context.project = _Projection(clock, "main_proj", 5120, bases=True)
    context.context_projection = _Projection(clock, "context_kv", 1536)
    norm = SimpleNamespace(add_residual=False, stage_source=False, weight=None, eps=1e-6,
                           plan=SimpleNamespace(handle=3), zero_centered=False)
    context.norm = type("N", (), {**vars(norm), "__call__": lambda self, x: (kernel(clock, "main_norm"), x.clone())[1]})()
    monkeypatch.setattr(torch.ops.b12x, "hyperconnection_grouped_rmsnorm",
                        lambda *a: kernel(clock, "main_norm"), raising=False)
    context.rotary = SimpleNamespace(apply=lambda *a, **k: kernel(clock, "rope"))
    context.stages = [((lambda x: (kernel(clock, f"norm{i}"), x)[1]),
                       (lambda x, slots, i=i: kernel(clock, f"write{i}"))) for i in range(3)]
    context.side_task = side_streams.register(context._run_side, context.SIDE_PURPOSE) if side else None
    return context


def test_context_side_off_has_no_start_work(clock, monkeypatch):
    context = _fake_context(clock, monkeypatch, side=False)
    assert context.start(torch.zeros(ROWS, 15360)) is None and clock.log == []


def test_context_side_projects_beside_verification_and_writes_after_the_counts(clock, monkeypatch):
    m = SimpleNamespace(positions=torch.zeros(ROWS, dtype=torch.int64), starts=torch.zeros(1, dtype=torch.int64),
                        requests=torch.zeros(ROWS, dtype=torch.int64), state_rows=torch.zeros(1, dtype=torch.int64))
    serial = _fake_context(clock, monkeypatch, side=False)
    auxiliary = torch.zeros(ROWS, 15360, dtype=torch.bfloat16)
    serial(auxiliary, torch.ones(1, dtype=torch.int32), m)
    serial_order = names(clock)
    clock.log.clear()
    context = _fake_context(clock, monkeypatch, side=True)
    pending = context.start(auxiliary)
    kernel(clock, "verify+commit")
    counts = torch.ones(1, dtype=torch.int32)
    context(auxiliary, counts, m, pending=pending)
    side = "side100"
    assert names(clock, side) == ["wait-stream:main", "main_proj", "main_norm", "context_kv", "record:e0"]
    main = names(clock, "main")
    assert main[:2] == ["verify+commit", "wait-event:e0"]
    # the per-layer norms, rotations and ring writes are the serial path's, in the same order, after the join
    assert main[2:] == [name for name in serial_order if name not in ("main_proj", "main_norm", "context_kv")]
    inputs, outputs = pending
    assert inputs == [auxiliary] and all(t is not auxiliary for t in outputs)


def test_parallel_draft_hook_is_inert_for_contexts_without_start():
    import inspect

    from ignition.components import parallel_draft

    source = inspect.getsource(parallel_draft.PreparedParallelDraftRound.verify)
    assert 'getattr(self.context, "start", None)' in source
    assert "self.context(auxiliary, counts, metadata)" in source  # unchanged call when nothing started early
