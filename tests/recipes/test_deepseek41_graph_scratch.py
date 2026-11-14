# SPDX-License-Identifier: Apache-2.0
"""TP3 graph-scratch savings on the functional forwards (S5 of the 2026-11-14 port, CPU only).

``engram_chunk_rows`` runs the Engram projection, its all-reduce and the mix in row chunks inside wide graphs, so the
Engram layer's Inductor-planned pool and its declaration operands hold one chunk of ``local``/``partial``/``projected``.
``share_engram_residual`` declares ``engram_residual`` on ``ffn_hc.residual``: preparation memory only, because the
planned serving graphs already place the Engram output on dead storage (pinned below by compiling a prefill stretch
with Inductor's memory planner on fake tensors).
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.components.parallel import TensorParallelShard
from ignition.recipes.deepseek41 import engram as engram_module
from ignition.recipes.deepseek41.decoder import Scratch
from ignition.recipes.deepseek41.tp3_config import TP3Config

MiB = 1 << 20


# --- configuration ---

def test_switches_default_off_parse_from_the_environment_and_are_ported():
    assert TP3Config().engram_chunk_rows == 0 and TP3Config().share_engram_residual is False
    config = TP3Config.from_environment({"FLARE_DS41_ENGRAM_CHUNK_ROWS": "1024",
                                         "FLARE_DS41_SHARE_ENGRAM_RESIDUAL": "1"})
    assert config.engram_chunk_rows == 1024 and config.share_engram_residual is True
    assert config.unported() == {} and config.require_ported() is config
    record = config.to_metadata()  # tp3-config.json record unchanged from tp3-main
    assert record["engram_chunk_rows"] == 1024 and record["share_engram_residual"] is True
    assert "engram_chunk_rows=1024" in config.describe() and "share_engram_residual=True" in config.describe()


@pytest.mark.parametrize("env,error", [
    ({"FLARE_DS41_ENGRAM_CHUNK_ROWS": "1000"}, "multiple of 128"),
    ({"FLARE_DS41_ENGRAM_CHUNK_ROWS": "-128"}, "non-negative integer"),
])
def test_invalid_chunk_rows_raise(env, error):
    with pytest.raises(ValueError, match=error):
        TP3Config.from_environment(env)


# --- declaration operands (preparation memory) ---

def _bytes(scratch):
    bases = {}

    def walk(value):
        if isinstance(value, torch.Tensor):
            base = value if value._base is None else value._base
            bases[id(base)] = base.nbytes
        elif hasattr(value, "__dataclass_fields__"):
            for name in value.__dataclass_fields__:
                walk(getattr(value, name))
    walk(scratch)
    return sum(bases.values())


def _prefill(**options):
    return Scratch.allocate(4096, 1, 3, "meta", candidate_tokens=4096, **options)


def test_default_declarations_are_the_base_layout():
    base = _prefill()
    assert _bytes(base) == _bytes(_prefill(engram_chunk_rows=0, share_engram_residual=False)) == 2_105_507_328
    assert base.engram.partial.shape == base.engram.projected.shape == (4096, 25600)
    assert base.engram.local.shape == (4096, 2048)
    assert base.engram_residual.untyped_storage() is not base.ffn_hc.residual.untyped_storage()


def test_chunked_declarations_hold_one_chunk_and_save_312_mib_per_prefill_scratch():
    chunked = _prefill(engram_chunk_rows=1024)
    engram = chunked.engram
    assert engram.mask.shape == (4096,) and engram.gathered.shape == (4096, 6144)
    assert engram.local.shape == (1024, 2048)
    assert engram.partial.shape == engram.projected.shape == (1024, 25600)
    # partial and projected 200 -> 50 MiB each, local 16 -> 4 MiB (world 3)
    assert _bytes(_prefill()) - _bytes(chunked) == 312 * MiB
    # Capacity views: wide graphs keep one chunk, narrow ones every row (one pass).
    assert chunked.rows(2048, 1).engram.partial.shape == (1024, 25600)
    assert chunked.rows(512, 1).engram.partial.shape == (512, 25600)
    assert Scratch.allocate(192, 64, 3, "meta", engram_chunk_rows=1024).engram.partial.shape == (192, 25600)
    # Unsharded: chunks project row views of ``gathered`` (no chunk copy); partial is the projected value.
    single = Scratch.allocate(4096, 1, 1, "meta", candidate_tokens=4096, engram_chunk_rows=1024).engram
    assert single.local.shape == (1024, 6144) and single.local._base is single.gathered
    assert single.partial is single.projected


def test_shared_residual_declaration_aliases_ffn_residual_in_every_view_and_saves_160_mib():
    shared = _prefill(share_engram_residual=True)
    assert shared.engram_residual is shared.ffn_hc.residual

    def base(value):
        return value if value._base is None else value._base
    view = shared.rows(512, 1)
    assert view.engram_residual.shape == view.ffn_hc.residual.shape == (512, 4, 5120)
    assert base(view.engram_residual) is base(view.ffn_hc.residual) is shared.ffn_hc.residual
    for scratch in (shared, view):
        for other in (scratch.residual, scratch.attention_hc.residual, scratch.mlp_output):
            assert base(other) is not shared.ffn_hc.residual
    assert _bytes(_prefill()) - _bytes(shared) == 160 * MiB
    # Both switches: 472 MiB less preparation memory per 4096-row prefill scratch (tp3-main: 465 MiB of graph scratch).
    assert _bytes(_prefill()) - _bytes(_prefill(engram_chunk_rows=1024, share_engram_residual=True)) == 472 * MiB


# --- chunked Engram through the real PreparedEngram (functional stand-ins) ---

class _Projection:
    created = []

    def __init__(self, source, weight, output, *, session, name, workspace, activation_mode, examples=None,
                 example_role=None):
        assert output.shape == (source.shape[0], 25600)
        self.rows, self.name, self.role, self.calls = source.shape[0], name, example_role, 0
        self.width = source.shape[1]
        _Projection.created.append(self)

    def functional(self, *, session):
        return self

    def __call__(self, source):
        assert source.shape == (self.rows, self.width)
        self.calls += 1
        repeats = -(-25600 // source.shape[1])
        return (source.float().repeat(1, repeats)[:, :25600] * 0.25 + source.float()[:, :1]).bfloat16()


class _Reduce:
    created = []

    def __init__(self, source, output, *, runtime, session, coordinator, name):
        self.rows, self.name = source.shape[0], name
        _Reduce.created.append(self)

    def functional(self, *, session):
        return self

    def __call__(self, source):
        assert source.shape == (self.rows, 5120)
        return (source.float() * 3).bfloat16()  # three ranks holding the same partial


class _Lookup:
    def __init__(self, inputs, gathered, live, **_):
        self.shape = tuple(gathered.shape)

    def functional(self, *, session):
        return self

    def __call__(self, source, count):
        rows = source.shape[0]
        assert (rows, 6144) == self.shape
        return (source.view(rows, -1)[:, :6144].float() * 0.5 + torch.arange(rows)[:, None] * 0.01).bfloat16()


class _Plan:
    def __init__(self, caps, invocation):
        self.caps = caps

    def request(self, name, prepare_call):
        return name


def _mix(state, projected, norm_weight, *, eps, plan, out, token_mask):
    assert state.shape[0] <= plan.caps.max_tokens and projected.shape[0] == state.shape[0] == out.shape[0]
    out.copy_(state.float() + projected[:, :state.shape[1]].float() * token_mask[:, None] * norm_weight[None])
    return out


def _engram(monkeypatch, tokens, chunk_rows, world):
    declared = []
    _Projection.created, _Reduce.created = [], []
    monkeypatch.setattr(engram_module, "QuantizedProjection", _Projection)
    monkeypatch.setattr(engram_module, "PreparedChunkedAllReduce", _Reduce)
    monkeypatch.setattr(engram_module, "PreparedEngramLookup", _Lookup)
    monkeypatch.setattr(engram_module, "declare", lambda session, requests: declared.extend(requests))
    monkeypatch.setattr(engram_module, "hyperconnection", SimpleNamespace(
        Caps=lambda **caps: SimpleNamespace(**caps), plan=_Plan, run_engram_mix=_mix))
    s = Scratch.allocate(tokens, 1, world, "cpu", candidate_tokens=tokens, engram_chunk_rows=chunk_rows)
    shard = TensorParallelShard(0, world)
    columns = torch.tensor([head * 256 + column for head in range(shard.rank, 24, world) for column in range(256)])
    tables = SimpleNamespace(inputs={1: None}, token_map=(), geometry=None, readiness_runtime=None,
                             columns=columns, shard=shard)
    metadata = SimpleNamespace(token_ids=torch.arange(tokens) * 7 % 129300, positions=torch.arange(tokens) - 2,
                               live_counts=torch.tensor([tokens]))
    pool = SimpleNamespace(config=SimpleNamespace(text={"rms_norm_eps": 1e-6}))
    weights = {"projection": None, "weight": torch.linspace(0.5, 1.5, 20480)}
    prepared = engram_module.PreparedEngram(1, weights, tables, s.residual, s.engram_residual, metadata, pool,
        scratch=s.engram, session=None, name="prefill.layer1.engram", workspace=None,
        runtime=object() if world > 1 else None, examples={})
    source = torch.randn(s.residual.shape, generator=torch.Generator().manual_seed(11)).bfloat16()
    return prepared, metadata, source, declared


@pytest.mark.parametrize("world", [3, 1])
@pytest.mark.parametrize("tokens,chunk_rows,parts", [
    (12, 4, 3), (10, 4, 3), (8, 8, 1), (8, 16, 1), (1300, 512, 3), (1024, 384, 3), (640, 128, 5)])
def test_chunked_engram_is_bit_identical_to_one_pass(monkeypatch, world, tokens, chunk_rows, parts):
    """PLUMBING ONLY. The stand-in kernels are row-independent CPU functions, so this proves the chunk orchestration
    (bounds, row slices of source/mask/output, column selection, one op pair per height, tail naming) and nothing
    about the real kernels: a b12x projection or mix that misbehaves at a row offset or a tail height would pass
    here. GPU numerics need the chunked-vs-one-pass export A/B (consistency.py, lavd; S5-REVIEW risk 1)."""
    whole, metadata, source, declared_whole = _engram(monkeypatch, tokens, 0, world)
    reference = whole(source, metadata)
    assert not whole.chunked and [p.name for p in _Projection.created] == ["prefill.layer1.engram.projection"]
    chunked, metadata, source_again, declared = _engram(monkeypatch, tokens, chunk_rows, world)
    assert torch.equal(source, source_again)
    result = chunked(source_again, metadata)
    assert result.shape == source.shape and result.data_ptr() != source.data_ptr()
    assert torch.equal(result, reference)
    assert len(chunked.bounds) == parts and chunked.chunked == (parts > 1)
    assert chunked.bounds[0][0] == 0 and chunked.bounds[-1][1] == tokens
    assert all(a[1] == b[0] for a, b in zip(chunked.bounds, chunked.bounds[1:]))
    names = [p.name for p in _Projection.created]
    roles = [p.role for p in _Projection.created]
    if tokens % chunk_rows and parts > 1:  # a shorter tail chunk: its own pair, its own declaration role
        assert names == ["prefill.layer1.engram.projection", "prefill.layer1.engram.projection.tail"]
        assert roles == ["engram.projection", "engram.projection.tail"]
        assert [p.rows for p in _Projection.created] == [chunk_rows, tokens % chunk_rows]
    else:
        assert names == ["prefill.layer1.engram.projection"] and roles == ["engram.projection"]
    assert sum(p.calls for p in _Projection.created) == parts
    assert [r.name for r in _Reduce.created] == ([name.replace("projection", "reduce") for name in names]
                                                 if world > 1 else [])
    assert declared == declared_whole == ["prefill.layer1.engram.mix"]
    assert chunked.mix_plan.caps.max_tokens == min(tokens, chunk_rows)


def test_masked_rows_and_row_order_reach_every_chunk(monkeypatch):
    """Negative control: a mix that ignored chunk offsets (rows of chunk 0 for every chunk) would differ."""
    prepared, metadata, source, _ = _engram(monkeypatch, 10, 4, 3)
    good = prepared(source, metadata)
    swapped = SimpleNamespace(**{**vars(metadata), "positions": metadata.positions.flip(0)})
    assert not torch.equal(prepared(source, swapped), good)  # mask rows are taken per chunk
    prepared.bounds = ((0, 4), (0, 4), (8, 10))  # broken bounds
    assert not torch.equal(prepared(source, metadata), good)


@pytest.mark.parametrize("chunk_rows", [0, 4])
def test_engram_forward_refuses_rows_other_than_declared(monkeypatch, chunk_rows):
    prepared, metadata, source, _ = _engram(monkeypatch, 10, chunk_rows, 3)
    prepared(source, metadata)
    wider = torch.cat((source, source[:2]))
    wide_metadata = SimpleNamespace(token_ids=torch.arange(12), positions=torch.arange(12),
                                    live_counts=torch.tensor([12]))
    with pytest.raises(ValueError, match="Engram forward got 12 rows; it was declared for 10"):
        prepared(wider, wide_metadata)
    with pytest.raises(ValueError, match="got 8 rows"):
        prepared(source[:8], SimpleNamespace(token_ids=metadata.token_ids[:8], positions=metadata.positions[:8],
                                             live_counts=torch.tensor([8])))


def test_engram_rejects_inconsistent_chunk_scratch(monkeypatch):
    _engram(monkeypatch, 8, 4, 3)
    s = Scratch.allocate(8, 1, 3, "cpu", candidate_tokens=8, engram_chunk_rows=4)
    bad = engram_module.EngramScratch(s.engram.mask, s.engram.gathered, s.engram.local[:2], s.engram.partial,
                                      s.engram.projected)
    tables = SimpleNamespace(inputs={1: None}, token_map=(), geometry=None, readiness_runtime=None,
                             columns=torch.arange(2048), shard=TensorParallelShard(0, 3))
    metadata = SimpleNamespace(token_ids=torch.zeros(8, dtype=torch.int64), live_counts=torch.ones(1))
    pool = SimpleNamespace(config=SimpleNamespace(text={"rms_norm_eps": 1e-6}))
    with pytest.raises(ValueError, match="contiguous lookup and projection geometry"):
        engram_module.PreparedEngram(1, {"projection": None, "weight": torch.ones(20480)}, tables, s.residual,
            s.engram_residual, metadata, pool, scratch=bad, session=None, name="x", workspace=None, runtime=object())
    with pytest.raises(ValueError, match="non-negative integer"):
        engram_module.EngramScratch.allocate(8, "cpu", chunk_rows=-1)


# --- Inductor-planned pool of a prefill stretch around the Engram layer ---

@torch.library.custom_op("flare_test_s5::kernel", mutates_args=("outs",))
def _kernel(ins: list[torch.Tensor], outs: list[torch.Tensor]) -> None:
    pass


class _PlannedOp:
    """Functional stand-in: allocates its output inside the forward and fills it with an opaque kernel."""

    def __init__(self, *args, **kwargs):
        target = args[2] if len(args) > 2 else args[1]
        self.shape = tuple(target.shape)

    def functional(self, *, session):
        return self

    def __call__(self, source, *_):
        out = torch.empty(self.shape, dtype=torch.bfloat16, device=source.device)
        _kernel([source], [out])
        return out


class _PlannedTorch:
    """``torch`` as engram.py sees it, with ``zeros_like`` optionally redirected (tp3-main's hand alias)."""

    alias = None

    def __getattr__(self, name):
        return getattr(torch, name)

    def zeros_like(self, value):
        return torch.zeros_like(value) if self.alias is None else self.alias.zero_()


def _mhc(incoming, tokens):
    outputs = (torch.empty(tokens, 4, 5120, dtype=torch.bfloat16), torch.empty(tokens, 4, dtype=torch.float32),
               torch.empty(tokens, 4, 4, dtype=torch.float32), torch.empty(tokens, 5120, dtype=torch.bfloat16))
    _kernel(list(incoming), list(outputs))
    return outputs


def _planned_pool(monkeypatch, tokens, chunk_rows, *, share=False):
    from torch._inductor import config as inductor_config
    from torch._inductor.codegen.memory_planning import MemoryPlanner
    from torch._inductor.compile_fx import compile_fx
    from torch._inductor.virtualized import V
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx

    from ignition import runtime_patches

    runtime_patches.apply()  # the memory-planning patch that records _flare_pool_sizes
    sizes, plan = [], MemoryPlanner.plan

    def record(self, lines):
        result = plan(self, lines)
        sizes.append(tuple(int(size) for _, size in V.graph._flare_pool_sizes))
        return result
    monkeypatch.setattr(MemoryPlanner, "plan", record)
    shim = _PlannedTorch()
    for name in ("QuantizedProjection", "PreparedChunkedAllReduce"):
        monkeypatch.setattr(engram_module, name, _PlannedOp)
    monkeypatch.setattr(engram_module, "PreparedEngramLookup",
                        lambda inputs, gathered, live, **_: _PlannedOp(None, gathered))
    monkeypatch.setattr(engram_module, "declare", lambda session, requests: None)
    monkeypatch.setattr(engram_module, "torch", shim)

    def mix(state, projected, norm_weight, *, eps, plan, out, token_mask):
        _kernel([state, projected, norm_weight, token_mask], [out])
        return out
    monkeypatch.setattr(engram_module, "hyperconnection", SimpleNamespace(
        Caps=lambda **caps: SimpleNamespace(**caps), run_engram_mix=mix,
        plan=lambda caps, invocation: SimpleNamespace(caps=caps, request=lambda **_: None)))
    s = Scratch.allocate(tokens, 1, 3, "meta", candidate_tokens=tokens, engram_chunk_rows=chunk_rows)
    mode = FakeTensorMode()
    columns = mode.from_tensor(torch.tensor([head * 256 + column for head in range(0, 24, 3) for column in range(256)]))
    tables = SimpleNamespace(inputs={1: None}, token_map=(), geometry=None, readiness_runtime=None,
                             columns=columns, shard=TensorParallelShard(0, 3))
    metadata = SimpleNamespace(token_ids=torch.zeros(tokens, dtype=torch.int64, device="meta"),
                               live_counts=torch.zeros(1, dtype=torch.int32, device="meta"))
    pool = SimpleNamespace(config=SimpleNamespace(text={"rms_norm_eps": 1e-6}))
    prepared = engram_module.PreparedEngram(1, {"projection": None, "weight": mode.from_tensor(torch.ones(20480))},
        tables, s.residual, s.engram_residual, metadata, pool, scratch=s.engram, session=None, name="l1.engram",
        workspace=None, runtime=object())

    def forward(previous, branch, token_ids, positions, live_counts):
        # Layer 0's FFN mHC and post (its ffn residual is dead after the post), layer 1 = Engram, its attention mHC.
        ffn_residual, post, comb, _ = _mhc((previous,), tokens)
        residual = torch.empty(tokens, 4, 5120, dtype=torch.bfloat16)
        _kernel([branch, ffn_residual, post, comb], [residual])
        shim.alias = ffn_residual if share else None
        output = prepared(residual, SimpleNamespace(token_ids=token_ids, positions=positions,
                                                    live_counts=live_counts))
        attention = _mhc((output,), tokens)
        return attention[0], attention[3]

    with mode:
        inputs = [torch.empty(tokens, 4, 5120, dtype=torch.bfloat16), torch.empty(tokens, 5120, dtype=torch.bfloat16),
                  torch.empty(tokens, dtype=torch.int64), torch.empty(tokens, dtype=torch.int64),
                  torch.empty(1, dtype=torch.int32)]
    graph = make_fx(forward, tracing_mode="fake", _allow_non_fake_inputs=True)(*inputs)
    with inductor_config.patch(memory_planning=True, memory_pool="combined", compile_threads=1), mode:
        compile_fx(graph, inputs)
    (planned,) = sizes[-1]
    return planned


def test_planned_pool_drops_with_chunking_and_the_hand_alias_saves_nothing(monkeypatch):
    """Inductor plans the stretch's combined pool from live ranges (stand-in kernels: no workspaces).

    At 4096 rows the one-pass peak is the reduce (residual 160 + partial 200 + projected 200 MiB); 1024-row chunks
    take 132 MiB off it. tp3-main's shared residual (the Engram mix writing into the previous layer's dead ffn
    residual by hand) plans the same pool as the fresh ``zeros_like`` output: Inductor already reuses that storage.
    """
    one_pass = _planned_pool(monkeypatch, 4096, 0)
    hand_alias = _planned_pool(monkeypatch, 4096, 0, share=True)
    chunked = _planned_pool(monkeypatch, 4096, 1024)
    chunked_alias = _planned_pool(monkeypatch, 4096, 1024, share=True)
    assert one_pass == hand_alias == 560 * MiB
    assert chunked == chunked_alias == 428 * MiB
    assert _planned_pool(monkeypatch, 2048, 0) - _planned_pool(monkeypatch, 2048, 1024) == 20 * MiB
