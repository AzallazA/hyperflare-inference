# SPDX-License-Identifier: Apache-2.0
"""DS41 TP3 prefill all-reduce parts (CPU only): the part plans the decoder (``ar_part_rows``) and the Engram projection
(``engram_part_rows``) declare for every prefill graph height at 768 rows vs halving, the plumbing from TP3Config into
PreparedLayer / PreparedEngram, and the record keys of both switches and of ``mhc_pdl`` (B12X_MHC_PDL)."""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import collectives
from ignition.recipes.deepseek41 import decoder as decoder_module
from ignition.recipes.deepseek41 import engram as engram_module
from ignition.recipes.deepseek41.tp3_config import MHC_PDL_ENV, TP3Config

AR_ROWS = 1024           # FLARE_DS41_AR_ROWS of every TP3 export: the shared runtime takes 1024 x 5120 BF16
ENGRAM_CHUNK = 1024      # FLARE_DS41_ENGRAM_CHUNK_ROWS of every TP3 export
PREFILL = (4096, 3584, 3072, 2560, 2048, 1536, 1024, 512, 256)


class _Channel:
    def __init__(self, rows):
        self.capacity = rows * 5120 * 2

    def should_allreduce(self, tensor):
        return tensor.numel() * tensor.element_size() <= self.capacity


class _Runtime:
    """One-shot PCIe runtime stand-in sized like the TP3 export's (AR_ROWS rows of 5120 BF16)."""
    algorithm = "oneshot"

    def __init__(self, rows=AR_ROWS):
        self.channel = _Channel(rows)

    def for_stream(self):
        return self.channel


class _Part:
    made = []

    def __init__(self, source, output, **kwargs):
        assert source.shape == output.shape
        _Part.made.append((source.shape[0], kwargs["name"]))

    def functional(self, *, session):
        return self


@pytest.fixture
def real_parts(monkeypatch):
    """The real PreparedChunkedAllReduce part cutting with PreparedAllReduce stubbed (records each part)."""
    _Part.made = []
    monkeypatch.setattr(collectives, "PreparedAllReduce", _Part)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: False)
    return _Part.made


def _decoder_parts(rows, cap):
    source = torch.empty(rows, 5120, dtype=torch.bfloat16, device="meta")
    reduce = collectives.PreparedChunkedAllReduce(source, torch.empty_like(source), runtime=_Runtime(), session=None,
                                                  coordinator=None, name="prefill.layer0.attention_reduce",
                                                  max_part_rows=cap)
    return [end - begin for begin, end in reduce.bounds]


# Decoder attention/MLP reduction of a prefill graph (layers 0-19 reduce at graph rows): halving (151744) vs 768.
DECODER = {
    4096: ([1024] * 4, [768] * 5 + [256]),
    3584: ([896] * 4, [768] * 4 + [512]),
    3072: ([768] * 4, [768] * 4),
    2560: ([640] * 4, [768] * 3 + [256]),
    2048: ([1024] * 2, [768] * 2 + [512]),
    1536: ([768] * 2, [768] * 2),
    1024: ([1024], [768, 256]),
    512: ([512], [512]),
    256: ([256], [256]),
}


@pytest.mark.parametrize("rows", PREFILL)
def test_decoder_part_plan_at_768(real_parts, rows):
    halving, capped = DECODER[rows]
    assert _decoder_parts(rows, 0) == halving
    assert _decoder_parts(rows, 768) == capped
    assert sum(capped) == rows and max(capped) <= 768
    # every part size the 768 plan declares is one a 151744 graph already declares (256 / 512 / 768 rows)
    assert set(capped) <= {256, 512, 768}


@pytest.mark.parametrize("rows", [1, 6, 24, 48, 96, 128])
def test_decode_and_verify_widths_are_untouched(real_parts, rows):
    assert _decoder_parts(rows, 768) == _decoder_parts(rows, 0) == [rows]


class _Projection:
    def __init__(self, local, weights, partial, **kwargs):
        self.rows = local.shape[0]

    def functional(self, *, session):
        return self


class _MixPlan:
    def __init__(self, caps, invocation):
        self.caps = caps

    def request(self, name, prepare_call):
        return name


def _engram(monkeypatch, tokens, cap, chunk_rows=ENGRAM_CHUNK, world=3):
    """A real PreparedEngram (meta tensors, real PreparedChunkedAllReduce) with its projection/lookup/mix stubbed."""
    monkeypatch.setattr(engram_module, "QuantizedProjection", _Projection)
    monkeypatch.setattr(engram_module, "PreparedEngramLookup",
                        lambda *args, **kwargs: SimpleNamespace(functional=lambda session: None))
    monkeypatch.setattr(engram_module, "declare", lambda session, requests: None)
    monkeypatch.setattr(engram_module, "hyperconnection", SimpleNamespace(
        Caps=lambda **caps: SimpleNamespace(**caps), plan=_MixPlan, run_engram_mix=None))
    s = decoder_module.Scratch.allocate(tokens, 1, world, "meta", candidate_tokens=tokens, engram_chunk_rows=chunk_rows)
    shard = decoder_module.TensorParallelShard(0, world)
    tables = SimpleNamespace(inputs={1: None}, token_map=(), geometry=None, readiness_runtime=None,
                             columns=torch.arange(6144 // world), shard=shard)
    metadata = SimpleNamespace(token_ids=torch.empty(tokens, dtype=torch.int64, device="meta"),
                               positions=torch.empty(tokens, dtype=torch.int64, device="meta"),
                               live_counts=torch.empty(1, dtype=torch.int32, device="meta"))
    pool = SimpleNamespace(config=SimpleNamespace(text={"rms_norm_eps": 1e-6}))
    options = {} if cap is None else {"max_part_rows": cap}
    return engram_module.PreparedEngram(1, {"projection": None, "weight": torch.ones(20480, device="meta")}, tables,
        s.residual, s.engram_residual, metadata, pool, scratch=s.engram, session=None,
        name=f"prefill{tokens}.layer1.engram", workspace=None, runtime=_Runtime(), coordinator=None, **options)


def _engram_plan(prepared):
    """{chunk height (tokens): part rows of its (height x 5, 5120) reduction view}, plus the chunk heights in order."""
    plan = {rows: [end - begin for begin, end in reduce.bounds] for rows, (_, reduce) in prepared.chunk_ops.items()}
    return plan, [end - begin for begin, end in prepared.bounds]


# Engram projection all-reduce of a prefill graph (1024-token chunks; a reduction view has 5 rows per token).
ENGRAM = {
    4096: ({1024: [640] * 8}, {1024: [768] * 6 + [512]}),
    3584: ({1024: [640] * 8, 512: [640] * 4}, {1024: [768] * 6 + [512], 512: [768] * 3 + [256]}),
    3072: ({1024: [640] * 8}, {1024: [768] * 6 + [512]}),
    2560: ({1024: [640] * 8, 512: [640] * 4}, {1024: [768] * 6 + [512], 512: [768] * 3 + [256]}),
    2048: ({1024: [640] * 8}, {1024: [768] * 6 + [512]}),
    1536: ({1024: [640] * 8, 512: [640] * 4}, {1024: [768] * 6 + [512], 512: [768] * 3 + [256]}),
    1024: ({1024: [640] * 8}, {1024: [768] * 6 + [512]}),
    512: ({512: [640] * 4}, {512: [768] * 3 + [256]}),
    256: ({256: [640] * 2}, {256: [768, 512]}),
}


@pytest.mark.parametrize("tokens", PREFILL)
def test_engram_part_plan_at_768(monkeypatch, real_parts, tokens):
    halving, capped = ENGRAM[tokens]
    base, heights = _engram_plan(_engram(monkeypatch, tokens, None))   # default argument = halving (151744)
    assert base == halving and sum(heights) == tokens
    assert _engram_plan(_engram(monkeypatch, tokens, 0)) == (halving, heights)
    real_parts.clear()
    plan, heights_768 = _engram_plan(_engram(monkeypatch, tokens, 768))
    assert plan == capped and heights_768 == heights   # the projection chunking itself does not move
    for height, parts in plan.items():
        assert sum(parts) == 5 * height and set(parts) <= {256, 512, 768}
    # one prepared all-reduce per part, named like the decoder's parts (prefill4096.layer1.engram.reduce.0 ... .6;
    # a tail chunk's reduction is ...engram.reduce.tail.N)
    assert [rows for rows, _ in real_parts] == [r for parts in plan.values() for r in parts]
    full = plan[heights[0]]
    assert [name for _, name in real_parts][:len(full)] == [f"prefill{tokens}.layer1.engram.reduce.{i}"
                                                            for i in range(len(full))]


@pytest.mark.parametrize("tokens,rows", [(1, 5), (6, 30), (24, 120), (48, 240), (96, 480), (128, 640)])
def test_engram_decode_widths_are_untouched(monkeypatch, real_parts, tokens, rows):
    for cap in (0, 768):
        assert _engram_plan(_engram(monkeypatch, tokens, cap))[0] == {tokens: [rows]}


def test_engram_part_cut_is_independent_of_the_decoder_cap(monkeypatch, real_parts):
    """ar_part_rows alone leaves the Engram reduction halving (and the reverse): two separate knobs."""
    config = TP3Config.from_environment({"FLARE_DS41_AR_PART_ROWS": "768"})
    assert (config.ar_part_rows, config.engram_part_rows) == (768, 0)
    config = TP3Config.from_environment({"FLARE_DS41_ENGRAM_PART_ROWS": "768"})
    assert (config.ar_part_rows, config.engram_part_rows) == (0, 768)


class _Stub:
    """Accepts any construction / attribute / item / call; for PreparedLayer's other components."""

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return _Stub()

    def __getitem__(self, key):
        return _Stub()

    def __call__(self, *args, **kwargs):
        return _Stub()


class _Stop(Exception):
    pass


@pytest.mark.parametrize("ar,eg", [(0, 0), (768, 0), (0, 768), (768, 768), (512, 640)])
def test_prepared_layer_hands_the_part_caps_down(monkeypatch, ar, eg):
    """PreparedLayer passes tp3.engram_part_rows to PreparedEngram and tp3.ar_part_rows to both decoder reductions."""
    seen = {}

    def engram(*args, **kwargs):
        seen["engram"] = kwargs.get("max_part_rows")
        return _Stub()

    def chunked(source, output, **kwargs):
        name = kwargs["name"].rsplit(".", 1)[-1]
        seen[name] = kwargs.get("max_part_rows")
        if name == "mlp_reduce":
            raise _Stop
        return _Stub()

    monkeypatch.setattr(decoder_module, "PreparedEngram", engram)
    monkeypatch.setattr(decoder_module, "PreparedChunkedAllReduce", chunked)
    for name in ("PreparedMHC", "PreparedAttention", "UnquantizedProjection", "PreparedRouter",
                 "PreparedRoutedExperts", "PreparedMLP", "ConcurrentSharedExpert", "PreparedMHCPost"):
        monkeypatch.setattr(decoder_module, name, _Stub)
    monkeypatch.setattr(decoder_module, "_mhc_weights", lambda *args: None)
    tokens = 4096
    scratch = SimpleNamespace(residual=torch.empty(tokens, 4, 8, device="meta"), engram_residual=_Stub(),
                              engram=_Stub(), mlp_output=_Stub(), ffn_hc=_Stub(), attention_hc=_Stub(),
                              pre_mix=_Stub(), attn_mix=_Stub(), attention=_Stub(), attention_output=_Stub(),
                              router_logits=_Stub(), router=_Stub(), routed=_Stub(), shared=_Stub(),
                              mlp_partial=_Stub())
    metadata = SimpleNamespace(token_ids=torch.empty(tokens, dtype=torch.int64, device="meta"),
                               positions=torch.empty(tokens, dtype=torch.int64, device="meta"))
    weights = SimpleNamespace(engram={"projection": None}, attention=None,
                              experts=SimpleNamespace(vision_correction=None, router=None, correction=None,
                                                      experts=None, shared=None))
    pool = SimpleNamespace(config=SimpleNamespace(text=_Stub()))
    with pytest.raises(_Stop):
        decoder_module.PreparedLayer(1, weights, None, scratch, metadata, pool, None, shard=SimpleNamespace(world_size=3),
            mode="prefill", runtime=_Runtime(), coordinator=None, session=None, name="prefill4096.layer1",
            workspace=_Stub(), carry=False, defer_post=True, tp3=TP3Config(ar_part_rows=ar, engram_part_rows=eg))
    assert seen == {"engram": eg, "attention_reduce": ar, "mlp_reduce": ar}


def test_record_keys_and_defaults():
    base = TP3Config()
    record = base.to_metadata()
    assert record["ar_part_rows"] == 0 and record["engram_part_rows"] == 0 and record["mhc_pdl"] is False
    assert base.describe() == "TP3 config: defaults"
    on = TP3Config.from_environment({"FLARE_DS41_AR_PART_ROWS": "768", "FLARE_DS41_ENGRAM_PART_ROWS": "768",
                                     MHC_PDL_ENV: "1"})
    record = on.to_metadata()
    assert (record["ar_part_rows"], record["engram_part_rows"], record["mhc_pdl"]) == (768, 768, True)
    assert set(k for k in record if record[k] != base.to_metadata()[k]) == {"ar_part_rows", "engram_part_rows", "mhc_pdl"}
    assert "engram_part_rows=768" in on.describe() and "mhc_pdl=True" in on.describe()
    # an empty / unset FLARE_DS41_ENGRAM_PART_ROWS is 0 (halving), as every integer switch
    assert TP3Config.from_environment({"FLARE_DS41_ENGRAM_PART_ROWS": ""}).engram_part_rows == 0


@pytest.mark.parametrize("value", [-1, -768, 1.5, "768", True])
def test_engram_part_rows_must_be_a_non_negative_integer(value):
    with pytest.raises(ValueError, match="engram_part_rows must be a non-negative integer"):
        TP3Config(engram_part_rows=value)


@pytest.mark.parametrize("environ,expected", [
    ({}, False), ({MHC_PDL_ENV: "0"}, False), ({MHC_PDL_ENV: "1"}, True), ({MHC_PDL_ENV: "true"}, True),
    ({MHC_PDL_ENV: ""}, True),   # b12x: os.getenv(name, "0") != "0", so an empty value is ON in b12x
    ({"FLARE_DS41_MHC_PDL": "1"}, False)])   # only b12x's own variable counts
def test_mhc_pdl_follows_b12x_reading_rule(environ, expected):
    assert TP3Config.mhc_pdl_from_environment(environ) is expected
    assert TP3Config.from_environment(environ).mhc_pdl is expected


def test_mhc_pdl_must_be_boolean():
    with pytest.raises(TypeError, match="mhc_pdl must be a boolean"):
        TP3Config(mhc_pdl=1)


@pytest.mark.parametrize("record,compiled", [(False, False), (True, True)])
def test_check_mhc_pdl_accepts_a_matching_import(record, compiled):
    config = TP3Config(mhc_pdl=record)
    assert config.check_mhc_pdl(compiled) is config


@pytest.mark.parametrize("record,compiled", [(False, True), (True, False)])
def test_check_mhc_pdl_refuses_a_mismatch(record, compiled):
    with pytest.raises(ValueError, match="mhc_pdl"):
        TP3Config(mhc_pdl=record).check_mhc_pdl(compiled)


def test_check_mhc_pdl_reads_the_imported_b12x_module(monkeypatch):
    from b12x.norm.mhc import _kernels
    expected = _kernels._MHC_PDL
    assert TP3Config(mhc_pdl=expected).check_mhc_pdl() is not None
    monkeypatch.setattr(_kernels, "_MHC_PDL", not expected)
    with pytest.raises(ValueError, match="mhc_pdl"):
        TP3Config(mhc_pdl=expected).check_mhc_pdl()


def test_release_export_turns_the_three_switches_on():
    from tests.recipes.test_deepseek41_tp3_config import RELEASE_ENV, export_environment
    config = TP3Config.from_environment(RELEASE_ENV)
    assert (config.ar_part_rows, config.engram_part_rows, config.mhc_pdl) == (768, 768, True)
    # each one has its own off switch in export.sh (0 = the 20261111 layout)
    off = TP3Config.from_environment(export_environment(AR_PART_ROWS="0", ENGRAM_PART_ROWS="0", MHC_PDL="0"))
    assert (off.ar_part_rows, off.engram_part_rows, off.mhc_pdl) == (0, 0, False)
