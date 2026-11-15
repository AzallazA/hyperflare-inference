# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DS41 TP3 launch merging (TP3Config.mma_pad_once), on CPU.

The decode/verification/draft forwards zero every padded MXFP8 MMA-scale tile shape once per forward and every
layer's quantizer reuses the tile, instead of one Inductor zero fill per layer and tile (``mxfp8_dense_bases``).

- switch: default off, parsing, record;
- bases: ``dense_bases_with_pads`` hands out the shared tile only for a padded layout it was given;
- every reader sees its writer first: with poisoned allocations, N layers quantizing through ONE shared tile leave
  each layer's GEMM exactly the bytes a fresh zeroed tile per layer gives (b12x's CuTe quantizer emulated byte for
  byte, a GEMM stand-in that reads the whole tile, padding included);
- Inductor (torch 2.14.1, CPU): the compiled chain keeps one zero fill per tile shape (not one per layer), mutates the
  shared tile in place (no copy of it), and matches eager and the per-layer-fill forward bit for bit;
- wiring: PreparedAttention._run, PreparedLayer._run, the decoder helpers and PreparedDraft pass the tiles through
  only when the switch is on (switch off: the calls are the ones of today).
"""

from types import SimpleNamespace

import pytest
import torch

from ignition.components import linear
from ignition.components import mla_norm_quant
from ignition.components.mla_norm_quant import dense_bases_with_pads, padded_mma_layout
from ignition.recipes.deepseek41 import attention as attention_module
from ignition.recipes.deepseek41 import decoder as decoder_module
from ignition.recipes.deepseek41 import dspark as dspark_module
from ignition.recipes.deepseek41.tp3_config import TP3Config
from tests.components.test_uninitialised_encodings import POISON, _layout, _poisoning, _quantize_like_cute


# --------------------------------------------------------------------------------------------------------------------
# switch


def test_switch_default_off_parses_and_is_recorded():
    assert TP3Config().mma_pad_once is False
    assert TP3Config.from_environment({}).mma_pad_once is False
    config = TP3Config.from_environment({"FLARE_DS41_MMA_PAD_ONCE": "1"})
    assert config.mma_pad_once is True
    assert config.to_metadata()["mma_pad_once"] is True and TP3Config().to_metadata()["mma_pad_once"] is False
    assert "mma_pad_once=True" in config.describe() and "mma_pad_once" not in TP3Config().describe()
    for off in ("0", "", "true"):  # release convention: only "1" is on
        assert TP3Config.from_environment({"FLARE_DS41_MMA_PAD_ONCE": off}).mma_pad_once is False
    with pytest.raises(TypeError):
        TP3Config(mma_pad_once=1)


# --------------------------------------------------------------------------------------------------------------------
# bases


@pytest.mark.parametrize("rows", [1, 5, 21, 43, 127])
def test_dense_bases_with_pads_hands_out_the_shared_tile_for_padded_layouts(rows):
    layout = _layout(rows, 256)
    key = padded_mma_layout(layout)
    assert key == (tuple(layout[2][0]), layout[2][1])
    pad = torch.zeros(key[0], dtype=key[1])
    values, scale_rows, mma = dense_bases_with_pads(layout, torch.device("cpu"), (pad,))
    assert mma is pad
    assert (tuple(values.shape), values.dtype) == layout[0] and (tuple(scale_rows.shape), scale_rows.dtype) == layout[1]
    # a pad of another shape is not taken; no pads = mxfp8_dense_bases exactly
    other = torch.zeros((1, 1, 9, 32, 4, 4), dtype=torch.uint8)
    assert dense_bases_with_pads(layout, torch.device("cpu"), (other,))[2] is not other
    assert dense_bases_with_pads(layout, torch.device("cpu"), (other, pad))[2] is pad
    assert dense_bases_with_pads(layout, torch.device("cpu"), None)[2].eq(0).all()


@pytest.mark.parametrize("rows", [128, 256])
def test_full_tiles_have_no_padded_layout(rows):
    assert padded_mma_layout(_layout(rows, 256)) is None


def test_dense_bases_without_pads_is_mxfp8_dense_bases(monkeypatch):
    seen = []
    monkeypatch.setattr(mla_norm_quant, "mxfp8_dense_bases", lambda layout, device: seen.append(layout) or "bases")
    assert dense_bases_with_pads(_layout(5, 256), torch.device("cpu")) == "bases"
    assert dense_bases_with_pads(_layout(5, 256), torch.device("cpu"), ()) == "bases"
    assert len(seen) == 2


# --------------------------------------------------------------------------------------------------------------------
# every reader sees its writer first (poisoned allocator, emulated CuTe quantizer, whole-tile GEMM reader)


def _gemm_reads(values, scale_rows, scale_mma):
    """What a GEMM stand-in sees: every byte it loads (values, compact scales, the WHOLE MMA tile incl. padding)."""
    return values.view(torch.uint8).clone(), scale_rows.clone(), scale_mma.clone()


@pytest.mark.parametrize("rows", [1, 5, 21, 43])
def test_layers_sharing_one_tile_read_what_fresh_zeroed_tiles_give(monkeypatch, rows):
    layers = 6
    layouts = (_layout(rows, 256), _layout(rows, 128))  # qkv_a-like and q_b-like K
    sources = [[torch.randn(rows, layout[0][0][1], dtype=torch.bfloat16) * (layer + 1) for layout in layouts]
               for layer in range(layers)]
    expected = []
    for layer in range(layers):
        per_layer = []
        for layout, source in zip(layouts, sources[layer], strict=True):
            bases = [torch.zeros(shape, dtype=dtype) for shape, dtype in layout]
            _quantize_like_cute(source, *bases)
            per_layer.append(_gemm_reads(*bases))
        expected.append(per_layer)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", _poisoning(torch.empty))
        pads = decoder_module.allocate_mma_pads(tuple(padded_mma_layout(layout) for layout in layouts),
                                                torch.device("cpu"))
        assert all(pad.eq(0).all() for pad in pads)
        for layer in range(layers):
            for index, (layout, source) in enumerate(zip(layouts, sources[layer], strict=True)):
                bases = dense_bases_with_pads(layout, torch.device("cpu"), pads)
                assert bases[2] is pads[index]
                assert bases[0].view(torch.uint8).eq(POISON).all() and bases[1].eq(POISON).all()
                _quantize_like_cute(source, *bases)
                for actual, reference in zip(_gemm_reads(*bases), expected[layer][index], strict=True):
                    assert torch.equal(actual, reference), (layer, index)


def test_negative_control_a_poisoned_tile_is_caught(monkeypatch):
    """The comparison above detects stale padding: a tile that is not zeroed differs from the reference."""
    rows, layout = 5, _layout(5, 256)
    source = torch.randn(rows, 256, dtype=torch.bfloat16)
    reference = [torch.zeros(shape, dtype=dtype) for shape, dtype in layout]
    _quantize_like_cute(source, *reference)
    stale = torch.full(layout[2][0], POISON, dtype=torch.uint8)
    bases = dense_bases_with_pads(layout, torch.device("cpu"), (stale,))
    _quantize_like_cute(source, *bases)
    assert not torch.equal(bases[2], reference[2])


# --------------------------------------------------------------------------------------------------------------------
# Inductor: one fill per tile shape, the tile mutated in place, same values


@torch.library.custom_op("lmtest::quantize_rows", mutates_args=("values", "scale_rows", "scale_mma"))
def _quantize_op(source: torch.Tensor, values: torch.Tensor, scale_rows: torch.Tensor, scale_mma: torch.Tensor) -> None:
    _quantize_like_cute(source, values, scale_rows, scale_mma)


@_quantize_op.register_fake
def _(source, values, scale_rows, scale_mma):
    pass


@torch.library.custom_op("lmtest::tile_gemm", mutates_args=("output",))
def _gemm_op(values: torch.Tensor, scale_mma: torch.Tensor, output: torch.Tensor) -> None:
    # every output element depends on its row's values and on EVERY byte of the MMA tile (padding included)
    rows = output.shape[0]
    tile = scale_mma.view(-1).to(torch.float32)
    output.copy_(values.view(torch.uint8)[:, :output.shape[1]].to(torch.float32)
                 + (tile * torch.arange(tile.numel(), dtype=torch.float32)).sum() / 1e6
                 + torch.arange(rows, dtype=torch.float32)[:, None])


@_gemm_op.register_fake
def _(values, scale_mma, output):
    pass


@torch.library.custom_op("lmtest::join", mutates_args=("output",))
def _join_op(dependency: torch.Tensor, output: torch.Tensor, values: torch.Tensor, scale_rows: torch.Tensor,
             scale_mma: torch.Tensor) -> None:
    pass  # like flare::join_encoded_projection: names the bases (keeps them alive) without writing them


@_join_op.register_fake
def _(dependency, output, values, scale_rows, scale_mma):
    pass


LAYERS, ROWS = 8, 5


def _chain(sources, layouts, shared):
    """Per layer: quantize A -> GEMM A -> quantize B (input depends on GEMM A) -> GEMM B -> join B; the next layer's
    inputs depend on this layer's outputs (as the decoder's residual chain does)."""
    pads = decoder_module.allocate_mma_pads(tuple(padded_mma_layout(layout) for layout in layouts),
                                            sources.device) if shared else None
    carry = torch.zeros(ROWS, 64)
    outputs = []
    for layer in range(LAYERS):
        out = []
        for index, layout in enumerate(layouts):
            source = (sources[layer, :, :layout[0][0][1]].float() + carry.sum(1, keepdim=True) * 1e-3).bfloat16()
            bases = dense_bases_with_pads(layout, sources.device, pads)
            _quantize_op(source, *bases)
            output = torch.empty(ROWS, 64)
            _gemm_op(bases[0], bases[2], output)
            if index == 1:
                _join_op(carry, output, *bases)
            out.append(output)
            carry = output
        outputs.append(torch.stack(out))
    return torch.stack(outputs)


def _compiled(fn, *args):
    from torch._inductor.utils import run_and_get_code

    torch._dynamo.reset()
    result, codes = run_and_get_code(torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False), *args)
    return result, "\n".join(codes)


def test_inductor_keeps_one_fill_per_tile_shape_and_mutates_the_shared_tile_in_place():
    torch.manual_seed(5)
    layouts = (_layout(ROWS, 256), _layout(ROWS, 128))
    sources = torch.randn(LAYERS, ROWS, 256, dtype=torch.bfloat16)
    eager_shared = _chain(sources, layouts, True)
    eager_fresh = _chain(sources, layouts, False)
    assert torch.equal(eager_shared, eager_fresh)
    compiled_shared, code_shared = _compiled(lambda s: _chain(s, layouts, True), sources)
    compiled_fresh, code_fresh = _compiled(lambda s: _chain(s, layouts, False), sources)
    assert torch.equal(compiled_shared, eager_fresh) and torch.equal(compiled_fresh, eager_fresh)

    import re

    def fills(code):
        # Inductor's CPU kernels for zero-initialised tiles: generated kernels named after the fill, or a
        # zero/fill call in the wrapper
        kernels = set(re.findall(r"cpp_fused_[a-z_]*zeros[a-z_0-9]*", code))
        return len(kernels) + len(re.findall(r"\.zero_\(\)|aten\.zeros\.|full\.default", code))
    shared_fills, fresh_fills = fills(code_shared), fills(code_fresh)
    assert fresh_fills >= LAYERS * len(layouts) // 2, (fresh_fills, code_fresh[:2000])  # the per-layer fills exist
    assert shared_fills <= len(layouts), (shared_fills, code_shared[:4000])
    # the shared tile is mutated in place: no copy of a tile-sized buffer appears in the shared chain
    assert "clone" not in code_shared and code_shared.count("copy_") <= code_fresh.count("copy_")


# --------------------------------------------------------------------------------------------------------------------
# wiring


def _fake_attention(seen):
    attention = object.__new__(attention_module.PreparedAttention)
    attention.mode, attention.ratio, attention.owns_window_metadata = "verify", 0, False
    attention.layer, attention.candidate_source = 3, 20
    rows = 5
    qkv_layout, q_layout = _layout(rows, 5120), _layout(rows, 1280)

    class QKV:
        encoded_layout, quantizer_plan = qkv_layout, object()

        def __call__(self, source, encoded_bases=None):
            seen.append(("qkv_a", encoded_bases))
            return torch.zeros(rows, 1792)

    class NormQuant:
        layouts = (((rows, 1280), torch.bfloat16), ((rows, 512), torch.bfloat16), *q_layout)
        padded_mma_layout = mla_norm_quant.PreparedMLANormQuant.padded_mma_layout

        def __call__(self, projected, **kwargs):
            seen.append(("norm_quant", kwargs))
            return torch.zeros(rows, 1280), torch.zeros(rows, 512), ("v", "r", "m")

    attention.projections = {"qkv_a": QKV(),
                             "q_b": lambda q, encoded_bases=None: torch.zeros(rows, 8 * 512)}
    attention.norm_quant, attention.concurrent_query, attention.index_side = NormQuant(), None, None
    attention.window_norm = None
    attention.query_shape = (rows, 8, 512)
    attention.rotary = SimpleNamespace(apply=lambda *a, **k: None)
    attention.window_writer = lambda *a, **k: None
    attention.attention = lambda *a, **k: torch.zeros(rows, 8, 512)
    attention.external_tail, attention.selectors, attention.global_kv = True, [], None
    return attention, qkv_layout, q_layout


def _run_attention(attention, **kwargs):
    rows = 5
    indices = SimpleNamespace(position=torch.zeros(rows, dtype=torch.int64), window_slots=None, window_indices=None,
                              window_lengths=None)
    metadata = SimpleNamespace(state_rows=None, token_page_table=None)
    return attention._run(torch.zeros(rows, 5120).bfloat16(), metadata, (None, None, None), indices=indices,
                          window_cache=torch.zeros(rows, 528), **kwargs)


def test_attention_layouts_and_pads_reach_both_quantizers():
    seen = []
    attention, qkv_layout, q_layout = _fake_attention(seen)
    layouts = attention.padded_mma_layouts()
    assert layouts == (padded_mma_layout(qkv_layout), padded_mma_layout(q_layout))
    pads = decoder_module.allocate_mma_pads(layouts, torch.device("cpu"))
    _run_attention(attention, mma_pads=pads)
    (name, bases), (norm, kwargs) = seen
    assert name == "qkv_a" and bases[2] is pads[0]
    assert norm == "norm_quant" and kwargs == {"mma_pads": pads}


def test_attention_without_pads_is_unchanged():
    seen = []
    attention, qkv_layout, _ = _fake_attention(seen)
    _run_attention(attention)
    (name, bases), (norm, kwargs) = seen
    assert bases[2].eq(0).all() and tuple(bases[2].shape) == qkv_layout[2][0]
    assert kwargs == {}  # today's call: norm_quant(projected)


def test_mma_pad_layouts_dedupe_in_first_use_order_and_allocate_zeroed():
    a, b = ((1, 1, 40, 32, 4, 4), torch.uint8), ((1, 1, 10, 32, 4, 4), torch.uint8)
    attentions = [SimpleNamespace(padded_mma_layouts=lambda: (a, b)) for _ in range(3)]
    attentions.append(SimpleNamespace(padded_mma_layouts=lambda: ()))
    assert decoder_module.mma_pad_layouts(attentions) == (a, b)
    assert decoder_module.allocate_mma_pads((), torch.device("cpu")) is None
    pads = decoder_module.allocate_mma_pads((a, b), torch.device("cpu"))
    assert [tuple(p.shape) for p in pads] == [a[0], b[0]] and all(p.eq(0).all() for p in pads)


def _fake_layer(seen):
    layer = object.__new__(decoder_module.PreparedLayer)
    layer.probe = SimpleNamespace(record=lambda *a: None, router=lambda *a: None, full=lambda *a: None)
    layer.probe_index, layer.probe_mode, layer.no_engram = 3, "verify", False
    layer.engram = None
    layer.attention_hc = lambda *incoming, pre_mix, pre_out: (None, None, None, None)
    layer.attention = SimpleNamespace(_run=lambda *a, **k: (seen.append(k), (torch.zeros(1), None, None, {}))[1])
    layer.brrr = lambda *a: (torch.zeros(1), (None, None, None), None)
    layer.post = None
    return layer


def test_layer_passes_pads_only_when_on():
    seen = []
    layer = _fake_layer(seen)
    indices = SimpleNamespace(position=None)
    layer._run((torch.zeros(1),), torch.zeros(1), None, indices, None)
    pads = (torch.zeros(4, dtype=torch.uint8),)
    layer._run((torch.zeros(1),), torch.zeros(1), None, indices, None, mma_pads=pads)
    assert "mma_pads" not in seen[0] and seen[1]["mma_pads"] is pads


def test_draft_shares_one_tile_set_between_its_layers_and_off_passes_nothing():
    calls = []

    class Layer:
        def __call__(self, incoming, pre_mix, metadata, indices, selection, **kwargs):
            calls.append(kwargs)
            return incoming, pre_mix, None, selection

    a = ((1, 1, 40, 32, 4, 4), torch.uint8)
    draft = object.__new__(dspark_module.PreparedDraft)
    rows = 6
    draft.embedding = lambda token_ids, num_rows: torch.zeros(rows, 16, dtype=torch.bfloat16)
    draft.window_metadata = lambda metadata: None
    draft.layers = (Layer(), Layer(), Layer())
    draft.weights = SimpleNamespace(draft=SimpleNamespace(norm=torch.ones(16)), config=SimpleNamespace(text={"rms_norm_eps": 1e-6}))
    draft.head = lambda hidden: hidden
    metadata = SimpleNamespace(token_ids=torch.zeros(rows, dtype=torch.int64), live_counts=torch.tensor([rows]),
                               sample_indices=torch.arange(rows), starts=torch.zeros(1))
    draft(metadata)  # class default: off
    assert calls == [{}, {}, {}]
    calls.clear()
    draft.mma_pad_layouts = (a,)
    draft(metadata)
    assert len(calls) == 3 and all(len(k["mma_pads"]) == 1 for k in calls)
    assert calls[0]["mma_pads"] is calls[1]["mma_pads"] is calls[2]["mma_pads"]


def test_decoder_and_draft_default_to_per_layer_fills():
    assert decoder_module.PreparedDecoder.mma_pad_layouts == ()
    assert dspark_module.PreparedDraft.mma_pad_layouts == ()
    # linear.mxfp8_dense_bases (GLM and every other user) is untouched
    assert linear.mxfp8_dense_bases.__module__ == "ignition.components.linear"
