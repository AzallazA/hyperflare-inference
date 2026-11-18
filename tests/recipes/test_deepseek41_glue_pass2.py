# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""DS41 TP3 ``glue_pass2``: the second decode-glue pass is one export-time switch, off by default (CPU).

Each item keeps every value a reader sees and removes launches only:
- the verification window staging and the draft window cache start uninitialised (every byte a reader
  addresses is written first: checked here against the window-index and ring-commit readers);
- the draft metadata's state rows are realised once (G1 pattern);
- the first layer's mHC pre-mix, DSpark's feedback and the proposal admission check are built without
  in-place updates.
Off, every changed function traces to the graph it traced to before the switch existed.
"""

import dataclasses
import random
import re
from types import SimpleNamespace

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.lowering import register_lowering
from torch._inductor.utils import run_and_get_code

from ignition.components import paged_batch
from ignition.components.paged_batch import Metadata
from ignition.components.sampling import RUNNING
from ignition.preparation import compilation
from ignition.recipes.deepseek41.attention import AttentionIndices
from ignition.recipes.deepseek41.decoder import initial_mix
from ignition.recipes.deepseek41.tp3_config import TP3Config


# --- switch ---------------------------------------------------------------------------------------------------

def test_switch_is_off_by_default_parsed_and_recorded():
    assert TP3Config().glue_pass2 is False
    assert TP3Config.from_environment({}).glue_pass2 is False
    assert TP3Config.from_environment({"FLARE_DS41_GLUE_PASS2": "0"}).glue_pass2 is False
    config = TP3Config.from_environment({"FLARE_DS41_GLUE_PASS2": "1"})
    assert config.glue_pass2 is True
    assert config.to_metadata()["glue_pass2"] is True and "glue_pass2" not in TP3Config().to_metadata()  # recorded only when on
    assert "glue_pass2=True" in config.describe()
    with pytest.raises(TypeError):
        TP3Config(glue_pass2=1)


# --- window staging and draft window cache: every read byte is written first ----------------------------------

RECORD = 528


def _metadata(rng, rows, tokens, mode):
    """Random packed requests: lengths per request (0 = idle row), positions continuing each request's history."""
    lengths = [rng.randint(0, tokens // rows) if rows > 1 else rng.randint(1, tokens) for _ in range(rows)]
    if not any(lengths):
        lengths[0] = 1
    starts = [rng.choice((0, 3, 127, 128, 129, 400, 1000)) for _ in range(rows)]
    states = [rng.randint(1, 30) for _ in range(rows)]
    m = Metadata.allocate(tokens, rows, 8, torch.device("cpu"), max_query_tokens=max(1, tokens // rows))
    requests, positions = [], []
    for request, (length, start) in enumerate(zip(lengths, starts)):
        for column in range(length):
            requests.append(request)
            positions.append(start + column)
    pad = tokens - len(positions)
    m.positions.copy_(torch.tensor(positions + [-1] * pad))
    m.requests.copy_(torch.tensor(requests + [rows - 1] * pad))
    m.starts.copy_(torch.tensor(starts))
    m.lengths.copy_(torch.tensor(lengths, dtype=torch.int32))
    m.state_rows.copy_(torch.tensor([s if n else 0 for s, n in zip(states, lengths)]))
    return m, lengths, starts


def _written(m, window, page_size):
    """Bytes the b12x snapshot writer stores: all 128 snapshot records of every request whose state row is valid
    (b12x SnapshotAndCacheDsv41SwaKernel: 66 x 1 KiB tiles per request, skipped only for a negative or
    out-of-capacity row), and the record of every live token at its window slot."""
    rows = m.starts.numel()
    written = torch.zeros(rows * page_size, dtype=torch.bool)
    for request in range(rows):
        if int(m.state_rows[request]) >= 0:
            written[request * page_size:request * page_size + 128] = True
    slots = window.window_slots
    written[slots[slots >= 0]] = True
    return written


@pytest.mark.parametrize("mode", ("verify", "draft"))
@pytest.mark.parametrize("rows,tokens", ((1, 5), (1, 6), (4, 24), (8, 48)))
def test_every_window_record_a_reader_addresses_is_written(mode, rows, tokens):
    rng = random.Random(f"{mode}{rows}{tokens}")
    for _ in range(25):
        m, lengths, starts = _metadata(rng, rows, tokens, mode)
        page_size = 128 + m.max_query_tokens
        width = page_size if mode == "draft" else 128
        indices = AttentionIndices(m, mode=mode, tokens=tokens, page_size=page_size, width=width)
        window = indices(m)
        written = _written(m, window, page_size)
        # Attention reads window_indices[token, :window_lengths[token]] (the rest are -1).
        read = window.window_indices
        assert (read[read >= 0] < rows * page_size).all()
        unwritten = read[(read >= 0)][~written[read[read >= 0].long()]]
        assert unwritten.numel() == 0, (mode, lengths, starts, unwritten)
        if mode == "verify":
            # Ring commit (_commit and attention._commit_window): accepted tokens < count <= query length, rows > 0.
            for request, length in enumerate(lengths):
                row = int(m.state_rows[request])
                for count in range(length + 1):
                    for item in range(min(count, 128)):
                        token = max(0, count - 128) + item
                        if token < count and row > 0:
                            assert written[request * page_size + 128 + token], (request, count, token)


# --- draft state rows: realised once for every reader ---------------------------------------------------------

@torch.library.custom_op("flare_test_glue2::state_reader", mutates_args=("out",))
def _state_reader(x: torch.Tensor, rows: torch.Tensor, out: torch.Tensor) -> None:
    out.copy_(x + rows.sum())


@_state_reader.register_fake
def _(*args):
    pass


def _draft_forward(states, lengths, x, snapshot):
    tokens = torch.zeros((states.numel(), 6), dtype=torch.int64)
    m = Metadata.pack(torch.zeros_like(states), lengths, states, torch.zeros((states.numel(), 4), dtype=torch.int32),
                      tokens, torch.tensor([states.numel()], dtype=torch.int32), token_capacity=tokens.numel())
    if snapshot:
        m = dataclasses.replace(m, state_rows=torch.ops.flare.snapshot(m.state_rows))
    for _ in range(3):  # three draft layers, each reading the state rows through an opaque writer
        out = torch.empty_like(x)
        _state_reader(x, m.state_rows, out)
        x = out * 1
    return x


def _state_row_writes(code):
    call = code[code.index("def call("):]
    return len([line for line in call.splitlines() if re.search(r"cpp_fused\w*(gt|where)\w*\(", line)])


def test_draft_state_rows_are_realised_once_with_the_same_values():
    # The traced c1 round (one request) ran the draft state-row expression once per draft layer (GPU Inductor,
    # triton_poi_fused_..._where_zeros_20 x3); CPU Inductor realises it once either way, so this checks the values
    # and that the snapshot leaves exactly one realisation. The launch count is checked on the exported forward.
    register_lowering(torch.ops.flare.snapshot.default)(compilation._snapshot_lowering)
    for states, lengths in (([7], [6]), ([7, 9], [6, 0]), ([3, 0, 5, 2], [6, 6, 0, 6])):
        states = torch.tensor(states, dtype=torch.int64)
        lengths = torch.tensor(lengths, dtype=torch.int32)
        x = torch.randn(4)
        expected = _draft_forward(states, lengths, x, False)
        torch._dynamo.reset()
        actual, (code, *_) = run_and_get_code(torch.compile(
            lambda: _draft_forward(states, lengths, x, True), backend="inductor", fullgraph=True))
        assert torch.equal(actual, expected)
        assert _state_row_writes(code) == 1


# --- first-layer mHC pre-mix -------------------------------------------------------------------------------------

def _inline_mix(rows):  # the code initial_mix replaced (decoder.py / dspark.py before glue_pass2)
    pre_mix = torch.zeros((rows, 4), dtype=torch.float32, device="cpu")
    pre_mix[:, 0].fill_(1)
    return pre_mix


def _graph(fn, *args):
    module = make_fx(fn)(*args)
    return re.sub(r"_tensor_constant\d+", "_c", module.code), module


def _storage_targets(module):
    calls = compilation._lower_storage(module)
    return [str(call.target) for call in calls]


@pytest.mark.parametrize("rows", (1, 5, 40))
def test_initial_mix_values_and_launches(rows):
    expected = _inline_mix(rows)
    assert torch.equal(initial_mix(rows, torch.device("cpu")), expected)
    assert torch.equal(initial_mix(rows, torch.device("cpu"), functional=True), expected)
    # Off: the same traced graph as the inline code. On: no opaque in-place fill.
    off, off_module = _graph(lambda: initial_mix(rows, torch.device("cpu")))
    inline, inline_module = _graph(lambda: _inline_mix(rows))
    assert off == inline
    assert _storage_targets(off_module) == _storage_targets(inline_module) == ["aten.fill_.Scalar"]
    _, on_module = _graph(lambda: initial_mix(rows, torch.device("cpu"), functional=True))
    assert _storage_targets(on_module) == []


# --- proposal admission check ------------------------------------------------------------------------------------

def _batch(rng, rows, capacity, functional, *, mode="decode", width=6):
    blocks, page_tokens = 4, 128
    torch.manual_seed(rng.randrange(1 << 30))  # the tensors below follow the Python generator's state
    slots = list(range(capacity))
    rng.shuffle(slots)
    row_slots = torch.tensor([slots[i] if rng.random() > .1 else rng.choice((-1, capacity, slots[0]))
                              for i in range(rows)], dtype=torch.int32)
    inputs = SimpleNamespace(
        row_slot_ids=row_slots,
        row_generations=torch.randint(0, 2, (rows,), dtype=torch.int64),
        live_rows=torch.tensor([rng.randint(0, rows + 1)], dtype=torch.int32),
        query_lengths=torch.randint(0, width + 2, (rows,), dtype=torch.int32),
        page_tables=torch.randint(-1, 40, (capacity, blocks), dtype=torch.int32),
        request_state_rows=torch.randint(-1, 40, (capacity, 8), dtype=torch.int64),
        token_ids=torch.randint(-1, 100, (rows * width,), dtype=torch.int64))
    requests = SimpleNamespace(
        generation=torch.randint(0, 2, (capacity,), dtype=torch.int64), capacity=capacity,
        consumed_tokens=torch.randint(-2, 600, (capacity,), dtype=torch.int64),
        status=torch.tensor([RUNNING if rng.random() > .2 else RUNNING + 1 for _ in range(capacity)], dtype=torch.int32),
        pending_token=torch.randint(-1, 101, (capacity,), dtype=torch.int64), vocab_size=100)
    return SimpleNamespace(
        inputs=inputs, requests=requests, rows=rows, mode=mode, verification_width=width, tokens=rows * width,
        target_token_capacity=rows * width - rng.randint(0, rows), page_tokens=page_tokens,
        pool=SimpleNamespace(memory_capacity=torch.tensor([32, 36])), input_vocabulary=100, validate_prefill=None,
        functional_validation=functional)


@pytest.mark.parametrize("rows,capacity", ((1, 1), (1, 16), (4, 8), (8, 16), (16, 32)))
def test_functional_admission_check_matches_and_has_no_in_place_updates(rows, capacity):
    rng = random.Random(rows * 100 + capacity)
    for _ in range(30):
        state = rng.getstate()
        base = _batch(rng, rows, capacity, False)
        rng.setstate(state)
        functional = _batch(rng, rows, capacity, True)
        expected = paged_batch.PreparedBatch.validate(base)
        actual = paged_batch.PreparedBatch.validate(functional)
        for a, e in zip(actual, expected, strict=True):
            assert a.dtype == e.dtype and a.shape == e.shape and torch.equal(a, e)
    off = _storage_targets(make_fx(lambda: paged_batch.PreparedBatch.validate(base))())
    on = _storage_targets(make_fx(lambda: paged_batch.PreparedBatch.validate(functional))())
    # Base: six in-place mask updates, each an opaque eager kernel (plus, in the served graph, the two copies into
    # the validation row that its all-reduce reads: storage calls 7/8 of the c1 forward). Functional: none.
    assert off.count("aten.bitwise_and_.Tensor") == 6
    assert on == []


# --- DSpark feedback: three slice copies vs one concatenation ----------------------------------------------------

def _feedback_slices(feedbacks, shape):  # decoder.py base: copy each layer's mean into its 5120-column slice
    auxiliary = torch.empty(shape, dtype=torch.bfloat16)
    for column, feedback in enumerate(feedbacks):
        auxiliary[:, column * 5120:(column + 1) * 5120].copy_(feedback)
    return auxiliary


def _feedback_cat(feedbacks, shape):  # glue_pass2
    return torch.cat(list(feedbacks), dim=1).to(torch.bfloat16)


@pytest.mark.parametrize("rows", (1, 5, 48))
def test_concatenated_feedback_matches_the_slice_copies(rows):
    generator = torch.Generator().manual_seed(rows)
    sources = [torch.randn((rows, 4, 5120), generator=generator) * scale for scale in (1e-3, 1., 3e4)]
    feedbacks = [source.float().mean(1) for source in sources]
    feedbacks[1][0, :7] = torch.tensor([0., -0., 1e-40, -1e-40, 3.4e38, float("inf"), -float("inf")])
    shape = (rows, 3 * 5120)
    assert torch.equal(_feedback_cat(feedbacks, shape).view(torch.int16), _feedback_slices(feedbacks, shape).view(torch.int16))
    slices = _storage_targets(make_fx(lambda *f: _feedback_slices(f, shape))(*feedbacks))
    concatenated = _storage_targets(make_fx(lambda *f: _feedback_cat(f, shape))(*feedbacks))
    assert slices.count("aten.copy_.default") in (0, 3) and concatenated == []
