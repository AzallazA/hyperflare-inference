# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 tp3.embed_rows: the vectorised host-row gather equals b12x's embedding lookup, and the switch wires it into
decode/verification/draft graphs only (CPU; the Triton kernel itself runs in Triton's CPU interpreter).

Reference semantics = ``PreparedEmbedding``: the output is zero-filled, then b12x's ``embedding_out`` writes
``table[ids[r]]`` for rows below the device live count. The CUDA comparison against b12x on a host-registered table is
test_deepseek41_host_embedding_cuda.py.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ignition.recipes.deepseek41 import decoder as decoder_module, dspark as dspark_module
from ignition.recipes.deepseek41.host_embedding import HostRowGather, device_noise_row
from ignition.recipes.deepseek41.tp3_config import TP3Config

NOISE, TABLE_ROWS = 97, 200


def reference(table, ids, live):
    """b12x semantics after PreparedEmbedding's zero fill."""
    expected = torch.zeros((ids.numel(), table.shape[1]), dtype=table.dtype)
    expected[:live] = table[ids[:live]]
    return expected


def draft_ids(requests, block, seed, live):
    """DSpark draft inputs: accepted token then block-1 noise tokens per request; dead rows hold arbitrary ids."""
    generator = torch.Generator().manual_seed(seed)
    ids = torch.full((requests, block), NOISE, dtype=torch.int64)
    ids[:, 0] = torch.randint(0, TABLE_ROWS, (requests,), generator=generator)
    ids = ids.view(-1)
    ids[live:] = torch.where(torch.arange(ids.numel())[live:] % 3 == 0, NOISE, 5)
    if requests > 1:
        ids[block] = NOISE  # an accepted token equal to the noise id
    return ids


def table(width=16, dtype=torch.bfloat16):
    return torch.randn((TABLE_ROWS, width), generator=torch.Generator().manual_seed(1)).to(dtype)


@pytest.mark.parametrize("requests,block,live", [(8, 5, 40), (8, 5, 23), (1, 5, 5), (4, 5, 20), (2, 6, 7), (8, 5, 0)])
@pytest.mark.parametrize("noise", [None, NOISE])
def test_gather_matches_the_b12x_lookup(requests, block, live, noise):
    weights = table()
    ids = draft_ids(requests, block, seed=requests * 31 + live, live=live)
    gather = HostRowGather(weights, rows=ids.numel(), noise_token=noise)
    for _ in range(2):  # replays reuse the same object
        result = gather(ids, torch.tensor([live], dtype=torch.int32))
        assert result.shape == (ids.numel(), weights.shape[1]) and result.dtype == weights.dtype
        assert torch.equal(result, reference(weights, ids, live))
    assert torch.equal(gather(ids), reference(weights, ids, ids.numel()))  # no live count: every row


def test_gather_int32_ids_and_fp32_tables():
    weights = table(width=24, dtype=torch.float32)
    ids = torch.tensor([3, NOISE, 0, TABLE_ROWS - 1], dtype=torch.int32)
    gather = HostRowGather(weights, rows=4, noise_token=NOISE)
    assert torch.equal(gather(ids, torch.tensor([3], dtype=torch.int32)), reference(weights, ids.long(), 3))


def test_noise_row_is_one_device_copy_per_table_and_token():
    weights = table()
    first = HostRowGather(weights, rows=5, noise_token=NOISE)
    second = HostRowGather(weights, rows=40, noise_token=NOISE)
    assert first.noise_row is second.noise_row  # shared by every graph of the rank
    assert device_noise_row(weights, NOISE) is first.noise_row
    saved = weights[NOISE].clone()
    assert torch.equal(first.noise_row[0], saved)
    weights[NOISE].fill_(0.5)  # an independent copy
    assert torch.equal(first.noise_row[0], saved)
    assert HostRowGather(table(), rows=5, noise_token=NOISE).noise_row is not first.noise_row
    assert HostRowGather(weights, rows=5).noise_row is None


def test_gather_rejects_bad_geometry():
    weights = table()
    with pytest.raises(ValueError, match="contiguous rows"):
        HostRowGather(weights.t(), rows=4)
    with pytest.raises(ValueError, match="positive row capacity"):
        HostRowGather(weights, rows=0)
    with pytest.raises(ValueError, match="noise token"):
        HostRowGather(weights, rows=4, noise_token=TABLE_ROWS)
    with pytest.raises(TypeError, match="BF16 or FP32"):
        HostRowGather(weights.half(), rows=4)
    gather = HostRowGather(weights, rows=4)
    with pytest.raises(ValueError, match="4 flat"):
        gather(torch.zeros(5, dtype=torch.int64))


def test_tp3_config_reads_embed_rows():
    assert TP3Config().embed_rows is False
    assert TP3Config.from_environment({}).embed_rows is False
    on = TP3Config.from_environment({"FLARE_DS41_EMBED_HOST": "1", "FLARE_DS41_EMBED_ROWS": "1"})
    assert on.embed_rows is True and on.to_metadata()["embed_rows"] is True
    assert "embed_rows=True" in on.describe()
    with pytest.raises(ValueError, match="embed_rows needs embed_host"):
        TP3Config.from_environment({"FLARE_DS41_EMBED_ROWS": "1"})
    with pytest.raises(TypeError, match="embed_rows must be a boolean"):
        TP3Config(embed_host=True, embed_rows=1)


def test_kernel_in_the_triton_interpreter():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH")))))
    completed = subprocess.run([sys.executable, "-m", "tests.recipes.test_deepseek41_host_embedding"],
                               cwd=root, env=env, capture_output=True, text=True, timeout=900)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "interpreter ok" in completed.stdout


def _interpret():
    """Run _gather_host_rows itself (DS41 width, BF16) on outputs poisoned with 0x7F/0xFF: every element is stored."""
    from ignition.recipes.deepseek41.host_embedding import _gather_host_rows

    weights = torch.randn((TABLE_ROWS, 5120), generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    cases = [(draft_ids(8, 5, seed=3, live=40), 40), (draft_ids(4, 5, seed=4, live=13), 13),
             (draft_ids(1, 5, seed=5, live=5), 0)]
    wild = torch.tensor([-4, 3, TABLE_ROWS + 9, NOISE], dtype=torch.int64)  # clamped into the table
    for ids, live in cases:
        for noise in (None, NOISE):
            gather = HostRowGather(weights, rows=ids.numel(), noise_token=noise)
            for count in (torch.tensor([live], dtype=torch.int32), None):
                expected = reference(weights, ids, live if count is not None else ids.numel())
                for fill in (0x7F, 0xFF):
                    out = torch.empty((ids.numel(), 5120), dtype=torch.bfloat16)
                    out.view(torch.uint8).fill_(fill)
                    _gather_host_rows[(ids.numel(),)](
                        weights, ids, ids if count is None else count, weights if noise is None else gather.noise_row,
                        out, ids.numel(), TABLE_ROWS, -1 if noise is None else noise, WIDTH=5120, STRIDE=5120,
                        BLOCK=8192, HAS_COUNT=count is not None, HAS_NOISE=noise is not None)
                    assert torch.equal(out, expected), (live, noise, fill)
    out = torch.empty((4, 5120), dtype=torch.bfloat16)
    _gather_host_rows[(4,)](weights, wild, torch.tensor([9], dtype=torch.int32), weights, out, 4, TABLE_ROWS, -1,
                            WIDTH=5120, STRIDE=5120, BLOCK=8192, HAS_COUNT=True, HAS_NOISE=False)
    assert torch.equal(out, weights[wild.clamp(0, TABLE_ROWS - 1)])
    print("interpreter ok")


# --- wiring (stubs; no kernels) -------------------------------------------------------------------------------------


class Component:
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def functional(self, **kwargs):
        return self


class Embedding(Component):
    built = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Embedding.built.append(self)

    def __call__(self, ids=None, *, token_ids=None, num_rows=None):
        return reference(self.args[0], token_ids, int(num_rows[0]))


class RingCommit(Component):
    staging = torch.zeros((1, 1), dtype=torch.uint8)


class Stop(Exception):
    pass


class Indices(Component):
    def __call__(self, metadata):
        raise Stop  # the forward has computed its embedding by now


def _metadata(ids, live):
    return SimpleNamespace(token_ids=ids, live_counts=torch.tensor([live, 0], dtype=torch.int32),
                           positions=torch.zeros(ids.numel(), dtype=torch.int64), max_query_tokens=6,
                           state_rows=torch.zeros(1, dtype=torch.int64), starts=torch.zeros(1, dtype=torch.int64))


@pytest.fixture
def stubs(monkeypatch):
    Embedding.built = []
    for module in (decoder_module, dspark_module):
        for name in ("PreparedEmbedding", "Rotary", "AttentionIndices", "PreparedLayer",
                     "PreparedVocabularyProjection", "PreparedTableRequests", "PreparedRingCommit"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, Component)
        monkeypatch.setattr(module, "PreparedEmbedding", Embedding)
        monkeypatch.setattr(module, "AttentionIndices", Indices)
        if hasattr(module, "PreparedRingCommit"):
            monkeypatch.setattr(module, "PreparedRingCommit", RingCommit)
        monkeypatch.setattr(module, "register_finalizer", lambda *args, **kwargs: None)
    monkeypatch.setattr(decoder_module.PreparedTableRequests, "__call__", lambda self, metadata: None, raising=False)
    monkeypatch.setattr(decoder_module.PreparedDecoder, "_prepare_head", lambda self, **kwargs: None)
    return Embedding


def _weights():
    config = SimpleNamespace(decoder_start=None, sources=(), text={"compress_ratios": []},
                             draft=SimpleNamespace(noise_token=NOISE, target_layers=()))
    draft = SimpleNamespace(layers=(), draft_head=None, norm=None)
    return SimpleNamespace(config=config, layers=(), draft=draft, embedding=table(width=5120), head=None, shard=None)


def _scratch(rows, dtype):
    return SimpleNamespace(local_logits=torch.zeros(1, dtype=dtype), logits=torch.zeros(1, dtype=dtype),
                           embedding=torch.zeros((rows, 5120), dtype=torch.bfloat16),
                           hidden=torch.zeros((rows, 5120), dtype=torch.bfloat16))


def _draft(weights, ids, tp3):
    return dspark_module.PreparedDraft(weights, _scratch(ids.numel(), torch.bfloat16), _metadata(ids, 0), None,
        runtime=None, vocabulary_runtime=None, coordinator=None, session=None, name="d", workspace=None, tp3=tp3)


def _decoder(weights, ids, mode, tp3):
    weights = SimpleNamespace(**{**vars(weights), "draft": None})
    pool = SimpleNamespace(reserved_backing=None, layout=None)
    return decoder_module.PreparedDecoder(weights, None, _scratch(ids.numel(), torch.float32), _metadata(ids, 0), pool,
        mode=mode, runtime=None, vocabulary_runtime=None, coordinator=None, session=None, name="t", workspace=None,
        tp3=tp3)


ON = TP3Config(embed_host=True, embed_rows=True)


@pytest.mark.parametrize("tp3", [None, TP3Config(), TP3Config(embed_host=True)])
def test_default_keeps_the_b12x_lookup_everywhere(stubs, tp3):
    weights, ids = _weights(), draft_ids(4, 5, seed=1, live=20)
    draft = _draft(weights, ids, tp3)
    assert draft.host_rows is None and isinstance(draft.embedding, Embedding)
    for mode in ("prefill", "decode", "verify"):
        target = _decoder(weights, ids, mode, tp3)
        assert target.host_rows is None and isinstance(target.embedding, Embedding)
    assert len(stubs.built) == 4


def test_switch_takes_draft_decode_and_verification_graphs(stubs):
    weights, ids = _weights(), draft_ids(4, 5, seed=1, live=20)
    draft = _draft(weights, ids, ON)
    assert draft.embedding is None and draft.host_rows.noise_token == NOISE and draft.host_rows.rows == 20
    assert draft.host_rows.table is weights.embedding
    for mode in ("decode", "verify"):
        target = _decoder(weights, ids, mode, ON)
        assert target.embedding is None and target.host_rows.noise_token is None and target.host_rows.rows == 20
    prefill = _decoder(weights, ids, "prefill", ON)
    assert prefill.host_rows is None and isinstance(prefill.embedding, Embedding)
    assert len(stubs.built) == 1  # only the prefill graph declared b12x's lookup


@pytest.mark.parametrize("tp3", [TP3Config(), ON])
@pytest.mark.parametrize("live", [20, 13, 0])
def test_forwards_embed_the_same_rows_either_way(stubs, monkeypatch, tp3, live):
    weights, ids = _weights(), draft_ids(4, 5, seed=live, live=live)
    seen = []
    original = HostRowGather.__call__

    def recording(self, token_ids, num_rows=None):
        seen.append(original(self, token_ids, num_rows))
        return seen[-1]

    monkeypatch.setattr(HostRowGather, "__call__", recording)
    draft = _draft(weights, ids, tp3)
    draft.metadata = _metadata(ids, live)
    target = _decoder(weights, ids, "verify", tp3)
    target.probe = SimpleNamespace(step=lambda *args: None, positions=lambda *args: None)
    for forward in (draft, target):
        with pytest.raises(Stop):
            forward(_metadata(ids, live))
    if tp3.embed_rows:
        assert len(seen) == 2
        for result in seen:
            assert torch.equal(result, reference(weights.embedding, ids, live))
    else:
        assert seen == []


if __name__ == "__main__":
    _interpret()
