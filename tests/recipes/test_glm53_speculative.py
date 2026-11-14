# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM packed verification and shared commit semantics."""

from types import SimpleNamespace

import pytest
import torch

from tests.components.completion_support import RecordOnlyLogprobs, completion_outputs

from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.components.speculation import PreparedSpeculativeCommit, SpeculativeAcceptance
from ignition.recipes.glm53.batch import BatchInputs
from ignition.recipes.glm53.dcp import ContextShards
from ignition.recipes.glm53.speculative import PreparedRound, retain_selection, reuse_selection
from ignition.recipes.glm53.mtp import DraftMetadata


from ignition.recipes.glm53.stack import DecoderScratch
from ignition.components.parallel import TensorParallelShard


def test_shared_commit_without_previous_token_embeddings():
    state = DeviceRequestState.allocate("cpu", vocab_size=32, capacity=2)
    state.admit(1, RequestAdmission(generation=7, seed=19, temperature=0.7, max_new_tokens=100), [1, 2], consumed_tokens=2)
    state.pending_token[1] = 3
    state.generated_tokens[1] = 1
    acceptance = SpeculativeAcceptance.allocate(1, 6, "cpu")
    acceptance.token_ids[0, :3] = torch.tensor([4, 5, 17])
    acceptance.proposed_count.fill_(3)
    acceptance.accepted_drafts.fill_(2)
    acceptance.offered_drafts.fill_(5)
    acceptance.valid.fill_(True)
    records = torch.zeros((2, 2, 32), dtype=torch.int64)
    commit = PreparedSpeculativeCommit(state, acceptance, torch.tensor([[3, 4, 5, 6, 7, 8]]),
        torch.tensor([1], dtype=torch.int32), torch.tensor([7]), torch.tensor([1], dtype=torch.int32),
        completion_outputs(records), torch.tensor([0], dtype=torch.int32), torch.tensor([1]), logits=torch.zeros((1, 6, state.vocab_size)))
    commit()
    assert state.consumed_tokens.tolist() == [0, 5]
    assert state.pending_token[1] == 17
    assert records[0, 1, 13] == 4
    assert records[0, 1, 16:19].tolist() == [4, 5, 17]


@pytest.mark.parametrize("width,limits,positions,token_slots,state_indices", [
    # Width one decodes each live row in place through a single state column.
    (1, [1, 1], [255, 510], [10 * 256 + 255, 21 * 256 + 254], [[1], [2]]),
    (6, [6, 2], [255, 256, 257, 258, 259, 260, 510, 511, -1, -1, -1, -1],
     [10 * 256 + 255, *range(11 * 256, 11 * 256 + 5), 21 * 256 + 254, 21 * 256 + 255],
     # Verification records tokens against each request's live row only.
     [[1, 0, 0, 0, 0, 0, 0, 0], [2, 0, 0, 0, 0, 0, 0, 0]]),
])
def test_verification_packs_unequal_lengths_and_retains_request_output_rows(
        width, limits, positions, token_slots, state_indices):
    op = PreparedRound.__new__(PreparedRound)
    op.batch, op.width = 2, width
    op.state_columns = len(state_indices[0])
    inputs = BatchInputs.allocate(128, 2, "cpu", capacity=2)
    inputs.page_tables.copy_(torch.tensor([[10, 11], [20, 21]]))
    inputs.live_rows.fill_(2)
    inputs.request_state_rows.copy_(torch.tensor([[1, 8, 9, 10, 11, 12, 13, 14], [2, 15, 16, 17, 18, 19, 20, 21]]))
    op.owner = SimpleNamespace(inputs=inputs, requests=SimpleNamespace(capacity=2), shards=ContextShards())
    args = (torch.tensor([0, 1]), torch.tensor(limits, dtype=torch.int32), torch.tensor([True, True]),
            torch.tensor([255, 510]), torch.arange(2 * width).view(2, width))
    metadata = op.target_metadata(*args)
    assert metadata.recurrent.query_start_loc.tolist() == [0, limits[0], sum(limits)]
    assert metadata.recurrent.num_tokens.item() == sum(limits)
    assert metadata.indexed.positions.tolist() == positions
    assert metadata.indexed.token_slots.tolist()[:sum(limits)] == token_slots
    # Columns past the round's width stay on the null row instead of reaching the next slot's rows.
    assert metadata.recurrent.state_indices.tolist() == state_indices
    repeated = op.target_metadata(*args)
    assert metadata.token_ids.data_ptr() != repeated.token_ids.data_ptr()
    torch.testing.assert_close(metadata.token_ids, repeated.token_ids)


def test_later_draft_passes_reuse_each_requests_first_selection():
    slots, starts = torch.tensor([2, 0, 1]), torch.tensor([255, 510, 300])
    state_rows = torch.arange(1, 4)[:, None].expand(3, 8)
    tables = torch.tensor([[10, 11], [20, 21], [30, 31]])
    hidden = torch.zeros((3, 4096), dtype=torch.bfloat16)
    tokens = torch.tensor([7, 8, 9])
    # The first pass packs requests one and two; request zero has no draft to propose.
    first_pass = DraftMetadata.pack(slots, torch.tensor([0, 1, 1]), starts, tokens, hidden, tables,
                                    capacity=1, state_rows=state_rows)
    first = {1: torch.tensor([3, 5, 9]), 2: torch.tensor([4, 6, 8, 12])}
    physical = torch.full((3, 2051), -1, dtype=torch.int32)
    lengths = torch.zeros(3, dtype=torch.int32)
    for row, request in enumerate((1, 2)):
        physical[row, :first[request].numel()] = first[request]
        lengths[row] = first[request].numel()
    retained = retain_selection((physical, lengths), first_pass.output_indices)
    # The second pass keeps only request two, which now occupies packed row zero.
    second_pass = DraftMetadata.pack(slots, torch.tensor([0, 0, 1]), starts + 1, tokens, hidden, tables,
                                     capacity=1, state_rows=state_rows)
    reused_physical, reused_lengths = reuse_selection(retained, second_pass.indexed)
    assert reused_lengths.tolist() == [4, 0, 0] and reused_lengths.dtype == torch.int32
    assert reused_physical[0, :4].tolist() == first[2].tolist()
    # Negative control: without the retained copy, packed row zero would read request one's selection.
    assert physical[0, :3].tolist() == first[1].tolist()


@pytest.mark.parametrize("reuse", [False, True])
def test_draft_rows_exclude_exhausted_requests_and_restore_output_order(reuse):
    scratch = DecoderScratch.allocate(3, 2, "prefill", "cpu", shard=TensorParallelShard(0, 2), sample_rows=1)
    storage = scratch.attention_hc.normalized if reuse else None
    metadata = DraftMetadata(3, 3, 2, "cpu", hidden=storage)
    metadata.hidden.fill_(float("nan"))
    slots = torch.tensor([2, 0, 1])
    state_rows = torch.arange(1, 4)[:, None].expand(3, 8)
    tables = torch.tensor([[10, 11], [20, 21], [30, 31]])
    hidden = torch.arange(3, dtype=torch.bfloat16)[:, None].expand(3, 4096)
    metadata.set(slots, torch.tensor([0, 1, 1]), torch.tensor([512, 255, 510]),
                 torch.tensor([7, 8, 9]), hidden, tables, capacity=1, state_rows=state_rows)
    assert metadata.indexed.query_start_loc.tolist() == [0, 0, 1, 2]
    assert metadata.indexed.positions.tolist() == [255, 510, -1]
    assert metadata.indexed.token_slots.tolist() == [10 * 256 + 255, 21 * 256 + 254, -1]
    assert metadata.token_ids[:2].tolist() == [8, 9]
    assert metadata.hidden[metadata.output_indices][1:, 0].tolist() == [1, 2]
    assert hidden[:, 0].tolist() == [0, 1, 2]
    if reuse:
        assert torch.equal(storage, metadata.hidden)
        assert storage.data_ptr() == metadata.hidden.data_ptr()
    metadata.set(slots, torch.zeros(3, dtype=torch.int64), torch.full((3,), 512),
                 torch.tensor([7, 8, 9]), hidden, tables, capacity=1, state_rows=state_rows)
    assert metadata.num_tokens.item() == 0
    assert not metadata.hidden.count_nonzero()
    assert metadata.indexed.positions.tolist() == [-1, -1, -1]
    assert metadata.indexed.token_slots.tolist() == [-1, -1, -1]


@pytest.mark.parametrize("compiled", [False, True])
def test_draft_metadata_is_local_and_traces_with_runtime_lengths(compiled):
    slots = torch.tensor([2, 0, 1])
    state_rows = torch.arange(1, 4)[:, None].expand(3, 8)
    tables = torch.tensor([[10, 11], [20, 21], [30, 31]])
    tokens = torch.tensor([[7, 8], [9, 10], [11, 12]])
    hidden = torch.arange(6, dtype=torch.bfloat16)[:, None].expand(6, 4096).reshape(3, 2, 4096)

    def pack(lengths):
        metadata = DraftMetadata.pack(slots, lengths, torch.tensor([254, 510, 0]), tokens, hidden,
            tables, capacity=2, state_rows=state_rows)
        return (metadata.token_ids, metadata.hidden, metadata.output_indices, metadata.num_tokens,
                metadata.indexed.positions, metadata.indexed.token_slots)

    run = torch.compile(pack, backend="eager", fullgraph=True) if compiled else pack
    lengths = torch.tensor([2, 1, 0])
    actual = run(lengths)
    assert actual[0][:3].tolist() == [7, 8, 9]
    assert actual[1][:3, 0].tolist() == [0, 1, 2]
    assert actual[2].tolist() == [1, 2, 2]
    assert actual[3].item() == 3
    assert actual[4].tolist() == [254, 255, 510, -1, -1, -1]
    for result, reference in zip(actual, pack(lengths), strict=True):
        torch.testing.assert_close(result, reference)
    empty = run(torch.zeros_like(lengths))
    assert empty[3].item() == 0
    assert empty[4].tolist() == [-1] * 6
    assert actual[0].data_ptr() != empty[0].data_ptr()


@pytest.mark.parametrize("hidden", [torch.empty((2, 4096)), torch.empty((1, 4096), dtype=torch.bfloat16),
    torch.empty((4096, 2), dtype=torch.bfloat16).t(), torch.empty((2, 4096), device="meta", dtype=torch.bfloat16)])
def test_draft_hidden_storage_rejects_incompatible_views(hidden):
    with pytest.raises(ValueError, match="draft hidden storage"):
        DraftMetadata(2, 1, 2, "cpu", hidden=hidden)


@pytest.fixture(autouse=True)
def records_only_cpu_commit(monkeypatch):
    # These state-machine tests run without CUDA; reduction and graph ordering
    # have independent CUDA coverage.
    monkeypatch.setattr("ignition.components.speculation.PreparedLogprobs", RecordOnlyLogprobs)


class _Recorder:
    """Stand-in for a prepared component: records its constructor's keyword arguments."""

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    @staticmethod
    def allocate(*args, **kwargs):
        return SimpleNamespace(args=args, kwargs=kwargs, token_ids=None, indexed=None,
                               state_indices=torch.zeros((2, 8)), num_seqs=torch.zeros(1))

    @staticmethod
    def scratch_layout(*args):
        return args


@pytest.mark.parametrize("width", [1, 2, 3, 6])
@pytest.mark.parametrize("share", [False, True])
def test_round_constructor_sets_width_one_in_place_and_shared_selection(monkeypatch, width, share):
    """PreparedRound.__init__ itself (not an injected field) chooses #217's in-place width one and #218's selection
    sharing: width one keeps one state column, no KDA records, no commit and no draft; wider rounds record eight
    columns; draft passes share the first selection only from width three when the checkpoint asks for it."""
    import ignition.recipes.glm53.speculative as speculative

    for name in ("PreparedSpeculativeDistributions", "PreparedStochasticVerification", "PreparedDraftDistribution",
                 "PreparedBatch", "PreparedAllReduce", "PreparedDecoder", "PreparedKDACommit"):
        monkeypatch.setattr(speculative, name, _Recorder)
    class Metadata(_Recorder):
        def __init__(self, token_ids, rows, recurrent, indexed):
            self.token_ids, self.rows, self.recurrent, self.indexed = token_ids, rows, recurrent, indexed

    monkeypatch.setattr(speculative, "DecoderMetadata", Metadata)
    monkeypatch.setattr(speculative, "DeltaMetadata", _Recorder)
    monkeypatch.setattr(speculative, "DraftMetadata", _Recorder)
    monkeypatch.setattr(PreparedRound, "_sampling_scratch", lambda self, device: {})
    drafts = []
    kda_records = SimpleNamespace(layers=(0,))
    backing = SimpleNamespace(device=torch.device("cpu"))
    owner = SimpleNamespace(
        pool=SimpleNamespace(backing=backing, decoder=None, memory_capacity=(0, 1),
                             draft=SimpleNamespace(index=SimpleNamespace(tail=torch.zeros((4, 3))))),
        inputs=SimpleNamespace(page_tables=torch.zeros((2, 5))), requests=SimpleNamespace(vocab_size=32, capacity=2),
        convolution=None, share_draft_selection=share, weights=SimpleNamespace(shard=None), shards=None,
        collectives={"runtime": None, "coordinator": None}, session=None, scratch=SimpleNamespace(
            rows=lambda *args, **kwargs: None), workspace=None, max_cache_pages=None, context=None,
        kda_records=kda_records, draft=lambda metadata, name, **kwargs: drafts.append((name, kwargs)) or SimpleNamespace(
            logits=torch.zeros((2, 32))))
    monkeypatch.setattr(speculative.PreparedDecoder, "layers", {0: SimpleNamespace(
        attention=SimpleNamespace(recurrent_plan=None))}, raising=False)
    round_ = PreparedRound(owner, 2, width, "round")
    assert round_.recorded is (width > 1)
    assert round_.state_columns == (8 if width > 1 else 1)
    assert round_.metadata.recurrent.kwargs == {"verification_columns": round_.state_columns}
    assert round_.decoder.kwargs["kda_records"] is (kda_records if width > 1 else None)
    assert (round_.kda_commit is not None) is (width > 1)
    assert round_.share_selection is (share and width > 2)
    assert (round_.draft is not None) is (width > 1)
    assert [name for name, _ in drafts] == ([] if width == 1 else ["round.draft"]) + ["round.absorb"]
