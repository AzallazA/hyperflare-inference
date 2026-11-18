# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Pack a prefill chunk's decoder suffix while preserving absolute cache positions.

Decoder SWA bounded replay is an approximation described in section 3.2.2 of
the DeepSeek V4.1 technical report. Global KV must be produced from every
encoder row before gathering the suffix. Chunks that fit the replay window
retain their complete continuation history; a truncated chunk establishes a
fresh SWA boundary at its first retained row.
"""

from dataclasses import dataclass

import torch

from ignition.components.paged_batch import Metadata


@dataclass(frozen=True)
class DecoderSuffix:
    metadata: Metadata
    indices: torch.Tensor
    live: torch.Tensor
    replay_start: torch.Tensor

    def gather(self, source):
        live = self.live.view((-1,) + (1,) * (source.ndim - 1))
        return torch.where(live, source[self.indices], 0)


class PreparedDecoderSuffix:
    def __init__(self, metadata, *, window=128):
        if metadata.starts.numel() != 1 or metadata.positions.numel() <= window or window <= 0:
            raise ValueError("decoder suffix packing requires one prefill chunk larger than its window")
        self.window = window
        self.metadata = Metadata.allocate(window, 1, metadata.page_table.shape[1], metadata.positions.device,
                                          samples=metadata.sample_indices.numel())
        self.replay_start = torch.zeros_like(metadata.starts)

    def functional(self):
        self.metadata = self.replay_start = None
        return self

    def begin(self, source):
        offsets = torch.arange(self.window, device=source.positions.device)
        count = source.lengths[0].long()
        kept = count.clamp(max=self.window)
        skipped = count - kept
        indices = (source.query_start_loc[0] + skipped + offsets).clamp(0, source.positions.numel() - 1)
        live = (offsets < kept) & (source.positions[indices] >= 0)
        starts = source.starts + skipped
        pages = source.page_table
        metadata = Metadata(
            token_ids=torch.where(live, source.token_ids[indices], 0),
            sample_indices=(source.sample_indices - skipped).clamp(0, self.window - 1),
            query_start_loc=torch.cat((torch.zeros(1, dtype=torch.int32, device=offsets.device), kept.view(1).int())),
            state_rows=source.state_rows,
            starts=starts,
            lengths=kept.view(1).int(),
            requests=torch.zeros(self.window, dtype=torch.int64, device=offsets.device),
            positions=torch.where(live, source.positions[indices], -1),
            page_table=pages,
            token_page_table=pages.expand(self.window, -1),
            live_counts=torch.cat((kept.view(1).int(), source.live_counts[1:])),
            max_query_tokens=self.window,
        )
        return DecoderSuffix(metadata, indices, live, torch.where(skipped > 0, starts, 0))


def encoder_capacities(requested, prefill_capacities, *, chunk):
    """TP3 (``tp3.prefill_skip_suffix``): the prefill capacities that get an encoder-only twin.

    Every requested size must be a captured prefill capacity above the decoder suffix window, so the twin has the
    same token capacity and inputs as an ordinary prefill graph. None or () -> no twins."""
    if not requested:
        return ()
    missing = [capacity for capacity in requested if capacity not in prefill_capacities or capacity <= 128]
    if missing:
        raise ValueError(f"FLARE_DS41_PREFILL_SKIP_SUFFIX {','.join(map(str, requested))}: "
                         f"{','.join(map(str, missing))} not captured prefill capacities above 128 "
                         f"(ladder {','.join(map(str, prefill_capacities))} for chunk {chunk})")
    return tuple(sorted(requested, reverse=True))


class PreparedEncoderGeneration:
    """TP3 encoder-only prefill (graph kind ``encoder_prefill``, ``tp3.prefill_skip_suffix``).

    The ordinary prefill generation (``PreparedGeneration``) without the decoder suffix: validate and pack the
    chunk, run ``PreparedDecoder(encoder_only=True)`` (layers through the global-KV boundary, Engram history
    commit), consume the prompt tokens and record completion without a vocabulary projection.

    State contract (Helios ``encoder_only_chunk`` decides when this may run):
    - written as by the ordinary graph: every row's KV/window/compressor/Engram state of layers 0-19, layer 20's
      compressed KV, index keys and latent pages (the global-KV sources 2/8/14/20 serve every later layer), the
      consumed-token count and the completion record (no token: the chunk is not the prompt's last);
    - left stale: the SWA windows of layers 20-39, the DSpark draft ring and ``last_hidden``. The next chunk is a
      prefill longer than the window (Helios refuses a <= 128-row successor and any snapshot at the stale
      endpoint), so it starts a fresh replay boundary for layers >= 20, rewrites all 128 ring slots and saves
      ``last_hidden``.
    """

    def __init__(self, decoder, batch, records, frame, sequence, *, runtime, coordinator, session, name):
        from ignition.components.collectives import PreparedAllReduce
        from ignition.components.sampling import PreparedSampling

        if not getattr(decoder, "encoder_only", False):
            raise ValueError("encoder-only generation requires an encoder-only decoder")
        self.decoder, self.batch = decoder, batch
        self.validation = PreparedAllReduce(batch.validation_input, batch.validation_sum, runtime=runtime,
            coordinator=coordinator, session=session, name=name + ".validation")
        b, rows = batch.inputs, batch.rows
        # Only the record kernel (SAMPLE=False) runs: it writes the consumption record of each live row.
        self.completion = PreparedSampling(decoder.scratch.logits, batch.requests, b.row_slot_ids[:rows],
            b.row_generations[:rows], b.live_rows, records, frame, sequence, sample_enabled=b.sample_enabled[:rows])
        self.completion.prepare()
        self.completion.functional(session=session)
        decoder.release_outputs()
        self.validation.functional(session=session)
        self.batch.functional()

    def __call__(self):
        self._run()

    def observe(self):
        """Forward-local packed metadata for diagnostic consumers (there is no hidden state or logits)."""
        return self._run(observe=True)

    def _run(self, *, observe=False):
        validation, present = self.batch.validate()
        metadata = self.batch.begin(self.validation(validation), present=present)
        self.decoder(metadata)
        self.batch.finish(metadata)
        self.completion.record_completion()
        if observe:
            return {"target." + name: getattr(metadata, name) for name in
                    ("live_counts", "starts", "lengths", "query_start_loc", "token_ids", "positions")}
