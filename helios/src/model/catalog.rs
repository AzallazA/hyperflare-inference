/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared validation and selection for every prepared model's graph catalog.

use crate::{
    Result,
    execution::{
        BATCH_CAPACITIES, ExecutionManifest, GraphKind, SamplingMode,
    },
    model::ModelConfig,
    scheduler::speculation,
};

pub fn validate(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
    validate_inner(config, manifest).map_err(|error| match error {
        crate::Error::Execution(crate::ExecutionError::Manifest(_)) => error,
        _ => crate::ExecutionError::Manifest(crate::Failure::caused_by(error.to_string(), error))
            .into(),
    })
}

fn validate_inner(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
    manifest.validate()?;
    if config.max_top_logprobs != manifest.max_top_logprobs()?
        || config.logprob_token_id_capacity != manifest.logprob_token_id_capacity()
    {
        return Err("model logprob capacity differs from its declared output buffers".into());
    }
    config.contract.validate(manifest)?;
    validate_context_variants(manifest)?;
    validate_context_variant_plans(config, manifest)?;
    config.sampling_ranges.validate(&config.sampling_defaults)?;
    if config
        .sampling_ranges
        .top_k
        .is_some_and(|range| range.max > config.contract.vocabulary as i32)
    {
        return Err("sampling top_k range exceeds the prepared vocabulary".into());
    }
    config.limits.validate_prefill()?;
    let page_tokens = crate::model::scheduler_capabilities(config).page_tokens;
    if config.model_id.is_empty()
        || config.limits.max_active == 0
        || config.limits.max_active > manifest.max_active
        || config.limits.request_slots != manifest.request_capacity()
        || config.limits.max_sequence_tokens == 0
        || config.limits.max_sequence_tokens > config.prepared_sequence_tokens
        // A request's last page may be partial: context-parallel pages (768 positions over three
        // ranks) do not divide a 2^20 context, and every admission rounds its pages up.
        || page_tokens == 0
        || config.limits.pages == 0
        || config.limits.pages > i32::MAX as u32
        || config.limits.snapshots > i32::MAX as u32 - (config.limits.request_capacity() + 1) as u32
        || config.max_new_tokens == 0
        || config.max_new_tokens > i32::MAX as u32
        || config.sampling_defaults.top_k > config.contract.vocabulary as i32
        || config.eos_token_ids.len() > 8
        || config
            .eos_token_ids
            .iter()
            .any(|id| *id as usize >= config.contract.vocabulary)
    {
        return Err("invalid prepared model configuration".into());
    }
    speculation::validate_widths(&config.speculative_widths)?;
    if config.completion_token_capacity != manifest.completion_token_capacity
        || config.speculative_widths.iter().any(|&width| width > manifest.completion_token_capacity) {
        return Err("model widths exceed the declared completion capacity".into());
    }
    if !config.speculative_widths.is_empty() && config.speculative_token_capacity == 0 {
        return Err("speculative token capacity must be positive".into());
    }
    if config
        .speculative_fixed_width
        .is_some_and(|w| !config.speculative_widths.contains(&w))
    {
        return Err(
            "speculative_fixed_width must belong to the prepared speculative_widths".into(),
        );
    }
    if config.speculative_widths.is_empty() {
        if config.speculative_proposal_source == speculation::ProposalSource::Copy {
            return Err("copy proposal source requires speculative graphs".into());
        }
    }
    for graph in &manifest.graphs {
        if matches!(graph.kind, GraphKind::Speculative | GraphKind::Mixed)
            && graph.token_capacity > 1
            && (config.history_drafts
                || config.speculative_proposal_source == speculation::ProposalSource::Copy)
        {
            use super::contract::ValueSource::*;
            let inputs = &config.contract.graph(&graph.name)?.inputs;
            let mut required = vec![
                CopyTokenIds,
                CopyTokenCount,
                CopyConsumedTokens,
                CopyGeneration,
            ];
            if config.speculative_proposal_source == speculation::ProposalSource::Copy {
                required.extend([CopyGrammarMask, CopyGrammarCount]);
            }
            if required
                .iter()
                .any(|source| !inputs.iter().any(|input| input.source == *source))
            {
                return Err(
                    "copied proposals require prepared token, generation, cursor, and mask inputs"
                        .into(),
                );
            }
        }
        select_shape_with_drafts(
            config,
            manifest,
            graph.kind,
            graph.batch_capacity,
            graph.token_capacity,
            graph.prefill_capacity,
            graph.packed_token_capacity,
            graph.draft_tokens,
            graph.sampling_mode,
        )?;
    }
    for kind in [
        GraphKind::Init,
        GraphKind::Restore,
        GraphKind::Snapshot,
        GraphKind::Resume,
        GraphKind::Decode,
    ] {
        if kind == GraphKind::Decode && !config.speculative_widths.is_empty() {
            continue;
        }
        for batch in batches(config) {
            select(config, manifest, kind, batch, 1, 0, SamplingMode::General)?;
        }
    }
    select(
        config,
        manifest,
        GraphKind::Prefill,
        1,
        config.limits.prefill_chunk_size,
        0,
        SamplingMode::General,
    )?;
    for &tokens in &config.contract.vision.patch_capacities {
        select(
            config,
            manifest,
            GraphKind::Vision,
            1,
            tokens,
            0,
            SamplingMode::General,
        )?;
    }
    if !config.speculative_widths.is_empty() {
        for batch in batches(config) {
            if !manifest.graphs.iter().any(|graph| graph.kind == GraphKind::Speculative
                && graph.batch_capacity == batch && graph.token_capacity == 1) {
                return Err(format!("prepared model lacks width-one speculative graph for batch {batch}").into());
            }
            if batch * 2 <= config.speculative_token_capacity
                && !manifest.graphs.iter().any(|graph| graph.kind == GraphKind::Speculative
                    && graph.batch_capacity == batch && graph.token_capacity > 1) {
                return Err(format!("prepared model lacks speculative graph for batch {batch}").into());
            }
        }
    }
    Ok(())
}

/// TP3 fork (DS41 `tp3.index_row_split_min_context`): a prefill or encoder-only graph named
/// `<base>-split-ctx<tokens>` is a long-context variant of the graph named `<base>`: same kind,
/// shape and uploads, a different kernel plan (the indexer row split). Ordinary selection never
/// returns it; [`select_context_variant`] swaps it in for a chunk that starts at `tokens` or later.
/// Keep the mark in step with `index_row_split.CONTEXT_VARIANT_MARK` (ignition).
pub const CONTEXT_VARIANT_MARK: &str = "-split-ctx";

/// `(base name, minimum start position)` of a context variant graph name, else `None`.
pub fn context_variant(name: &str) -> Option<(&str, u64)> {
    let (base, tail) = name.rsplit_once(CONTEXT_VARIANT_MARK)?;
    if base.is_empty() || tail.is_empty() || !tail.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    tail.parse::<u64>().ok().filter(|&min| min > 0).map(|min| (base, min))
}

/// TP3 fork: the graph a one-lane prefill chunk starting at position `start` runs, given the
/// graph `base` ordinary selection chose: the context variant of `base` with the largest minimum
/// start position not above `start`, else `base` itself.
pub fn select_context_variant(manifest: &ExecutionManifest, base: usize, start: u64) -> usize {
    let Some(chosen) = manifest.graphs.get(base) else { return base };
    if !matches!(chosen.kind, GraphKind::Prefill | GraphKind::EncoderPrefill) || chosen.batch_capacity != 1 {
        return base;
    }
    let mut best = (base, 0u64);
    for (index, graph) in manifest.graphs.iter().enumerate() {
        if let Some((name, min)) = context_variant(&graph.name)
            && name == chosen.name
            && min <= start
            && min > best.1
            && same_shape(graph, chosen)
        {
            best = (index, min);
        }
    }
    best.0
}

/// TP3 fork: true for a context variant graph name (see [`CONTEXT_VARIANT_MARK`]).
pub fn is_context_variant(name: &str) -> bool {
    context_variant(name).is_some()
}

/// TP3 fork: the graphs the scheduler plans with. A context variant shares its base's shape (the
/// runtime swaps it in at submit), so the shape catalogs and the cost model see the base only.
pub fn scheduling_graphs(graphs: &[crate::execution::GraphDescriptor]) -> Vec<crate::execution::GraphDescriptor> {
    graphs.iter().filter(|graph| !is_context_variant(&graph.name)).cloned().collect()
}

fn same_shape(a: &crate::execution::GraphDescriptor, b: &crate::execution::GraphDescriptor) -> bool {
    a.kind == b.kind
        && a.batch_capacity == b.batch_capacity
        && a.token_capacity == b.token_capacity
        && a.prefill_capacity == b.prefill_capacity
        && a.packed_token_capacity == b.packed_token_capacity
        && a.draft_tokens == b.draft_tokens
        && a.sampling_mode == b.sampling_mode
}

/// TP3 fork: every context variant names an existing one-lane prefill / encoder-only base graph
/// of the same shape and binds exactly its uploads (the runtime packs the step before it picks).
pub(crate) fn validate_context_variants(manifest: &ExecutionManifest) -> Result<()> {
    fn uploads(graph: &crate::execution::GraphDescriptor) -> Vec<&str> {
        let mut names: Vec<_> = graph.input_bindings.iter().map(|b| b.buffer_name.as_str()).collect();
        names.sort_unstable();
        names
    }
    for graph in &manifest.graphs {
        let Some((base, _)) = context_variant(&graph.name) else { continue };
        let ok = matches!(graph.kind, GraphKind::Prefill | GraphKind::EncoderPrefill)
            && graph.batch_capacity == 1
            && manifest.graphs.iter().any(|other| other.name == base && same_shape(other, graph)
                && uploads(other) == uploads(graph));
        if !ok {
            return Err(format!(
                "context variant {} lacks a one-lane prefill base {base} with the same shape and uploads",
                graph.name
            )
            .into());
        }
    }
    Ok(())
}

/// TP3 fork: a context variant's semantic input plan (source, indexing, offset, padding, tensor
/// of every upload) equals its base's: the runtime packs a step with the plan of the shape's
/// first contract graph and swaps the variant in afterwards (review C2; as the encoder twins).
pub(crate) fn validate_context_variant_plans(config: &ModelConfig, manifest: &ExecutionManifest) -> Result<()> {
    let plan = |name: &str| -> Result<Vec<_>> {
        let mut inputs: Vec<_> = config.contract.graph(name)?.inputs.iter().map(|b| (b.buffer.clone(),
            b.source as i32, b.indexing as i32, b.value_offset, b.padding, b.tensor.clone())).collect();
        inputs.sort();
        Ok(inputs)
    };
    for graph in &manifest.graphs {
        let Some((base, _)) = context_variant(&graph.name) else { continue };
        if plan(&graph.name)? != plan(base)? {
            return Err(format!("context variant {} binds a different input plan than {base}", graph.name).into());
        }
    }
    Ok(())
}

fn speculative_shape(config: &ModelConfig, batch: usize, tokens: usize) -> bool {
    batches(config).any(|capacity| capacity == batch)
        && config.speculative_widths.contains(&tokens)
        && (tokens == 1 || (batch * tokens <= config.speculative_token_capacity && config
            .speculative_fixed_width
            .is_none_or(|width| width == tokens)))
}

fn batches(config: &ModelConfig) -> impl Iterator<Item = usize> {
    BATCH_CAPACITIES
        .into_iter()
        .filter(|batch| *batch <= config.limits.max_active.next_power_of_two())
}

pub fn select(
    config: &ModelConfig,
    manifest: &ExecutionManifest,
    kind: GraphKind,
    batch: usize,
    tokens: usize,
    prefill: usize,
    mode: SamplingMode,
) -> Result<usize> {
    if kind == GraphKind::Speculative {
        let mut matching = manifest.graphs.iter().enumerate().filter(|(_, graph)|
            graph.kind == kind && graph.batch_capacity == batch && graph.token_capacity == tokens
                && graph.prefill_capacity == prefill && graph.sampling_mode == mode);
        let (index, _) = matching.next().ok_or_else(|| format!("prepared model lacks speculative graph for batch {batch}, width {tokens}"))?;
        if matching.next().is_some() {
            return Err("speculative graph selection requires explicit draft and target-row capacities".into());
        }
        return Ok(index);
    }

    select_shape(config, manifest, kind, batch, tokens, prefill, 0, mode)
}

pub fn select_shape(
    config: &ModelConfig, manifest: &ExecutionManifest, kind: GraphKind,
    batch: usize, tokens: usize, prefill: usize, packed: usize, mode: SamplingMode,
) -> Result<usize> {
    select_shape_with_drafts(config, manifest, kind, batch, tokens, prefill, packed, 0, mode)
}

pub fn select_shape_with_drafts(
    config: &ModelConfig, manifest: &ExecutionManifest, kind: GraphKind,
    batch: usize, tokens: usize, prefill: usize, packed: usize, drafts: usize, mode: SamplingMode,
) -> Result<usize> {
    let declared = batches(config).any(|capacity| capacity == batch)
        && match kind {
            GraphKind::Mixed => {
                batch >= 2
                    && prefill > 0
                    && prefill.is_multiple_of(64)
                    && prefill <= config.limits.prefill_chunk_size
                    && if config.speculative_widths.is_empty() {
                        tokens == 1
                    } else {
                        speculative_shape(config, batch, tokens)
                    }
            }
            GraphKind::Prefill | GraphKind::ScoredPrefill | GraphKind::EncoderPrefill => {
                batch == 1
                    && tokens > 0
                    && tokens <= config.limits.prefill_chunk_size
                    && tokens.is_multiple_of(64)
            }
            GraphKind::Vision => {
                batch == 1 && config.contract.vision.patch_capacities.contains(&tokens)
            }
            GraphKind::Speculative => {
                config.speculative_widths.contains(&tokens)
                    && drafts < config.completion_token_capacity
                    && packed >= batch && packed <= batch.saturating_mul(tokens)
                    && (tokens == 1 || packed <= config.speculative_token_capacity)
            }
            _ => tokens == 1,
        };
    if !declared || (kind != GraphKind::Mixed && prefill != 0) {
        return Err(format!(
            "undeclared prepared graph kind or capacity: {kind:?}, batch {batch}, tokens {tokens}"
        )
        .into());
    }
    let mut matches = manifest.graphs.iter().enumerate().filter(|(_, graph)| {
        graph.kind == kind
            && context_variant(&graph.name).is_none()
            && graph.batch_capacity == batch
            && graph.token_capacity == tokens
            && graph.prefill_capacity == prefill
            && graph.packed_token_capacity == packed
            && graph.draft_tokens == drafts
            && graph.sampling_mode == mode
    });
    let (index, _) = matches.next().ok_or_else(|| {
        format!("prepared model lacks {kind:?}/{mode:?} graph with batch {batch}, tokens {tokens}")
    })?;
    if matches.next().is_some() {
        return Err("ambiguous prepared model graph capacity".into());
    }
    Ok(index)
}

#[cfg(test)]
mod context_variant_tests {
    use super::*;
    use crate::execution::{GraphDescriptor, InputBinding};

    fn graph(name: &str, kind: GraphKind, tokens: usize, inputs: &[&str]) -> GraphDescriptor {
        GraphDescriptor {
            tuned_bindings: vec![], confidence_width: 0, draft_tokens: 0, packed_token_capacity: 0,
            name: name.into(), kind, sampling_mode: SamplingMode::General, handle: 0x5000,
            batch_capacity: 1, token_capacity: tokens, prefill_capacity: 0, estimated_duration_ns: 0,
            input_bindings: inputs.iter().map(|b| InputBinding { buffer_name: (*b).into(), host_offset: 0, nbytes: 4 }).collect(),
        }
    }

    fn manifest(graphs: Vec<GraphDescriptor>) -> ExecutionManifest {
        ExecutionManifest { completion_token_capacity: 16, tuning_graphs: vec![], engram: None, collectives: None,
            cache: None, offload: None, abi_version: crate::execution::ABI_VERSION, context: 1, device: 0,
            max_active: 1, input_frame_bytes: 0, buffers: vec![], graphs }
    }

    fn ds41() -> ExecutionManifest {
        manifest(vec![
            graph("prefill3584", GraphKind::Prefill, 3584, &["tokens", "positions"]),
            graph("prefill4096", GraphKind::Prefill, 4096, &["tokens", "positions"]),
            graph("prefill4096-split-ctx32768", GraphKind::Prefill, 4096, &["positions", "tokens"]),
            graph("encoder_prefill4096", GraphKind::EncoderPrefill, 4096, &["tokens", "positions"]),
            graph("encoder_prefill4096-split-ctx32768", GraphKind::EncoderPrefill, 4096, &["tokens", "positions"]),
            graph("encoder_prefill4096-split-ctx262144", GraphKind::EncoderPrefill, 4096, &["tokens", "positions"]),
        ])
    }

    #[test]
    fn context_variant_names_parse_strictly() {
        assert_eq!(context_variant("prefill4096-split-ctx32768"), Some(("prefill4096", 32768)));
        assert_eq!(context_variant("encoder_prefill4096-no-mtp-split-ctx65536"), Some(("encoder_prefill4096-no-mtp", 65536)));
        for name in ["prefill4096", "prefill4096-no-mtp", "prefill4096-split-ctx", "prefill4096-split-ctx0",
                     "prefill4096-split-ctx12x", "-split-ctx4096", "prefill4096-split-ctx-5"] {
            assert_eq!(context_variant(name), None, "{name}");
        }
    }

    #[test]
    fn variants_are_chosen_by_start_position_and_never_by_ordinary_selection() {
        let m = ds41();
        let index = |name: &str| m.graphs.iter().position(|g| g.name == name).unwrap();
        let (prefill, encoder) = (index("prefill4096"), index("encoder_prefill4096"));
        // ordinary selection: the replicated graph, never a variant, and no ambiguity
        let mut config_free = m.graphs.iter().enumerate().filter(|(_, g)| g.kind == GraphKind::Prefill
            && g.token_capacity == 4096 && context_variant(&g.name).is_none());
        assert_eq!(config_free.next().map(|(i, _)| i), Some(prefill));
        assert!(config_free.next().is_none());
        for (start, want) in [(0, "prefill4096"), (32767, "prefill4096"), (32768, "prefill4096-split-ctx32768"),
                              (1 << 20, "prefill4096-split-ctx32768")] {
            assert_eq!(m.graphs[select_context_variant(&m, prefill, start)].name, want, "start {start}");
        }
        for (start, want) in [(4096, "encoder_prefill4096"), (40960, "encoder_prefill4096-split-ctx32768"),
                              (262143, "encoder_prefill4096-split-ctx32768"), (262144, "encoder_prefill4096-split-ctx262144")] {
            assert_eq!(m.graphs[select_context_variant(&m, encoder, start)].name, want, "start {start}");
        }
        // a capacity without a variant stays put at any start
        let other = index("prefill3584");
        assert_eq!(select_context_variant(&m, other, 1 << 20), other);
        assert!(validate_context_variants(&m).is_ok());
    }

    #[test]
    fn variants_must_match_their_base() {
        let mut orphan = ds41();
        orphan.graphs.push(graph("prefill2048-split-ctx32768", GraphKind::Prefill, 2048, &["tokens", "positions"]));
        assert!(validate_context_variants(&orphan).is_err());
        let mut shape = ds41();
        shape.graphs[2].token_capacity = 3584;
        assert!(validate_context_variants(&shape).is_err());
        let mut uploads = ds41();
        uploads.graphs[2].input_bindings.pop();
        assert!(validate_context_variants(&uploads).is_err());
        let mut decode = ds41();
        decode.graphs.push(graph("decode1-split-ctx32768", GraphKind::Decode, 1, &[]));
        decode.graphs.push(graph("decode1", GraphKind::Decode, 1, &[]));
        assert!(validate_context_variants(&decode).is_err());
    }
}
