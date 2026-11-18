/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! TP3 fork: encoder-only prefill twins in the prepared contract, the execution manifest and the
//! graph catalog.

use super::contract::{GraphOperation, ValueSource};
use super::*;
use crate::execution::{GraphDescriptor, GraphKind};

const TWIN: &str = "EncoderPrefill-1-512";

/// The test model plus an encoder-only twin of its 512-token prefill graph.
fn with_twin() -> (ModelConfig, ExecutionManifest) {
    let mut config = super::tests::config();
    let mut manifest = super::testing::manifest(&config);
    // The shared test catalog also lists multi-row prefill shapes no prepared model has.
    manifest.graphs.retain(|g| g.kind != GraphKind::Prefill || g.batch_capacity == 1);
    config.contract.graphs.retain(|g| manifest.graphs.iter().any(|d| d.name == g.graph));
    let mut twin = config.contract.graphs.iter().find(|g| g.kind == GraphOperation::Prefill).unwrap().clone();
    twin.graph = TWIN.into();
    twin.kind = GraphOperation::EncoderPrefill;
    config.contract.graphs.push(twin);
    let descriptor = manifest.graphs.iter().find(|g| g.kind == GraphKind::Prefill).unwrap().clone();
    assert_eq!(descriptor.token_capacity, 512);
    manifest.graphs.push(GraphDescriptor { name: TWIN.into(), kind: GraphKind::EncoderPrefill, ..descriptor });
    for (index, graph) in manifest.graphs.iter_mut().enumerate() {
        graph.handle = index + 1;
    }
    (config, manifest)
}

#[test]
fn encoder_prefill_is_a_prefill_operation_that_does_not_sample() {
    assert_eq!(GraphKind::try_from("encoder_prefill").unwrap(), GraphKind::EncoderPrefill);
    assert_eq!(GraphOperation::try_from(14).unwrap(), GraphOperation::EncoderPrefill);
    assert_eq!(GraphOperation::try_from(GraphKind::EncoderPrefill).unwrap(), GraphOperation::EncoderPrefill);
    assert!(GraphKind::EncoderPrefill.is_prefill());
    assert!(!GraphKind::EncoderPrefill.samples());
    // It reads the Engram host tables like every prompt graph.
    assert!(crate::engram::consumes(GraphKind::EncoderPrefill));
}

#[test]
fn a_twin_with_its_prefill_graphs_shape_and_uploads_validates_and_selects_apart() {
    let (config, manifest) = with_twin();
    manifest.validate().unwrap();
    config.contract.validate(&manifest).unwrap();
    let twin = manifest.graphs.iter().position(|g| g.name == TWIN).unwrap();
    let prefill = manifest.graphs.iter().position(|g| g.kind == GraphKind::Prefill).unwrap();
    let select = |kind| catalog::select(&config, &manifest, kind, 1, 512, 0, crate::execution::SamplingMode::General);
    assert_eq!(select(GraphKind::EncoderPrefill).unwrap(), twin);
    assert_eq!(select(GraphKind::Prefill).unwrap(), prefill);
}

#[test]
fn a_twin_without_a_matching_prefill_graph_is_refused() {
    // Different uploads in the execution manifest.
    let (_, mut manifest) = with_twin();
    manifest.graphs.last_mut().unwrap().input_bindings.retain(|b| b.buffer_name != "prefill_positions");
    assert!(manifest.validate().unwrap_err().to_string().contains("lacks a prefill twin"));
    // No prefill graph of its capacity.
    let (_, mut manifest) = with_twin();
    manifest.graphs.retain(|g| g.kind != GraphKind::Prefill);
    assert!(manifest.validate().unwrap_err().to_string().contains("lacks a prefill twin"));
    // A window-sized twin.
    let (_, mut manifest) = with_twin();
    manifest.graphs.last_mut().unwrap().token_capacity = 128;
    assert!(manifest.validate().unwrap_err().to_string().contains("unsupported graph capacities"));
    // Same buffers, different semantic source: the runtime would pack it with the wrong plan.
    let (mut config, manifest) = with_twin();
    let twin = config.contract.graphs.last_mut().unwrap();
    let binding = twin.inputs.iter_mut().find(|b| b.source == ValueSource::PromptPosition).unwrap();
    binding.source = ValueSource::PromptToken;
    assert!(config.contract.validate(&manifest).unwrap_err().to_string()
        .contains("prefill twin's uploads"));
}
