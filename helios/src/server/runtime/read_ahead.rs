/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! TP3 fork (perf/engram-readahead, after c2606aae `prefetch_rows`): hands prompt positions of
//! prefill lanes to the disk-backed Engram tables' reader threads before each step is launched,
//! so their rows are read while earlier steps replay instead of while this step's graph waits.
//! Prompt tokens are host-known at admission; decode, speculative and draft rows are graph-
//! produced and are not predicted. All ranks of the TP3/TP2 process share one table set and its
//! readers, so rank zero's binding reaches every rank. The graph stays the authority: it
//! publishes the ids its lookup consumes and the gather serves exactly those.

use crate::scheduler::{RequestHandle, RequestScheduler, ScheduledPlan};
use std::sync::Arc;

pub(super) struct ReadAhead {
    shared: Arc<crate::engram::Shared>,
    /// Per request slot, the prompt position through which rows were handed to the readers.
    progress: Vec<Option<(RequestHandle, usize)>>,
    /// Per table, the read-ahead tokens this step's prefill lanes have left to share.
    budget: Vec<usize>,
}

impl ReadAhead {
    /// Some only when a disk-backed table reads ahead (FLARE_ENGRAM_READ_AHEAD=1).
    pub(super) fn new(shared: &Arc<crate::engram::Shared>, slots: usize) -> Option<Self> {
        shared.reads_ahead().then(|| Self {
            shared: shared.clone(),
            progress: vec![None; slots],
            budget: Vec::new(),
        })
    }

    /// Called in `Worker::submit` for the step with submission sequence `sequence`, before launch.
    pub(super) fn submit(
        &mut self,
        step: &ScheduledPlan,
        sequence: u64,
        scheduler: &RequestScheduler,
        admissions: &[Option<crate::model::Admission>],
    ) {
        let shared = &*self.shared;
        // Requests that left since the last step: their queued windows are dropped unread.
        for (slot, entry) in self.progress.iter_mut().enumerate() {
            if entry.is_some_and(|(request, _)| scheduler.request(request).is_err()) {
                shared.read_ahead_release(slot);
                *entry = None;
            }
        }
        if !crate::engram::consumes(step.kind) {
            return;
        }
        let tables = shared.index.tables.len();
        // One bound per table for every prompt of the step together, so several requests
        // prefilling at once cannot each pin a full share of the row cache.
        self.budget.clear();
        self.budget.extend((0..tables).map(|table| shared.read_ahead_tokens(table)));
        for (row, lane) in step.lanes.iter().enumerate() {
            if !step.lane_kind(row).is_prefill() {
                continue;
            }
            let slot = lane.request.slot as usize;
            let Ok(live) = scheduler.request(lane.request) else {
                continue;
            };
            let tokens = live.tokens();
            let admission = admissions.get(slot).and_then(Option::as_ref);
            let excluded = |position: usize| {
                admission.is_some_and(|admission| admission.data.is_image_position(position))
            };
            let first = lane.start_position as usize;
            let launched = (first + lane.tokens).min(tokens.len());
            let generation = lane.request.generation;
            // The reader drops windows of positions this step launched once its gather is served.
            shared.read_ahead_launch(slot, generation, sequence, launched);
            let progress = match self.progress.get(slot) {
                Some(Some((request, end))) if *request == lane.request => (*end).max(first),
                _ => first,
            };
            let mut reached = usize::MAX;
            for table in (0..tables).filter(|table| shared.table_reads_ahead(*table)) {
                let share = self.budget[table];
                // Never stop short of the launching chunk: its rows are needed now. Windows are
                // split at the chunk end so none straddles two gathers.
                let limit = (first + share).max(launched).min(tokens.len());
                let mut position = progress;
                for end in [launched, limit] {
                    if position < end {
                        let reached = shared.read_ahead(table, slot, generation, tokens, position, end, excluded);
                        if reached < end {
                            position = reached;
                            break;
                        }
                        position = end;
                    }
                }
                reached = reached.min(position);
                // Rows queued in earlier steps stay pinned until used, so the lane's whole
                // extent past the graph's position counts against the step's bound.
                self.budget[table] = share.saturating_sub(position.saturating_sub(first));
            }
            if reached != usize::MAX {
                if let Some(entry) = self.progress.get_mut(slot) {
                    *entry = Some((lane.request, reached));
                }
            }
        }
    }
}
