/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! TP3 fork: encoder-only prefill chunks (`encoder_only_chunk`, `encoder_prefill_step`).
//!
//! An encoder-only chunk leaves the decoder-suffix state (layer >= 20 windows, draft ring,
//! `last_hidden`) stale at its endpoint. These tests hold the scheduler side of that contract:
//! only non-final chunks longer than the window, ending at no checkpoint and followed by more
//! than a window of prompt, are eligible; the next chunk is never window-sized; no snapshot or
//! checkpoint is ever taken at a stale endpoint; the switch and the package's twin catalog gate
//! the decision, and the counters count what ran.

use super::*;

const WINDOW: u64 = SUFFIX_WINDOW as u64;

fn ds41_engine(encoders: &[usize]) -> RequestScheduler {
    let mut s = tp3_engine(1, |kind, batch, tokens| match kind {
        GraphKind::Prefill => 27_400_000 + tokens * 67_880,
        GraphKind::Decode => 12 * MS + (batch - 1) * MS / 4,
        GraphKind::Speculative => 12_500_000 + (tokens - 1) * 300_000 + (batch - 1) * 300_000,
        _ => MS,
    });
    s.encoder_prefill_graphs(encoders).unwrap();
    s
}

fn admit_with(s: &mut RequestScheduler, prompt: usize, boundaries: Vec<usize>) -> RequestHandle {
    s.admit(RequestSpec {
        prompt: (0..prompt as u32).map(|t| t % 1000).collect(),
        message_boundaries: boundaries,
        conditioning: vec![],
        max_new_tokens: 4,
        grammar_constrained: false,
        host_stops: false,
        prompt_logprobs: false,
        growable_output: false,
        tools: false,
        reasoning_prompt: false,
    })
    .unwrap()
}

/// Completes a committed step as `ds41_step` does (speculative records offer at most the lane's drafts).
fn finish_step(s: &mut RequestScheduler, plan: &ScheduledPlan, fence: Fence) {
    finish_step_timed(s, plan, fence, plan.estimated_duration_ns());
}

/// `finish_step` with a measured GPU time of `gpu_ns`.
fn finish_step_timed(s: &mut RequestScheduler, plan: &ScheduledPlan, fence: Fence, gpu_ns: u64) {
    s.advance_to(s.now_ns + plan.estimated_duration_ns()).unwrap();
    let mut records = completion_records(s, plan, fence, 0);
    for lane in &plan.lanes {
        if let Some(vector) = records[lane.request.slot as usize].vector.as_mut() {
            vector.offered_drafts = vector.offered_drafts.min(lane.tokens as i64 - 1);
        }
    }
    for token in s.complete(fence, &records, gpu_time(gpu_ns)).unwrap() {
        s.published_output(token.request, fence).unwrap();
    }
    let finished: Vec<_> = s.slots.iter().filter_map(|slot| slot.request.as_ref())
        .filter(|r| r.phase == RequestPhase::Finishing && r.pending == 0).map(|r| r.handle).collect();
    for h in finished {
        s.finish(h).unwrap();
    }
}

/// One planned step: plan, commit (with the encoder decision the runtime would take), complete.
/// Returns the plan and whether the runtime would have run the encoder-only twin.
fn run_step(s: &mut RequestScheduler) -> (ScheduledPlan, bool) {
    let plan = s.plan().expect("runnable trace must make progress");
    let encoder = s.encoder_prefill_step(&plan).unwrap();
    let fence = Fence(s.submitted.0 + 1);
    s.commit(&plan, fence).unwrap();
    finish_step(s, &plan, fence);
    (plan, encoder)
}

#[test]
fn encoder_only_chunks_need_a_window_exceeding_successor_and_no_checkpoint() {
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 1400, 32);
    let r = s.request(h).unwrap();
    assert!(encoder_only_chunk(r, 0, 512));
    assert!(encoder_only_chunk(r, 512, 512));
    // Final chunk, window-sized chunk, or a successor that fits the window: ordinary graph.
    assert!(!encoder_only_chunk(r, 1024, 376));
    assert!(!encoder_only_chunk(r, 0, 128));
    assert!(!encoder_only_chunk(r, 760, 512));
    assert!(encoder_only_chunk(r, 759, 512)); // 1271 + 128 < 1400
    s.request_mut(h).unwrap().message_boundaries = vec![512, 700, 1100];
    let r = s.request(h).unwrap();
    // A checkpoint is captured at 512, and the next cut (700) is within the window of 512 + 64.
    assert!(!encoder_only_chunk(r, 0, 512));
    assert!(!encoder_only_chunk(r, 64, 512));
    // Ends at the 700 checkpoint; without checkpoints the same chunk qualifies.
    assert!(!encoder_only_chunk(r, 188, 512));
    s.request_mut(h).unwrap().message_boundaries = vec![1100];
    assert!(encoder_only_chunk(s.request(h).unwrap(), 188, 512));
    // 1100 is a cut: a chunk ending at 1000 would leave a 100-token successor.
    assert!(!encoder_only_chunk(s.request(h).unwrap(), 488, 512));
    s.request_mut(h).unwrap().prompt_logprobs = true;
    assert!(!encoder_only_chunk(s.request(h).unwrap(), 0, 512));
    s.request_mut(h).unwrap().prompt_logprobs = false;
    s.request_mut(h).unwrap().prefill_spans = vec![900..1000];
    assert!(!encoder_only_chunk(s.request(h).unwrap(), 0, 512));
    s.request_mut(h).unwrap().prefill_spans.clear();
    s.request_mut(h).unwrap().vision_patches.push_back(576);
    assert!(!encoder_only_chunk(s.request(h).unwrap(), 0, 512));
}

#[test]
fn encoder_graphs_must_twin_prefill_capacities_above_the_window_before_admission() {
    let mut s = ds41_engine(&[]);
    assert!(s.encoder_prefill_graphs(&[128]).is_err());
    assert!(s.encoder_prefill_graphs(&[4000]).is_err());
    assert!(s.encoder_prefill_graphs(&[4096, 4096]).is_err());
    s.encoder_prefill_graphs(&[4096, 3072, 3584]).unwrap();
    assert_eq!(s.encoder_prefill_catalog, [3072, 3584, 4096]);
    admit_with(&mut s, 1000, vec![]);
    assert!(s.encoder_prefill_graphs(&[4096]).is_err());
}

/// Every chunk of a cold prompt: (start, tokens, graph capacity, encoder-only, marker after commit).
fn prompt_chunks(s: &mut RequestScheduler, h: RequestHandle) -> Vec<(u64, usize, usize, bool)> {
    let mut chunks = Vec::new();
    for _ in 0..400 {
        if s.request(h).is_err() || s.request(h).unwrap().phase == RequestPhase::Decode {
            break;
        }
        let (plan, encoder) = run_step(s);
        if plan.kind == GraphKind::Prefill && plan.lanes[0].request == h {
            let lane = &plan.lanes[0];
            chunks.push((lane.start_position, lane.tokens, plan.token_capacity, encoder));
        }
    }
    chunks
}

#[test]
fn cold_prompt_runs_every_eligible_non_final_chunk_encoder_only() {
    let prompt = 30_000;
    let mut s = ds41_engine(&[3072, 3584, 4096]);
    let h = admit_with(&mut s, prompt, vec![]);
    let chunks = prompt_chunks(&mut s, h);
    assert_eq!(chunks.last().map(|&(start, tokens, _, _)| start + tokens as u64), Some(prompt as u64));
    let encoders: Vec<_> = chunks.iter().filter(|chunk| chunk.3).collect();
    assert!(!encoders.is_empty(), "no encoder-only chunk ran: {chunks:?}");
    for (index, &(start, tokens, capacity, encoder)) in chunks.iter().enumerate() {
        let end = start + tokens as u64;
        let eligible = [3072, 3584, 4096].contains(&capacity) && tokens as u64 > WINDOW
            && end + WINDOW < prompt as u64;
        assert_eq!(encoder, eligible, "chunk {index} {start}+{tokens} in {capacity}: {chunks:?}");
        if encoder {
            // The successor is longer than the window, so it replays its own suffix state.
            assert!(chunks[index + 1].1 as u64 > WINDOW);
        }
    }
    assert!(!chunks.last().unwrap().3, "the final chunk always samples");
    let stats = s.stats();
    assert_eq!(stats.encoder_prefill_chunks, encoders.len() as u64);
    assert_eq!(stats.encoder_prefill_tokens, encoders.iter().map(|chunk| chunk.1 as u64).sum::<u64>());
    // The prompt still reaches its first token after the last, ordinary chunk.
    assert!(s.request(h).is_err() || s.request(h).unwrap().generated_tokens > 0
        || s.request(h).unwrap().phase == RequestPhase::Decode);
}

#[test]
fn the_switch_and_the_twin_catalog_gate_encoder_only_chunks() {
    for (enabled, catalog) in [(false, vec![3072, 3584, 4096]), (true, vec![]), (true, vec![256])] {
        let mut s = ds41_engine(&catalog);
        s.configure_policy(PolicyConfig { encoder_prefill: enabled, ..s.policy.clone() }).unwrap();
        let h = admit_with(&mut s, 20_000, vec![]);
        let chunks = prompt_chunks(&mut s, h);
        assert!(chunks.iter().all(|chunk| !chunk.3), "{enabled} {catalog:?}: {chunks:?}");
        assert_eq!(s.stats().encoder_prefill_chunks, 0);
    }
}

#[test]
fn chunks_ending_at_checkpoints_or_near_cuts_run_the_ordinary_graph() {
    // Message boundaries at 4096 (a checkpoint end) and 8250 (a cut 58 tokens after 8192).
    let mut s = ds41_engine(&[3072, 3584, 4096]);
    let h = admit_with(&mut s, 20_000, vec![4096, 8250]);
    let chunks = prompt_chunks(&mut s, h);
    for &(start, tokens, _, encoder) in &chunks {
        let end = start + tokens as u64;
        if [4096, 8250].contains(&end) || (end < 8250 && end + WINDOW >= 8250) || end == 20_000 {
            assert!(!encoder, "{start}+{tokens}: {chunks:?}");
        }
    }
    assert!(chunks.iter().any(|chunk| chunk.3));
}

#[test]
fn a_chunk_after_an_encoder_only_chunk_is_never_window_sized() {
    let mut s = cold_engine(1, false);
    s.prefill_graphs(&[64, 128, 512]).unwrap();
    let h = add(&mut s, 1, 1400, 32);
    assert_eq!(measured_step(&mut s, MS / 10, 0).kind, GraphKind::Init);
    // Force the illegal shape: a cut 100 tokens ahead while the suffix state is stale.
    s.request_mut(h).unwrap().message_boundaries = vec![100];
    s.request_mut(h).unwrap().encoder_only_through = Some(0);
    assert!(s.request(h).unwrap().suffix_stale());
    for capacity in [64, 128, 512] {
        assert_eq!(s.prefill_work(s.request(h).unwrap(), capacity), 0);
    }
    assert!(s.plan().is_none_or(|plan| plan.kind != GraphKind::Prefill));
    s.request_mut(h).unwrap().message_boundaries = vec![300];
    // A longer successor is still allowed, and a window-sized capacity cannot serve it.
    assert_eq!(s.prefill_work(s.request(h).unwrap(), 128), 0);
    assert_eq!(s.prefill_work(s.request(h).unwrap(), 512), 300);
    s.request_mut(h).unwrap().message_boundaries = vec![100];
    s.request_mut(h).unwrap().encoder_only_through = None;
    let plan = s.plan().unwrap();
    assert_eq!((plan.kind, plan.lanes[0].tokens), (GraphKind::Prefill, 100));
}

#[test]
fn the_marker_is_replaced_by_the_next_chunk_and_never_outlives_it() {
    let mut s = ds41_engine(&[3072, 3584, 4096]);
    let h = admit_with(&mut s, 12_000, vec![]);
    let mut seen_stale = false;
    for _ in 0..100 {
        if s.request(h).unwrap().phase == RequestPhase::Decode {
            break;
        }
        let (plan, encoder) = run_step(&mut s);
        let r = s.request(h).unwrap();
        if plan.kind == GraphKind::Prefill {
            let end = plan.lanes[0].start_position + plan.lanes[0].tokens as u64;
            assert_eq!(r.encoder_only_through, encoder.then_some(end));
            assert_eq!(r.suffix_stale(), encoder);
            seen_stale |= encoder;
        }
    }
    assert!(seen_stale);
    assert_eq!(s.request(h).unwrap().encoder_only_through, None, "the final chunk clears the marker");
}

#[test]
fn stale_endpoints_are_never_snapshotted() {
    // finish_early (pages exhausted, cancellation) keeps a terminal snapshot of an ordinary
    // endpoint but never of a stale one.
    for stale in [false, true] {
        let mut s = ds41_engine(&[3072, 3584, 4096]);
        let h = admit_with(&mut s, 12_000, vec![]);
        loop {
            let (plan, encoder) = run_step(&mut s);
            if plan.kind == GraphKind::Prefill && plan.lanes[0].request == h {
                if !stale {
                    // An ordinary endpoint: pretend the chunk was not encoder-only.
                    s.request_mut(h).unwrap().encoder_only_through = None;
                }
                assert!(encoder, "the first prompt chunk of a 12K cold prompt is eligible");
                break;
            }
        }
        assert_eq!(s.request(h).unwrap().suffix_stale(), stale);
        s.finish_early(h, false).unwrap();
        let r = s.request(h).unwrap();
        assert_eq!(r.terminal_snapshot.is_some(), !stale);
        assert_eq!(r.phase, if stale { RequestPhase::Finishing } else { RequestPhase::Snapshot });
    }
}

#[test]
fn a_checkpoint_at_a_stale_endpoint_is_not_captured() {
    // encoder_only_chunk never admits a checkpoint end; should a boundary appear at one, the
    // completion still captures nothing there.
    let mut s = ds41_engine(&[3072, 3584, 4096]);
    let h = admit_with(&mut s, 12_000, vec![]);
    loop {
        let plan = s.plan().unwrap();
        let encoder = s.encoder_prefill_step(&plan).unwrap();
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        if encoder {
            let end = plan.lanes[0].start_position as usize + plan.lanes[0].tokens;
            s.request_mut(h).unwrap().message_boundaries = vec![end];
        }
        finish_step(&mut s, &plan, fence);
        if encoder {
            let r = s.request(h).unwrap();
            assert!(r.suffix_stale());
            assert!(r.boundary_capture.is_none() && r.message_snapshot.is_none());
            assert_eq!(r.phase, RequestPhase::Prefill);
            break;
        }
    }
}

#[test]
fn randomized_prompts_keep_the_stale_state_contract() {
    // Several concurrent prompts with random lengths and message boundaries under the TTFT guard:
    // every encoder-only decision satisfies the rule, every chunk after one is longer than the
    // window, no checkpoint is captured at a stale endpoint, and every prompt finishes.
    for case in 0..12u64 {
        let mut seed = 0x5eed_u64 + case * 7919;
        let mut random = |low: usize, high: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            low + (seed >> 33) as usize % (high - low)
        };
        let mut s = tp3_engine(4, |kind, batch, tokens| match kind {
            GraphKind::Prefill => 27_400_000 + tokens * 67_880,
            GraphKind::Decode => 12 * MS + (batch - 1) * MS / 4,
            GraphKind::Speculative => 12_500_000 + (tokens - 1) * 300_000 + (batch - 1) * 300_000,
            _ => MS,
        });
        s.encoder_prefill_graphs(&[3072, 3584, 4096]).unwrap();
        s.configure_policy(PolicyConfig { ttft_target_ms: 50.0, prefill_merge_min_rows: 512,
            prefill_tail_merge: PrefillTailMerge::On, ..s.policy.clone() }).unwrap();
        let mut handles = Vec::new();
        for _ in 0..4 {
            let prompt = random(300, 24_000);
            let mut boundaries: Vec<usize> = (0..random(0, 6)).map(|_| random(1, prompt)).collect();
            boundaries.sort_unstable();
            boundaries.dedup();
            handles.push(admit_with(&mut s, prompt, boundaries));
        }
        let mut encoders = 0;
        let finished = |s: &RequestScheduler, h: &RequestHandle| s.request(*h)
            .map_or(true, |r| r.consumed_tokens >= r.prompt_tokens as u64);
        for _ in 0..4000 {
            if handles.iter().all(|h| finished(&s, h)) {
                break;
            }
            let Some(plan) = s.plan() else { break };
            let encoder = s.encoder_prefill_step(&plan).unwrap();
            for (row, lane) in plan.lanes.iter().enumerate() {
                let r = s.request(lane.request).unwrap();
                if plan.lane_kind(row).is_prefill() && r.suffix_stale() {
                    assert!(lane.tokens as u64 > WINDOW, "window-sized successor of a stale chunk");
                    assert_eq!(lane.start_position, r.consumed_tokens);
                }
            }
            if encoder {
                encoders += 1;
                let lane = &plan.lanes[0];
                assert!(encoder_only_chunk(s.request(lane.request).unwrap(), lane.start_position, lane.tokens));
            }
            let fence = Fence(s.submitted.0 + 1);
            s.commit(&plan, fence).unwrap();
            finish_step(&mut s, &plan, fence);
            for h in &handles {
                if let Ok(r) = s.request(*h) && r.suffix_stale() {
                    assert!(r.boundary_capture.is_none_or(|c| c.consumed_tokens != r.consumed_tokens));
                    assert!(r.terminal_snapshot.is_none());
                    assert_eq!(r.phase, RequestPhase::Prefill);
                }
            }
        }
        assert!(encoders > 0);
        assert_eq!(s.stats().encoder_prefill_chunks, encoders);
        assert!(handles.iter().all(|h| finished(&s, h)), "every prompt reaches its end");
    }
}

#[test]
fn image_requests_keep_the_ordinary_graph_after_their_patches_are_drained() {
    // Vision graphs drain `vision_patches` before any prefill; the image slot stays held.
    let mut s = engine(1, false);
    let h = add(&mut s, 1, 1400, 32);
    assert!(encoder_only_chunk(s.request(h).unwrap(), 0, 512));
    s.request_mut(h).unwrap().image_slot = Some(0);
    assert!(!encoder_only_chunk(s.request(h).unwrap(), 0, 512));
}

#[test]
fn commit_refuses_a_window_sized_or_non_prefill_lane_on_a_stale_request() {
    let mut s = ds41_engine(&[3072, 3584, 4096]);
    let h = admit_with(&mut s, 12_000, vec![]);
    loop {
        let (_, encoder) = run_step(&mut s);
        if encoder {
            break;
        }
    }
    assert!(s.request(h).unwrap().suffix_stale());
    let plan = s.plan().unwrap();
    assert!(plan.lanes[0].tokens > SUFFIX_WINDOW);
    let fence = Fence(s.submitted.0 + 1);
    let mut step = plan.step.clone();
    step.lanes[0].tokens = SUFFIX_WINDOW;
    let short = ScheduledPlan { step, ..plan.clone() };
    assert!(s.commit(&short, fence).unwrap_err().to_string().contains("longer than the decoder window"));
    // Nothing changed: the planned chunk still commits.
    assert!(s.request(h).unwrap().suffix_stale());
    s.commit(&plan, fence).unwrap();
    finish_step(&mut s, &plan, fence);
    assert!(!s.request(h).unwrap().suffix_stale() || s.stats().encoder_prefill_chunks == 2);
}

#[test]
fn the_unguarded_negative_control_lets_a_window_sized_chunk_read_a_stale_suffix() {
    // 4096 + 64: guarded, the first chunk runs the ordinary graph; unguarded, it runs the twin and
    // the 64-row tail follows a stale endpoint (what the GPU negative control must detect).
    for guarded in [true, false] {
        let mut s = ds41_engine(&[3072, 3584, 4096]);
        if !guarded {
            s.unguarded_encoder_prefill();
        }
        let h = admit_with(&mut s, 4096 + 64, vec![]);
        let chunks = prompt_chunks(&mut s, h);
        assert_eq!(chunks.iter().map(|c| (c.0, c.1)).collect::<Vec<_>>(), [(0, 4096), (4096, 64)]);
        assert_eq!(chunks[0].3, !guarded);
        assert!(!chunks[1].3);
    }
}

#[test]
fn encoder_only_durations_do_not_train_the_full_prefill_estimate() {
    // 12,000 tokens: the first chunk runs encoder-only; 4,196 tokens: the 4096 chunk is ordinary.
    let mut seen = [false; 2];
    for prompt in [12_000, 4_196] {
        let mut s = ds41_engine(&[3072, 3584, 4096]);
        admit_with(&mut s, prompt, vec![]);
        let plan = loop {
            let plan = s.plan().expect("the prompt makes progress");
            if plan.kind == GraphKind::Prefill {
                break plan;
            }
            let fence = Fence(s.submitted.0 + 1);
            s.commit(&plan, fence).unwrap();
            finish_step(&mut s, &plan, fence);
        };
        assert_eq!((plan.kind, plan.token_capacity), (GraphKind::Prefill, 4096));
        let fence = Fence(s.submitted.0 + 1);
        s.commit(&plan, fence).unwrap();
        let encoder = s.submitted_encoder(fence).unwrap();
        let before = s.estimated_duration_ns(&plan);
        finish_step_timed(&mut s, &plan, fence, plan.estimated_duration_ns() / 2);
        assert_eq!(s.estimated_duration_ns(&plan) == before, encoder, "encoder {encoder}");
        seen[usize::from(encoder)] = true;
    }
    assert_eq!(seen, [true, true]);
}

#[test]
fn encoder_capacities_used_under_mixed_traffic() {
    // Evidence for the twin set: which prefill capacities ever run encoder-only under the TTFT
    // guard, prefill merge and tail merge with random prompts and message boundaries.
    let mut used = std::collections::BTreeMap::<usize, u64>::new();
    for case in 0..24u64 {
        let mut seed = 0xcafe_u64 + case * 104_729;
        let mut random = |low: usize, high: usize| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            low + (seed >> 33) as usize % (high - low)
        };
        let mut s = tp3_engine(4, |kind, batch, tokens| match kind {
            GraphKind::Prefill => 27_400_000 + tokens * 67_880,
            GraphKind::Decode => 12 * MS + (batch - 1) * MS / 4,
            GraphKind::Speculative => 12_500_000 + (tokens - 1) * 300_000 + (batch - 1) * 300_000,
            _ => MS,
        });
        s.encoder_prefill_graphs(&[3072, 3584, 4096]).unwrap();
        s.configure_policy(PolicyConfig { ttft_target_ms: 50.0, prefill_merge_min_rows: 512,
            prefill_tail_merge: PrefillTailMerge::On, ..s.policy.clone() }).unwrap();
        for _ in 0..4 {
            let prompt = random(300, 60_000);
            let mut boundaries: Vec<usize> = (0..random(0, 8)).map(|_| random(1, prompt)).collect();
            boundaries.sort_unstable();
            boundaries.dedup();
            admit_with(&mut s, prompt, boundaries);
        }
        for _ in 0..4000 {
            let Some(plan) = s.plan() else { break };
            let fence = Fence(s.submitted.0 + 1);
            s.commit(&plan, fence).unwrap();
            if s.submitted_encoder(fence).unwrap() {
                *used.entry(plan.token_capacity).or_default() += 1;
            }
            finish_step(&mut s, &plan, fence);
        }
    }
    eprintln!("encoder-only chunks by capacity: {used:?}");
    assert!(used.contains_key(&4096));
}

/// A fresh TP3 server: every graph cost unmeasured, the package's encoder twin of 4096 listed (as
/// `runtime.rs` gives the scheduler every manifest graph, twins included).
fn fresh_ds41_server() -> RequestScheduler {
    let mut s = RequestScheduler::new([41; 32],
        SchedulerLimits { max_active: 1, request_slots: 1, max_sequence_tokens: 65536,
            prefill_chunk_size: 4096, pages: 32768, snapshots: 256 },
        SchedulerCapabilities { completion_token_capacity: 16, vocabulary: 1024, page_tokens: 64,
            vision: crate::model::VisionCapabilities::disabled(), draft_pages_per_request: 0,
            completion_inputs: vec![], speculative_allocations: vec![], state_rows_per_request: None }).unwrap();
    s.prefill_graphs(&prefill_capacities(4096).collect::<Vec<_>>()).unwrap();
    s.enable_speculation(&[1, 6]).unwrap();
    let mut graphs = super::tests::duration_graphs(1);
    let mut twin = graphs.iter().find(|g| g.kind == GraphKind::Prefill && g.token_capacity == 4096).unwrap().clone();
    twin.kind = GraphKind::EncoderPrefill;
    twin.name = "encoder_prefill4096".into();
    graphs.push(twin);
    s.graph_durations(&graphs).unwrap();
    s.encoder_prefill_graphs(&[4096]).unwrap();
    s
}

/// The suffix-proof warm-up on a fresh server: unique raw prompts of `tokens`, one after another.
/// GPU model: a full chunk costs 1 ms + 20 us per graph row, its encoder twin `ratio` of that; the
/// first launch of `cold` (the first large full prefill, or the twin) pays 50 % more (lazy
/// loading and first touches after boot). Returns per prompt (encoder chunks, (capacity, encoder)).
fn warm_up(ratio: f64, cold: Option<bool>, tokens: usize) -> (RequestScheduler, Vec<(u64, Vec<(usize, bool)>)>) {
    let mut s = fresh_ds41_server();
    let mut cold = cold;
    let mut prompts = Vec::new();
    for n in 0..4u32 {
        let before = s.stats().encoder_prefill_chunks;
        let h = s.admit(RequestSpec {
            prompt: (0..tokens as u32).map(|t| (t * 7 + n * 13 + t / 97) % 1000 + 1).collect(),
            message_boundaries: vec![], conditioning: vec![], max_new_tokens: 1, grammar_constrained: false,
            host_stops: false, prompt_logprobs: false, growable_output: false, tools: false,
            reasoning_prompt: false }).unwrap();
        let mut chunks = Vec::new();
        for _ in 0..400 {
            if s.request(h).is_err() {
                break;
            }
            let plan = s.plan().expect("the warm-up prompt makes progress");
            let fence = Fence(s.submitted.0 + 1);
            s.commit(&plan, fence).unwrap();
            let encoder = s.submitted_encoder(fence).unwrap();
            let mut ns = match plan.kind {
                GraphKind::Prefill => 1_000_000.0 + 20_000.0 * plan.token_capacity as f64,
                GraphKind::Decode | GraphKind::Speculative => 12.0 * MS as f64,
                _ => MS as f64 / 10.0,
            } * if encoder { ratio } else { 1.0 };
            if plan.kind == GraphKind::Prefill && plan.token_capacity >= 1024 && cold == Some(encoder) {
                ns *= 1.5;
                cold = None;
            }
            if plan.kind == GraphKind::Prefill {
                chunks.push((plan.token_capacity, encoder));
            }
            finish_step_timed(&mut s, &plan, fence, ns as u64);
        }
        prompts.push((s.stats().encoder_prefill_chunks - before, chunks));
    }
    (s, prompts)
}

#[test]
fn a_slow_first_sample_does_not_lock_encoder_chunks_out_of_the_warm_up() {
    // The 11-17 mini warm-up: four unique 24,000-token prompts ran +1, +0, +0, +0 encoder chunks.
    // Discovery timed the first 4096 chunk (the first large prefill after boot) on the twin;
    // that one slow sample priced 4096 below 3584 per token, encoder samples were then dropped,
    // and every later prompt ran 3584-row chunks, which have no twin. Now discovery times a full
    // 4096 chunk first, the twin learns its own estimate, and an eligible chunk is priced at the
    // twin (never above the full graph): every prompt after the first runs the 5 chunks the
    // window rule allows (24,000 = 5 x 4096 + 3520).
    let full = cost::Key { drafts: 0, kind: GraphKind::Prefill, batch: 1, tokens: 4096, prefill: 0, packed: 0 };
    let twin = cost::Key { kind: GraphKind::EncoderPrefill, ..full };
    for ratio in [0.88, 0.5] {
        for cold in [None, Some(false), Some(true)] {
            let (s, prompts) = warm_up(ratio, cold, 24_000);
            let label = format!("ratio {ratio} cold {cold:?}: {prompts:?}");
            assert_eq!(prompts[0].1[0], (4096, false), "discovery times the full graph first: {label}");
            assert_eq!(prompts[0].1[1], (4096, true), "then the twin: {label}");
            for (encoders, chunks) in &prompts[1..] {
                assert_eq!(*encoders, 5, "{label}");
                assert_eq!(chunks.iter().filter(|chunk| chunk.1).count(), 5, "{label}");
                assert!(chunks.iter().filter(|chunk| chunk.1).all(|chunk| chunk.0 == 4096), "{label}");
            }
            // Each estimate holds only its own graph's samples.
            let full_ns = s.costs.duration_key(full).unwrap();
            let twin_ns = s.costs.duration_key(twin).unwrap();
            assert_eq!(full_ns, if cold == Some(false) { 124_380_000 } else { 82_920_000 }, "{label}");
            if cold != Some(true) {
                assert_eq!(twin_ns, (82_920_000.0 * ratio) as u64, "{label}");
            }
        }
    }
}

/// TP3 fork (review B1): a package with long-context variants (`<base>-split-ctx<N>`, same shape
/// as the base) passes the start-up catalog set-up the runtime runs (cost model on
/// `catalog::scheduling_graphs`, then `shape_catalogs`); handed every graph unfiltered, the cost
/// model and the prefill / encoder catalogs refuse the duplicate shapes.
#[test]
fn context_variants_stay_out_of_the_cost_model_and_the_shape_catalogs() {
    use crate::execution::{GraphDescriptor, SamplingMode};
    let graph = |name: String, kind: GraphKind, tokens: usize| GraphDescriptor {
        tuned_bindings: vec![], confidence_width: 0, draft_tokens: 0, packed_token_capacity: 0,
        name, kind, sampling_mode: SamplingMode::General, handle: 0x5000,
        batch_capacity: 1, token_capacity: tokens, prefill_capacity: 0, estimated_duration_ns: 1_000_000,
        input_bindings: vec![],
    };
    let mut graphs: Vec<_> = prefill_capacities(4096).map(|n| graph(format!("prefill{n}"), GraphKind::Prefill, n)).collect();
    for n in [3072, 3584, 4096] {
        graphs.push(graph(format!("encoder_prefill{n}"), GraphKind::EncoderPrefill, n));
    }
    // the engine runs speculation at widths 1 and 6 on batch 1 (tp3_engine)
    graphs.extend(super::tests::duration_graphs(1).into_iter()
        .filter(|g| g.kind == GraphKind::Speculative && [1, 6].contains(&g.token_capacity)));
    graphs.push(graph("prefill4096-split-ctx32768".into(), GraphKind::Prefill, 4096));
    graphs.push(graph("encoder_prefill4096-split-ctx32768".into(), GraphKind::EncoderPrefill, 4096));
    let mut s = ds41_engine(&[]);
    assert!(crate::scheduler::cost::ActionCosts::prepared(&graphs, 16).is_err());
    assert!(s.prefill_graphs(&graphs.iter().filter(|g| g.kind == GraphKind::Prefill)
        .map(|g| g.token_capacity).collect::<Vec<_>>()).is_err());
    let mut s = ds41_engine(&[]);
    s.graph_durations(&crate::model::catalog::scheduling_graphs(&graphs)).unwrap();
    s.shape_catalogs(&graphs).unwrap();
    assert_eq!(s.prefill_catalog, prefill_capacities(4096).collect::<Vec<_>>());
    assert_eq!(s.encoder_prefill_catalog, [3072, 3584, 4096]);
    assert_eq!(crate::model::catalog::scheduling_graphs(&graphs).len(), graphs.len() - 2);
    // the runtime counts what it swapped in
    s.record_context_variant(4096);
    s.record_context_variant(4096);
    assert_eq!((s.stats().context_variant_chunks, s.stats().context_variant_tokens), (2, 8192));
}
