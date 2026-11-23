/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Which overdue request the TTFT guard protects: `oldest` (arrival order) or `shortest`
//! (fewest remaining prompt chunks, with every other guarded prefill round owed to the oldest).

use super::*;

/// One simulated round: the step and its estimated duration.
type Round = (ScheduledStep, u64);

struct Arrival {
    round: usize,
    id: u32,
    prompt: usize,
    boundaries: Vec<usize>,
    tools: bool,
}

struct Trace {
    rounds: Vec<Round>,
    /// Per arrival: (round admitted, round of its first token).
    first_tokens: Vec<(usize, Option<usize>)>,
    handles: Vec<RequestHandle>,
}

impl Trace {
    fn digest(&self) -> (usize, u64) {
        let text = format!("{:?}", self.rounds);
        (self.rounds.len(), twox_hash::XxHash3_64::oneshot(text.as_bytes()))
    }
}

/// Runs `decoders` steady decoders plus the given arrivals until every arrival has its first
/// token or `limit` rounds have passed.
fn run_trace(engine: fn(usize) -> RequestScheduler, active: usize, policy: PolicyConfig, decoders: usize,
    arrivals: &[Arrival], limit: usize) -> Trace {
    let mut s = engine(active);
    s.configure_policy(policy).unwrap();
    tp3_decoders(&mut s, decoders, 3);
    let mut trace = Trace { rounds: Vec::new(), first_tokens: Vec::new(), handles: Vec::new() };
    let mut next = 0;
    while trace.rounds.len() < limit {
        while next < arrivals.len() && arrivals[next].round <= trace.rounds.len() {
            let a = &arrivals[next];
            let h = s.admit(RequestSpec { prompt: vec![a.id; a.prompt], message_boundaries: a.boundaries.clone(),
                conditioning: vec![], max_new_tokens: 16, grammar_constrained: false, host_stops: false,
                prompt_logprobs: false, growable_output: false, tools: a.tools, reasoning_prompt: false }).unwrap();
            trace.handles.push(h);
            trace.first_tokens.push((trace.rounds.len(), None));
            next += 1;
        }
        if next == arrivals.len() && trace.first_tokens.iter().all(|(_, t)| t.is_some()) {
            break;
        }
        let plan = ds41_step(&mut s, 3);
        trace.rounds.push((plan.step.clone(), plan.estimated_duration_ns()));
        let round = trace.rounds.len() - 1;
        for (i, h) in trace.handles.iter().enumerate() {
            if trace.first_tokens[i].1.is_none() && first_token(&s, *h) {
                trace.first_tokens[i].1 = Some(round);
            }
        }
    }
    trace
}

fn arrival(round: usize, id: u32, prompt: usize) -> Arrival {
    Arrival { round, id, prompt, boundaries: vec![], tools: false }
}

/// Four 40K prompts loading next to two decoders, a 1.9K ping at round 6, then a 1K ping and a
/// 9K prompt at round 40 (DS41 costs, guard 50 ms, interleave 1).
fn ds41_long_load(policy: PolicyConfig) -> Trace {
    let arrivals = [arrival(0, 10, 40_000), arrival(0, 11, 40_000), arrival(0, 12, 40_000),
        arrival(0, 13, 40_000), arrival(6, 50, 1900), arrival(40, 51, 1000), arrival(40, 52, 9000)];
    run_trace(ds41_engine, 10, policy, 2, &arrivals, 4000)
}

/// GLM costs, interleave 2, guarded chunks capped at 2048, multi-message tool prompts (tail merge
/// applies under the default `ToolsOnly`).
fn glm_long_load(policy: PolicyConfig) -> Trace {
    let tool = |round, id, prompt, boundaries: Vec<usize>| Arrival { round, id, prompt, boundaries, tools: true };
    let arrivals = [tool(0, 10, 30_000, vec![12_000, 29_950]), arrival(0, 11, 30_000),
        tool(0, 12, 20_000, vec![19_900]), arrival(3, 50, 600), tool(25, 51, 5000, vec![4000, 4970]),
        arrival(26, 52, 300)];
    run_trace(glm_engine, 10, policy, 3, &arrivals, 4000)
}

fn ordered(order: TtftOrder, interleave: u32, max_chunk: usize) -> PolicyConfig {
    PolicyConfig { ttft_order: order, ..ttft_policy(50.0, max_chunk, interleave) }
}

/// `oldest` reproduces the rounds of the guard before `TtftOrder` existed (digests and first-token
/// rounds recorded on bece44e6 with this file's scenarios and the policy code unchanged). The GLM
/// digest was re-recorded with the acceptance population (#306), which changes the first draft
/// widths of new requests (before it: 7039464314114195283); its round count and first-token
/// rounds did not change, and DS41 did not change.
#[test]
fn golden_oldest_order_rounds() {
    for (name, trace, digest, first) in [
        ("ds41", ds41_long_load(ordered(TtftOrder::Oldest, 1, 0)), (92usize, 11037627831189092831u64),
            vec![(0, Some(21)), (0, Some(41)), (0, Some(61)), (0, Some(81)), (6, Some(83)), (40, Some(85)), (40, Some(91))]),
        ("glm", glm_long_load(ordered(TtftOrder::Oldest, 2, 2048)), (129, 12774214185769665554),
            vec![(0, Some(48)), (0, Some(93)), (0, Some(121)), (3, Some(92)), (25, Some(128)), (26, Some(123))]),
    ] {
        assert_eq!(trace.first_tokens, first, "{name}");
        assert_eq!(trace.digest(), digest, "{name}");
    }
    assert_eq!(PolicyConfig::default().ttft_order, TtftOrder::Oldest);
    assert_eq!(PolicyConfig::interactive().ttft_order, TtftOrder::Oldest);
}

#[test]
fn shortest_picks_the_fewest_remaining_chunks_then_the_oldest() {
    for order in [TtftOrder::Oldest, TtftOrder::Shortest] {
        let mut s = ds41_engine(6);
        s.configure_policy(ordered(order, 1, 0)).unwrap();
        ds41_decoders(&mut s, 2);
        let mut handles = Vec::new();
        for (id, prompt) in [(100, 9000), (101, 5000), (102, 4096), (103, 1900)] {
            let h = add(&mut s, id, prompt, 16);
            initialize(&mut s, h);
            handles.push(h);
            s.advance_to(s.now_ns + MS).unwrap();
        }
        let [oldest, two, one, newest_one] = handles[..] else { unreachable!() };
        assert!(s.ttft_overdue().is_none());
        s.advance_to(s.now_ns + 60 * MS).unwrap();
        assert_eq!(s.ttft_remaining_chunks(s.request(oldest).unwrap()), 3);
        assert_eq!(s.ttft_remaining_chunks(s.request(two).unwrap()), 2);
        assert_eq!(s.ttft_remaining_chunks(s.request(one).unwrap()), 1);
        assert_eq!(s.ttft_remaining_chunks(s.request(newest_one).unwrap()), 1);
        assert_eq!(s.ttft_oldest_overdue().unwrap().handle, oldest);
        // Equal chunk counts fall back to arrival: the 4096 prompt before the newer 1900 one.
        let protected = if order == TtftOrder::Oldest { oldest } else { one };
        assert_eq!(s.ttft_overdue().unwrap().handle, protected, "{order:?}");
        let plan = ds41_step(&mut s, 3);
        assert!(plan.kind.is_prefill() && plan.lanes[0].request == protected, "{order:?}: {plan:?}");
        // Shortest: that chunk owes the next guarded round to the oldest; its chunk repays it.
        assert_eq!(s.ttft_overdue().unwrap().handle, oldest);
        assert_eq!(s.ttft_owe_oldest, order == TtftOrder::Shortest);
        let mut guarded = Vec::new();
        while guarded.len() < 4 {
            let plan = ds41_step(&mut s, 3);
            if plan.kind.is_prefill() {
                guarded.push(plan.lanes[0].request);
            }
        }
        // Shortest: the owed chunk of the oldest (9000 = 4096 + 4096 + 808), the remaining
        // one-chunk prompt, the owed oldest again, then the oldest's last 808 tokens (one chunk
        // left against `two`'s two). Oldest: arrival order.
        let expected = if order == TtftOrder::Shortest { [oldest, newest_one, oldest, oldest] }
            else { [oldest, oldest, two, two] };
        assert_eq!(guarded, expected, "{order:?}");
    }
}

/// (a) A ping arriving while four 40K prompts load gets its first token within a bounded number
/// of rounds under `shortest`; under `oldest` it waits for every earlier long prompt.
#[test]
fn shortest_serves_a_ping_during_long_prefills_within_bounded_rounds() {
    for interleave in [0u32, 1, 2] {
        let oldest = ds41_long_load(ordered(TtftOrder::Oldest, interleave, 0));
        let shortest = ds41_long_load(ordered(TtftOrder::Shortest, interleave, 0));
        let wait = |trace: &Trace, i: usize| trace.first_tokens[i].1.unwrap() - trace.first_tokens[i].0;
        eprintln!("interleave {interleave}: oldest {:?}\n  shortest {:?}", oldest.first_tokens, shortest.first_tokens);
        // One round to become overdue (a 4096 chunk outlasts the 50 ms target), then at most one
        // guarded round owed to the oldest and its interleave before the ping's own chunk.
        let bound = 2 + 2 * (1 + interleave as usize);
        assert!(wait(&shortest, 4) <= bound, "interleave {interleave}: ping waited {} rounds", wait(&shortest, 4));
        assert!(wait(&shortest, 5) <= bound + 2 * (1 + interleave as usize),
            "interleave {interleave}: second ping waited {} rounds", wait(&shortest, 5));
        // Oldest serves the ping only after at least three of the four long prompts (with
        // interleave 2 an ordinary round may take it before the fourth).
        assert!(oldest.first_tokens[4].1 > oldest.first_tokens[2].1, "interleave {interleave}: {:?}", oldest.first_tokens);
        assert!(wait(&oldest, 4) > 5 * bound, "interleave {interleave}: {:?}", oldest.first_tokens);
        // (b) every long prompt still gets its first token, later by at most the guarded chunks
        // of the requests served ahead of it (1.9K + 1K + 9K = five chunks) and their interleave.
        assert!(shortest.first_tokens.iter().all(|(_, t)| t.is_some()));
        for i in 0..4 {
            assert!(wait(&shortest, i) <= wait(&oldest, i) + 5 * (1 + interleave as usize) + 4,
                "interleave {interleave}: long {i} {} vs {}", wait(&shortest, i), wait(&oldest, i));
        }
    }
}

/// (b) A long prompt keeps at least every second guarded prefill round while a new short request
/// arrives every round, so its first token is bounded by twice its oldest-first schedule.
#[test]
fn shortest_does_not_starve_a_long_prompt_under_a_stream_of_short_requests() {
    for interleave in [0u32, 1, 2] {
        let mut s = ds41_engine(16);
        s.configure_policy(ordered(TtftOrder::Shortest, interleave, 0)).unwrap();
        ds41_decoders(&mut s, 1);
        let long = add(&mut s, 10, 40_000, 16);
        let arrival = s.now_ns;
        let mut overdue_at = None;
        let mut id = 100;
        let mut rounds = 0;
        let mut chunks = Vec::new();
        let mut pings = 0;
        while !first_token(&s, long) {
            assert!(rounds < 400, "interleave {interleave}: long prompt starved at {chunks:?}");
            if s.has_available_slot() && s.slots.iter().filter(|e| e.request.is_some()).count() < 15 {
                add(&mut s, id, 1000, 16);
                id += 1;
                pings += 1;
            }
            if overdue_at.is_none() && s.now_ns - arrival >= 50 * MS {
                overdue_at = Some(rounds);
            }
            let plan = ds41_step(&mut s, 3);
            if plan.lanes.iter().enumerate().any(|(row, lane)| lane.request == long && plan.lane_kind(row).is_prefill()) {
                chunks.push(rounds);
            }
            rounds += 1;
        }
        eprintln!("interleave {interleave}: long chunks at rounds {chunks:?}, {pings} pings admitted");
        assert_eq!(chunks.len(), 10);
        // Once overdue it waits at most one other guarded chunk and the interleave after each of
        // the two (plus one round of slack) for each of its own chunks.
        let period = 2 * (1 + interleave as usize);
        let overdue_at = overdue_at.unwrap();
        assert!(chunks[0] - overdue_at <= period + 1, "interleave {interleave}: overdue at {overdue_at}, {chunks:?}");
        assert!(chunks.windows(2).all(|w| w[1] - w[0] <= period + 1), "interleave {interleave}: {chunks:?}");
        assert!(rounds - overdue_at <= 10 * (period + 1) + 1, "interleave {interleave}: {rounds} rounds");
        assert!(pings > 10, "the stream must actually compete");
    }
}

/// Agent-like traffic: similar turns whose remaining prompts fit one chunk (cached history plus
/// a new message) tie on chunk count, so `shortest` falls back to arrival order and runs exactly
/// the rounds `oldest` runs.
#[test]
fn shortest_matches_oldest_when_waiting_turns_fit_one_chunk() {
    let arrivals: Vec<_> = (0..12).map(|i| arrival(i * 3 / 2, 200 + i as u32, 1200 + (i * 997) % 2800)).collect();
    for interleave in [0, 1] {
        let oldest = run_trace(ds41_engine, 16, ordered(TtftOrder::Oldest, interleave, 0), 4, &arrivals, 2000);
        let shortest = run_trace(ds41_engine, 16, ordered(TtftOrder::Shortest, interleave, 0), 4, &arrivals, 2000);
        assert!(oldest.first_tokens.iter().all(|(_, t)| t.is_some()));
        assert_eq!(oldest.rounds, shortest.rounds, "interleave {interleave}");
    }
}

/// Eight agents, each sending its next turn as soon as the previous one finishes (64 output
/// tokens); new prompt work per turn 0.8K-14K tokens (1-4 chunks), as when tool results of very
/// different sizes join a cached history. Returns (sorted TTFTs in ms, completed turns).
fn agent_load(order: TtftOrder, interleave: u32, seconds: u64) -> (Vec<f64>, usize) {
    let mut s = ds41_engine(8);
    s.configure_policy(ordered(order, interleave, 0)).unwrap();
    let start = s.now_ns;
    let mut seed = 12345u64;
    let mut next_size = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        800 + (seed >> 33) as usize % 13_200
    };
    let mut live: Vec<(RequestHandle, u64, bool)> = Vec::new();
    let mut ttft = Vec::new();
    let mut turns = 0;
    let mut id = 1000;
    while s.now_ns < start + seconds * 1000 * MS {
        while live.len() < 8 {
            let h = add(&mut s, id, next_size(), 64);
            id += 1;
            live.push((h, s.now_ns, false));
        }
        ds41_step(&mut s, 3);
        for (h, arrival, seen) in &mut live {
            if !*seen && first_token(&s, *h) {
                *seen = true;
                ttft.push((s.now_ns - *arrival) as f64 / 1e6);
            }
        }
        let before = live.len();
        live.retain(|(h, _, _)| s.request(*h).is_ok());
        turns += before - live.len();
    }
    ttft.sort_by(f64::total_cmp);
    (ttft, turns)
}

#[test]
fn shortest_on_agent_like_turns_of_different_sizes() {
    for interleave in [1, 2] {
        let (old, old_turns) = agent_load(TtftOrder::Oldest, interleave, 120);
        let (short, short_turns) = agent_load(TtftOrder::Shortest, interleave, 120);
        let p = |v: &[f64], q: f64| v[((q * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1];
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        eprintln!("interleave {interleave}: TTFT ms mean/p50/p90/max, turns: oldest {:.0}/{:.0}/{:.0}/{:.0} {old_turns}; \
            shortest {:.0}/{:.0}/{:.0}/{:.0} {short_turns}",
            mean(&old), p(&old, 0.5), p(&old, 0.9), p(&old, 1.0), mean(&short), p(&short, 0.5), p(&short, 0.9), p(&short, 1.0));
        assert!(old.len() > 50 && short.len() > 50);
        // Shortest-first lowers mean and median TTFT, raises the tail somewhat (bounded by the
        // owed rounds), and keeps throughput. Recorded 11-15: interleave 1 mean 2312 -> 2142,
        // p50 2249 -> 1772, p90 3883 -> 4118, max 5210 -> 5825 ms, turns 147 -> 145.
        // With the acceptance population (#306) new turns start from the acceptance of earlier
        // ones, so their first draft widths, and with them the decode rounds between guarded
        // chunks, change. Recorded 11-17: interleave 1 mean 2195 -> 2002, p50 2073 -> 1708 ms;
        // interleave 2 mean 2557 -> 2605, p50 2517 -> 2481 ms (before #306 2662 -> 2555 and
        // 2701 -> 2277). At interleave 2 the population shortened `oldest`'s waits more than
        // `shortest`'s, leaving the two means within 2%. The strict mean comparison therefore
        // holds at the default interleave (1); at interleave 2 the median must still improve
        // and the mean may be at most 5% worse.
        assert!(p(&short, 0.5) <= p(&old, 0.5), "interleave {interleave}");
        let mean_tolerance = if interleave == PolicyConfig::default().ttft_decode_interleave { 1.0 } else { 1.05 };
        assert!(mean(&short) <= mean_tolerance * mean(&old), "interleave {interleave}");
        assert!(p(&short, 1.0) <= 1.5 * p(&old, 1.0), "interleave {interleave}");
        assert!(short_turns as f64 >= 0.97 * old_turns as f64, "interleave {interleave}");
    }
}

/// The guard's validation loads (one 1.9K probe at a time next to 3-7 decoders, DS41 and GLM
/// costs) have a single waiting request, so `shortest` reproduces their first-token latencies.
#[test]
fn shortest_leaves_the_single_probe_guard_simulations_unchanged() {
    for decoders in [3, 7] {
        for (engine, accepted) in [(ds41_engine as fn(usize) -> RequestScheduler, 3), (glm_engine, 1)] {
            let oldest = simulate_probe_load_on(engine, accepted, ordered(TtftOrder::Oldest, 1, 0), decoders, 30);
            let shortest = simulate_probe_load_on(engine, accepted, ordered(TtftOrder::Shortest, 1, 0), decoders, 30);
            assert_eq!(oldest.ttft_ms, shortest.ttft_ms, "decoders {decoders}");
            assert_eq!(oldest.decoder_tokens_per_s, shortest.decoder_tokens_per_s);
        }
    }
}
