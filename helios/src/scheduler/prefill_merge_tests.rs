/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Prefill merge (`PolicyConfig::prefill_merge_min_rows`): small prefill pieces of a turn's new
//! input join a neighbouring chunk instead of paying a graph launch of their own.
//!
//! The simulations replay multi-turn agent conversations shaped like `ttft_under_load.py agent`
//! against DS41 / GLM TP3-like costs: a shared system prompt, then per turn the cached history
//! plus the previous answer's end-of-sentence token, a new user message (a "tool result") and the
//! two-token generation prompt. Each next turn is admitted against the prefix cache, so the
//! resident (reused) tokens show what the previous turn left behind.

use super::*;

/// DS41 TP3 prefill costs fitted on the 1241 agent8 trace (59 launches at 100-200K context:
/// 27.4 ms per launch + 67.88 µs per graph row); speculative rounds as `ds41_engine`. 256
/// snapshot rows (`tp3_engine`), so checkpoints are evicted and compacted as on the server
/// (DS41 151744: 484 rows, ~1,900 evictions per 300 s of agent8).
fn ds41_agent_engine(active: usize) -> RequestScheduler {
    tp3_engine(active, ds41_agent_cost)
}

fn ds41_agent_cost(kind: GraphKind, batch: u64, tokens: u64) -> u64 {
    match kind {
        GraphKind::Prefill => 27_400_000 + tokens * 67_880,
        GraphKind::Decode => 12 * MS + (batch - 1) * MS / 4,
        GraphKind::Speculative => 12_500_000 + (tokens - 1) * 300_000 + (batch - 1) * 300_000,
        _ => MS,
    }
}

/// `ds41_agent_engine` with the measured medians of the 1241 agent8 trace where it has them
/// (128: 36.6, 256: 45.8, 512: 61.1, 3,072: 236.5, 3,584: 282.7, 4,096: 322.6 ms per launch),
/// under which a 3,072-row chunk plus a stub out-rates one 3,584-row chunk per token.
fn ds41_measured_engine(active: usize) -> RequestScheduler {
    tp3_engine(active, |kind, batch, tokens| match (kind, tokens) {
        (GraphKind::Prefill, 128) => 36_582_000,
        (GraphKind::Prefill, 256) => 45_765_000,
        (GraphKind::Prefill, 512) => 61_083_000,
        (GraphKind::Prefill, 3072) => 236_532_000,
        (GraphKind::Prefill, 3584) => 282_656_000,
        (GraphKind::Prefill, 4096) => 322_556_000,
        _ => ds41_agent_cost(kind, batch, tokens),
    })
}

/// `ds41_agent_engine` with snapshot rows and pages for every checkpoint of the simulation, so no
/// checkpoint is evicted or compacted and resume points depend on the merge alone.
fn ds41_roomy_engine(active: usize) -> RequestScheduler {
    roomy_engine(active, ds41_agent_cost)
}

/// `glm_engine` costs without eviction.
fn glm_roomy_engine(active: usize) -> RequestScheduler {
    roomy_engine(active, |kind, batch, tokens| match kind {
        GraphKind::Prefill => 20 * MS + tokens * 88_000,
        GraphKind::Decode => 8 * MS + (batch - 1) * MS / 2,
        GraphKind::Speculative => (tokens - 1) * 250_000 + if batch <= 4 {
            8_050_000 + (batch - 1) * 1_330_000
        } else {
            12_050_000 + (batch - 4) * 2_430_000
        },
        _ => MS,
    })
}

fn roomy_engine(active: usize, cost: fn(GraphKind, u64, u64) -> u64) -> RequestScheduler {
    let mut s = RequestScheduler::new(
        [41; 32],
        SchedulerLimits {
            max_active: active,
            request_slots: active.next_power_of_two(),
            max_sequence_tokens: 65536,
            prefill_chunk_size: 4096,
            pages: 1 << 17,
            snapshots: 16384,
        },
        SchedulerCapabilities {
            completion_token_capacity: 16,
            vocabulary: 1024,
            page_tokens: 64,
            vision: crate::model::VisionCapabilities::disabled(),
            draft_pages_per_request: 0,
            completion_inputs: vec![],
            speculative_allocations: vec![],
            state_rows_per_request: None,
        },
    )
    .unwrap();
    s.prefill_graphs(&prefill_capacities(4096).collect::<Vec<_>>()).unwrap();
    s.enable_speculation(&[1, 6]).unwrap();
    let mut graphs = super::super::tests::duration_graphs(active);
    for g in &mut graphs {
        g.estimated_duration_ns = cost(g.kind, g.batch_capacity as u64, g.token_capacity as u64);
    }
    s.measured_graph_durations(&graphs).unwrap();
    s
}

const BOS: u32 = 1;
const EOS: u32 = 2;
const USER: u32 = 3;
const ASSISTANT: u32 = 5;
const THINK_END: u32 = 6;
const THINK_START: u32 = 8;
const SYSTEM: u32 = 7;
/// Generated tokens in `completion_records`.
const CONTENT: u32 = 99;

#[derive(Clone, Copy, Debug)]
struct Shape {
    system: usize,
    /// Inclusive range of new user-message tokens per turn.
    user: (usize, usize),
    /// Inclusive range of output tokens per turn.
    output: (u32, u32),
    /// A conversation restarts (as `--max-history-tokens`) once its prompt exceeds this.
    max_prompt: usize,
}

/// `ttft_under_load.py agent` at DS41: ~3,460-token tool results (1,900 "words"), outputs up to
/// 600 tokens (the replay's mean is 236), the system prompt scaled to the simulated context.
const REPLAY: Shape = Shape { system: 8000, user: (3250, 3720), output: (60, 420), max_prompt: 56_000 };

#[derive(Clone, Copy, Debug)]
struct Agents {
    engine: fn(usize) -> RequestScheduler,
    agents: usize,
    seconds: u64,
    tools: bool,
    reasoning: bool,
    shape: Shape,
}

/// One agent's conversation: the next prompt is the last prompt plus the answer, so the prefix
/// cache resumes it from what the previous turn published.
struct Conversation {
    prompt: Vec<u32>,
    boundaries: Vec<usize>,
    turn: usize,
    /// (turn index, prompt, boundaries, output) of the request in flight.
    live: Option<(RequestHandle, usize, u64, bool, u32)>,
    seed: u64,
    restarts: usize,
}

#[derive(Debug, Default)]
struct Run {
    turns: usize,
    output_tokens: u64,
    prefill_launches: usize,
    /// Prefill launches by graph capacity.
    launches_by_capacity: std::collections::BTreeMap<usize, usize>,
    prefill_tokens: u64,
    snapshot_launches: usize,
    message_checkpoints: u64,
    prompt_checkpoints: u64,
    /// Per (agent, restart, turn): resident tokens at admission, prompt tokens, launches,
    /// prefilled tokens, TTFT in ms, finished.
    turn_records: std::collections::BTreeMap<(usize, usize, usize), TurnRecord>,
    rounds_digest: u64,
    round_count: usize,
    /// Prompt tokens of the branch probes (one per agent) that the cache holds.
    branch_resident: Vec<(usize, u64, u64)>,
    sim_seconds: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct TurnRecord {
    resident: u64,
    prompt: usize,
    launches: usize,
    small_launches: usize,
    prefilled: u64,
    ttft_ms: f64,
    finished: bool,
}

impl Run {
    /// Finished turns after each conversation's first (whose system prompt may still be cold).
    fn finished(&self) -> impl Iterator<Item = &TurnRecord> {
        self.turn_records.iter().filter(|(key, t)| key.2 >= 1 && t.finished).map(|(_, t)| t)
    }
    fn per_turn(&self, f: impl Fn(&TurnRecord) -> f64) -> f64 {
        let n = self.finished().count().max(1);
        self.finished().map(f).sum::<f64>() / n as f64
    }
    fn ttft(&self, q: f64) -> f64 {
        let mut v: Vec<f64> = self.finished().map(|t| t.ttft_ms).collect();
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(f64::total_cmp);
        v[((q * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1]
    }
    fn summary(&self, label: &str) -> String {
        format!("{label}: {} turns ({:.1}/s), {:.0} output tok/s, launches/turn {:.2} (<=512 rows {:.2}), \
            prefilled/turn {:.0}, resident/turn {:.0}, TTFT ms mean {:.0} p50 {:.0} p90 {:.0} max {:.0}, \
            message/prompt checkpoints {}/{}, snapshot launches {}, launches by capacity {:?}",
            self.turns, self.turns as f64 / self.sim_seconds, self.output_tokens as f64 / self.sim_seconds,
            self.per_turn(|t| t.launches as f64), self.per_turn(|t| t.small_launches as f64),
            self.per_turn(|t| t.prefilled as f64), self.per_turn(|t| t.resident as f64),
            self.per_turn(|t| t.ttft_ms), self.ttft(0.5), self.ttft(0.9), self.ttft(1.0),
            self.message_checkpoints, self.prompt_checkpoints, self.snapshot_launches, self.launches_by_capacity)
    }
}

fn next_random(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed >> 33
}

fn between(seed: &mut u64, (low, high): (usize, usize)) -> usize {
    low + next_random(seed) as usize % (high - low + 1)
}

/// The generation prompt: `<｜Assistant｜>` plus `<think>` (thinking) or `</think>`.
fn generation_prompt(reasoning: bool) -> [u32; 2] {
    [ASSISTANT, if reasoning { THINK_START } else { THINK_END }]
}

impl Conversation {
    fn new(agent: usize, shape: Shape) -> Self {
        let mut c = Self { prompt: Vec::new(), boundaries: Vec::new(), turn: 0, live: None,
            seed: 7919 * agent as u64 + 17, restarts: 0 };
        c.restart(shape);
        c
    }

    fn restart(&mut self, shape: Shape) {
        self.prompt = std::iter::once(BOS).chain(std::iter::repeat_n(SYSTEM, shape.system)).collect();
        self.boundaries = vec![1, 1 + shape.system];
        self.turn = 0;
    }

    /// Appends the next user message (a tool result) and the generation prompt.
    fn user_turn(&mut self, agent: usize, shape: Shape, reasoning: bool) -> (Vec<u32>, Vec<usize>, u32) {
        let user = between(&mut self.seed, shape.user);
        let output = between(&mut self.seed, (shape.output.0 as usize, shape.output.1 as usize)) as u32;
        let value = 1000 + 100 * agent as u32 + (self.turn % 97) as u32;
        let mut prompt = self.prompt.clone();
        prompt.push(USER);
        prompt.extend(std::iter::repeat_n(value, user));
        let mut boundaries = self.boundaries.clone();
        boundaries.push(prompt.len());
        prompt.extend(generation_prompt(reasoning));
        (prompt, boundaries, output)
    }

    /// The history the next turn renders after this answer: DeepSeek V4.1 re-renders the
    /// generation prompt unchanged, except a thinking turn without tools whose `<think>` header
    /// becomes `</think>` (reasoning is dropped from history).
    fn answered(&mut self, mut prompt: Vec<u32>, mut boundaries: Vec<usize>, output: u32, tools: bool, reasoning: bool) {
        if reasoning && !tools {
            *prompt.last_mut().unwrap() = THINK_END;
        }
        prompt.extend(std::iter::repeat_n(CONTENT, output as usize));
        prompt.push(EOS);
        boundaries.push(prompt.len());
        self.prompt = prompt;
        self.boundaries = boundaries;
        self.turn += 1;
    }
}

fn agent_policy(tail: PrefillTailMerge, merge_rows: usize) -> PolicyConfig {
    merge_policy(PolicyConfig { prefill_tail_merge: tail, ..ttft_policy(50.0, 0, 1) }, merge_rows)
}

fn merge_policy(policy: PolicyConfig, merge_rows: usize) -> PolicyConfig {
    PolicyConfig { prefill_merge_min_rows: merge_rows, ..policy }
}

fn run_agents(a: Agents, policy: PolicyConfig) -> Run {
    let mut s = (a.engine)(a.agents);
    s.configure_policy(policy).unwrap();
    let start = s.now_ns;
    let mut conversations: Vec<_> = (0..a.agents).map(|i| Conversation::new(i, a.shape)).collect();
    let mut run = Run::default();
    // (agent, prompt, boundaries, output) of each request in flight, by slot.
    let mut flight: std::collections::HashMap<RequestHandle, (usize, Vec<u32>, Vec<usize>, u32)> = Default::default();
    let mut digest_text = String::new();
    while s.now_ns < start + a.seconds * 1000 * MS {
        for (agent, c) in conversations.iter_mut().enumerate() {
            if c.live.is_some() {
                continue;
            }
            if c.prompt.len() > a.shape.max_prompt {
                c.restarts += 1;
                c.restart(a.shape);
            }
            let (prompt, boundaries, output) = c.user_turn(agent, a.shape, a.reasoning);
            let h = s.admit(RequestSpec { prompt: prompt.clone(), message_boundaries: boundaries.clone(),
                conditioning: vec![], max_new_tokens: output, grammar_constrained: false, host_stops: false,
                prompt_logprobs: false, growable_output: false, tools: a.tools, reasoning_prompt: a.reasoning })
                .unwrap();
            let resident = s.request(h).unwrap().reused_prompt_tokens();
            run.turn_records.insert((agent, c.restarts, c.turn), TurnRecord { resident, prompt: prompt.len(),
                ..Default::default() });
            c.live = Some((h, c.turn, s.now_ns, false, output));
            flight.insert(h, (agent, prompt, boundaries, output));
        }
        let plan = ds41_step(&mut s, 3);
        digest_text.push_str(&format!("{:?}|", &*plan));
        run.round_count += 1;
        match plan.kind {
            GraphKind::Snapshot => run.snapshot_launches += 1,
            kind if kind.is_prefill() => {
                run.prefill_launches += 1;
                *run.launches_by_capacity.entry(plan.token_capacity).or_default() += 1;
            }
            _ => {}
        }
        for (row, lane) in plan.lanes.iter().enumerate() {
            if !plan.lane_kind(row).is_prefill() {
                continue;
            }
            run.prefill_tokens += lane.tokens as u64;
            let agent = flight[&lane.request].0;
            let c = &conversations[agent];
            let record = run.turn_records.get_mut(&(agent, c.restarts, c.turn)).unwrap();
            record.launches += 1;
            record.small_launches += usize::from(plan.token_capacity <= 512);
            record.prefilled += lane.tokens as u64;
        }
        for (agent, c) in conversations.iter_mut().enumerate() {
            let Some((h, turn, admitted, seen, output)) = c.live.as_mut() else { continue };
            if !*seen && first_token(&s, *h) {
                *seen = true;
                run.turn_records.get_mut(&(agent, c.restarts, *turn)).unwrap().ttft_ms = (s.now_ns - *admitted) as f64 / 1e6;
            }
            if s.request(*h).is_err() {
                let (h, output) = (*h, *output);
                let record = run.turn_records.get_mut(&(agent, c.restarts, c.turn)).unwrap();
                record.finished = true;
                run.turns += 1;
                run.output_tokens += u64::from(output);
                let (_, prompt, boundaries, output) = flight.remove(&h).unwrap();
                c.live = None;
                c.answered(prompt, boundaries, output, a.tools, a.reasoning);
            }
        }
    }
    run.sim_seconds = (s.now_ns - start) as f64 / 1e9;
    run.rounds_digest = twox_hash::XxHash3_64::oneshot(digest_text.as_bytes());
    let stats = s.stats();
    run.message_checkpoints = stats.message_checkpoints;
    run.prompt_checkpoints = stats.prompt_checkpoints;
    // Branch probe: each agent's history up to the end of its last answer, then a different
    // user message (an edited tool result or a regenerated step).
    for c in &conversations {
        if let Some((h, ..)) = c.live {
            s.cancel(h).unwrap();
        }
    }
    for _ in 0..64 {
        if s.plan().is_none() {
            break;
        }
        ds41_step(&mut s, 3);
    }
    for (agent, c) in conversations.iter().enumerate() {
        if c.turn < 2 {
            continue;
        }
        let answer_end = *c.boundaries.last().unwrap();
        let mut prompt = c.prompt[..answer_end].to_vec();
        prompt.push(USER);
        prompt.extend(std::iter::repeat_n(50_000 + agent as u32, 900));
        prompt.extend(generation_prompt(a.reasoning));
        let spec = RequestSpec { prompt, message_boundaries: vec![answer_end], conditioning: vec![], max_new_tokens: 8,
            grammar_constrained: false, host_stops: false, prompt_logprobs: false, growable_output: false,
            tools: a.tools, reasoning_prompt: a.reasoning };
        if let Ok(plan) = s.plan_admission(&spec) {
            run.branch_resident.push((agent, answer_end as u64, plan.resident_tokens()));
        }
    }
    run
}

const DS41_REPLAY: Agents = Agents { engine: ds41_agent_engine, agents: 8, seconds: 120, tools: false,
    reasoning: false, shape: REPLAY };

/// Flag off reproduces the rounds of the scheduler before the prefill merge existed (digests
/// recorded on 498d0ab6 with this file's scenarios and the policy code unchanged). The DS41
/// entries were re-recorded with the acceptance population (#306): new requests start from the
/// acceptance of earlier ones instead of the fixed prior, which changes their first draft widths.
/// Before it: replay and tools 477 rounds, digest 6826759491390006074; thinking 470 rounds,
/// digest 4936267365442837913. GLM tools did not change.
#[test]
fn golden_agent_rounds_without_prefill_merge() {
    let short = |a: Agents| Agents { seconds: 20, ..a };
    for (name, a, tail, rounds, digest) in [
        ("ds41 replay", short(DS41_REPLAY), PrefillTailMerge::On, 480, 4172790986037816443u64),
        ("ds41 thinking", short(Agents { reasoning: true, ..DS41_REPLAY }), PrefillTailMerge::On, 471, 16995297670789752731),
        ("ds41 tools", short(Agents { tools: true, reasoning: true, ..DS41_REPLAY }), PrefillTailMerge::ToolsOnly, 480,
            4172790986037816443),
        ("glm tools", short(Agents { engine: glm_engine, agents: 16, tools: true, ..DS41_REPLAY }), PrefillTailMerge::ToolsOnly,
            173, 9133372263739486257),
    ] {
        let run = run_agents(a, agent_policy(tail, 0));
        eprintln!("{name}: rounds {} digest {} | {}", run.round_count, run.rounds_digest, run.summary("off"));
        assert_eq!((run.round_count, run.rounds_digest), (rounds, digest), "{name}");
    }
    assert_eq!(PolicyConfig::default().prefill_merge_min_rows, 0);
    assert_eq!(PolicyConfig::interactive().prefill_merge_min_rows, 0);
    assert_eq!(PolicyConfig::throughput().prefill_merge_min_rows, 0);
}

/// A request resumed at `consumed` (its own checkpoint unless `foreign`), whose new input is the
/// previous answer's end-of-sentence token (boundary `consumed + 1`), then `messages` more
/// messages of the given lengths, then a two-token generation prompt.
fn resumed(s: &mut RequestScheduler, consumed: usize, messages: &[usize], tools: bool, reasoning: bool, foreign: bool)
    -> RequestHandle {
    let mut boundaries = vec![consumed + 1];
    for length in messages {
        boundaries.push(boundaries.last().unwrap() + length);
    }
    let prompt = boundaries.last().unwrap() + 2;
    let h = s.admit(RequestSpec { prompt: vec![11; prompt], message_boundaries: boundaries, conditioning: vec![],
        max_new_tokens: 16, grammar_constrained: false, host_stops: false, prompt_logprobs: false,
        growable_output: false, tools, reasoning_prompt: reasoning }).unwrap();
    let r = s.request_mut(h).unwrap();
    r.phase = RequestPhase::Prefill;
    r.consumed_tokens = consumed as u64;
    r.scheduled_consumed = consumed as u64;
    r.prefix_tokens = if foreign { 0 } else { consumed as u64 };
    h
}

fn configured(rows: usize, tail: PrefillTailMerge) -> RequestScheduler {
    let mut s = ds41_agent_engine(8);
    s.configure_policy(PolicyConfig { prefill_merge_min_rows: rows, prefill_tail_merge: tail,
        ..PolicyConfig::interactive() }).unwrap();
    s
}

fn works(s: &RequestScheduler, h: RequestHandle) -> Vec<(usize, usize)> {
    let r = s.request(h).unwrap();
    s.prefill_catalog.iter().map(|&c| (c, s.prefill_work(r, c))).collect()
}

/// The agent turn of `ttft_under_load.py agent`: one end-of-sentence token, a 3,461-token user
/// message, the generation prompt. Today the first token is its own 128-row launch.
#[test]
fn forward_merge_joins_the_answer_end_to_the_next_message() {
    for rows in [0, 512] {
        let mut s = configured(rows, PrefillTailMerge::On);
        let h = resumed(&mut s, 50_000, &[3461], false, false, false);
        let work = works(&s, h);
        let at = |c: usize| work.iter().find(|w| w.0 == c).unwrap().1;
        if rows == 0 {
            assert!(work.iter().all(|&(_, w)| w == 1), "{work:?}");
            continue;
        }
        // Answer end + user message + generation prompt (tail merge) in one 3,584-row launch.
        assert_eq!(at(3584), 3464);
        assert_eq!(at(4096), 3464);
        assert_eq!(at(128), 128);
        assert_eq!(at(2560), 2560);
        // Within a segment the chunk size is the scheduler's choice, as without the merge.
        assert_eq!(at(3072), 3072);
    }
}

#[test]
fn forward_merge_needs_a_chunk_start_at_the_request_checkpoint_and_skips_only_short_pieces() {
    let mut s = configured(512, PrefillTailMerge::On);
    // Not at its own checkpoint (a capture the cache already held): today's chunk.
    let foreign = resumed(&mut s, 20_000, &[3461], false, false, true);
    assert_eq!(s.prefill_work(s.request(foreign).unwrap(), 4096), 1);
    // Several short pieces accumulate until the merged piece reaches 512 tokens.
    let h = resumed(&mut s, 20_000, &[100, 300, 2500], false, false, false);
    assert_eq!(s.prefill_work(s.request(h).unwrap(), 4096), 1 + 100 + 300 + 2500 + 2);
    let h = resumed(&mut s, 30_000, &[700, 2500], false, false, false);
    // 1 + 700 >= 512: the merged piece stops at the end of the 700-token message.
    assert_eq!(s.prefill_work(s.request(h).unwrap(), 4096), 701);
}

/// The thinking rule: a request the tail merge does not apply to (thinking without tools under
/// `On`; every request without tools under `ToolsOnly`; all under `Off`) keeps its last message,
/// the checkpoint its next turn resumes from; earlier short pieces still merge.
#[test]
fn forward_merge_keeps_the_last_message_when_the_tail_merge_does_not_apply() {
    for (tail, tools, reasoning) in [(PrefillTailMerge::On, false, true), (PrefillTailMerge::ToolsOnly, false, false),
        (PrefillTailMerge::Off, true, false), (PrefillTailMerge::On, false, false), (PrefillTailMerge::On, true, true),
        (PrefillTailMerge::ToolsOnly, true, true)] {
        let mut s = configured(512, tail);
        // Answer end, then a 300-token user message: the whole new input is below 512 tokens.
        let h = resumed(&mut s, 10_000, &[300], tools, reasoning, false);
        let work = s.prefill_work(s.request(h).unwrap(), 4096);
        let expected = if tail.applies(tools, reasoning) { 303 } else { 301 };
        assert_eq!(work, expected, "{tail:?} tools={tools} reasoning={reasoning}");
    }
}

/// A segment longer than the threshold is chunked as without the merge, and a piece that starts
/// mid-segment (after a capacity-bound chunk) is not merged across the next boundary.
#[test]
fn forward_merge_leaves_long_segments_and_mid_segment_pieces_alone() {
    let mut s = configured(512, PrefillTailMerge::On);
    // Answer end + 4,300-token message + generation prompt: 4,096 tokens, then the rest.
    let h = resumed(&mut s, 10_000, &[4300, 3000], false, false, false);
    assert_eq!(s.prefill_work(s.request(h).unwrap(), 4096), 4096);
    assert_eq!(s.prefill_work(s.request(h).unwrap(), 3584), 3584);
    let r = s.request_mut(h).unwrap();
    r.consumed_tokens += 4096;
    r.scheduled_consumed = r.consumed_tokens;
    // 205 tokens left before the boundary, but the chunk does not start at a checkpoint.
    assert_eq!(s.prefill_work(s.request(h).unwrap(), 4096), 205);
}

/// Drives `h` to its end with `ds41_step` and returns the prompt-end resume point a follow-up
/// turn sharing `shared` tokens with it finds in the cache.
fn follow_up_resume(rows: usize, tail: PrefillTailMerge, tools: bool, reasoning: bool) -> (u64, u64) {
    let mut s = configured(rows, tail);
    let prompt = 1500;
    let h = s.admit(RequestSpec { prompt: (0..prompt as u32).collect(), message_boundaries: vec![1, 100, prompt - 2],
        conditioning: vec![], max_new_tokens: 3, grammar_constrained: false, host_stops: false, prompt_logprobs: false,
        growable_output: false, tools, reasoning_prompt: reasoning }).unwrap();
    let mut launches = 0;
    while s.request(h).is_ok() {
        let plan = ds41_step(&mut s, 3);
        launches += u64::from(plan.kind.is_prefill());
    }
    let shared = if reasoning && !tools { prompt - 1 } else { prompt };
    let follow_up = RequestSpec { prompt: (0..shared as u32).chain([7777, 7778, 7779]).collect(),
        message_boundaries: vec![], conditioning: vec![], max_new_tokens: 1, grammar_constrained: false,
        host_stops: false, prompt_logprobs: false, growable_output: false, tools, reasoning_prompt: reasoning };
    (s.plan_admission(&follow_up).unwrap().resident_tokens(), launches)
}

/// The next turn's resume point does not move under the merge (tail-merge cases of
/// `boundary_tests::tail_merge_keeps_the_follow_up_turn_resume_point`); the merge only removes
/// the one-token piece after the beginning of sequence.
#[test]
fn prefill_merge_keeps_the_follow_up_turn_resume_point() {
    for tail in [PrefillTailMerge::Off, PrefillTailMerge::ToolsOnly, PrefillTailMerge::On] {
        for (tools, reasoning) in [(false, true), (false, false), (true, true), (true, false)] {
            let (off, off_launches) = follow_up_resume(0, tail, tools, reasoning);
            let (on, on_launches) = follow_up_resume(512, tail, tools, reasoning);
            let case = format!("{tail:?} tools={tools} reasoning={reasoning}");
            assert_eq!(on, off, "{case}");
            assert!(on_launches < off_launches, "{case}: {on_launches} vs {off_launches}");
        }
    }
}

/// Flag off vs `prefill_merge_min_rows` 512 on the same conversations. `strict` (no eviction):
/// every turn resumes exactly where it does without the merge.
fn compare(name: &str, a: Agents, tail: PrefillTailMerge, strict: bool) -> (Run, Run) {
    let off = run_agents(a, agent_policy(tail, 0));
    let on = run_agents(a, agent_policy(tail, 512));
    eprintln!("{name}\n  {}\n  {}", off.summary("off"), on.summary("on 512"));
    let branch = |run: &Run| run.branch_resident.iter().map(|(_, end, resident)| end - resident).collect::<Vec<_>>();
    eprintln!("  branch at the last answer end, re-prefilled tokens: off {:?} on {:?}", branch(&off), branch(&on));
    // (b) Same prompts (agent, conversation, turn), so the resume point at admission is the
    // next-turn reuse each run left behind. Compared over the turns both runs finished.
    let (mut compared, mut later, mut earlier) = (0, 0, 0);
    let (mut resident, mut prefilled) = ([0u64; 2], [0u64; 2]);
    for (key, t) in &on.turn_records {
        if let Some(o) = off.turn_records.get(key).filter(|o| o.finished && t.finished) {
            assert_eq!(t.prompt, o.prompt);
            compared += 1;
            later += usize::from(t.resident > o.resident);
            earlier += usize::from(t.resident < o.resident);
            resident[0] += o.resident;
            resident[1] += t.resident;
            prefilled[0] += o.prefilled;
            prefilled[1] += t.prefilled;
        }
    }
    eprintln!("  {compared} turns in both runs: on resumed later in {later}, earlier in {earlier}; resident {resident:?}, \
        prefilled (incl. re-prefill) {prefilled:?}");
    assert!(compared > 2 * a.agents, "{name}: {compared} turns compared");
    // No turn resumes earlier because of the merge (equal prompts and cache pressure aside:
    // fewer captures can only leave more checkpoints alive), and no more tokens are prefilled.
    if strict {
        assert_eq!((later, earlier), (0, 0), "{name}");
    }
    assert!(resident[1] >= resident[0] && prefilled[1] <= prefilled[0], "{name}: {resident:?} {prefilled:?}");
    // Fewer launches, first tokens no later on average and sooner at the median, no fewer turns.
    assert!(on.per_turn(|t| t.launches as f64) <= off.per_turn(|t| t.launches as f64) - 0.8, "{name}");
    assert!(on.per_turn(|t| t.ttft_ms) <= 1.05 * off.per_turn(|t| t.ttft_ms) && on.ttft(0.5) < off.ttft(0.5), "{name}");
    assert!(on.turns >= off.turns, "{name}");
    // (c) No agent starves: each one finishes turns in both runs.
    for run in [&off, &on] {
        for agent in 0..a.agents {
            assert!(run.turn_records.iter().filter(|(k, t)| k.0 == agent && t.finished).count() >= 2,
                "{name}: agent {agent} starved");
        }
    }
    // A branch at a skipped boundary re-prefills fewer than 512 tokens more (without eviction;
    // with it, whether the branch point survives depends on which checkpoints were compacted).
    let lost = |run: &Run| run.branch_resident.iter().filter(|(_, end, resident)| end - resident >= 512).count();
    eprintln!("  branch probes re-prefilling >= 512 tokens: off {}/{}, on {}/{}", lost(&off), off.branch_resident.len(),
        lost(&on), on.branch_resident.len());
    if strict {
        for (agent, end, on_resident) in &on.branch_resident {
            let Some((_, _, off_resident)) = off.branch_resident.iter().find(|b| b.0 == *agent && b.1 == *end) else { continue };
            assert!(on_resident + 512 > *off_resident, "{name}: {end} {off_resident} {on_resident}");
        }
    }
    (off, on)
}

/// (b) The agent replay at 8 streams (no tools, thinking off: the measured agent8 traffic).
#[test]
fn prefill_merge_on_the_ds41_agent_replay() {
    let (off, on) = compare("ds41 replay, tail merge on", DS41_REPLAY, PrefillTailMerge::On, false);
    compare("ds41 replay, no eviction", Agents { engine: ds41_roomy_engine, ..DS41_REPLAY }, PrefillTailMerge::On, true);
    // The answer's end-of-sentence token no longer takes a launch and a message checkpoint.
    assert!(on.message_checkpoints * 4 < off.message_checkpoints);
    assert!(on.snapshot_launches < off.snapshot_launches);
}

/// (d) Thinking without tools keeps its last message (its next turn's resume point); the
/// answer piece before it (end of thinking-dropped history, ~n + 3 tokens) merges.
#[test]
fn prefill_merge_on_ds41_thinking_turns_without_tools() {
    compare("ds41 thinking, no tools, tail merge on", Agents { reasoning: true, ..DS41_REPLAY }, PrefillTailMerge::On, false);
    compare("ds41 thinking, no tools, no eviction", Agents { engine: ds41_roomy_engine, reasoning: true, ..DS41_REPLAY },
        PrefillTailMerge::On, true);
}

/// (d) Tools + thinking under the fork default `ToolsOnly`, and at 16 streams.
#[test]
fn prefill_merge_on_ds41_tool_turns() {
    compare("ds41 tools, tools-only", Agents { tools: true, reasoning: true, ..DS41_REPLAY }, PrefillTailMerge::ToolsOnly, false);
    compare("ds41 tools, no eviction", Agents { engine: ds41_roomy_engine, tools: true, reasoning: true, ..DS41_REPLAY },
        PrefillTailMerge::ToolsOnly, true);
    compare("ds41 replay c16", Agents { agents: 16, ..DS41_REPLAY }, PrefillTailMerge::On, false);
}

/// GLM costs, 16 and 32 agents with tools (GLM serves 32 slots).
#[test]
fn prefill_merge_on_glm_agent_turns() {
    for agents in [16, 32] {
        compare(&format!("glm tools c{agents}"), Agents { engine: glm_engine, agents, seconds: 240, tools: true,
            ..DS41_REPLAY }, PrefillTailMerge::ToolsOnly, false);
        compare(&format!("glm tools c{agents}, no eviction"), Agents { engine: glm_roomy_engine, agents, seconds: 240,
            tools: true, ..DS41_REPLAY }, PrefillTailMerge::ToolsOnly, true);
    }
}

/// (c) Long prompts with many short messages next to decoders and pings: every request gets
/// its first token, none later than without the merge plus a small slack.
#[test]
fn prefill_merge_does_not_starve_long_prompts_or_pings() {
    let mut first = Vec::new();
    for rows in [0, 512] {
        let mut s = ds41_agent_engine(10);
        s.configure_policy(merge_policy(ttft_policy(50.0, 0, 1), rows)).unwrap();
        tp3_decoders(&mut s, 2, 3);
        let mut handles = Vec::new();
        for (i, prompt) in [40_000usize, 40_000, 25_000, 1_900, 900, 9_000].into_iter().enumerate() {
            // A message every 150 tokens (selected down to the checkpoint budget) plus a 1-token BOS piece.
            let boundaries: Vec<usize> = std::iter::once(1).chain((150..prompt - 2).step_by(150)).collect();
            let h = s.admit(RequestSpec { prompt: vec![200 + i as u32; prompt], message_boundaries: boundaries,
                conditioning: vec![], max_new_tokens: 16, grammar_constrained: false, host_stops: false,
                prompt_logprobs: false, growable_output: false, tools: false, reasoning_prompt: false }).unwrap();
            handles.push((h, s.now_ns));
        }
        let mut seen = vec![None; handles.len()];
        for _ in 0..3000 {
            ds41_step(&mut s, 3);
            for (i, (h, admitted)) in handles.iter().enumerate() {
                if seen[i].is_none() && first_token(&s, *h) {
                    seen[i] = Some((s.now_ns - admitted) as f64 / 1e6);
                }
            }
            if seen.iter().all(Option::is_some) {
                break;
            }
        }
        eprintln!("rows {rows}: first tokens ms {seen:?}");
        assert!(seen.iter().all(Option::is_some), "rows {rows}: {seen:?}");
        first.push(seen.into_iter().map(Option::unwrap).collect::<Vec<_>>());
    }
    for (off, on) in first[0].iter().zip(&first[1]) {
        assert!(*on <= off * 1.05 + 100.0, "{:?}", first);
    }
}

/// The DS41 replay at 8, 2 and 1 agents under fitted and measured prefill costs, by threshold.
/// Prints only; the asserted comparisons are above. (11-15: a "stub" rule that refused chunks
/// leaving < N tokens of a segment for another launch was tried here and dropped: no change at
/// c8 once the answer end merges, and at c1/c2 the greedy choice found other splits.)
#[test]
fn prefill_merge_attribution_on_the_ds41_agent_replay() {
    for (engine, name, agents) in [(ds41_agent_engine as fn(usize) -> RequestScheduler, "fitted costs", 8),
        (ds41_measured_engine, "measured graph medians", 8), (ds41_measured_engine, "measured, c2", 2),
        (ds41_measured_engine, "measured, c1", 1)] {
        for (label, rows) in [("off", 0), ("merge 128", 128), ("merge 512", 512), ("merge 1024", 1024)] {
            let policy = agent_policy(PrefillTailMerge::On, rows);
            let run = run_agents(Agents { engine, agents, ..DS41_REPLAY }, policy);
            eprintln!("{}", run.summary(&format!("{name}, {label}")));
        }
    }
}
