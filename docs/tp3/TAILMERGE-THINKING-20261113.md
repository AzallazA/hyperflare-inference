# Tail merge `true` without the thinking regression, on tp3-main (2026-11-13)

> Commit ids shown as [unpublished] belong to the local history before the 2026-11-14 port and are not in the public repository.

Branch `ds41-tailmerge-thinking` (worktree `~/flare-up-tmfix`) on `ds41-upstream-285-286` ([unpublished] = tp3-main + #286,
GPU-validated 11-13). Local and Gitea only. Not merged: needs the GPU check
`~/ds41-tp3/analysis/work/tmfix/val-c.sh` (waiter `/tmp/gpuq-seq95.sh`).

## Why

Upstream 617ef929 (+#285) stays out (cold-start stall: probe TTFT 118-158 s, decoders ~1 tok/s). On tp3-main, with
`--scheduler-prefill-tail-merge true` (serve-best default since 11-13) the only broken case is a THINKING request
WITHOUT tools: DS41 drops reasoning from history when no tools are rendered, so this turn's `<｜Assistant｜><think>`
becomes `<｜Assistant｜></think>answer` in the next turn and that turn diverges right after the last message. The merge
runs the chunk that would stop at the last message straight to the prompt end, so the last-message checkpoint is never
captured and the next turn falls back to the previous selected boundary: it re-prefills the whole last user message
(val-b base arm: thinking 8K/32K follow-up gaps 14359 / 57898 on turn 2, ~370-430 later; refoff/tools-only: 2).

## What is ported from `sched-617-tailmerge` ([unpublished], the final form of [unpublished])

- `RequestSpec::reasoning_prompt` / `Prompt::reasoning_prompt`: the generation prompt opens a reasoning block, set in
  `server/request.rs` from the prepared output's start state (`OutputStart::Reasoning`).
- `PrefillTailMerge::applies(tools, reasoning_prompt)`: `On` = `tools || !reasoning_prompt`. A no-tool thinking
  request is not merged and keeps its last-message checkpoint, as under `tools-only`. `false` and `tools-only`
  unchanged. CLI help updated.
- Not ported: `tail_merge_cut` (prompt end as the anchor) and the merged-cut bookkeeping. They exist because 617ef929
  makes the anchor the only unbudgeted cut; tp3-main has no anchor and no work budget (below).

## Why tp3-main needs no guaranteed prompt-end checkpoint

On tp3-main a prefill chunk that ends at the prompt end captures a `BoundaryKind::Prompt` checkpoint whenever
`plans_prompt_capture()` holds (`checkpoint_budget >= 2 && max_new_tokens > 1`). `checkpoint_budget` is a static slot
share (`snapshots / max_active`, clamped 1..8) fixed at admission; the prompt end and the terminal snapshot are
reserved from it first, and only the remainder goes to message boundaries (`checkpoint_boundaries`). Nothing depends on
measured costs, freshness or an extra-work allowance, and the merge only moves the final chunk end onto the prompt end,
which is then captured as before (`all_selected_message_boundaries_are_offered`: merged requests publish `[130]`).
The cases where the prompt end is not captured (budget < 2, `max_new_tokens` 1, snapshot pool momentarily empty) are
the same in every merge mode. So the 617ef929 failure (a merged request with no checkpoint at all) cannot happen here.

Known limit, unchanged by this branch (same as `false` / `tools-only`): with more message boundaries than the budget,
`checkpoint_boundaries` spreads the selection evenly, so the last message boundary is not guaranteed to be selected.

## Resume point of the next turn (CPU scheduler test, 400 / 1500-token prompts, 512-token chunks)

System message ends at 100, last message 2 tokens before the prompt end. thinking = no tools + reasoning (next turn
diverges right after the last message); chat = no reasoning; tools = tools (+/- reasoning).

| mode | tp3-main thinking / chat / tools | this branch thinking / chat / tools |
| --- | --- | --- |
| false | 398 / 400 / 400 | 398 / 400 / 400 |
| tools-only | 398 / 400 / 400 | 398 / 400 / 400 |
| true | **100** / 400 / 400 | 398 / 400 / 400 |

Cost: a no-tool thinking request pays the 2-token tail graph again (as with tools-only). `llm_decode_bench` sends no
thinking flag and DS41 defaults to thinking, so its 8K prefill number may drop toward tools-only (INFO in val-c);
probe8 and the agent replay send thinking off and keep the merge.

## CPU checks (`CUDA_VISIBLE_DEVICES=`)

- helios `cargo test --lib --features server,runtime-core`: 1381 passed / 4 failed (the template-dir failures;
  branch A 1380 / 4). Ported/new: `tail_merge_keeps_the_follow_up_turn_resume_point` (all modes x tools x reasoning,
  400 and 1500 tokens), `tail_merge_on_keeps_the_last_message_only_for_thinking_without_tools`, `MERGE_CASES` with the
  reasoning flag (offered-boundary tests). TTFT guard tests (`ttft_*`, `mixed_candidates_satisfy_the_ttft_guard`,
  `glm_ttft_under_decode_load_cpu`) pass.
- `tests/deepseek_requests.rs` 15 passed, including
  `reasoning_prompt_marks_turns_whose_next_turn_rewrites_the_generation_prompt` (real DS V4/V4.1 template renders:
  flag values; which next turns extend the prompt). `message_boundaries` 6/3 and `deepseek_encoder` 14/1: the same
  template-dir / Jinja-image failures as branch A.
- flare-cli `cargo test`: 177 passed / 2 failed (the pre-existing Orbit fixture failures).
- Release build of flare-cli: 36 s (helios + flare-cli rebuilt on a target dir seeded from `~/flare-up-rust1113`).

## GPU check (val-c.sh)

DS41 1241 on this binary with serve-best.sh (tail-merge true): KV, verify_fix, tools, eval-43, multi-turn resume gaps
(thinking no-tools <= 16, tools+thinking 0, thinking off <= 0; 8K/32K, cold and warm), idle; same-session speed A/B vs
the tp3-main binary (c1/c4/c8, 2 rounds, probe8 p90 < 1.0 s gate); GLM TP3 130000 and TP2 130307 KV/probe/smoke.
Verdict line `TMFIX VERDICT: PASS|HOLD <reasons>`.
