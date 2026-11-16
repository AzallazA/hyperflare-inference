# TP3 fork handbook: orientation, workflows, rules and the mistakes behind them

For anyone (human or agent) working on DeepSeek-V4.1-Flash on NVIDIA Flare at TP3 (3x RTX PRO 6000 Blackwell
Workstation, 450 W, +75 core / +6000 mem). Part A says where things are and how to run them. Part B gives the
step-by-step for the recurring tasks. Part C is tips for agents. Part D is the rule list, each rule with the
incident that produced it (dates are 2026-11). Read the part you need before starting; read D.1 before any speed
claim, D.3 before any kernel edit, D.4 before any push.

---

## A. Orientation

### A.1 Where things live

| What | Where |
| --- | --- |
| **Main branch (since 2026-11-13)** | `tp3-main` = DS41 + GLM-5.3-Flash on one branch (merge of `ds41-tp3-sync` and `glm-hyperflare`, validated 11-13: speed level, DS41 eval 43/43, GLM probe/smoke, GLM TP2 bit-identical across binaries). Worktree `~/flare-up-main`, ONE binary `flare-cli/target/release/flare` serves both models (`serve-best.sh`, `serve-glm.sh`, `~/switch-model.sh`). Exports from it need `PYTHONPATH=~/flare-up-main` (GLM: `~/flare-up-main/.b12x-overlay:~/flare-up-main`) or `ignition` imports another worktree. Workflow: every change on its own branch (`ds41-*`, `glm-*`, `fix/*`), merged into `tp3-main` only after it is confirmed on the GPU; shared-code changes also need the other model's CPU tests + smoke/speed; upstream syncs go through a `sync-<date>` branch. Public fork gets scrubbed release snapshots only. **State 11-14:** `tp3-main` has NOT moved for the upstream port; it moves only after GLM passes on the port (user OK needed), and still serves GLM. |
| **Upstream port (2026-11-14, in progress)** | Port of our TP3 work onto upstream 70069281 (20 commits after the v0.1.0 tag; #285 and 617ef929 reverted). Worktrees: `~/flare-up-port` (branch `port-20261114`, own venv, kernel cache `~/.cache/flare/kernels-port`, config `~/.config/flare-port`), DS41 and GLM feature branches `port-20261114-ds41` / `port-20261114-glm`, and the merge of both in `~/flare-up-port-int` (branch `port-20261114-int`, own venv, overlay, wheel and binary; GLM config `~/.config/flare-port-glm`). **The DS41 local default serves from `~/flare-up-port-int` since 11-14 19:07** (B.8). GLM has NOT been exported from the port yet: the GLM defaults are still the tp3-main packages on the tp3-main binary. A Rust rebuild in `~/flare-up-port-int` changes what the next DS41 restart serves. |
| **State 2026-11-16 (release candidate)** | `tp3-main` = c0ff27c5 (on the port lineage: c8fb6185 = frontend reference fixes, required-thinking 400, CUDA stack limit, TTFT guard order; plus round-2 code and export KV hygiene, all round-2 switches default off in the code). The DS41 default serves from branch `perf/sched-prefill-merge-c8fb6185` (c8fb6185 + prefill merge a02548e1/593d2803, binary sha256 27864a1d79e4..., since 11-16 15:00) until that branch is merged into `tp3-main`; it merges cleanly and changes no Python, so the merge's Rust equals the serving binary's. Default package `ds41-1017-engram-fp8/gb202-tp3-202611160815` (exported from the round-2 branch 302c9b1a = `tp3-main`'s export path without the KV hygiene commit). `serve-best.sh` defaults: PCM 12G, BUDGET 32, `FLARE_CUDA_STACK_LIMIT=256`, `--scheduler-ttft-order shortest`, `FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS=512`, interactive + tail merge, private tables + fast exit; served KV 5,403,008. |
| Fork checkout (branch `ds41-tp3-v3`, pre-merge; history only) | `~/flare-up-v3` (`.venv/`, `flare/target/release/flare`, `.brrr-ext/` compiled kernels) |
| Other worktrees of the same repo | `~/flare-up` (old v2, its `serve-best.sh` just execs v3), `~/flare-up-base` (detached baseline), `~/flare-up-stage2` (kernel stage 2), `~/flare-up-vision` (vision staging). `git worktree list` from any of them. |
| Recipe (the Python that defines the model for Ignition) | `ignition/recipes/deepseek41/` (`decoder.py`, `attention.py`, `weights.py`, `brrr.py`, `bootstrap.py`, `fused_qkv_norm.py`) |
| BRRR fused layer tail kernel | `brrr/kernels/decoder.cu` (+ `flare_sparse_decode.cu`, opt-in) |
| Rust serving runtime | `helios/src/` (model/allocation accounting, Engram reader, protection), CLI in `flare/src/cli.rs` |
| FlashInfer (b12x) site-package patches | `tp3-patches/*.patch` (re-apply to `.venv` after every `uv sync`) |
| TP3 test tooling | `tp3-tools/` (`bench_steps.py`, `spec_metrics.py`, `steps_logger.py`, `test_fused_qkv_norm.py`, `test_flare_sparse_decode.py`); Engram precision eval in `tp3-tools/engram-eval/` |
| Release scripts and guides (single source) | `tp3-tools/release/` (`export.sh`, `serve.sh`, `make_checkpoint_view.sh`, `check_env.sh`, `expert-order/`), `docs/tp3/EXPORT.md`, `docs/tp3/ENGRAM-PRECISION.md`; the dated recipes folder carries copies made at tagging time |
| GPU unit tests | `tests/recipes/test_deepseek41_*_cuda.py`; the TP3 ones are `..._brrr_tail_cuda.py` (3 GPUs) and `..._expert_parking_cuda.py` (1 GPU) |
| Quality and consistency checks | `tp3-tools/verify_fix.py` (TRUE corruptions), `tp3-tools/consistency.py` (decode vs prefill logprobs), `~/ds41-tp3/bench.py` (fixed 43-answer set + c4) |
| KV capacity without a serve | `flare run <pkg> <serve flags> --dry-run` (per-rank memory plan + KV tokens + snapshot slots; exact on 5 recorded loads; `--dry-run-free-memory` overrides the assumed free VRAM) |
| Full qualification suite | release folder `qualification/qual_all.sh` (eval tools lavd hotel retrieval longgen eos livebench stress); needs `~/llm-inference-bench`, `~/LiveBench` |
| Checkpoints | full view `/m2-2/flare-ckpt/ds41-1017-engram-mxfp4` (built by `release/make_checkpoint_view.sh`; the release docs' `~/ds41-1017-engram-mxfp4` is an example path, not ours); truncated test models `/m2-2/flare-ckpt/ds41-mini6` (6 layers) and `ds41-mini8` (8 layers, has native scales, use for parking/BRRR tests) |
| Exported packages | `~/.cache/flare/packages/models/ds41-1017-engram-fp8/gb202-tp3-<timestamp>/` (FP8 Engram view, since 11-16; DS41 default `...202611160815`, served KV 5,403,008 with stack limit 256) and `.../ds41-1017-engram-mxfp4/...` (MXFP4 view: 11-14 default `...202611142139`, upstream port, HOT 342, served KV 5,024,384; rollback `...202611121241` (served KV 3,189,376; also serves on the port binary); each non-default package is ~4 GB on disk (shared blobs) |
| Server scripts | `~/flare-up-v3/serve-best.sh` (defaults since 11-14 19:07: `P` = pkg 142139, `FLARE_DIR` = `~/flare-up-port-int`; K=5, ALPHA=0.8, BUDGET=24, PCM=32G, SCHED=interactive, private Engram tables + fast exit), restart via `/tmp/restart-default.sh`. Rollback: `FLARE_DIR=~/flare-up-main P=<...121241> ./serve-best.sh`. Previous script: `serve-best.sh.bak-1114-1241default` |
| Release folder (public) | `~/rtx-recipes/DeepSeek-V4.1-Flash Hyperflare-tp3-YYYYMMDD/` from the Hyperflare release on (earlier: `... Flare-tp3-2026110x/`) (README, EXPORT, MEASUREMENTS, ENGRAM-PRECISION, export.sh, serve.sh, source.patch, qualification/) |
| Upstream proposals | `docs/upstream-proposals/` in the fork (public with the branch, not submitted upstream); mirrored on the `proposals` branch of the private staging repo |
| Scratch / logs / older vLLM-era work | `~/ds41-tp3/` (bench logs, eval_*.json, analyze_trace.py, compose files); `~/engram-eval/` (FP8 vs MXFP4 Engram perplexity eval) |
| Archive of old builds | `/mnt/ai_a` (AI_A), cold storage `/mnt/ai_b` |

### A.2 Server: start, stop, inspect

```bash
# is it up, what is it serving
curl -s localhost:15015/v1/models | grep -o '"id":"[^"]*"'
ps -eo args | grep "release/[f]lare run" | grep -o "gb202[^ ]*\|budget [0-9]*\|alpha [^ ]*\|memory [^ ]*"
curl -s localhost:15015/metrics | awk '/^flare_total_kv_tokens/{print $2}'     # KV pool in tokens

# stop (never pgrep -f a pattern that is in your own command line, see D.2.4)
for p in $(ps -eo pid,args | awk '$2=="./flare/target/release/flare" && $3=="run"{print $1}'); do kill -INT $p; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)" -lt 1000 ]; do sleep 2; done

# start the default (background, log in ~/flare-up-v3/serve-best.log); takes ~2 min to answer /v1/models
/tmp/restart-default.sh
# start a candidate package with overrides
cd ~/flare-up-v3 && P=~/.cache/flare/packages/models/ds41-1017-engram-mxfp4/gb202-tp3-2026110XXXXX BUDGET=24 nohup ./serve-best.sh > cand.log 2>&1 &
```

The server needs ~97 GiB host RAM for the two Engram tables (`FLARE_ENGRAM_RAM_TABLES=0`) plus the parked experts
and prefix cache. Chat requests: thinking is toggled with `chat_template_kwargs: {"enable_thinking": true|false}`.
Flare clamps temperature 0 to 1.0.

### A.3 Benchmarks

```bash
# decode speed, c1/c4/c8 at 32K context, 60 s each; writes JSON
python3 ~/llm-inference-bench/llm_decode_bench.py --port 15015 --model DeepSeek-V4.1-Flash \
    --concurrency 1,4,8 --contexts 32k --duration 60 --output $D/$LABEL-32k-r1.json
# long context c1
... --concurrency 1 --contexts 8k,32k,128k,512k --duration 30 --output $D/$LABEL-longctx.json
# coding peak (high-acceptance speculative workload)
... --concurrency 1 --contexts 8k,32k,128k,512k --coding-peak --output $D/$LABEL-codingpeak.json
# steps/s and acceptance from the server log
.venv/bin/python tp3-tools/spec_metrics.py <serve log>
```

Reference numbers for the default build at 450 W, cap 24, 32G prefix cache: c1/c4/c8 32K ≈ 298/738/959 tok/s,
coding c1 ≈ 456 tok/s. Spread between runs is 2-3%. See D.1 before interpreting anything. Port default 142139
against 1241 on the same binary (11-14, 2 alternating 30 s rounds): c1/c4/c8 32K 301/717/970 vs 318/720/979 tok/s;
c1 is 0.974x in steps/s (accepted by the user 11-14, speed work later, D.1.12).

### A.4 Quality checks (cheapest first)

1. `python3 tp3-tools/verify_fix.py 10` on the running server: look only at `TRUE corruptions` (must be 0).
2. `python3 ~/ds41-tp3/bench.py out.json` : 43 known-answer checks (expect 42-43/43), 1,500-token decodes, 32K needle,
   4 concurrent.
3. `.venv/bin/python tp3-tools/consistency.py 96 6 <tokenizer dir>` : decode-vs-prefill logprob agreement; compare against the
   same number from the default build in the same session (it is biased toward b12x rounding). The summary line ends
   with `empty one-token replies retried N, positions skipped M` (D.2.10); read both before the top-1 number.
4. `qualification/qual_all.sh` from the release folder (about 1 h at c8). Needs `FLARE_LOG`, `CKPT`, writes to
   `QUAL_OUT`. LiveBench needs `--livebench-release-option 2024-11-25`. Expected: eval 43/43, tools 60/60, lavd/hotel
   30/30, retrieval pass, longgen 4/4, LiveBench ~82, c8 stress PASS.
5. Reuse (user rule 2026-11-11): a full qual may skip any stage that already PASSED on the same build (package/image
   id + binary) and the same serve settings. Rerun a stage if anything changed or a related check failed. The internal qual
   report lists each reused stage and the result it came from; public docs just report the score (no reuse note needed),
   as long as that exact build + settings produced it.

### A.5 Export

```bash
cd ~/flare-up-v3 && CKPT=/m2-2/flare-ckpt/ds41-1017-engram-mxfp4 HOT=342 tp3-tools/release/export.sh   # the recipes folder ships the same script
```

Full export ≈ 25-40 min with a warm tuning cache; writes a new `gb202-tp3-<timestamp>` package. Knobs are env vars
(`HOT`, `PREFILL`, `DRAFTS`, `MAXC`, `FLARE_DS41_VISION_HOST`, plus the `FLARE_DS41_*` set in `export.sh`). `MAXC`
sets `--max-concurrency` (default 8; `MAXC=16` = the c16 package, warm its tuning cache first, D.2.2). Graph scratch
is on by default (`FLARE_DS41_ENGRAM_CHUNK_ROWS=1024`, `FLARE_DS41_SHARE_ENGRAM_RESIDUAL=1`; set both to 0 for the
previous layout). The recipe reads
them once, in `TP3Config.from_environment()` (`ignition/recipes/deepseek41/tp3_config.py`), validates them, prints
the non-default switches in the export progress and records every switch in `tp3-config.json` beside the package's
`model.safetensors` (and in each rank's memory report as `tp3_config`; paths appear as basename + fingerprint,
never absolute). Everything downstream receives the `TP3Config` instance; new switches are fields there, not new
`os.environ` reads. `FLARE_DS41_VISION_STAGE=1` implies `FLARE_DS41_VISION_HOST=1` (staging reads from the parked
copy; `VISION_HOST=0` alone keeps the tower in VRAM only when staging is off too). Export and serving cannot overlap
(VRAM). Mini export for kernel/wiring checks: `SNAP=<truncated view> LAYERS=<brrr layers> tp3-tools/mini-export.sh`
(export, serve, `tp3-tools/consistency.py`, stop; ~10 min; refuses to start next to a running server).

On the port (upstream memory scheme, D.2.18) read these export log lines after every full export: `before tuning:
... MiB free` (rank 0; 3,083 MiB at HOT 342 on 11-14), `Shared declaration examples: ...` (expect 0 MiB superseded),
`Graph pool requests (...)` (per-graph pools, largest first; the largest sets the capture pool), `[brrr] placement
rank <r>: ...` (one per rank; 11-14: `in_place=200` on all three) and `Routed-expert selections: ...` (which graphs
chose micro/tiny_decode MoE). Before any export from a port venv, check the native module is current (D.2.17).

### A.6 GPU tests and traces

```bash
# BRRR layer tail vs stock path, 3 GPUs (all widths 1..48, resident+parked, per-chunk on/off, negative control): the 5 base
# cases ~15 min (-k "perchunk or break-rope and not item6"); the item-6 switch cases (-k "item6 or split or fastroute or parity") add ~45-55 min
# CUDA 13.1 must be first on PATH: /usr/bin/nvcc is 12.4 and fails with "Unsupported gpu architecture 'compute_120a'"
export CUDA_HOME=/usr/local/cuda-13.1 PATH=/usr/local/cuda-13.1/bin:$PATH
cd ~/flare-up-v3 && CUDA_VISIBLE_DEVICES=0,1,2 FLARE_RUN_CUDA_TESTS=1 FLARE_DS41_SNAPSHOT=/m2-2/flare-ckpt/ds41-mini8 \
    .venv/bin/pytest -x -s tests/recipes/test_deepseek41_brrr_tail_cuda.py     # in a worktree, also TORCH_EXTENSIONS_DIR=$PWD/.brrr-ext
# expert parking (resident / split / merged bank) equivalence + byte accounting, 1 GPU
CUDA_VISIBLE_DEVICES=0 FLARE_RUN_CUDA_TESTS=1 FLARE_DS41_SNAPSHOT=/m2-2/flare-ckpt/ds41-mini8 \
    .venv/bin/pytest -x -s tests/recipes/test_deepseek41_expert_parking_cuda.py
# kernel micro-tests
.venv/bin/python tp3-tools/test_fused_qkv_norm.py ; .venv/bin/python tp3-tools/test_flare_sparse_decode.py
# GPU tests that session.tune() then compile_forward() a planned Triton kernel (GLM indexer/MLA, DS41 indexer)
# need the kernel caches pinned as `ignition export` pins them, or Inductor's first kernel repoints
# TRITON_CACHE_DIR and the deferred b12x load fails with "planned Triton program has not been compiled":
export TRITON_CACHE_DIR=~/.cache/flare/kernels/triton B12X_COMPILE_CACHE_DIR=~/.cache/flare/kernels/b12x \
       TORCHINDUCTOR_CACHE_DIR=~/.cache/flare/kernels/inductor      # tp3-tools/glm53-gpu-validate.sh does this
```

Traces: `nsys profile` on the serving process at c1 and c8 (existing captures `serve-node-v3-c1.nsys-rep`,
`serve-node-v3-c8.nsys-rep` in the fork root); summarise per-stream busy intervals, not just kernel sums (D.1.5).

### A.7 Repositories and where to push

| Repository | Where | Purpose and rules |
| --- | --- | --- |
| Upstream Flare | `origin` = https://github.com/NVIDIA/flare-local-inference (read-only for us) | source of truth; our branch `ds41-tp3-v3` sits on top of it (merge-base 972e28e4 as of 2026-11-08; upstream main has moved on, see B.5) |
| Local Gitea (private view) | remote `gitea` on the LAN (address, login and password location: agent memory note `gitea-local`); repos `flare-tp3-fork` and `rtx-recipes` | browse commits, diffs and Markdown; push every commit here (`git push gitea <branch>`); experiment branches welcome (`stage2-wide-rows`, `vision-staging`, `config-object`, `upstream-main`); container `gitea` |
| Public release (GitHub) | https://github.com/AzallazA/RTX-PRO-6000-TP3-Inference-Recipes (`origin` of `~/rtx-recipes`); private staging copy `RTX-PRO-6000-TP3-Inference-Recipes-staging` (`staging` remote, branch `proposals` holds the proposal drafts) | user-facing release folders (docs, export/serve scripts, source.patch, qualification results); push only with the user's explicit OK, after the hygiene grep (D.4.1) |
| Public fork (GitHub) | remote `public` = https://github.com/AzallazA/spark-local-inference (fork of NVIDIA/spark-local-inference, the renamed flare-local-inference; default branch `ds41-tp3-v3`) | reviewed main line and release tags only (`flare-tp3-YYYYMMDD`, each with a GitHub Release note); push with the user's explicit OK; upstream PRs for `docs/upstream-proposals` open from it; GitHub cannot make a fork private, hence Gitea for the private view |

The fork tree is kept publication-clean at all times: no experiment scripts at the root (they live outside the repo, e.g.
`~/ds41-tp3/fork-archive` and the untracked `serve-best.sh`), no `/home/<user>` paths, no hostnames. `git grep -n
"/home/\|$(hostname)" HEAD` must stay empty before any push.

### A.8 Knob glossary (what each lever does and what we know)

| Knob | Effect / verdict |
| --- | --- |
| `HOT` (hot experts per parked layer, layers 20-33) | 342 = 2.37M KV default; each expert fewer ≈ +92K KV tokens; 300 = 2.99M KV release build, ~4% slower. On the upstream port (11-14) HOT 342 serves 5,024,384 KV (1241 on tp3-main: 3,189,376) and HOT 300 8,708,992 |
| `BUDGET` (`--mtp-max-verification-token-budget`) | 32 since 11-15 (owner, for 16 slots): on pkg 152105 c16 1,375 vs 1,226 tok/s at 24, c12 1,234 vs 1,188, c1/c4/c8 level (315/749/1,024 vs 309/757/1,045), served KV 5,220,992 vs 5,257,344 (`ds41-exp-1115-1901` BUD rows). Before: 24 adopted for 8 slots (+1-2% at c1/c8 vs 128; drops 13 speculative graphs, ~76 MiB/rank = +87K KV tokens) |
| `PCM` (`--prefix-cache-memory`) | 12G since the FP8 Engram tables (11-16): 32G does not fit beside them in 251 GiB (dry run 237.3 GiB committed, predicted 0.4 GiB available). Before: 32G (~78 warm 127K prompts vs ~37 at 16G; ~10 MiB device headroom, -6K KV, host RAM otherwise) |
| `K` drafts | keep 5; coding acceptance is high; k7 was -3-6% at c4/c8 |
| `ALPHA` 0.6/0.8/1.0, scheduler balanced/throughput | within noise; defaults 0.8 / balanced |
| Power 350/450/600 W, core offset | 450 W ≈ +10-12% over 350 W; 600 W no gain; +100/+150 core no gain |
| `FLARE_DS41_BRRR_PARKED_MAX_ROWS` (`PARKED_ROWS` in `export.sh`, default 12) | gate for BRRR in parked layers; ungated = c8 +1%, c1 level (tabled); 24 on the c16 package = c16 +3.4%, c1-c8 level (11-09); c8 package: release-chain A/B |
| `--scheduler-policy interactive` (TTFT guard: balanced + `--scheduler-ttft-target-ms 50`) | adopted 11-10: probe TTFT p50 0.36-0.38 s vs 2.0-4.0 s at c4/c8, agent c4 turns 519 vs 448, decode level; `SCHED=balanced` reverts |
| `FLARE_DS41_BRRR_SPARSE`, `FLARE_DS41_FUSED_QKV_NORM` | opt-in experiments, measured level; default off, kept with their kernel tests (`TP3Config.brrr_sparse` / `fused_qkv_norm`) |
| BRRR prologue, `FLARE_DS41_QB_ATOMIC` | measured level; REMOVED 2026-11-08 (`TP3Config.from_environment` rejects the variables) |
| BRRR stage 2 (OTOK/SPAN 48 wide rows, worktree `~/flare-up-stage2`) | REGRESSION: c4 -5%, c1 -3% (11-08, real export); closed |
| Extra PDL trigger at the end of `layer_kernel` | cost ~2% at c8 (11-08 bisect); removed. Only the original trigger remains |
| `FLARE_DS41_VISION_STAGE=1` (vision tower weights staged through VRAM slots; implies `VISION_HOST=1`) | cold image 0.11 s vs 1.70 s (11-08); 190 MiB of activation-pool slots, KV unchanged |
| Engram: `FLARE_ENGRAM_RAM_TABLES=0` vs `--embedding-table-storage disk` | MXFP4 era: disk = -90 GiB RAM, -3-4% decode; MXFP4 vs FP8 tables: wiki ppl +2.6%, benchmarks within noise |
| FP8 Engram tables, both in host RAM (view `ds41-1017-engram-fp8`, default since 11-16 00:55) | DeepSeek's own shards 47-48 (189 GiB) instead of NVIDIA's MXFP4 (97 GiB). Host RAM committed (dry run) 217.3 GiB at PCM 12G vs 145.7 GiB for MXFP4 pkg 152105; lowest MemAvailable 21.0 GiB at boot (seq202 arm b), 10.7 GiB in the 16-stream stress (seq214): 256 GB is the minimum. vs 152105 at budget 32: c1 0.989, c4 1.019, c8 1.027, c16 0.989 tok/s, served KV -1.8% (seq202 `ds41-fp8-1115-2224`). Layer 1 in RAM + layer 14 on disk with a 32 GiB row cache (`FLARE_ENGRAM_RAM_LAYERS=0`, release `serve.sh` `ENGRAM_RAM=layer1`): 177.8 vs 237.3 GiB committed at PCM 32G, decode c1-c16 level, agent8 output tok/s x0.821. Both on disk: not measured with FP8 |
| `FLARE_DS41_EMBED_HOST=1`, `FLARE_DS41_VISION_HOST=1` | embeddings/vision in host RAM; images ~3 s slower each |
| `FLARE_DS41_AR_ROWS=1024` | chunked all-reduce sizing for prefill |
| `FLARE_DS41_ENGRAM_CHUNK_ROWS=1024`, `FLARE_DS41_SHARE_ENGRAM_RESIDUAL=1` | graph scratch, default on in `export.sh`: Engram projection/reduce/mix in 1024-row chunks (-300 MiB/rank) and `engram_residual` on `ffn_hc.residual` (-165 MiB/rank); KV 2.46M -> 3.01M at HOT 342, decode level. On the port there is no graph scratch pool: chunk rows act on the functional forwards (planned prefill4096 pool 560 MiB unchunked vs 428 MiB chunked) and the shared residual is only a declaration alias (the planner already reuses it) |
| `MAXC` (`--max-concurrency` at export) | 16 in the release `export.sh` since 20261116 (DS41 default since 11-15 16:25, pkg 152105 and later; with the VRAM reclaim switches the 16-slot package had more KV than the 8-slot 151744). Before: 8; 16 then cost ~650K KV (dry run, pre-port) |
| `FLARE_CUDA_STACK_LIMIT` / `flare run --cuda-stack-limit` (bytes) | 256 adopted 11-16: +247,552 KV tokens on 160815 (5,155,456 -> 5,403,008, seq214 `ds41-integration-1116-0614`). Packages replay the exporting context's stack limit (driver default 1,024 B); the load refuses an override below the largest stack of any registered kernel (DS41: 192 B, the BRRR layer kernel) or when one cannot be bounded. The dry run does not model it. Unset = recorded limit |
| `--scheduler-ttft-order oldest\|shortest` | which overdue request the interactive guard serves: `oldest` (default, the 20261111 behaviour) or `shortest` (fewest remaining prompt chunks; the oldest keeps at least every second guarded round). Adopted `shortest` 11-16 with the c8fb6185 binary. Simulation (8 agents, 0.8K-14K new tokens per turn): mean TTFT 2,312 -> 2,142 ms, p50 2,249 -> 1,772, p90 3,883 -> 4,118, max 5,210 -> 5,825, turns 147 -> 145 (commit 1e22ffc3) |
| `FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS` / `--scheduler-prefill-merge-min-rows` | a chunk starting at the request's own checkpoint runs past message boundaries closer than N tokens (0 = off, code default). 512 adopted by the owner 11-16 14:45 although the A/B rule said HOLD: 8-stream standard replay turns x1.086 against an 11.8% same-flag band (the 8-stream WNT fit came out negative); 16 streams WNT x1.157 (band 4.6%); TTFT p50 x0.75-0.83; prefill launches per turn 3.15 -> 1.00; decode c1 x1.023, c8 x1.011 (seq227 `ds41-pmerge-c8fb-1116-1252`, AB.md and AB-recomputed.md) |
| `tool_choice` required / named function with thinking on | HTTP 400 "Thinking mode does not support this tool_choice" since c8fb6185 (was: quietly automatic, then 500 when the model answered without a call), as DeepSeek's reference. tool-eval TC-45 probes in thinking mode, so it stays excluded (91 graded) |
| `FLARE_EXPORT_LIBRARY_SCRATCH` (export KV hygiene, 4b4bb69f) | on by default in the code (0 = off): cuBLAS's own 64 MiB pool no longer retained by a coincidental 8-byte value. Package 160815 predates it. On 161317 (same switches + Engram side stream 14): dry and served KV unchanged (4,996,096 / 5,155,456), one 44 MiB unclaimed span per rank still reported, so the chain said HOLD (seq217 `ds41-exp-fp8kv-1116-0655`) |

| BRRR item 6: `FLARE_DS41_BRRR_SPLIT_TAIL`, `_FAST_ROUTE`, `_COUNTER_PARITY` | default off, not adopted (11-09): split tail and counter parity slower in the layer A/B; fast route -0.5/-1.9/-2.0 µs/layer at m 1/6/12 but end to end level (c1 1.004x, c4 0.989x, c8 0.998x) |

**Round-2 and 11-15 export switches: what the release export sets.** All are default off in the code (`TP3Config`,
B12X environment); `tp3-tools/release/export.sh` turns on the ones marked ON (package 160815 = these values,
`tp3-config.json`). Measured as one combined export (seq209 `ds41-exp-fp8-1116-0241`, 160815 vs 160407 = the same FP8 package
without the round-2 switches, budget 32): c1 x1.076, c4 x1.011, c8 x0.992, prefill 8K x1.046, 32K x1.042; dry KV +0.6%.

| Switch | Release export | Note |
| --- | --- | --- |
| `FLARE_DS41_INDEX_SIDE_STREAM` | ON | indexer query chain on a side stream (bit-identical, GPU proof in the chain) |
| `FLARE_DS41_CONTEXT_SIDE_STREAM` | ON | DSpark context projection beside verification |
| `FLARE_DS41_AR_PART_ROWS`, `FLARE_DS41_ENGRAM_PART_ROWS` | 768, 768 | prefill all-reduce parts: 5 x 768 + 256 per 4096-row chunk |
| `B12X_MHC_PDL` | ON | PDL in the mHC norm kernels (20 prefill mHC races re-run) |
| `FLARE_DS41_BRRR_WOA_UNEVEN` / `_WOA_SMALL` | 1 / 0 | BRRR job layout at <= 8 rows (own extension build) |
| `FLARE_DS41_EMBED_ROWS` | ON | host-row embedding gather (11-15, pkg 151744) |
| `FLARE_DS41_VRAM_RECLAIM` | ON | four reclaim items (11-15, pkg 152105) |
| `B12X_PCIE_TWOSHOT_PULL` (overlay) | ON | needs `B12X_OVERLAY` with `tp3-patches/overlay/flashinfer-pcie-twoshot-pull-world3.patch` |
| `B12X_MLA_DECODE_REMAINDER_STREAM` | ON | attention 16 + 8-head grids on two streams (ignition hook) |
| `PARKED_ROWS` (`FLARE_DS41_BRRR_PARKED_MAX_ROWS`) | 0 | BRRR at every width in the parked layers |
| `MAXC` | 16 | request slots |
| `FLARE_DS41_ENGRAM_SIDE_LAYERS` | off | `14` measured level on 161317 (seq217), not adopted |
| `FLARE_DS41_BRRR_SPIN_NS`, `_TWO_SHOT_MIN`, `_KS_WOB`, `_PER_SM` | off (kernel defaults) | build-time tuning switches, not adopted |
| `FLARE_DS41_PACKED_KEEP_BATCHES` / `_WIDE_GRAPHS` / `_WIDE_TOKENS` | off | packed verification shapes above 8 slots, not adopted |

Ideas already closed (do not re-run without a new angle): FP8 LM head, fused layer L2 prefetch, EP all-reduce over
b12x, CPU-side expert compute, DCP3, dynamic expert cache (0.40% cold routes at H342), route reuse, sparse decode at
FP8, k7 drafts, 8192 prefill chunks, BRRR wide-row restreaming (stage 2), extra PDL trigger in the layer kernel, BRRR item 6
(split tail, fast route, counter parity). Details in `~/.claude/.../memory/` notes and MEASUREMENTS.md.

---

## B. Workflows

### B.1 Speed experiment (settings or build)

1. Write down the hypothesis and the expected gain in tok/s before touching anything.
2. Note the current confounders: power limit, offsets, BUDGET, PCM, K, package id (`ps` line in A.2).
3. Settings change: restart the server with the override. Build change: full export (mini first only for risky
   kernel/wiring edits), then serve the new package with the default's flags.
4. Run alternating A/B in one session: default, candidate, default, candidate, default, candidate (3 each, same
   bench command, same duration). Record tok/s per concurrency and coding.
5. Decide only on the 3-run means. A difference under ~3% at c1/c8 is "level"; stop there and do not look further.
6. If adopted: `verify_fix` (0 TRUE corruptions), then the relevant qualification stages, then update
   `serve-best.sh` comments, MEASUREMENTS.md and the memory note. If not adopted: record the verdict in the experiment's
   docstring or MEASUREMENTS.md so it is not repeated.
7. Restore the default server (`/tmp/restart-default.sh`).

### B.2 Kernel or recipe change

1. Work in a worktree (`git worktree add ~/flare-up-<name> -b <name>`), symlink `.venv`; keep a backup of the file you
   replace (`decoder.cu.pre-stage1` pattern) until the change is committed. Every export or script from the worktree
   needs `PYTHONPATH=<worktree>` (D.2.11) and its own `TORCH_EXTENSIONS_DIR=<worktree>/.brrr-ext`.
2. Compile standalone first (the BRRR extension builds via `torch.utils.cpp_extension`; `CUDA_HOME=/usr/local/cuda-13.1`).
3. Run the layer test (A.6) with the negative control; a per-row mismatch is usually the reference, see D.3.11.
4. Mini export + `tp3-tools/consistency.py` against the default's number.
5. Full export, `verify_fix`, A/B per B.1. Only then commit; then regenerate `source.patch` and the release docs if
   the default changed (B.3).
6. Anything that touched `.venv` site-packages must have a patch file in `tp3-patches/` before the session ends.

### B.3 Release folder update

1. Scripts and guides are edited in the fork (`tp3-tools/release/`, `docs/tp3/`) and copied into the dated folder
   `~/rtx-recipes/DeepSeek-V4.1-Flash Flare-tp3-YYYYMMDD/`; per-release docs (README, MEASUREMENTS, results) are edited
   in the folder. Regenerate `source.patch` from the fork
   (`git diff <upstream base>..HEAD` plus untracked new files) and verify it in a temp worktree of clean upstream:
   apply, then compare `git ls-files -s | md5sum` with the fork.
2. Hygiene pass: `grep -rn "/home/\|~/\|$(hostname)\|gmail" .`; `find . -name __pycache__`; scrub `hostname`/`uname`
   fields from result JSON. Scripts must write to `QUAL_OUT`/run dirs.
3. Every number quoted in README/EXPORT/MEASUREMENTS must point at a log or `results/` file with its power, cap and
   PCM. Package/model name in docs = `ds41-1017-engram-mxfp4`.
4. Commit with the identity configured in the checkout. Push only with the maintainer's explicit OK (each public push).

### B.4 Code review of the fork

1. Give fresh reviewers the diff and the files, not your conclusions. Two passes: first "bugs and wrong
   implementations" (including unadopted opt-in paths), then "what could be implemented better".
2. Verify every finding against the code before fixing; discard the ones that do not reproduce.
3. Fix, rebuild (`cargo build --release` for Helios, re-export for recipe/kernel changes), re-run `verify_fix`.
4. Record verdicts (fixed / no change needed / tabled) in the commit message.

### B.5 Publishing a release

1. Validate the state (B.1/B.2), merge every branch the release serves into `tp3-main` (since the 2026-11-14 port the
   release branch; `ds41-tp3-v3` is history), push to Gitea. *2026-11-16:* the serving binary came from
   `perf/sched-prefill-merge-c8fb6185`, not yet in `tp3-main` c0ff27c5; the tag must contain it.
2. Tag it `hyperflare-tp3-YYYYMMDD` (from the 2026-11 Hyperflare release on; the published `flare-tp3-*` tags stay) (annotated: settings and measurements in the message), push branch and tag to `public`,
   and write the GitHub Release note (what changed, measured numbers, what was not adopted, link to the recipes folder).
3. The recipes folder is dated by the day the change was made (release convention): a new
   `DeepSeek-V4.1-Flash Hyperflare-tp3-YYYYMMDD` folder per release day (earlier folders keep `Flare-tp3`), copied from the previous one and edited; earlier
   folders stay as published. Its README describes the changes **since the previous folder** (not since the first
   one): intro list, "Coming from <previous>" migration notes, headline table labelled with the builds it holds; the
   cumulative feature description stays as "What this build contains". *Incident:* the 20261108 folder shipped for an
   hour with the 20261107 intro ("supersedes 20261105, +6-8%"). Regenerate its `source.patch` from the tag (`git diff --binary 70069281 <tag>` since the port; `972e28e44` up to 20261111), verify
   on a clean upstream worktree (tree-identical), cite the tag in the folder docs, add the row to the top-level
   README, run the hygiene grep, commit, push `origin` (GitHub) and `gitea`. The folder and the tag must always agree.
4. **Public branch after the port (open, owner decision, 2026-11-16).** The public default branch `ds41-tp3-v3`
   (618d3ee7 = tag `hyperflare-tp3-20261111`) is not an ancestor of `tp3-main`: the port rebuilt the history on
   upstream 70069281. Publishing needs either a force-push of `ds41-tp3-v3` (history rewrite: explicit OK and a
   force-push warning, D.4.6) or a new public branch (for example `tp3-main`) made the default. Published tags stay.
   Check first that no upstream PR is open from a public branch that would be rewritten.
5. **Qualification policy for a release (changed 2026-11-16 by the owner: "minimum, to release faster").** A release is
   qualified by tier, by what changed since the last fully qualified package (binary, export, weights, full), with the
   GPU queue chain `qual-chain.sh` (`~/ds41-tp3/analysis/work/flare-ds41-perf/qual-20261116/README.md`). The
   20261116 release ran `TIER=weights` (FP8 Engram tables = a quantisation change) plus `RETRIEVAL=1 LONG512=1`:
   smoke, image, tools 60, consistency, verify_fix, eval-43, tool-eval, decode 32K c1/c8/c16, 128K/512K c1, prefill 32K,
   agent8 replay, retrieval 128K/900K, GPQA Diamond, hotel-lights and KV stress at 16 streams. Not run: LiveBench,
   long generation, lavd, the natural-EOS decode bench, the 8-stream stress (the 16-stream stress replaces it), the
   128K c8 cell, the vision sets. The release notes and the folder README list what was not run.

### B.6 Sync from upstream (deliberate, never automatic)

1. `git fetch origin` and read `git log --oneline <merge-base>..origin/main` for anything that touches the DeepSeek
   recipe, Helios memory/Engram/package loading, b12x pins or the CLI. Note it in the status report.
2. While the public fork has no open upstream PRs, rebase: `git rebase origin/main ds41-tp3-v3` in a worktree,
   resolve conflicts, keep commit messages. After PRs are open from the public fork, merge instead
   (`git merge origin/main`) so published hashes stay stable.
3. Re-apply `tp3-patches/` to the venv if `uv.lock` or the FlashInfer pin changed (`uv sync --locked --extra prepare
   --extra test`, then the patches; check offsets/fuzz with `patch --dry-run` and regenerate a patch that only applies
   with an offset). Since the 11-11 sync there is no `ignition.patches` step: upstream's `ignition.runtime_patches`
   applies the torch/Triton fixes in memory. A FlashInfer bump gets its own worktree venv (never `uv sync` the serving
   venv), and a new MoE `TUNING_WORKLOAD_VERSION` or CUTLASS DSL version means a cold tuning cache (re-tune).
   Upstream renames: "package" became "core" in code (`runtime_core`, `CoreStorage`, `model_core`); the cache keeps
   `~/.cache/flare/packages` while no `cores/` directory exists; the `flare` binary is `flare-cli/target/release/flare`
   (`cargo build --locked --release --manifest-path flare-cli/Cargo.toml`), so `ps` patterns on
   `./flare/target/release/flare` miss it.
4. Validate before trusting it: `cargo build --release`, the 3-GPU layer test, a full export, `verify_fix` (0 true
   corruptions), one alternating A/B round vs the current default. Then commit, push to Gitea, and only then update
   the release folder (`source.patch` is regenerated from the new base).
5. **Comparing a branch with upstream: fetch first, and look for reverts, not only subjects.** Remote refs in a
   worktree are only as fresh as the last fetch of that repository, and "commit X is in" is true only if no later
   commit reverts it: check `git log --oneline <base>..<branch> | grep -i revert` and `git merge-base --is-ancestor`
   against a freshly fetched ref. *Incident (11-14):* the `upstream-sync/main` ref had not been fetched and still
   pointed at the v0.1.0 commit, so the port looked based on the tag when it is 20 commits later (the fork checkout's
   `origin/main` was two commits behind the port base 70069281 as well), and the first upstream check said the port carried #285 and 617ef929 because their subjects
   matched; the port reverts both (dc2fa0de, 7c4a0ba0) and the G2 preflight checks the binary has no 617ef929 flag.
   Corrected on the status board 6 minutes later.

### B.7 Upstream watch (what NVIDIA's branches hold for us, 2026-11-08)

- `astra/ds41-brrr-poc` (10-28, research-only): NVIDIA's own fused layer tail, same stage range as ours, kernel in
  FlashInfer's experimental tree, TP4 all-resident with native scales, selected per layer by a race against the stock
  path after FP64 per-operator bound checks and a held-out input; the stock path won at the one 8-row graph they tried,
  the whole-model export never finished, no tok/s. Our port is further along (BRRR-40 c1 +6.5%, c4 +2.2%, gated parked
  layers at c8). Borrowed: FP64 bounds + held-out check for our layer test; native-replay fixtures are a candidate. A
  contribution of ours would have to live in that shape: kernel in FlashInfer, a Flare component with a CLI flag in the
  signed rebuild manifest, and a per-layer race it must win.
- `fix/vocabulary-parallel-top-p`, `astra/startup-weight-storage-upstream-20261104`: already in our base in merged
  form (the explicit top_p range our bootstrap adds to the server config is now redundant; proposal 03 can drop it).
- `fix/checkpoint-gds-capability-probe`: forces the bounce read mode this host already uses; arrives with the next sync.
- Everything else (GLM, Windows, RoCE, multi-host, scheduler) does not touch our path.

### B.8 Switching the local default (procedure used 11-14)

1. Get the user's decision; record it on the status board with the time from `date` (C.34).
2. Back up `serve-best.sh` with a dated name (`serve-best.sh.bak-1114-1241default`).
3. Change only the defaults: `P=${P:-<new package>}` and `cd ${FLARE_DIR:-<worktree with the binary>}`, and add the
   rollback line as a comment.
4. Restart through `/tmp/restart-default.sh` (it calls `serve-best.sh`; wait for `/v1/models`, C.19).
5. Verify on the live server: package name in the `ps` line (A.2), the binary's sha256 against the one that passed
   the gates, served KV (`flare_total_kv_tokens`), and a test reply.
6. Update A.1 (packages, server scripts) and the memory note.
7. Afterwards: every chain's "restore default" restores the NEW default, and chain preflights that pin the old
   default (for example `g2-ds41-full.sh` checks `serve-best.sh` defaults to 1241) refuse until updated. If the
   default now runs from a worktree, a Rust rebuild there changes what the next restart serves.

*11-14:* package 142139 on the `~/flare-up-port-int` binary (sha256 67c691d29041...), live at 19:07 with served KV
5,024,384 and a correct test reply; `tp3-main` and GLM untouched.

---

## C. Tips for agents

1. **Check the GPUs before using them.** `nvidia-smi --query-gpu=memory.used --format=csv` ; if the server is up you
   have ~600 MiB free and nothing CUDA will start. Ask or coordinate before stopping the server; other agents may be
   benchmarking.
2. **Long jobs run as tracked background jobs** (`run_in_background`), each with its own log file; never a foreground
   command that can exceed the timeout. Chain scripts need `chmod +x` and `set -euo pipefail`.
3. **Save work incrementally.** Agents get killed by rate limits and context resets. Commit-ready files, patch files
   and result JSON on disk survive; a plan in your head does not. Write results to a file as soon as a measurement
   ends.
4. **Do not spawn a duplicate agent or chain** for a task that is already running; send the existing agent a message,
   and before launching a chain script list the running ones with `ps -eo pid,etime,args | grep '[c]hain-'` (never
   `pgrep -f`/`pkill -f` with a pattern that is in your own command line: that kills your shell, 11-08 again). Two
   chains driving the same compose project or server port corrupt each other's results. *Incident (11-08):* a second
   vLLM test chain was launched while the first was still in a restart loop; both brought the same container up and
   down and one of them lost both verify runs.
5. **Worktrees for parallel code work.** Multiple agents editing one checkout corrupt each other's diffs. One worktree
   per task; only `~/flare-up-v3` serves.
6. **Stop and look at the command line before killing processes**: `pgrep -f` matches your own shell. Use the `ps |
   awk` form in A.2.
7. **Read the memory notes first** (`MEMORY.md` in Claude Code's per-project memory folder for this working directory): they hold the
   closed ideas and the measured verdicts, so you do not rerun a dead end.
8. **Report numbers with their context**: package id, power, BUDGET, PCM, concurrency, context length, run count.
   "c1 +4%" alone is not a result.
9. **Docker (vLLM era, still used for comparisons):** one `docker compose` command at a time; prompt logprobs one
   request at a time; MMLU-style evals need 2,048 output tokens.
10. **Dynamo-traced code**: anything in the recipe's forward runs under torch.compile during export. No `data_ptr()`
    comparisons, dict writes or Python-side side effects there; use a custom op (runs at capture time) or a finalizer.
11. **Upstream rules** (AGENTS.md, CONTRIBUTING.md): no env-var-driven behaviour in Flare/Helios/components/recipes,
    no offload policy selected by model identity. Anything meant for a proposal must be a declaration, config value or
    CLI flag. The fork's `FLARE_DS41_*` switches are allowed only because they stay in the recipe + `export.sh`.
12. **Be honest about verdicts.** "Level" is a valid and common result; most fusions here measured level because the
    kernels overlap on side streams. Do not stretch a 1-run +4% into an adoption.
13. **Size KV with the dry run, not a load.** `flare run <pkg> <serve flags> --dry-run` prints each rank's memory plan,
    the KV tokens and snapshot slots in seconds, without a CUDA context, while the server keeps serving. Use it before
    a sweep, when choosing serve flags, and before crediting a build with a KV change (D.1 rule 4). Accuracy:
    exact on the five DeepSeek-V4.1-Flash TP3 loads it was calibrated on, but 0.2 % low on the later packages 081024 and
    090700 (2,461,184 vs 2,467,200 and 3,007,360 vs 3,014,656 served: quote the served `Allocate cache` count); 0.4 % low (TP3) and 1.3 % low (TP2) on
    GLM-5.3-Flash, which was not in the calibration, so read it as a slightly conservative estimate for other models.
    On the upstream port (11-14) it is further off: 4,754,944 vs 5,024,384 served at HOT 342 (5.4 % low) and
    8,419,328 vs 8,708,992 at HOT 300 (3.3 % low); quote the served number for port packages.
    `--dry-run-free-memory` replaces the assumed free memory (driver total minus 606 MiB) when a GPU is shared.
    Dry runs are cheap, so answer KV questions with them before an export: c16 graphs cost ~650K KV against c8, and
    a serve-time verification budget below the export's drops the unused verify graphs.
14. **Hide the GPUs from CPU test runs** while anything serves or benchmarks: prefix `CUDA_VISIBLE_DEVICES=`. Some
    `*_cuda` tests run whenever a GPU is visible, not only with `FLARE_RUN_CUDA_TESTS=1`. *Incident (11-08):* a CPU
    test pass put a CUDA test onto the GPUs for a few seconds in the middle of a benchmark sweep.
15. **Never edit a chain script that is running or waiting;** bash reads scripts as it goes. Kill and relaunch it, or
    write a new file and `mv` it over the old path (new inode) for a script that has not started yet. Gate each
    follow-on chain on a success marker, not a "done" marker. *Incident (11-08):* a comparison chain waited for
    "export done", which the failed TP3 export also wrote, and benchmarked TP2 alone.
16. **A new rank or head count meets kernel limits that tests on other shapes never hit.** Check the qualification
    lists before exporting: b12x's BF16 x MXFP8 batched GEMM takes 8, 16 or 32 heads only; the sparse-MLA kernel
    mis-tiles 22 heads; the shipped TP3 runtime collective ladder stops at 24 MiB (a 4096-row prefill at hidden 4096
    reduces 32 MiB). Check per-layer formats too: a draft layer can be stored differently from the target layers
    (GLM-5.3-Flash FP4-Local: layer 45's `kv_b_proj` is MXFP8, the target layers' are BF16). Each of these failed a
    TP3 export about 11 minutes in; a mini export or a format scan of the index finds them sooner.
17. **Speculation settings trade tokens per step against steps per second.** Read both before choosing. *Example
    (GLM-5.3-Flash TP3, 5 drafts):* adaptive verification off raised tokens per step from 2.23 to 2.72 and cut c1
    steps/s from 112 to 76, i.e. -16 % tok/s; alpha 0.6 reached 2.49 tokens per step at level tok/s.
18. **One Flare binary serves packages from other branches** when their Rust is unchanged: check
    `git diff --stat <base> HEAD -- flare helios heliopause` before building a second binary.
19. **Chain wait-conditions and failure watchers.** Never wait for the default-restart script to exit: it stays
    alive as the server's parent; wait for `/v1/models` instead. Failure watchers must pre-seed the log's existing
    lines (old failures re-trigger otherwise) and match failure words only, not words such as "skipped" that normal
    verdict lines contain.
20. **Profiling Flare** (one process, CUDA graphs, no NVTX ranges): start the server under `nsys launch` with
    collection off, then `nsys start` / `nsys stop` around a steady decode window; trace kernels only, with
    per-node CUDA graph tracing (`--cuda-graph-trace=node`), or the graph shows up as one opaque launch.
21. **Shell environment traps.** `~/.local/bin/env` (uv's sourced-only script) sits first on `PATH` and shadows
    `/usr/bin/env`, so `env VAR=x cmd` silently runs nothing: write `VAR=x cmd` or `/usr/bin/env VAR=x cmd`. The login
    shell also exports `IMG=<another vLLM image>`: a chain variable named `IMG` (`${IMG:-default}`) silently picks it up
    (2026-11-09: the vLLM A/B began pulling the wrong image). Give script variables specific names.
22. **One GPU queue, not parallel chains.** Queue GPU jobs through `/tmp/gpuq-runner.sh` (log `~/bench-results/gpuq.log`):
    an item runs only after `/tmp/gpuq/ready-<name>` exists, which is touched after its prep is reviewed; `done-<name>` is
    written after it ran; moving `ready-*` into `/tmp/gpuq/held/` pauses the queue. Edit the runner only while it is idle
    (kill it, edit, relaunch): bash reads a running script incrementally.
23. **Upstream reproductions need a pristine tree.** The fork venv carries `tp3-patches/` (until the 2026-11-14
    port also the tiny-decode clamp, now done by FlashInfer aa6c4221 itself), so a bug check on upstream needs its own clone and venv (`uv sync --locked --extra prepare --extra test`).
    For correctness-only exports use `--autotune-effort off --no-mtp` (minutes instead of hours; without MTP every
    decode step is a 1-row launch). DeepSeek-V4 (`deepseek4`) rejects `--language-model-only` (no separable vision
    encoder), and on main 8c2e782 its export needs `engram` dropped from its `LayerWeights` entry in
    `ignition/components/weights.py` (hyperflare#249 notes it).
24. **DeepSeek request options.** The DeepSeek frontends accept `chat_template_kwargs` `enable_thinking`/`thinking_mode`,
    not `thinking` (HTTP 400 "unsupported DeepSeek prompt option").
25. **Token-id prompts must not contain DS41's image placeholder 129264** (`<｜deepseek_image｜>`): the frontend counts
    placeholders against attached images and returns HTTP 400 "image placeholder count does not match input images"
    without a server log line. Prompts built by tokenizing source text can contain it (Helios `decode.rs` has the
    literal); 11-13 the ideas chain's 128K capture failed this way. Strip it, and print `e.read()` on HTTP errors.
    DeepSeek requests always run at temperature 1.0 / top_p 1.0: the recipe gives no range, so the server pins any
    requested value to the default (`helios/src/server/request.rs` `sampling_number`).
26. **`local x=$1 y="...$x..."` in one statement fails under `set -u`:** bash expands the whole line before assigning,
    so `$x` is still unset (11-13: the vLLM .12.1 test died right after a 4-minute boot with "arm: unbound variable").
    Write `local x=$1; local y=...`. A `--dry-run` that never calls the function does not catch it.
27. **Bit-identical gates across restarts do not hold for DS41 or GLM TP2 (11-13):** DS41 differs at token 0 between two
    boots of the same package (b12x MoE atomics; requested temperature 0 is also clamped to 1.0), and GLM TP2 130307 on
    its own original binary matched only 8/12 first tokens (5/12 exact, max |dlogprob| 2.14) against its own probe from
    the day before, while repeats on one server are 12/12. Compare against a same-session control (or a tolerance /
    the eval suites), never against a probe recorded on another boot. Also: exported KV varies run to run (+-0.5%, and
    once -1.9% from the exporter's conservative pointer scan keeping a stray 64 MiB block), so gate KV with ">=" a
    reference minus a tolerance and re-export on a miss, not "=".
28. **Rare startup failure "CUDA address collision: requested 0x..., reserved 0x..." (11-13, once in ~60 restarts):** a core's
    fixed VMM addresses were already partly mapped in the new process (address-space randomization). Fails at "Allocate
    model memory" in 0.0 s with nothing loaded; restart. Chains should retry a serve once on this exact message before
    counting it as a failure. Investigate if it repeats.
29. **Host memory for hot CPU-side tables: no transparent huge pages (11-14).** DS41 private Engram tables
    (`FLARE_ENGRAM_PRIVATE_TABLES=1`) on THP "always" came out ~60% huge pages (fragmented host) and cost 32K c4 -15%
    (640 vs 749 tok/s, dTLB load misses x2.6); c1/c8 looked level, so a c1-only or c8-only bench would have missed it.
    They now use `MADV_NOHUGEPAGE` (same fast load, 13 s vs 26 s). Bench c1, c4 and c8 for any host-memory change, and
    test THP effects with `prctl(PR_SET_THP_DISABLE)` (41) on a wrapper around the launcher (inherited across exec).
    serve-best.sh runs private tables + `FLARE_FAST_EXIT=1` by default (DS41 restart ~55 s vs ~80 s); off:
    `PRIVTABLES=0 FASTEXIT=0`.
30. **Chain dry runs take `--dry-run` as an argument, not `DRY=1` in the environment.** The chain scripts set
    `DRY` themselves from `$1` (`DRY=0; [ "${1:-}" = --dry-run ] && DRY=1`), so an exported `DRY=1` is overwritten
    and the chain runs for real, including its EXIT trap that restarts the default. Read a script's usage line before
    a dry run. *Incident (11-14 12:45):* testing the G1 chain's new preflight with `DRY=1` ran it for real; it refused
    in preflight (stale wheel, D.2.17), and its exit trap restarted the live DS41 server once (about 1 min down).
31. **Under `set -o pipefail`, never `cmd | grep -q pattern`.** `grep -q` exits at the first match, the writer gets
    SIGPIPE, and the pipeline returns non-zero although the pattern was found, so the test reads as "not found" at
    random (more often on long output). Grep the file directly (`grep -qa pattern file`; `-a` because logs carry
    `\r` progress and binary bytes) or a here-string (`grep -q pattern <<< "$var"`), as the 11-14 port chains'
    preflights do.
32. **GPU tests that need "raw replay" need the qualification build of the native module.** `flare_heliopause` built
    without the `qualification` Cargo feature raises "raw replay requires the native qualification Cargo feature"
    (or skips); that is a test-environment limit, not a product failure. Run those tests from a separate venv with
    the qualification build; do not rebuild the export/serving venv's wheel for them. Chains mark them optional
    (`G3_QUAL_WHEEL=1` runs them). *Incident (11-14):* the first G2 chain stopped before its export on
    `T-vocab_tp3` for this reason (10 min lost), and GLM `kda_tp`/`mlp_tp` failed the same way in G3. Upstream's own
    baseline has 21 `native_recipe_contract` failures from the same missing feature.
33. **A CUDA illegal memory access poisons the rest of its process.** Every later test in the same pytest process
    fails (first `torch.zeros`, fixture setup "illegal memory access"), so only the first failure carries
    information. Rerun suspect cases one per process, with `CUDA_LAUNCH_BLOCKING=1` to name the faulting kernel.
    *Incident (11-14):* in G3 one GLM reused-selection case faulted and took the other world sizes with it; run
    alone, one case per process, the crashing cases passed (cause still open; suspected cold-cache tuning).
34. **Status board times come from `date`, never from an estimate.** *Incident (11-14):* two entries (18:05,
    18:10) were written with guessed times and had to be corrected afterwards.

---

## D. Rules and the incidents behind them

### D.1 Measuring speed

1. **A/B in one session, alternating.** Default, candidate, default, candidate (3 runs each, same power, same serve
   flags, same host prefix cache). Never compare against numbers from another day, another power limit or another
   cap. *Incident:* the ungated parked BRRR build looked 3.6% slower at c1 against the default's earlier runs; the
   alternating A/B showed c1 identical (296 vs 296).
2. **Single runs decide nothing.** Run-to-run spread is 2-3% at c1 and c8 and more in coding. *Incident:* the
   route-reuse quick bench showed c1 +4.7%; three user runs showed level. The first settings sweep crowned alpha 1.0 /
   throughput on one lucky run because the ranking mixed single runs with multi-run averages.
3. **Judge by tok/s, but read steps/s and tokens/step next to it.** A row cap or BRRR gate shifts tokens per step,
   so steps/s alone overstates a change (*incident:* BRRR at c8 showed higher steps/s but lower tok/s). The converse
   also happens: tok/s moves while steps/s does not, which means draft acceptance (sampling at temperature 1) moved,
   not the build (*incident, 11-08:* the config-object package looked 1.5% slower at c8 over 4 rounds at equal or
   higher steps/s; tok/step 15.8-16.8 vs 16.3-17.7 explained all of it). Chain logs print both via
   `tp3-tools/bench_cell.py`; `tp3-tools/bench_steps.py` gives the full per-cell table.
4. **Know the confounders before reading a KV or speed number:** power and offsets; `BUDGET` 24 vs 128 (the cap
   drops 13 speculative graphs, ~76 MiB/rank: 2,373,760 KV at 128 vs 2,461,184 at 24 for the same package);
   `--prefix-cache-memory` (32G costs ~10 MiB device headroom); whether the bench respects EOS. *Incident:* three
   builds were credited with "+58K / +73K / +87K KV" that was entirely the serve flags; the first explanation ("the
   32G prefix cache") was itself wrong and only the KV dry run (`flare run ... --dry-run`) settled it. Run the dry
   run before attributing any KV change to a build.
5. **Know the two background variances before calling a delta real.** Export-to-export (identical code, tuning
   races): 0.5-1% at c8 (11-08: 080909 vs its re-export 081134, 983 vs 988 tok/s over 4 rounds). Session-to-session
   for the same package: ~2% (080909 ran 986-1003 one session, 979-991 the next). A delta inside these bands needs a
   re-export or another session before it means anything; one re-export of the baseline is the cheapest check.
6. **Bisect a regression before dropping a whole change set.** Stage 1 measured -2% at c8; one of its six parts (an
   added PDL trigger) was the entire cost, found with one extra export. The other five were kept.
7. **Trace estimates are upper bounds.** Attention-side kernels overlap on side streams; a kernel that looks like 6%
   of the step may be hidden. Measure per-stream overlap before building a fusion. *Incidents:* BRRR prologue
   (estimated -5%, measured 0), sparse decode (estimated 2-3%, level), fused qkv_a norm (2.6% on paper, level).
8. **`verify_fix` "natural-uncertainty flags" are noise.** One build gave 4, 8, 4, 27, 16, 17 across runs. Only
   `TRUE corruptions` counts; natural-EOS benches and the qualification suite decide quality.
9. **Ignore-EOS bench loops are a bench artifact**, not corruption. Re-check with `--respect-eos` (always 0 loops).
10. **Engine rounds/s is not a speed metric across builds; gate decode on tok/s plus per-request steps.** Flare's
    `verification_graphs_per_second` (the `NNN.Ns` field of `bench_cell.py`) counts engine rounds; a build that packs
    more requests into one round shows fewer rounds/s at equal per-request speed. Gate on `aggregate_tps` AND the
    bench's `server_steps_per_s` (draft_request_rounds per second, i.e. per-request steps summed over live requests;
    what llm-inference-bench reports); keep rounds/s as an informational column. *Incident (2026-11-12):* the DS41
    sync package 0438 showed c8 "steps" 0.94x while the user's llm-inference-bench showed identical per-request steps
    and tok/s. The DS41 sync qual (stage A), the GLM sync scripts and the GLM pick script were switched to the right
    metric. The standing decode gate (set 11-11) is per cell: steps/s >= 0.98x AND tok/s >= 0.96x. Chains must use
    exactly this rule. *Incident (11-14):* the G2 chain's own looser rule (fail only if both are below 0.97x) passed
    c1 at 0.974x steps / 0.947x tok/s, which the standing rule fails; the miss was found by recomputing from the raw
    files.
11. **Determinism probes must bypass the prefix cache.** Identical repeats are served from Flare's prefix cache, so
    "first token stable x8 within a run" is an artifact; only fresh computation tests determinism (decode repeats are
    what exposed the real non-determinism). Flare's OpenAI server has no per-request cache bypass (no `cache_salt`),
    so give every repetition its own short nonce prefix (the same nonce for repetition i in every session), check
    `usage.prompt_tokens_details.cached_tokens` is 0, and compare repetition i across a restart. Judge a build as
    "lost determinism" only against a base measured the same way (fresh, before and after its own restart): FAIL
    where the candidate is unstable and the base stable, WARN when both are unstable. *Incident (2026-11-12):* the
    11-11 sync runs failed "not bit-identical across a restart where 1227 is", but 1227 had been probed once, from the
    cache.
12. **Before blaming a binary or package for a decode delta, compare GPU time per step at equal tokens per step.**
    tok/s alone swings with draft acceptance: 1241 by itself ranged 2.42-2.80 generated tokens per step over four
    c1 rounds on 11-14. Take `verification_gpu_nanoseconds / verification_graphs` from the bench counters, fit it
    against verified tokens per graph for the reference, and read the candidate's residual. *Incidents (11-14):* G0
    suggested the new port binary cost ~2% c1 steps; at equal verified tokens per step it was within 0.2 %, the old
    binary had simply verified fewer tokens per step in those rounds. The port package's real c1 cost (+0.21/+0.25 ms
    GPU per step, all inside graph replay, host side 0.015 ms on both) survived that check. Lead for the later speed
    work (not fixed): decode graphs on the new scheme carry several hundred zero-fill kernels (429-467 per compiled
    decode forward vs 0 in the 1241-era code; speculative1x1 1,531 graph nodes vs 885), from functional components
    allocating custom-op outputs with `torch.zeros`. Details and next tests: `~/ds41-tp3/analysis/work/port1114/G2-C1-ANALYSIS.md`.

### D.2 Exporting and serving

1. **Mini export first only for risky kernel or wiring changes**; settings and placement changes go straight to a
   full export (working policy). A mini run catches compile/trace/wiring failures, not speed.
2. **Warm the tuning cache before an H342 export.** A cold tuning cache runs out of VRAM at HOT=342; export HOT=300
   first. Needed again after every tuning version or draft-count change, and for a new `MAXC`: the c16 package's
   new verify shapes ran out of memory in tuning at HOT=342 until a HOT=300 c16 export had warmed the cache.
   A new kernel cache (the port's `kernels-port`) or FlashInfer pin is cold too. *Data point (11-14, port):* the cold
   HOT 300 export had 6,003 MiB free before tuning and tuned for 955 s from scratch; the HOT 342 export after it had
   3,083 MiB free before tuning (1241: 3,235) and tuned in 227 s from the warm cache, 41 min for the whole export,
   no memory error.
3. **The GPUs are exclusive.** Stop the server before tests, traces or exports and wait for memory.used < 1000 MiB.
4. **Never `pgrep -f`/`pkill -f` a pattern that appears in your own command line.** It killed the working shell
   five times (twice more on 11-12). The `grep '[x]yz'` bracket trick does NOT help when the same command line also
   names the script (`ps | grep '[g]lm-budget-ab.sh' | xargs kill` while launching `glm-budget-ab.sh` killed the
   shell). Use `ps -eo pid,args | awk '$2=="./flare/target/release/flare" && $3=="run"'`, or look up the PID in one
   call and `kill <PID>` in the next.
5. **`chmod +x` new scripts, including ones generated with `sed ... > new.sh` or a heredoc**, and check the exit code of a
   detached launch (a `Permission denied` exit 126 leaves no log and no process). Run chains as tracked background jobs. **Always `/usr/bin/env`, never bare `env`:** uv
   puts its own `env` shim first on PATH and a command run through it exits 0 having done nothing (the release
   scripts already say so; the 11-08 bisect chain lost a run to it anyway). *Incident:* a confirmation chain "ran" three
   times with an empty log (exit 126) and the launch method was misdiagnosed first.
6. **One compose command at a time.** `down` racing `up` leaves "removal already in progress" and a hung start.
7. **vLLM prompt logprobs one request at a time**; MMLU-style evals need 2,048 output tokens (20% of replies reason
   first; 207 answers were unparsed with an 8-token cap).
8. **No `flare models gc` during an export.** 
9. **Check free space on the package store before every export.** A GLM-5.3-Flash TP3 variant writes about 73 GB of new
   blobs (experts in a new layout share nothing), a DeepSeek re-export about 4 GB. Version files are hardlinks into the
   shared blob store, so `models list` shows almost nothing reclaimable per version; space comes back only through
   `flare models remove` plus `flare models gc` (never during an export). Archive first with `rsync -aH` of the whole
   store to the archive drive, check the archive lists every version, then remove and gc. *Incident (11-08):* the boot
   drive reached 100 % and a GLM export failed while writing its package ("No space left on device", reported by the
   other ranks as a broken pipe); gc of orphans from failed exports freed 93 GB, archiving 19 non-default versions 250 GB.
12. **`/tmp` is a 126 GB tmpfs: it is RAM.** Never point a large or long-running log, download, clone or checkpoint
    at `/tmp`; use `~/ds41-tp3/` or `/m2-2/`. `aria2c --log` writes at debug level unless `--log-level=warn` is given.
    *Incident (11-08):* a 30 GB aria2c debug log in `/tmp` starved the vLLM TP3 image of pinned host RAM (driver
    "out of memory" while pinning the Engram tables, kernel OOM kills, a restart loop) during the .11/.12 check.
13. **A package exported into a separate `--config` package store is signed with that store's development key.**
    Serve it with `--publisher-keys <store>/development-publisher.json`; the first use also asks for explicit key
    approval and the checkpoint license acceptance. Copying the default store's signing key into the other store
    before exporting avoids both.
14. **A new Flare config directory generates a new signing key on its first export.** Packages from it then fail
    under the default config with "core signer is not a configured publisher". Before the first export into a new
    `--config` dir (or `CONFIG=`), copy `development-signing-key.pem` and `development-publisher.json` from the default
    config dir; for a package already signed with another key, approve it with
    `--dangerously-approve-key <sha256 of the public key bytes> --dangerously-accept-publisher local-development`.
    *(2026-11-12)*
15. **KV regressions between packages: compare the operator scratch first.** Diff the export log's
    `Operator scratch:` line and the per-rank `memory_map` allocations from `flare models inspect --json --all` between
    the two packages. The shared operator scratch is sized by the largest selected workspace and the log names only
    its owner, so a different kernel choice elsewhere can be the cause. *Incident (2026-11-12):* FlashInfer
    0.7.1 -> 0.7.3 raised it 296 -> 438 MiB (-2.9% KV), most likely the prefill-4096 MoE tile choice. Patch df4e92cd
    (branch ds41-tp3-sync) makes tuning prefer lower-scratch kernels within a tolerance of the fastest:
    `FLARE_TUNE_SCRATCH_TOLERANCE` (default 0.02, 0 = off) and `FLARE_TUNE_SCRATCH_MIN_SAVING_MIB` (default 16).
    On the upstream port the memory lines are different; see D.2.18.
17. **Reinstall the native module after every Rust or proto change.** `flare_heliopause` is a non-editable wheel in
    the venv, so `cargo build` does not update what `ignition export` loads. After any change under `helios`,
    `heliopause` or `flare`: `uv pip install --python .venv/bin/python --no-deps --reinstall --refresh ./heliopause`
    (about 41 s). Chains pre-flight this: the wheel's `.so` must contain the newest symbol (`grep -qa
    MXFP4_E2M1_E8M0 <so>`) and be newer than the last Rust commit (`git log -1 --format=%ct -- helios heliopause
    flare`). *Incident (11-14):* the first G1 mini export failed after 28 min at the native core export with
    "unrecognized enum value 'MXFP4_E2M1_E8M0'" (the MXFP4 Engram encoding): the port venv's wheel was built at 07:48,
    the Rust commit that added the encoding (ee9c2479) landed at 07:56.
18. **The upstream memory scheme (in v0.1.0; 2b38bf9d "Let Inductor plan model temporary storage", 4a24d78a), on
    the port since 11-14.** There is no graph scratch pool any more. Each compiled forward's temporaries are planned
    by the compiler into one pool per graph; captured graphs share ONE capture pool, allocated largest first, so its
    size is the largest per-graph pool. Op shapes come from shared "declaration examples" that are released before
    capture. Persistent storage (the bump-allocated arena) is never freed, so anything loaded there to be parked
    (embedding, vision tower) must load as a temporary, or it stays as a dead device copy (~2 GiB/rank). KV gets what
    is left after capture. *Effect (DS41, HOT 342, 11-14):* persistent memory per rank 1,602 MiB lower than 1241, capture
    pool 650 MiB (1241: graph scratch 1,343.5 MiB + capture pool 42 MiB), served KV 5,024,384 vs 3,189,376 (+58 %).
    Read the export lines in A.5 on every port export: free memory before tuning, `Shared declaration examples`
    (superseded must be 0 MiB: build the largest shapes first), `Graph pool requests`, `[brrr] placement rank`, the
    routed-expert selections. *Incident (11-14):* the first full export on the new scheme ran out of GPU memory at
    "Construct graph catalog" (78/99, all ranks) with 5.74 GiB of Torch preparation memory against ~2.3 GiB on
    tp3-main: the parked embedding's 1.23 GiB load copy stayed cached in the open load pool and a long-lived buffer
    pinned 1.06 GiB of it (`empty_cache` does not help; a pool of its own destroyed after parking does), and ~2.5 GiB of
    declaration operands were allocated per graph and per layer instead of shared. Fix 22960712 (own short load pool
    for parked embedding/vision, declaration operands shared through declaration examples, one vision scratch, CPU
    test) brought the catalog to 99/99 with 2.64 GiB free before tuning; S5 and the review improvements later reached
    3,083 MiB at HOT 342. Report: `~/ds41-tp3/analysis/work/port1114/G1-OOM-REPORT.md`. The ~27 GiB that looked
    "unaccounted" in the failure was the merged bank's hot experts (26.33 GiB), mapped outside Torch.
16. **Queue scripts: never let a waiter see its own trigger, and never start a server inside `$(...)`.**
    (a) A waiter that greps the shared queue log for "X done" must not log any line containing "X done" (or anchor the
    pattern to the exact line format, e.g. `grep -qE "^[0-9-]{5} [0-9:]{5} seq71 done$"`); free-text notes in the log
    must avoid trigger phrases too. (b) A shell function that starts a background server must not run inside a command
    substitution: the server inherits the pipe and the caller waits forever; redirect the subshell
    (`( ... & ) > /dev/null 2>&1`) and return values through variables. (c) Every serve wait loop fails fast when the
    server process exits. (d) `chmod +x` before `setsid nohup` (a non-executable waiter silently never starts).
    *Incidents (2026-11-12):* a 30-min hang on a server that was already up (b), a 30-min wait on a TP2 package refused
    on 3 GPUs (c), and a DS41 chain that started early on its own "(waits for seq71 done)" line and stopped a GLM qual
    mid-GPQA (a).
    (e) Two waiters on the same end line both fire when it appears: launch the second only after the first has
    written its start line (and make it wait for the first's end), or give it a guard that checks nothing else holds
    the GPUs. *11-14:* seq115 and seq114 both followed seq113; seq114 was queued only after seq115 had logged its start
    and ran after seq115's end. (f) A failed hard gate in a chain's test stage stops the chain before its export by
    design; read the gate, fix or reclassify the test (known upstream failure = INFO, environment-only = skipped),
    then rerun with the later stages only (`STAGES=...`). *11-14:* G2 stopped at `T-vocab_tp3` (C.32) and G3 at two
    GLM attention tests, both before exporting.
11. **Exports from a worktree must put the worktree first on the import path.** The shared venv's `ignition` console
    script imports the editable install, which points at `~/flare-up-v3`, whatever the current directory is; pytest
    imports from the cwd, so tests look right while exports silently build the main checkout's code. Run
    `PYTHONPATH=$PWD export.sh` (or `.venv/bin/python -m ignition export`) and log `ignition.__file__` next to the
    package id. *Incident (11-08):* the "stage 2" and "vision staging" packages were stage 1 and plain v3; a
    "compiled cache hit" for a graph that should have changed and a missing progress line were the tells.
9. **Scripts that are running must not be edited in place**; copy to a new name (`sweep2.sh`) and start that.
   **Never switch branches in a repository a script is running from**: `git checkout` of a branch that lacks the
   folder deletes and recreates it, the running process keeps a dead cwd, and every relative path fails from then on.
   Use a worktree per branch (`~/rtx-recipes-proposals` for `proposals`). *Incident (11-08):* a qualification run lost
   hotel-lights, retrieval, long generation and stress to a branch switch made to update a proposal file.
   **A chain's "wait for empty GPUs" loop only knows the processes it stops.** If a manual test leaves a container or
   server on the GPUs, the chain waits forever with no log line (`sleep 2` child, no output). Before launching a chain,
   check `nvidia-smi` and `docker ps`; never run a chain's steps by hand while another chain is queued on the GPUs.
10. **Qualification tooling quirks**: `long_context.py` asserted power ≤ 400 W (now 600) and refused existing output
    dirs; LiveBench needs `--livebench-release-option 2024-11-25`; docker engine crashes restart the container, so
    re-check which model is up after an OOM. **`tp3-tools/consistency.py` and empty one-token replies:** when the
    sampled token is a terminating end-of-sequence the server writes no logprob record (`helios/src/server/runtime.rs`,
    terminating EOS), and DS41 always samples at temperature 1.0 (C.25), so a one-token probe can come back empty at
    random. The tool now retries up to 4 times and counts `retried` / `skipped` positions (e90adcc7). A chain gate
    that "did not evaluate" because its tool crashed is neither a pass nor a fail: rerun it, with the reference in
    the same session. *Incident (11-14):* the G2 consistency gate crashed (IndexError) on both the new package and
    1241; the rerun with the fixed tool gave 0.962 vs 0.963 top-1 with 0 retries and 0 skips.

### D.3 Kernel and runtime code

1. **Keyed registries must not use shared pointers.** Scratch arenas hand every projection the same base address and
   sibling graphs reuse offsets. Key by the consumer's output buffer and scope the entry to the graph (arm before the
   producer, consume right after); reset on every non-deferred run. *Incident:* the fused qkv_a norm's split-K
   deferral would have skipped the reduce of any 4-way GEMM in the arena.
2. **Read GPU-produced inputs only after `pdl_wait()`.** *Incident:* sparse decode read per-row lengths before the
   wait; safe only because an unrelated kernel sat in between.
3. **Contract-check kernel constants against the model config at register time** (TOPK, MIX, ROPE, swiglu limit,
   routing scale, Sinkhorn/eps, scratch shapes and strides). A silent upstream scratch change must fail loudly.
4. **Cross-rank decisions must be identical** (which layers use BRRR, at which widths); divergence deadlocks the
   in-kernel all-reduce. Keep the all-gather check in construction code, never in a traced forward.
5. **Assert that parking stayed parking.** b12x's repack is in place for our geometry but would copy to VRAM for a
   ceil-tiled one (TP4). The post-`prepare_weights` check turns silent over-commit into an export error.
6. **Count mixed VRAM/host allocations by their device bytes at every site** (six in Helios; three were missed in the
   first pass, one legacy branch in the second).
7. **No `data_ptr()` or registry work inside Dynamo-traced code**; use a custom op or a binding finalizer.
8. **Every "identical" claim needs a bit-exact test; every equivalence test needs a negative control** shown to fail
   (`FLARE_DS41_BRRR_BREAK=rope`, `TP3Config.brrr_break`). The control's difference must clear the build's own
   run-to-run floor: the BRRR split-tail control (4-22K differing elements) did not clear the plain build's 10-24K
   and is report-only.
9. **Env-var-driven behaviour stays in the recipe and the CLI shim** (upstream rule, D/C.11). In the fork the only
   reader is `TP3Config.from_environment()`; components take the config object (`tp3=`), tests build it from their
   own environment.
10. **Save every site-packages edit as a patch in `tp3-patches/` immediately.** *Incident:* the b12x split-K deferral
    existed only in the venv for a day.
11. **A per-row mismatch in a kernel-vs-reference test is a routing or quantizer effect until proven otherwise.**
    Check which rows differ, the top-k key margins of those rows, and the noise of two independent activation
    quantizers before suspecting the kernel. *Incident:* the layer test's `mlp_output` cosine of 0.95-0.998 cost a
    debugging session; every bad row had a 6th/7th expert key margin ≤ 0.036 (the two mHC/RMSnorm implementations
    round the FFN input differently), and the residual ~5% on matched rows is two e4m3 32-group quantizers. The fix
    was in the test (probe the kernel's routing per row, reconcile near-ties, tolerance 0.995 on `mlp_output`), not in
    the kernel. Swaps chain across rows, so compare per row, not as a multiset.
12. **Rebuild and re-serve after Helios changes**: a Rust edit is not live until `cargo build --release` and a server
    restart; a recipe or kernel edit is not live until a new export.
13. **Several compile-time kernel variants in one process need `-fno-gnu-unique`** (or no function-local static
    guards in inline templates). GCC makes such guards GNU-unique symbols shared by every loaded `.so`, so only the
    first variant set its shared-memory attribute; the others failed with "layer kernel fits 0 CTAs per SM".
14. **"auto" activation precision makes graphs of different widths differ by design.** With `activation_mode="auto"`
    (GLM asks for it on every KDA projection) the tuner races BF16 activations against on-the-fly 8-bit activations
    per GEMM shape, so the one-row decode graph and the six-row verification graph can land on different precisions,
    and a FlashInfer pin that bumps the candidate contract (55 -> 56 in aa6c4221) re-races every shape. A test that
    compares those two shapes then reports a precision mix as an error. Check like-for-like (autotune off, or 8-bit
    activations everywhere) before calling it a kernel regression. *Incident (11-14):* the GLM learned-KDA verification
    test failed on the port at 0.026 (world 1) / 0.040 (world 4) relative error against a 0.01 tolerance (2.6-4.4 %
    per row): the one-row `o_proj` ran 8-bit, the six-row one BF16. With 8-bit everywhere the error was 0.000000,
    with autotune off 0.000119. Open product question (not decided): whether the mix costs GLM draft acceptance.
    Analysis: `~/ds41-tp3/analysis/work/port1114/GLM-KDA-MLA-PIN-ANALYSIS.md`.

### D.4 Release folder

1. **No personal data**: grep for `/home/`, the hostname, the email and `~/`; delete `__pycache__`; scrub
   `hostname`/`uname` from bench JSON. *Incidents:* shipped `.pyc` with a home path, a result file with the machine
   name, a script writing to a literal `~/ds41-tp3/...`.
2. **Scripts write to a run directory, never into shipped results or RESULTS.md.**
3. **Model name = checkpoint directory name** (`ds41-1017-engram-mxfp4`) in docs, `serve.sh` and the view.
4. **`source.patch` must reproduce the fork tree on clean upstream**; verify with a temp worktree and
   `git ls-files -s | md5sum`. Quote the correct file count.
5. **Every number traces to a log or `results/` file**, with power, cap and PCM; say when it came from the pre-rebase
   fork.
6. **Commit with the checkout's configured identity** (agent memory notes hold it). Commit and push only when asked;
   public pushes need an explicit OK; history rewrites need an explicit OK and a force-push warning.

### D.5 Reviews

1. Two passes with fresh reviewers who have not seen the author's conclusions; the second pass found the legacy
   accounting branch and the cross-graph aliasing the first missed.
2. Separate "bug hunt" from "how could this be better".
3. Unadopted experiments stay only behind default-off switches, with the verdict in the docstring, and get reviewed
   like production code (they produced two latent bugs).

### D.6 Quality checks per change (tiered)

Small changes get quick checks before they become the local default; the full suite runs once, on the exact package
being published, before any public release. Every change stays behind its own switch, so a full-suite failure on a
combined build is found by exporting with switches toggled, not by reverting code.

0. **Measure the benefit first.** Before any quality check, show that the change improves what it is for (speed in an
   alternating A/B, KV in a dry run, prefill, host RAM) beyond the variances in D.1, with no loss elsewhere. A change
   that does not pay off stays behind its default-off switch with the verdict recorded, and gets no qualification run.
   Only correctness tests that guard the measurement itself (the layer test, the mini-export consistency check) run
   before the A/B, so a broken build is not timed.
1. **Every change:** its own targeted test (layer test with FP64 bounds and the negative control, or the mini-export
   `consistency.py` check), `verify_fix` with 0 TRUE corruptions, the 43-question eval and the 60 tool-call prompts.
2. **Decode-path changes** (experts, BRRR tail, sampling, drafts): add hotel-lights at c8 (short prompt, long
   generated reasoning; catches drift in the decode path).
3. **Prefill, attention or context changes** (prefill chunking, Engram lookups over the prompt, indexer, KV writes):
   add hotel-lights and lavd at c8 (lavd is a long structured CSV context; it exercises prefill and long-context
   consistency, which hotel-lights does not).
4. **Memory, KV or graph-layout changes:** add the c8 KV stress test, where they fail.
5. **Before a public release:** the tier of the release chain (B.5.5, owner policy since 2026-11-16). *Until 2026-11-16:* the full `qual_all.sh` suite (eval, tools, lavd and hotel at c8, retrieval 128K and
   900K, long generation, EOS, LiveBench, stress) on the package being published.
6. **Full suite on a single change before it becomes the default** when it touches sampling or verification math, the
   chat template or tool parsing, or when any quick check flagged it even once.
