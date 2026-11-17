<!-- GitHub Release note for tag hyperflare-tp3-20261116. Title: "Hyperflare TP3 2026-11-16: upstream port, FP8 Engram, 16 slots, 5.40M KV"
     The annotated tag message is the commented block at the end of this file. -->

**Flare = Hyperflare.** The binary, CLI and cache paths are still `flare`. **New base:** this release is built on
upstream Hyperflare `70069281` (2026-11-14), not `972e28e4`. The fork's history was rebuilt on that base, so this tag
does not descend from `hyperflare-tp3-20261111`; the earlier tags stay as published. Release folder:
`DeepSeek-V4.1-Flash Hyperflare-tp3-20261116` in https://github.com/AzallazA/RTX-PRO-6000-TP3-Inference-Recipes
(`source.patch` against `70069281`, regenerated from this tag).

Changes since `hyperflare-tp3-20261111`:

- **Port onto upstream 70069281.** Upstream's compiler-planned memory (one planned pool per compiled forward, one
  shared capture pool) replaces the graph scratch pool; the TP3 work (expert parking with merged banks, BRRR fused
  layer tail, vocabulary-parallel sampling, Engram table storage, vision staging, TTFT guard) was carried over. On the
  day of the port the same settings served 58% more KV than the previous base.
- **Upstream sync items:** FlashInfer `aa6c4221` (0.7.3; `flashinfer-tiny-decode-swiglu.patch` no longer needed),
  `--mtp-only` packages, the export's lower-scratch kernel preference (`FLARE_TUNE_SCRATCH_TOLERANCE=0.05`).
- **Scheduler:** prefill tail merge (`--scheduler-prefill-tail-merge true`; thinking requests without tools keep their
  last-message checkpoint); selectable TTFT guard order (`--scheduler-ttft-order oldest|shortest`, `serve.sh` uses
  `shortest`); prefill merge of short message pieces at a request's own checkpoint
  (`--scheduler-prefill-merge-min-rows` / `FLARE_SCHEDULER_PREFILL_MERGE_MIN_ROWS`, `serve.sh` uses 512; code default 0 = off).
- **Runtime:** CUDA stack-limit override (`--cuda-stack-limit` / `FLARE_CUDA_STACK_LIMIT`, `serve.sh` uses 256: the
  load refuses a value below the largest registered kernel stack); private Engram tables on 4 KiB pages and fast exit
  (`FLARE_ENGRAM_PRIVATE_TABLES=1`, `FLARE_FAST_EXIT=1`).
- **DeepSeek request handling follows DeepSeek's reference:** `required` or named `tool_choice` with thinking on returns
  HTTP 400 "Thinking mode does not support this tool_choice" (was HTTP 500 after generation); a named tool choice forces
  the call in chat mode; `reasoning_effort` accepts `none`, `minimal` and `medium`; malformed tool-call arguments and
  non-object history arguments pass through as text instead of failing the request.
- **Decode path:** decode glue cleanup (several hundred fewer kernel launches per decode graph); two-shot pull all-reduce
  for prefill (b12x overlay patch `tp3-patches/overlay/flashinfer-pcie-twoshot-pull-world3.patch`); BRRR in the parked
  layers at every width (`PARKED_ROWS=0`); attention 16 + 8-head grids on two streams; host-row embedding gather.
- **Round-2 export switches** (all default off in the code, on in `export.sh`): index and DSpark-context side streams,
  768-row prefill all-reduce parts (decoder and Engram), PDL in the mHC norm kernels, BRRR uneven wo_a split. Engram side
  stream (`FLARE_DS41_ENGRAM_SIDE_LAYERS=14`) measured level and left off.
- **Memory:** four VRAM reclaim switches (`FLARE_DS41_VRAM_RECLAIM=1`); 16 request slots (`MAXC=16`); export KV hygiene
  (cuBLAS's own pool kept out of the export's reachability; `FLARE_EXPORT_LIBRARY_SCRATCH=0` turns it off; the release
  package was exported before it and the same settings gave the same KV with it).
- **Weights:** DeepSeek's own FP8 Engram tables (189 GiB, both in host RAM) instead of NVIDIA's MXFP4 tables (97 GiB).
  The checkpoint view is DeepSeek's snapshot alone (`make_checkpoint_view.sh`, view `ds41-1017-engram-fp8`). **Host RAM:
  256 GB is the minimum** (the server commits 217.3 GiB at a 12G prefix cache); `serve.sh ENGRAM_RAM=layer1` keeps one
  table on disk.
- **Serve defaults:** verification budget 32 (was 24), prefix cache 12G (was 16G in `serve.sh`; 32G does not fit beside
  the FP8 tables).

**Package `ds41-1017-engram-fp8@gb202-tp3-202611160815`** (`export.sh` defaults: HOT 342, MAXC 16, PARKED_ROWS 0,
round-2 switches as above; served at 450 W, k5, alpha 0.8, budget 32, prefix cache 12G, interactive scheduler, TTFT
order shortest, prefill merge 512, stack limit 256): **KV 5,403,008 tokens** (20261111: 3,014,656 at 8 slots, budget 24,
32G prefix cache). `export.sh` pins the package's Engram chunk setting (`FLARE_DS41_ENGRAM_CHUNK_ROWS=1024`); a
value set by the caller is not passed through.

Decode 32K (`llm_decode_bench`, ignore-EOS, one 30 s pass per cell): c1 / c8 / c16 **319.5 / 978.8 / 1,331.5 tok/s**.
Single passes vary: four passes of the same binary and package earlier the same day gave c1 293.7-317.2 and c8
989.2-1,024.4. c4 (not in the release run): 781.2 on the same package with the binary before prefill merge. 128K / 512K
c1: 320.8 / 292.7 (one run). Prefill 32K: 15,766 tok/s. 8-stream agent replay (300 s): 649 turns, 464.6 output tok/s,
first token p50 0.42 s / p90 0.72 s. 20261111 published 301 / 734 / 988 at c1 / c4 / c8 (means of alternating rounds of
60 s cells), 289 / 271 at 128K / 512K, prefill 32K 14,547: different method (single 30 s passes here), so compare the
ranges, not single numbers.

**Qualification (tier "weights": the FP8 tables are a quantisation change):** corruption check 0 true corruptions (10
runs); decode-vs-prefill consistency top-1 0.951 (gate 0.95); eval-43 42/43; tool calls 60/60; tool-eval 90/100 (77/91
graded, TC-45 excluded as before; baseline 91); image smoke 5/5; long retrieval 128K / 900K pass / pass; GPQA Diamond
91.4% (181/198, one sample; WARN: 4 answers truncated, gate ≤ 3; baseline 90.4%); hotel-lights at 16 streams 30/30.
During qualification, a hardware fault on one GPU (Xid 79, the card dropped off the PCIe bus) interrupted the first
16-stream KV stress attempt after 42 seconds; the stress was rerun on the same build and settings, and its result
follows: **16-stream KV stress PASS** (2 waves, 32/32 long requests, 212/212 pings, 0 bad log lines, lowest free VRAM
64 MiB: above the 16 MiB gate, below the kit's 480 MiB margin; 25.6 min); post-run smoke 2/2, served KV unchanged, 0
true corruptions; server and kernel log scan clean.

**Not run for this release** (owner's tiered policy of 2026-11-16, to release faster): LiveBench, long generation,
lavd, the natural-EOS decode bench, the 8-stream KV stress (the 16-stream stress replaces it), the 128K c8 cell, the
vision image sets.

**Not adopted:** Engram side stream on layer 14 (level); BRRR build-time tuning switches (spin, two-shot crossover,
wo_b slices, CTAs per SM); packed verification shapes above 8 slots.

<!-- Annotated tag message (git tag -a hyperflare-tp3-20261116 <tag commit> -F <file>):

hyperflare-tp3-20261116: DeepSeek-V4.1-Flash-1017 at TP3 on Hyperflare 70069281

Base: upstream Hyperflare 70069281 (port of the TP3 work, 2026-11-14).
Package: ds41-1017-engram-fp8@gb202-tp3-202611160815, export.sh defaults (HOT 342, MAXC 16, PARKED_ROWS 0,
--mtp-only, two-shot pull overlay, attention remainder stream, mHC PDL, AR/Engram parts 768, index + context
side streams, BRRR uneven wo_a, embed rows, VRAM reclaim, scratch tolerance 0.05); FP8 Engram tables in host RAM.
Serve: 450 W, k5, alpha 0.8, budget 32, prefix cache 12G, interactive + tail merge, TTFT order shortest,
prefill merge 512, CUDA stack limit 256. KV 5,403,008.
Decode 32K c1/c8/c16 319.5/978.8/1,331.5 tok/s (one 30 s pass; same-day spread c1 293.7-317.2, c8 989.2-1,024.4);
128K/512K c1 320.8/292.7; prefill 32K 15,766.
Qualification tier weights: verify 0 TRUE, consistency 0.951, eval 42/43, tools 60/60, tool-eval 90, retrieval
pass/pass, GPQA 91.4% (WARN 4 truncated), hotel16 30/30, stress16 PASS (rerun after a GPU hardware fault, Xid 79,
interrupted the first attempt: 32/32, 212/212, lowest free VRAM 64 MiB), post checks and log scan PASS.
Not run: LiveBench, longgen, lavd, EOS bench, 8-stream stress.
-->
