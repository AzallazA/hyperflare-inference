<!-- GitHub Release note for tag hyperflare-glm-20261116. Title: "Hyperflare GLM-5.3-Flash 2026-11-16: TP3 14.93M / TP2 3.11M KV, upstream 70069281"
     The annotated tag message is the commented block at the end of this file. -->

**Flare = Hyperflare.** The binary, CLI and cache paths are still `flare`. **New base:** this GLM release is built on
upstream Hyperflare `70069281` (2026-11-14), the base of `hyperflare-tp3-20261116`; it does not descend from
`hyperflare-glm-20261112` (base `cbf21a12`), which stays as published. This tag is `hyperflare-tp3-20261116` plus the
GLM VRAM reclaim switches (one commit) and the GLM release scripts and texts. Release folder:
`GLM-5.3-Flash Hyperflare-tp3-20261116` in https://github.com/AzallazA/RTX-PRO-6000-TP3-Inference-Recipes
(`source.patch` against `70069281`, regenerated from this tag).

**Published after the exports and their quick checks passed; a minimal tier "export" qualification followed on
2026-11-17** (eval-43 43/43 TP3 and 42/43 TP2, long retrieval 128K / 900K pass, TP2 KV stress at 8 streams PASS;
**the TP3 KV stress at 32 streams was not run**). What ran, and on which binary, is listed below.

Changes for GLM-5.3-Flash since `hyperflare-glm-20261112`:

- **Port onto upstream 70069281** (`hyperflare-tp3-20261116`): the GLM TP3/TP2 work (TP3 shard geometry, DCP3, world-3
  pull gather overlay, KDA side chain, 704 expert columns, token embedding in host memory at TP2, vision tower staging)
  was carried onto upstream's compiler-owned storage. The vision tail is now staged per image as well
  (`FLARE_GLM53_VISION_STAGE=1` = blocks and tail); the fork's lean vision working memory was not carried (upstream's
  compiler plans that memory now). On the port the 20261112 settings already served more KV: 13,833,984 tokens at TP3
  (+5.4% over 13,130,496) and 2,518,016 at TP2 (+12.8% over 2,231,808), packages not published.
- **Port fixes found on GLM:** the context-parallel candidate gathers write plain BF16 row slices (`f23ab9ab`; torch
  2.14.1's Inductor miscompiled the FP32 dtype views, which gave wrong answers on prompts above 1,024 tokens in the
  first port export, caught by a prefill probe that is now an export gate); GLM's BF16 head BMMs race Triton tiles
  only at export (`FLARE_BF16_BMM_CUBLAS=0`, `b9b59ee7`: contains a cuBLAS over-read seen on SM120); decode glue cleanup
  (unfilled fully-written scratch, fewer copies per step).
- **Export-time VRAM reclaim switches** (`7a4c4b21`, fork-only, all default off in the code; the release export scripts
  turn them on). None changes what a graph computes; they change which allocations an export keeps:
  - `FLARE_GLM53_TRIAL_POOL=separate` (TP3 and TP2): the eager "Measure graph memory" trial graphs are captured into a
    pool of their own that is released before the serving capture, so the serving graphs no longer carve their pool
    out of trial segments the package then keeps whole. TP3: 3,574 MiB returned, retained graph pool 1,110 MiB
    (was 3,216); TP2: 2,990 MiB returned (pool 1,110 MiB, was 2,442).
  - `FLARE_GLM53_SLAB_SEGMENT=1` (TP3): the preparation-only context-parallel prefill slab (528 MiB) takes whole
    2 MiB granules, so no later tensor shares its last segment and keeps the 530 MiB mapping in the package.
  - `FLARE_GLM53_ONESHOT_FIT=1` (TP2): the one-shot all-reduce is sized for the 512 rows that use it (prefill of 1,024
    rows and more uses the copy-engine ring) instead of 4,096 rows: TP communication 138 MiB per GPU (was 502).
  - `FLARE_GLM53_AR_ROWS=N` (TP3, opt-in): 1,024-row prefill all-reduce parts; not used by this release (needs a
    prefill A/B).
- **Serving (binary of `hyperflare-tp3-20261116`):** the GLM serve scripts now pass `--scheduler-ttft-order shortest`
  (the TTFT guard serves the overdue request with the fewest prompt chunks left first; `TTFT_ORDER=oldest` = 20261112
  behaviour) and `FLARE_CUDA_STACK_LIMIT=256`. The prefill merge and tail merge of the DeepSeek release are not turned
  on for GLM.
- **Release scripts in the fork:** `tp3-tools/release/glm/` (export and serve, TP3 and TP2); the recipes folder copies
  them.

**Packages** (checkpoint `nvidia/GLM-5.3-Flash-FP4-Local` @ `76348ddb`, model `glm53-fp4-local`, 450 W):

| | TP3 `gb202-tp3-202611170053` (32 slots, 4 image slots) | TP2 `gb202-tp2-202611170201` (8 slots, 2 image slots) |
| --- | --- | --- |
| KV served, release binary with the serve scripts (stack limit 256) | **14,929,920** | **3,113,472** |
| KV served on the export chain binary (no stack limit) / dry run | 14,834,688 / 15,025,152 | 3,049,984 / 3,177,472 |
| 20261112 published (130000 / 130307), served (dry run) | 13,130,496 (13,393,152): this release +13.7% | 2,231,808 (2,414,592): this release +39.5% |
| Same settings without the reclaim switches, export chain binary (151109 / 151621) | 13,833,984 (14,035,968): reclaim +1,000,704 (+7.2%) | 2,518,016 (2,653,184): reclaim +531,968 (+21.1%) |
| Decode 32K, one 30 s pass, release binary (tok/s) | c1 / c8 / c32 **290.3 / 878.8 / 1,827.6** | c1 / c8 **245.4 / 750.2** |
| Decode 32K, one 30 s pass, export chain binary (tok/s) | c1 / c8 / c32 279.1 / 886.6 / 1,812.5 | c1 / c4 / c8 246.5 / 519.2 / 762.8 |
| Prefill 8K / 32K, release binary (tok/s) | 10,180 / 11,701 | 11,583 / 12,275 |

Speed: the reclaim switches change memory only; the export chain passes are level with or above the same-binary
passes of 151109 / 151621 (c1 1.02× / 1.00×, c8 1.02× / 1.02×). The 20261112 folder's decode numbers (TP3 279 / 866 / 1,790 at c1 /
c8 / c32, TP2 248 / 512 / 727 at c1 / c4 / c8) are averages of repeated 60 s runs on another binary; single 30 s
passes vary by a few percent, so read these as level, not as a speed-up.

**Checks that ran on these packages (export chains, binary `67c691d29041`: a build of the port before the
`hyperflare-tp3-20261116` scheduler and runtime changes; served with the interactive scheduler, default TTFT order,
no stack override):** TP3: served KV, image latency (first image 0.14 s, 8,000-token image 1.46 s) and image smoke
5/5, first-token probe 24/24 with repeat 12/12, prefill probe PASS with one marginal case (33K needle 30/32 found
against 32/32 for 151109, p = 0.25, not significant), one decode and prefill pass. TP2: served KV, first-token probe
24/24 with repeat 12/12, image smoke 5/5, corruption check 0 true corruptions (20 runs), tool calls 60/60 at
concurrency 8, prefill probe PASS (no marginal case), one decode and prefill pass.

**Boot check on the release binary** (tested build `27864a1d79e4`; this tag's Rust code is that of
`hyperflare-tp3-20261116`; stack limit 256, TTFT order shortest), one boot per package: PASS. TP3: up in 30 s, KV
14,929,920, stack line 256 B (largest kernel stack 32 B), text smoke, image smoke 5/5, first-token probe 24/24 with
repeat 12/12, decode and prefill pass as above, server and kernel log scans clean. TP2: up in 29 s, KV 3,113,472
(largest kernel stack 48 B), the same checks PASS.

**Tier "export" qualification, 2026-11-17** (binary `1bdcce7378e4`, built from fork `main` `036cde24`, a later
commit than this tag: its Rust changes are DeepSeek Engram code and a KV-layout fix, and the GLM packages serve the
same KV; `serve-glm-tp*.sh` settings; temperature 1.0, top-p 0.95). TP3 at 32 streams: served KV 14,929,920 (14,834,688
without the stack limit), first-token probe 24/24 with repeat 12/12, decode-vs-prefill consistency top-1 0.953 (info),
eval-43 **43/43**, 128K decode c1 287 tok/s, tool-eval-bench 93/100 (info, no earlier GLM run to compare), long
retrieval 128K / 900K pass / pass, hotel-lights 30 runs: 26 exact, 2 wrong, 2 truncated. **The TP3 KV stress at 32
streams was not run.** TP2 at 8 streams: served KV 3,113,472 (3,049,984 without), probe 24/24 with repeat 12/12,
consistency 0.958, eval-43 **42/43**, long retrieval pass / pass, hotel-lights 29 exact, 1 wrong, **KV stress PASS**
(16/16 long requests, 92/92 pings), KV unchanged after the run, server and kernel log scans clean. Hotel-lights: earlier
GLM runs ranged from 26 to 30 exact out of 30 (20261112 TP2: 26/30); one miss in each run here is "expected 48, got
49", an item an earlier run also missed. Results:
[`qualification/RESULTS.md`](https://github.com/AzallazA/RTX-PRO-6000-TP3-Inference-Recipes/blob/main/GLM-5.3-Flash%20Hyperflare-tp3-20261116/qualification/RESULTS.md).

**Not run for these packages:** the TP3 KV stress at 32 streams, request checks (13 requests), tool calls at TP3,
corruption check at TP3, image latency at TP2, the prefill-merge arm (TP3), GPQA Diamond (same weights and
quantisation), long generation, the image question sets.

**Not adopted:** `FLARE_GLM53_AR_ROWS=1024` at TP3 (about +191K KV, prefill A/B not run); token embedding in host
memory at TP3 (read every decode step); prefill merge for GLM (one pass measured only).

<!-- Annotated tag message (git tag -a hyperflare-glm-20261116 <tag commit> -F <file>):

hyperflare-glm-20261116: GLM-5.3-Flash at TP3 and TP2 on Hyperflare 70069281

Base: upstream Hyperflare 70069281 = hyperflare-tp3-20261116 + GLM export-layout VRAM switches (7a4c4b21) + GLM
release scripts (tp3-tools/release/glm) and texts.
Packages (nvidia/GLM-5.3-Flash-FP4-Local @ 76348ddb, 450 W):
  TP3 gb202-tp3-202611170053: DCP3, a16, KDA side chain, 704 expert columns, 32 slots, 4 image slots, --mtp-only,
  FLARE_GLM53_TRIAL_POOL=separate FLARE_GLM53_SLAB_SEGMENT=1 FLARE_BF16_BMM_CUBLAS=0. KV 14,929,920
  served (20261112: 13,130,496). Decode 32K c1/c8/c32 290.3/878.8/1,827.6 tok/s (one 30 s pass), prefill 32K 11,701.
  TP2 gb202-tp2-202611170201: DCP2, a16, KDA side chain, embedding in host memory, 8 slots, 2 image slots,
  --mtp-only, FLARE_GLM53_TRIAL_POOL=separate FLARE_GLM53_ONESHOT_FIT=1 FLARE_BF16_BMM_CUBLAS=0. KV 3,113,472
  served (20261112: 2,231,808). Decode 32K c1/c8 245.4/750.2 tok/s (one 30 s pass), prefill 32K 12,275.
Serve: interactive scheduler, TTFT order shortest, CUDA stack limit 256, prefix cache 32G, 5 drafts, TP3 alpha 0.8.
Checks on these packages (export chains, binary 67c691d29041): TP3 KV, image latency + smoke 5/5, probe 24/24,
prefill probe PASS (one marginal case), speed pass; TP2 KV, probe 24/24, image smoke 5/5, verify 0 TRUE, tools 60/60,
prefill probe PASS, speed pass. Release binary 27864a1d79e4: one boot per package PASS (KV, smoke, image smoke,
probe, speed pass, log scans); the KV and speed numbers above are from it.
Not run yet: tiered qualification (tier export: hotel-lights, KV stress, consistency, eval-43, tool-eval, 128K,
retrieval, request checks, merge arm) on the release binary.
-->
