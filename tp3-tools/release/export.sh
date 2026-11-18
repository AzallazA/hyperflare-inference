#!/bin/bash
# Export the package from a checkpoint view (make_checkpoint_view.sh). Run it from the fork checkout with its .venv
# (uv sync, then re-apply tp3-patches/ to the installed FlashInfer; Hyperflare's torch/Triton fixes are applied in memory
# by ignition.runtime_patches). The defaults reproduce the release package (16 request slots, FP8 Engram view, round-2
# switches on, Engram side stream off). Every switch below is recorded in the package's tp3-config.json.
#   CKPT     checkpoint view directory (required; the directory name becomes the model name, default view: ds41-1017-engram-fp8)
#   B12X_OVERLAY  b12x copy with tp3-patches/overlay/flashinfer-pcie-twoshot-pull-world3.patch (required while TWOSHOT=1):
#            VENV=$PWD/.venv B12X_OVERLAY_PATCHES=$PWD/tp3-patches/overlay/flashinfer-pcie-twoshot-pull-world3.patch \
#              tp3-tools/make-b12x-overlay.sh $PWD/.b12x-overlay-twoshot
#   HOT      hot experts per parked layer (default 342; each expert fewer adds ~92K KV tokens, see README)
#   PARK     parked (host-expert) layers (default 20-33). Prefill runs layers 20-39 on the last 128 rows of a chunk only, so
#            only those are candidates; e.g. PARK=20-39 HOT=354 keeps the release's host-expert rows (14 x 42 = 588) spread
#            over 20 layers (20 x 30 = 600; perf/ds41-tp3-park, not measured on the GPU yet)
#   MAXC     request slots the package captures graphs for (default 16; 8 = fewer graphs, not measured on this build)
#   PREFILL  prefill chunk size (default 4096)
#   DRAFTS   DSpark drafts (default 5); serve with the same DRAFTS (serve.sh)
#   PARKED_ROWS  BRRR row gate of the parked layers 20-33 (default 0 = BRRR at every width, as on the resident layers;
#            12 = the 20261111 gate)
#   TWOSHOT  1 (default) = race the BF16 two-shot pull all-reduce beside the upstream candidates (B12X_PCIE_TWOSHOT_PULL,
#            needs B12X_OVERLAY); 0 = upstream candidate set, no overlay
#   ATTN     1 (default) = each attention decode's 16-head grid on a side stream beside the 8-head remainder grid
#            (B12X_MLA_DECODE_REMAINDER_STREAM; ignition hook, no b12x change); 0 = serial
#   MHC_PDL  1 (default) = programmatic dependent launch in b12x's mHC norm kernels (B12X_MHC_PDL); 0 = off
#   AR_PART_ROWS, ENGRAM_PART_ROWS  prefill all-reduce part caps for the decoder and the Engram projection (default
#            768 each = 5 x 768 + 256 per 4096-row chunk instead of 4 x 1024; 0 = halving, the 20261111 layout)
#   INDEX_SIDE   1 (default) = indexer query chain on a side stream (FLARE_DS41_INDEX_SIDE_STREAM)
#   CONTEXT_SIDE 1 (default) = DSpark context projection on a side stream beside verification (FLARE_DS41_CONTEXT_SIDE_STREAM)
#   ENGRAM_SIDE  layer list for the Engram side stream (FLARE_DS41_ENGRAM_SIDE_LAYERS; default empty = off; "14" measured
#            level, not adopted)
#   WOA_UNEVEN   BRRR wo_a job split for <= 8 rows (FLARE_DS41_BRRR_WOA_UNEVEN; default 1); WOA_SMALL default 0
#   EMBED_ROWS   1 (default) = vectorised host-row embedding gather (FLARE_DS41_EMBED_ROWS)
#   RECLAIM  1 (default) = the four VRAM reclaim switches (FLARE_DS41_VRAM_RECLAIM: shared Engram token map, trimmed
#            vocabulary gather, no draft-vocabulary channel, shared vocabulary columns); 0 = off
#   LIBRARY_SCRATCH  unset = code default (on): cuBLAS's own 64 MiB pool is kept out of the export's reachability
#            (FLARE_EXPORT_LIBRARY_SCRATCH); 0 = off, as the code the release package was exported from
#   PAD_ONCE     1 = one MXFP8 MMA-scale zero fill per decode/verify/draft forward instead of one per layer
#            (FLARE_DS41_MMA_PAD_ONCE); default 0 (ds41-tp3-next 2026-11-17, not measured on the GPU yet)
#   VISION_ARENA_HOST  1 = image feature arena (160 MiB/GPU) in device-visible host memory (FLARE_DS41_VISION_ARENA_HOST);
#            default 0 (VRAM map 2026-11-15 item 3)
#   MARKOV_HOST  1 = DSpark Markov-head embedding (63 MiB/GPU) in device-visible host memory (FLARE_DS41_MARKOV_EMBED_HOST);
#            default 0 (VRAM map item 6)
#   MARKOV_STAGE 1 = with MARKOV_HOST=1, copy each request's Markov predecessor row into VRAM once per draft step
#            (FLARE_DS41_MARKOV_STAGE_ROWS); keeps MARKOV_HOST's ~74K KV without its decode cost (11-18 microbench:
#            b1 +0.01 ms, b16 -0.00 ms per round vs the table in VRAM); default 0
#   GLUE2    1 = second decode-glue pass (FLARE_DS41_GLUE_PASS2: no zero fill of the verification staging / draft window
#            cache, draft state rows once, one-kernel pre-mix, concatenated draft feedback, functional admission check;
#            same bytes, GPU proof padglue2-proof-1117-2320); default 0
#   SKIP_SUFFIX  prefill capacities that get an encoder-only twin (FLARE_DS41_PREFILL_SKIP_SUFFIX, e.g. 4096; "1" =
#            3072,3584,4096): non-final chunks of long cold prompts skip layers 21-39 and the head (Helios decides per
#            chunk; serve with --scheduler-encoder-prefill false to turn it off); default unset = no twins
#   ORDER    expert order shipped in tp3-tools/release/expert-orders/<ORDER> (sets COUNTS when COUNTS is unset), e.g.
#            order-mix-provisional-exl3S (blend order of the 11-18 default, fingerprint a70f5e67d564)
#   SHARE_OPERANDS  1 = upstream a386501f (one computed copy per shared pointwise operand of opaque calls;
#            FLARE_EXPORT_SHARE_COMPUTED_OPERANDS, part of the compiler cache key); default 0
#   DRAFT_VOCAB  draft vocabulary JSON (tp3-tools/ds41_draft_vocab.py build; tp3-tools/release/draft-vocab/, held-out
#            miss rates inside each table): the NVFP4 DSpark draft head keeps only those tokens (FLARE_DS41_DRAFT_VOCAB);
#            default unset = full head
#   CONFIG   Flare config file for the export (--config; default: Flare's default config)
#   COMPILE_WORKERS  CPU compiler processes (--compile-workers; default automatic). The release export used 3 to stay
#            inside the test system's power-supply limit
#   FLARE_EXPORT_FORWARD_WORKERS=N, FLARE_EXPORT_FORWARD_WORKER_GIB=G  (fork, export speed; unset = as before) forward
#            compiler processes per host for "Compile model forwards" and their host-RAM admission estimate (upstream 32 GiB).
#            They apply only after kernel tuning (GPUs idle; COMPILE_WORKERS still caps tuning). N=9 G=16: expected
#            ~13 -> ~5-6 min for 30 forwards. Device/host admission still decides: read "N compiler workers; per rank"
#   FLARE_EXPORT_FORWARD_WORKER_VRAM_MIB=M  (fork) per-worker GPU-memory admission estimate (upstream 1024; a worker holds
#            only its CUDA context, ~700 MiB measured 11-17): 768 admits 2 workers per GPU at ~3.2 GiB free (1 GiB margin
#            kept), 3 at >= 3,328 MiB. Export-time VRAM only. Log line: "Forward compiler admission: ..."
#            On an out-of-memory retry, unset every FLARE_EXPORT_FORWARD_* variable.
#   FLARE_EXPORT_TUNE_ONLY=1  (fork) stop after "Prepare and tune kernels" (selections flushed to the kernel cache, exit 0,
#            no package): the warm-up before a cold-cache HOT 342 export (HOT=300 FLARE_EXPORT_TUNE_ONLY=1, then the export)
#   FLARE_DS41_VISION_STAGE 1 (default) = vision tower in host RAM, weights staged through VRAM slots per block
#   GPUS     default 0,1,2
# A cold tuning cache needs more free VRAM than serving. If tuning runs out of memory, export once with HOT=300,
# then export the target. The second export reuses the shared tuning cache.
set -euo pipefail
: "${CKPT:?set CKPT to the checkpoint view directory}"
HERE=$(dirname "$(readlink -f "$0")")
TWOSHOT=${TWOSHOT:-1}
if [ "$TWOSHOT" = 1 ]; then
    : "${B12X_OVERLAY:?TWOSHOT=1 needs B12X_OVERLAY (see the header), or export with TWOSHOT=0}"
fi
if [ -n "${B12X_OVERLAY:-}" ]; then
    [ -d "$B12X_OVERLAY/b12x_tree" ] || { echo "B12X_OVERLAY=$B12X_OVERLAY has no b12x_tree" >&2; exit 1; }
    export PYTHONPATH="$B12X_OVERLAY${PYTHONPATH:+:$PYTHONPATH}" B12X_PCIE_DCP_GATHER_PULL=0
fi
if [ -n "${ORDER:-}" ] && [ -z "${COUNTS:-}" ]; then
    COUNTS=$HERE/expert-orders/$ORDER
    [ -d "$COUNTS" ] || { echo "ORDER=$ORDER: no $COUNTS" >&2; exit 1; }
fi
if [ "${MARKOV_STAGE:-0}" = 1 ] && [ "${MARKOV_HOST:-0}" != 1 ]; then
    echo "MARKOV_STAGE=1 needs MARKOV_HOST=1 (it stages rows of the parked table)" >&2; exit 1
fi
export FLARE_DS41_TP3_GRAPH_GIB=1 FLARE_DS41_HOST_EXPERT_LAYERS=${PARK:-20-33} FLARE_DS41_HOT_EXPERTS=${HOT:-342} \
       FLARE_DS41_MERGED_BANK=1 FLARE_DS41_TP3_VOCAB_PARALLEL=1 \
       FLARE_DS41_EXPERT_COUNTS_DIR=${COUNTS:-$HERE/expert-order} HF_HUB_OFFLINE=1 \
       FLARE_DS41_BRRR_LAYERS=0-39 FLARE_DS41_BRRR_CKPT="$CKPT" FLARE_DS41_BRRR_PARKED_MAX_ROWS=${PARKED_ROWS:-0} \
       FLARE_DS41_EMBED_HOST=1 FLARE_DS41_VISION_HOST="${FLARE_DS41_VISION_HOST:-1}" FLARE_DS41_VISION_STAGE="${FLARE_DS41_VISION_STAGE:-1}" \
       FLARE_DS41_AR_ROWS=1024 FLARE_DS41_ENGRAM_CHUNK_ROWS=1024 FLARE_DS41_SHARE_ENGRAM_RESIDUAL=1 \
       FLARE_DS41_AR_PART_ROWS=${AR_PART_ROWS:-768} FLARE_DS41_ENGRAM_PART_ROWS=${ENGRAM_PART_ROWS:-768} \
       FLARE_DS41_BRRR_WOA_UNEVEN=${WOA_UNEVEN:-1} FLARE_DS41_BRRR_WOA_SMALL=${WOA_SMALL:-0} \
       FLARE_DS41_VRAM_RECLAIM=${RECLAIM:-1} \
       FLARE_TUNE_SCRATCH_TOLERANCE=0.05 FLARE_TUNE_SCRATCH_MIN_SAVING_MIB=16 \
       B12X_PCIE_TWOSHOT_PULL=$TWOSHOT B12X_MLA_DECODE_REMAINDER_STREAM=${ATTN:-1} B12X_MHC_PDL=${MHC_PDL:-1}
# On/off switches that TP3Config reads as "1" = on; set only when on, so an off export records the 20261111 values.
[ "${INDEX_SIDE:-1}" = 1 ] && export FLARE_DS41_INDEX_SIDE_STREAM=1
[ "${CONTEXT_SIDE:-1}" = 1 ] && export FLARE_DS41_CONTEXT_SIDE_STREAM=1
[ "${EMBED_ROWS:-1}" = 1 ] && export FLARE_DS41_EMBED_ROWS=1
[ -n "${ENGRAM_SIDE:-}" ] && export FLARE_DS41_ENGRAM_SIDE_LAYERS=$ENGRAM_SIDE
[ -n "${LIBRARY_SCRATCH:-}" ] && export FLARE_EXPORT_LIBRARY_SCRATCH=$LIBRARY_SCRATCH
[ "${PAD_ONCE:-0}" = 1 ] && export FLARE_DS41_MMA_PAD_ONCE=1
[ "${VISION_ARENA_HOST:-0}" = 1 ] && export FLARE_DS41_VISION_ARENA_HOST=1
[ "${MARKOV_HOST:-0}" = 1 ] && export FLARE_DS41_MARKOV_EMBED_HOST=1
[ "${MARKOV_STAGE:-0}" = 1 ] && export FLARE_DS41_MARKOV_STAGE_ROWS=1
[ "${GLUE2:-0}" = 1 ] && export FLARE_DS41_GLUE_PASS2=1
[ -n "${SKIP_SUFFIX:-}" ] && export FLARE_DS41_PREFILL_SKIP_SUFFIX=$SKIP_SUFFIX
[ "${SHARE_OPERANDS:-0}" = 1 ] && export FLARE_EXPORT_SHARE_COMPUTED_OPERANDS=1
[ -n "${DRAFT_VOCAB:-}" ] && export FLARE_DS41_DRAFT_VOCAB=$DRAFT_VOCAB
exec .venv/bin/ignition export "$CKPT" --gpus "${GPUS:-0,1,2}" --max-concurrency "${MAXC:-16}" \
  --prefill-chunk-size "${PREFILL:-4096}" --mtp-draft-tokens "${DRAFTS:-5}" --draft-head-quantization nvfp4 \
  --markov-head-quantization fp8 --vocabulary-parallel-sampling --mtp-only --verbose \
  ${COMPILE_WORKERS:+--compile-workers "$COMPILE_WORKERS"} ${CONFIG:+--config "$CONFIG"}
