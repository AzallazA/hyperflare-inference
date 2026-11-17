#!/bin/bash
# exp/ds41-exl3: export DeepSeek-V4.1-Flash-1017 from diffbot's EXL3 3-bit pack at TP2 (two 96 GB RTX PRO 6000).
# The pack is the checkpoint as published (no view): EXL3 MUL1 routed experts (xmoe kernel, recipe-local), EXL3 MCG
# drafter experts, DeepSeek's FP8 elsewhere, int4 Engram tables (encoding INT4_FP16_BLOCK32 = 16; RAM or disk + row
# cache is chosen when serving, the package is the same). BRRR is off (it runs MXFP4 experts only).
#
#   CKPT   pack directory (required); a truncated view of it for the mini export (MINI=1)
#   GPUS   default 0,1
#   PARK   parked MoE layers (FLARE_DS41_HOST_EXPERT_LAYERS, default 2-39): hot experts in VRAM, the rest in host memory
#   HOT    hot experts per parked layer (default 300): each expert fewer frees ~6.6 MB x parked layers per rank
#   EXPORT_HOT  VRAM rows mapped per parked bank DURING the export (FLARE_DS41_EXL3_EXPORT_HOT; default = HOT): the
#          package still declares HOT VRAM rows and the kernels use HOT (exl3/experts.py export_hot); keeps the export's
#          tuning headroom of a lower hot count
#   MOE_PREFILL  prefill MoE path, group (default) or do (FLARE_DS41_EXL3_PREFILL, decode-once kernel)
#          (38 layers: ~252 MB, ~ KV) and moves ~0.3 % more routes onto the host path
#   MAXC   request slots (default 8)   PREFILL  prefill chunk (default 4096)   DRAFTS  DSpark drafts (default 5 = the
#          released head at k=5, the pack's own 2x default; their k=7 code head is the optional drafter-code-k7/)
#   MINI   1 = truncated view (FLARE_DS41_MINI)
#   MTP_ONLY  1 (default, as TP3's 160815) = --mtp-only: no "-no-mtp" graphs (~25 fewer graphs, shorter export); the
#          package then serves with drafts only (every serve here uses --mtp-draft-tokens 5); 0 = both catalogs
#   SIDE_STREAMS  1 (default) = FLARE_DS41_INDEX_SIDE_STREAM=1 + FLARE_DS41_CONTEXT_SIDE_STREAM=1 (TP3's values): the
#          index chain forks after the q/KV norm and joins before the selectors (inside attention); the DSpark context
#          projections fork after the target decoder and join before the ring writes; neither is open while an xmoe
#          kernel runs. 0 = off. (Engram side layers stay off.)
#   EXL3_SPEED  extra exp/ds41-exl3-speed switches, each default off: FLARE_DS41_EXL3_HOT_LAYERS (per-layer hot counts),
#          FLARE_DS41_EXL3_COLD_SIDE (cold launch on a side stream with k SMs), FLARE_DS41_EXL3_SHARED_SIDE (shared expert
#          beside the router, joined before xmoe) are passed through from the environment as set
#   IMAGE_CONCURRENCY  --image-concurrency n image feature slots (unset = recipe default min(4, MAXC) = 4 x 40 MiB
#          VisionState.arena per GPU; 1 = 40 MiB, -120 MiB per GPU). A second concurrent image request then waits in the
#          scheduler (Capacity -> back to pending, text requests still admitted) until the first one's slot retires
#   CONFIG optional Flare config (--config)
# Format-independent switches of the TP3 default package 160815, on here: embed_rows (host-row embedding gather),
# VRAM reclaim (shared Engram token map, trimmed vocabulary gather, no draft-vocabulary channel, shared vocabulary
# columns), B12X_MHC_PDL=1, index + context side streams (SIDE_STREAMS), scratch preference 0.05 / 16 MiB.
# Deliberately off: Engram side layers (beside the MoE: xmoe's grid barriers need all their blocks resident), AR_ROWS / AR_PART_ROWS / ENGRAM_PART_ROWS (the TP3 shared PCIe
# all-reduce; at TP2 Flare uses its decode all-reduce + DMA prefill all-reduce), BRRR switches, 16-slot packed shapes.
#   DEBUG_SERVE  1 = `ignition debug-serve` (prepare in this process and serve HTTP on HOST:PORT, default
#          127.0.0.1:15015) with exactly the export's environment and preparation flags, instead of `ignition export`;
#          needs heliopause's qualification build on HELIOPAUSE_PATH (a directory placed after this tree on PYTHONPATH).
#          The only use is FLARE_DS41_ROUTE_COUNTS=<dir> (routecount.py, expert-order calibration), which an export
#          refuses.
#   --dry-run  print the environment and the exact export argv, run nothing (CPU only)
#   --plan-only  CPU plan check (tp3-tools/ds41_exl3_plan_check.py) with the same environment; stops before any GPU work
# Run from this checkout with its .venv (heliopause from this tree knows
# encoding 16). Logs go to stdout; the package path is printed at the end by ignition.
set -euo pipefail
ROOT=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
cd "$ROOT"
: "${CKPT:?set CKPT to the EXL3 pack directory}"
python3 -I -c "import json,sys; q=json.load(open(sys.argv[1]))['quantization_config']; sys.exit(q.get('quant_method') != 'exl3')" \
  "$CKPT/config.json" || { echo "$CKPT is not an EXL3 pack" >&2; exit 2; }
for v in $(compgen -e | grep -E '^FLARE_DS41_BRRR' || true); do echo "refusing: $v is set (BRRR is MXFP4-only)" >&2; exit 2; done
if [ "${DEBUG_SERVE:-0}" != 1 ]; then   # s1: FLARE_DS41_EXL3_ROUTE_CAPTURE (routecapture.py) and the debug-serve
  # server overrides (FLARE_DEBUG_SERVE_SERVER_JSON, merged into debug-serve's native server options) are debug-serve only too
  for v in FLARE_DS41_ROUTE_COUNTS FLARE_DS41_EXL3_ROUTE_CAPTURE FLARE_DEBUG_SERVE_SERVER_JSON FLARE_DS41_DEBUG_KV_PAGES; do
    [ -n "${!v:-}" ] && { echo "refusing: $v is set (debug-serve only, never export with it)" >&2; exit 2; }; done
fi
export PYTHONPATH="$ROOT${HELIOPAUSE_PATH:+:$HELIOPAUSE_PATH}" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.1}" HF_HUB_OFFLINE=1 \
       TORCH_EXTENSIONS_DIR="$ROOT/.brrr-ext" TORCH_CUDA_ARCH_LIST=12.0a MAX_JOBS=${MAX_JOBS:-4} \
       FLARE_DS41_EXL3=1 FLARE_DS41_TP3_GRAPH_GIB=1 \
       FLARE_DS41_HOST_EXPERT_LAYERS="${PARK-2-39}" FLARE_DS41_HOT_EXPERTS="${HOT:-300}" FLARE_DS41_MERGED_BANK=1 \
       FLARE_DS41_EXPERT_COUNTS_DIR="${COUNTS:-$ROOT/tp3-tools/release/expert-order}" \
       FLARE_DS41_EMBED_HOST=1 FLARE_DS41_EMBED_ROWS="${FLARE_DS41_EMBED_ROWS:-1}" \
       FLARE_DS41_VRAM_RECLAIM="${FLARE_DS41_VRAM_RECLAIM:-1}" B12X_MHC_PDL="${B12X_MHC_PDL:-1}" \
       FLARE_DS41_VISION_HOST="${FLARE_DS41_VISION_HOST:-1}" \
       FLARE_DS41_VISION_STAGE="${FLARE_DS41_VISION_STAGE:-1}" \
       FLARE_DS41_ENGRAM_CHUNK_ROWS="${FLARE_DS41_ENGRAM_CHUNK_ROWS:-1024}" \
       FLARE_DS41_SHARE_ENGRAM_RESIDUAL="${FLARE_DS41_SHARE_ENGRAM_RESIDUAL:-1}"
[ "${MINI:-0}" = 1 ] && export FLARE_DS41_MINI=1
if [ "${SIDE_STREAMS:-1}" = 1 ]; then export FLARE_DS41_INDEX_SIDE_STREAM=1 FLARE_DS41_CONTEXT_SIDE_STREAM=1; fi
export FLARE_TUNE_SCRATCH_TOLERANCE="${FLARE_TUNE_SCRATCH_TOLERANCE:-0.05}" \
       FLARE_TUNE_SCRATCH_MIN_SAVING_MIB="${FLARE_TUNE_SCRATCH_MIN_SAVING_MIB:-16}"
[ -n "${EXPORT_HOT:-}" ] && export FLARE_DS41_EXL3_EXPORT_HOT="$EXPORT_HOT"
[ -n "${MOE_PREFILL:-}" ] && export FLARE_DS41_EXL3_PREFILL="$MOE_PREFILL"
export PATH="$CUDA_HOME/bin:$PATH"
.venv/bin/python -c 'import ignition, flare_heliopause; print("ignition", ignition.__file__, "heliopause", flare_heliopause.__file__)'
/usr/bin/env | grep -E '^(FLARE_DS41_|FLARE_TUNE_|B12X_)' | sort || true   # /usr/bin/env: another `env` (e.g. uv's) can come first on PATH
VERB=export; [ "${DEBUG_SERVE:-0}" = 1 ] && VERB=debug-serve
ARGV=(.venv/bin/ignition "$VERB" "$CKPT" --gpus "${GPUS:-0,1}" --max-concurrency "${MAXC:-8}"
  --prefill-chunk-size "${PREFILL:-4096}" --mtp-draft-tokens "${DRAFTS:-5}" --draft-head-quantization nvfp4
  --markov-head-quantization fp8 --vocabulary-parallel-sampling --verbose ${CONFIG:+--config "$CONFIG"})
[ "${MTP_ONLY:-1}" = 1 ] && ARGV+=(--mtp-only)
[ -n "${IMAGE_CONCURRENCY:-}" ] && ARGV+=(--image-concurrency "$IMAGE_CONCURRENCY")
[ "${DEBUG_SERVE:-0}" = 1 ] && ARGV+=(--host "${HOST:-127.0.0.1}" --port "${PORT:-15015}")
if [ "${1:-}" = --plan-only ] && [ "${DEBUG_SERVE:-0}" = 1 ]; then echo "--plan-only checks exports only" >&2; exit 2; fi
if [ "${1:-}" = --plan-only ]; then
  CUDA_VISIBLE_DEVICES= exec .venv/bin/python tp3-tools/ds41_exl3_plan_check.py "$CKPT" \
    --world "$(tr ',' '\n' <<< "${GPUS:-0,1}" | grep -c .)" -- "${ARGV[@]:1}"
fi
if [ "${1:-}" = --dry-run ]; then printf 'export argv:'; printf ' %q' "${ARGV[@]}"; echo; exit 0; fi
exec "${ARGV[@]}"
