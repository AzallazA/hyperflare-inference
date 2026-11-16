#!/bin/bash
# Export the package from a checkpoint view (make_checkpoint_view.sh). Run it from the fork checkout with its .venv
# (uv sync, then re-apply tp3-patches/ to the installed FlashInfer; Flare's torch/Triton fixes are applied in memory
# by ignition.runtime_patches, there is no separate ignition.patches step any more).
#   CKPT     checkpoint view directory (required)
#   HOT      hot experts per parked layer (default 342 = 2.37M KV; each expert fewer adds ~92K KV tokens, see EXPORT.md)
#   PREFILL  prefill chunk size (default 4096)
#   DRAFTS   DSpark drafts (default 5); serve with the same DRAFTS (serve.sh)
#   MAXC     maximum concurrent requests the package captures graphs for (default 8; 16 = the c16 package)
#   FLARE_DS41_VISION_STAGE 1 (default) = vision tower parked in host RAM and its weights staged through VRAM slots per
#            block (image 0.11 s instead of 1.70 s, no KV cost); 0 = no staging
#   FLARE_DS41_ENGRAM_CHUNK_ROWS 1024 (default) + FLARE_DS41_SHARE_ENGRAM_RESIDUAL 1 (default) = graph scratch: chunked Engram
#            projection and engram_residual on ffn_hc.residual (KV 2.46M -> 3.01M at HOT 342, decode level); 0/0 = previous layout
#   B12X_OVERLAY  optional patched b12x copy put ahead of the venv (tp3-tools/make-b12x-overlay.sh); default none
#   B12X_PCIE_TWOSHOT_PULL 1 = with the flashinfer-pcie-twoshot-pull-world3 overlay patch (B12X_OVERLAY), race the BF16
#            two-shot pull all-reduce beside the upstream candidates; unset/0 = upstream set (default)
#   B12X_MLA_DECODE_REMAINDER_STREAM 1 = run each attention decode's 16-head grid on a side stream beside the 8-head
#            remainder grid (ignition/components/mla_remainder_stream.py; no overlay, no b12x change); unset/0 = serial
#   FLARE_DS41_AR_PART_ROWS  decoder prefill all-reduce part cap (0 = halving to what the runtime accepts, default)
#   FLARE_DS41_ENGRAM_PART_ROWS  the same cap for the Engram projection all-reduce (0 = halving, default; 768 = 6 x 768
#            + 512 per 1024-token chunk instead of 8 x 640)
#   B12X_MHC_PDL 1 = programmatic dependent launch in b12x's mHC norm kernels (new mHC compile/tuning keys: the prefill
#            mHC races run again); unset/0 = off (default). Recorded as tp3-config.json mhc_pdl
#   FLARE_DS41_VISION_HOST  1 (default) = vision tower in host RAM (implied by VISION_STAGE=1); VISION_HOST=0 together
#            with VISION_STAGE=0 keeps the tower in VRAM, which needs a lower HOT (unmeasured, no longer useful)
#   PARKED_ROWS  BRRR row gate of the parked layers 20-33 (FLARE_DS41_BRRR_PARKED_MAX_ROWS; default 12 = graphs of at most
#            12 rows run BRRR there, wider verification graphs Flare's MoE path); 0 = ungated (BRRR at every width, as on
#            the resident layers). No kernel, Rust or VRAM change (BRRR buffers are sized for 128 rows); graphs only.
# A cold tuning cache needs more free VRAM than serving. If tuning runs out of memory, export once with HOT=300,
# then export the target. The second export reuses the shared tuning cache.
set -euo pipefail
: "${CKPT:?set CKPT to the checkpoint view directory}"
HERE=$(dirname "$(readlink -f "$0")")
export FLARE_DS41_TP3_GRAPH_GIB=1 FLARE_DS41_HOST_EXPERT_LAYERS=20-33 FLARE_DS41_HOT_EXPERTS=${HOT:-342} \
       FLARE_DS41_MERGED_BANK=1 FLARE_DS41_TP3_VOCAB_PARALLEL=1 \
       FLARE_DS41_EXPERT_COUNTS_DIR=${COUNTS:-$HERE/expert-order} HF_HUB_OFFLINE=1 \
       FLARE_DS41_BRRR_LAYERS=0-39 FLARE_DS41_BRRR_CKPT="$CKPT" FLARE_DS41_BRRR_PARKED_MAX_ROWS=${PARKED_ROWS:-12} \
       FLARE_DS41_EMBED_HOST=1 FLARE_DS41_VISION_HOST="${FLARE_DS41_VISION_HOST:-1}" FLARE_DS41_VISION_STAGE="${FLARE_DS41_VISION_STAGE:-1}" \
       FLARE_DS41_AR_ROWS=1024 FLARE_DS41_ENGRAM_CHUNK_ROWS="${FLARE_DS41_ENGRAM_CHUNK_ROWS:-1024}" \
       FLARE_DS41_SHARE_ENGRAM_RESIDUAL="${FLARE_DS41_SHARE_ENGRAM_RESIDUAL:-1}"
if [ -n "${B12X_OVERLAY:-}" ]; then
    [ -d "$B12X_OVERLAY/b12x_tree" ] || { echo "B12X_OVERLAY=$B12X_OVERLAY has no b12x_tree" >&2; exit 1; }
    export PYTHONPATH="$B12X_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
fi
exec .venv/bin/ignition export "$CKPT" --gpus "${GPUS:-0,1,2}" --max-concurrency "${MAXC:-8}" \
  --prefill-chunk-size "${PREFILL:-4096}" --mtp-draft-tokens "${DRAFTS:-5}" --draft-head-quantization nvfp4 \
  --markov-head-quantization fp8 --vocabulary-parallel-sampling --verbose
