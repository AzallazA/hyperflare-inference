// SPDX-License-Identifier: MIT
// diffbot recipe/kernels/xmoe_bulk/quant/exl3_moe_transfer.cu (MIT, Copyright (c) 2025 Turboderp and Copyright (c) 2026
// Diffbot: ../LICENSE.exllamav3, ../LICENSE.diffbot), ported by exp/ds41-exl3-final-bulk (2026-11-17). Changes: the
// N40_* build flags of their xmoe_bulk build (-DN40_COMPUTE_THREADS=256 -DN40_CTA_MULT=1 -DN40_PRODUCER_WARPS=1) are set
// here, local to this translation unit, and exl3_moe_transfer_threads() reports the block size to the dispatch
// (exl3_moe.cu, xmoe_set_flat_bulk). Kernel code unchanged.
#define N40_COMPUTE_THREADS 256
#define N40_CTA_MULT 1
#define N40_PRODUCER_WARPS 1
#define EXL3_GEMM_BASE_THREADS 128
#define N40_BUFFERS 3
#define N40_BULK_B 1
#define XMOE_FLAT_MAX_PAIRS 384
#define XMOE_FLAT_MAX_JOBS 384
#include "comp_units/exl3_moe_instances.cuh"
#include "exl3_moe_transfer_kernel.cuh"
fp_exl3_moe_kernel exl3_moe_transfer_k3_cb2() { return exl3_moe_transfer_kernel<3,128,2>; }
int exl3_moe_transfer_smem() { return exl3_moe_transfer_dyn_smem<3,128>(); }
int exl3_moe_transfer_threads() { return N40_COMPUTE_THREADS + 32 * N40_PRODUCER_WARPS; }
int exl3_moe_transfer_max_pairs() { return XMOE_FLAT_MAX_PAIRS; }
