// SPDX-License-Identifier: MIT
// Copied unchanged below this header from diffbot recipe/kernels/xmoe_bulk/quant/exl3_flat128_jobs.cuh (MIT, Copyright (c) 2025 Turboderp
// and Copyright (c) 2026 Diffbot: ../LICENSE.exllamav3, ../LICENSE.diffbot). Ported by exp/ds41-exl3-final-bulk (2026-11-17).
#pragma once

// xmoe flat scheduler: grouped variant of exl3_gemm_kernel_inner. One GEMM shape (size_k x size_n) is applied to
// num_jobs * num_mats independent problems (A_j, B_j, C_j) with m_j <= 16 rows each. The (job, n-tile, k-tile) space
// is flattened into one slice range split evenly over ALL blocks of the grid, so every SM gets the same number of
// k-tiles no matter how many experts are active (the group kernel gives each expert to one 8-SM group and idles the
// rest of the GPU in the last round). Split-K partial sums use the original per-column lock protocol, with the column
// index running over all jobs: locks[(mat * num_jobs + job) * tiles_n + n].

#include "../ptx.cuh"
#include "exl3_dq.cuh"
#ifndef EXL3_GEMM_H_ACC
#define EXL3_GEMM_H_ACC 0
#endif

struct FlatGemmJobs
{
    const int* start;             // [num_jobs] first pair row of the job
    const int* rows;              // [num_jobs] rows, 1..16
    const int* expert;            // [num_jobs] expert index
    int num_jobs;                 // jobs per matrix
    int num_mats;                 // 1, or 2 for gate+up (job q >= num_jobs uses matrix 1)
    const half* A[2];             // row-major fp16 activations, row stride size_k, indexed by pair row
    half* C[2];                   // row-major fp16 output, row stride size_n, indexed by pair row
    const uint16_t* const* B[2];  // per-expert trellis pointer tables
};

// xmoe_k64 (2026-10-17): restructured main loop. TILESIZE_K 64 (each sub_k group owns TILEBLOCKS_K / 2 k-blocks per
// stage, so the per-stage bookkeeping is amortized over twice the weights), global->shared copies spread over all 512
// threads (the original issued them from sub_k 0 only, which then arrived late at every barrier), ONE barrier per stage
// (the refill issued after the barrier targets the buffer every thread finished in the previous stage), and fragments
// decoded and consumed in the same stage (FRAG_STAGES 1: frees the registers of the 3-deep fragment ring).
