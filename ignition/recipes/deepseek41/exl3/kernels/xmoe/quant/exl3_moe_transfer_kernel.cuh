// SPDX-License-Identifier: MIT
// Copied unchanged below this header from diffbot recipe/kernels/xmoe_bulk/quant/exl3_moe_transfer_kernel.cuh (MIT, Copyright (c) 2025 Turboderp
// and Copyright (c) 2026 Diffbot: ../LICENSE.exllamav3, ../LICENSE.diffbot). Ported by exp/ds41-exl3-final-bulk (2026-11-17).
#pragma once

// xmoe flat scheduler for decode-sized batches. The group kernel (exl3_moe_kernel) hands each active expert to one
// 8-SM group and processes experts in rounds, so with 36 active experts on 23 groups the second round idles 10 groups
// and every expert pays 5 group barriers. Here the whole grid (one block per SM) runs five phases over ALL active
// experts at once, separated by 4 grid barriers:
//   1 gather + input Hadamard (g, u)   2 gate+up GEMM, flattened over experts   3 output Had + act + down input Had
//   4 down GEMM, flattened              5 output Had + weighted scatter-add
// Rows stay in token_sorted (expert-sorted) pair order in the temp buffers, so expert e's rows are contiguous.
// Experts with more than 16 rows are split into 16-row jobs.

#include <cuda_bf16.h>
#include "exl3_moe_common.cuh"
#include "../util.h"
#include "../util.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner_transfer.cuh"   // xmoe_ds41p: t stage loop with group barriers / mbarrier ring (bit-identical)
#include "exl3_devctx.cuh"
#include "../ptx.cuh"

#ifndef XMOE_FLAT_SH_STAGES
#define XMOE_FLAT_SH_STAGES N40_BUFFERS   // xmoe_ds41p: stage buffers of the flat GEMM ring (2 * XMOE_P_G, or XMOE_P_NBUF)
#endif
#ifndef XMOE_FLAT_TK
#define XMOE_FLAT_TK 128     // xmoe_k64: flat GEMM k-tile (block size stays 256 * MOE_TILESIZE_K / 16 = 512)
#endif
#ifndef XMOE_FLAT_MAX_JOBS
#define XMOE_FLAT_MAX_JOBS 640    // 8x profile build (xmoe_p8k): 1536
#endif
#ifndef XMOE_FLAT_MAX_PAIRS
#define XMOE_FLAT_MAX_PAIRS 1024  // 8x profile build (xmoe_p8k): 8192, so k7 x 128 seqs (1,152 rows) stays on the flat kernel
#endif
#define XMOE_FLAT_MAX_EXPERTS 512
#define XMOE_FLAT_BARRIER_ID 200  // group_barrier slot; the group kernel uses ids < MOE_MAX_GROUPS

static_assert(EXL3_GEMM_BASE_THREADS == 128 || EXL3_GEMM_BASE_THREADS == 256);
template<int t_bits, int MOE_TILESIZE_N, int cb>
__global__ __launch_bounds__(N40_COMPUTE_THREADS + 32 * N40_PRODUCER_WARPS, N40_CTA_MULT)
void exl3_moe_transfer_kernel(EXL3_MOE_KERNEL_ARGS)
{
    static_assert(t_bits > 0, "xmoe flat: fixed bitrate only");
    __shared__ int job_start[XMOE_FLAT_MAX_JOBS];
    __shared__ int job_rows[XMOE_FLAT_MAX_JOBS];
    __shared__ int job_expert[XMOE_FLAT_MAX_JOBS];
    __shared__ short pair_expert[XMOE_FLAT_MAX_PAIRS];
    __shared__ int warp_tot[2][32];
    __shared__ int s_tot[2];

    int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;
    auto grid_sync = [&] () { group_barrier(XMOE_FLAT_BARRIER_ID, gridDim.x, barrier_counters_sense); };

    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp_id = t / 32;
    const int warps_per_block = blockDim.x / 32;

    // Job table, built redundantly by every block from expert_count (no inter-block dependency).
    // Exclusive scans over experts of (row count, 16-row job count); invalid experts (0 rows, or more than
    // max_tokens_per_expert, which the caller runs on its fat path) contribute rows but no jobs.
    int cpart[2] = {}, npart[2] = {};
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        int e = 2 * t + h;
        if (e < num_experts) {
            int count = (int) expert_count[e];
            bool skip = count < 0;
            cpart[h] = skip ? -count : count;
            npart[h] = !skip && count > 0 && count <= max_tokens_per_expert ? (count + 15) / 16 : 0;
        }
    }
    int c = cpart[0] + cpart[1], nj = npart[0] + npart[1];
    int ic = c, inj = nj;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1)
    {
        int a = __shfl_up_sync(0xffffffff, ic, o);
        int b = __shfl_up_sync(0xffffffff, inj, o);
        if (lane >= o) { ic += a; inj += b; }
    }
    if (lane == 31) { warp_tot[0][warp_id] = ic; warp_tot[1][warp_id] = inj; }
    __syncthreads();
    if (warp_id == 0)
    {
        int a = lane < warps_per_block ? warp_tot[0][lane] : 0;
        int b = lane < warps_per_block ? warp_tot[1][lane] : 0;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1)
        {
            int x = __shfl_up_sync(0xffffffff, a, o);
            int y = __shfl_up_sync(0xffffffff, b, o);
            if (lane >= o) { a += x; b += y; }
        }
        if (lane < warps_per_block) { warp_tot[0][lane] = a; warp_tot[1][lane] = b; }
        if (lane == 31) { s_tot[0] = a; s_tot[1] = b; }
    }
    __syncthreads();
    const int pos = ic - c + (warp_id ? warp_tot[0][warp_id - 1] : 0);
    const int jpos = inj - nj + (warp_id ? warp_tot[1][warp_id - 1] : 0);
    const int num_pairs = s_tot[0];   // rows over real experts; sentinel rows (expert == num_experts) come after
    const int num_jobs = s_tot[1];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        int e = 2 * t + h;
        int posh = pos + (h ? cpart[0] : 0);
        int jposh = jpos + (h ? npart[0] : 0);
        if (e < num_experts) {
            for (int i = 0; i < npart[h]; ++i) {
                job_start[jposh + i] = posh + 16 * i;
                job_rows[jposh + i] = MIN(16, cpart[h] - 16 * i);
                job_expert[jposh + i] = e;
            }
            short pe = npart[h] ? (short)e : (short)-1;
            for (int p = posh; p < posh + cpart[h]; ++p) pair_expert[p] = pe;
        }
    }
    __syncthreads();
    if (num_jobs == 0) return;  // uniform across the grid: no barrier is entered

    const int gwarp = blockIdx.x * warps_per_block + warp_id;
    const int gwarps = gridDim.x * warps_per_block;

    // 1. Gather + input Hadamard for g, u
    {
        const int wpt = hidden_dim / 128;
        for (int w = gwarp; w < num_pairs * wpt; w += gwarps)
        {
            int p = w / wpt;
            int off = w % wpt;
            int e = pair_expert[p];
            if (e < 0) continue;
            const half* in_ptr = hidden_state + token_sorted[p] * hidden_dim + off * 128;
            had_hf_r_128_inner<true, false>(in_ptr, temp_state_g + (size_t) p * hidden_dim + off * 128,
                                            gate_suh[e] + off * 128, 0.088388347648f);
            had_hf_r_128_inner<true, false>(in_ptr, temp_state_u + (size_t) p * hidden_dim + off * 128,
                                            up_suh[e] + off * 128, 0.088388347648f);
        }
    }
    grid_sync();

    // 2. Gate + up GEMM over all jobs
    {
        FlatGemmJobs J;
        J.start = job_start; J.rows = job_rows; J.expert = job_expert;
        J.num_jobs = num_jobs; J.num_mats = 2;
        J.A[0] = temp_state_g; J.A[1] = temp_state_u;
        J.C[0] = temp_intermediate_g; J.C[1] = temp_intermediate_u;
        J.B[0] = gate_trellis; J.B[1] = up_trellis;
        exl3_gemm_kernel_inner_transfer<t_bits, false, cb, MOE_TILESIZE_M, XMOE_FLAT_TK, MOE_TILESIZE_N,
                                      XMOE_FLAT_SH_STAGES, 1>(J, hidden_dim, intermediate_dim, locks);
    }
    grid_sync();

    // 3. Output Hadamard for g, u + activation + input Hadamard for d (in place into intermediate_g)
    {
        const int wpt = intermediate_dim / 128;
        for (int w = gwarp; w < num_pairs * wpt; w += gwarps)
        {
            int p = w / wpt;
            int off = w % wpt;
            int e = pair_expert[p];
            if (e < 0) continue;
            half* ig = temp_intermediate_g + (size_t) p * intermediate_dim + off * 128;
            half* iu = temp_intermediate_u + (size_t) p * intermediate_dim + off * 128;
            had_hf_r_128_guad_inner(ig, iu, ig, gate_svh[e] + off * 128, up_svh[e] + off * 128,
                                    down_suh[e] + off * 128, 0.088388347648f, act_limit, act_function);
        }
    }
    grid_sync();

    // 4. Down GEMM over all jobs, output into temp_state_g (free after phase 2)
    {
        FlatGemmJobs J;
        J.start = job_start; J.rows = job_rows; J.expert = job_expert;
        J.num_jobs = num_jobs; J.num_mats = 1;
        J.A[0] = temp_intermediate_g; J.A[1] = nullptr;
        J.C[0] = temp_state_g; J.C[1] = nullptr;
        J.B[0] = down_trellis; J.B[1] = nullptr;
        exl3_gemm_kernel_inner_transfer<t_bits, false, cb, MOE_TILESIZE_M, XMOE_FLAT_TK, MOE_TILESIZE_N,
                                      XMOE_FLAT_SH_STAGES, 1>(J, intermediate_dim, hidden_dim, locks);
    }
    grid_sync();

    // 5. Output Hadamard for d + weighted scatter-add into the fp32 output
    {
        const int wpt = hidden_dim / 128;
        for (int w = gwarp; w < num_pairs * wpt; w += gwarps)
        {
            int p = w / wpt;
            int off = w % wpt;
            int e = pair_expert[p];
            if (e < 0) continue;
            float* out_ptr = output_state + token_sorted[p] * hidden_dim + off * 128;
            had_hf_r_128_d_inner(temp_state_g + (size_t) p * hidden_dim + off * 128, out_ptr,
                                 down_svh[e] + off * 128, 0.088388347648f * __half2float(weight_sorted[p]));
        }
    }
}

// Dynamic shared memory for the flat kernel: GEMM stages + reduction buffer; phase 5 needs 128 floats per warp
template<int t_bits, int MOE_TILESIZE_N>
constexpr int exl3_moe_transfer_dyn_smem()
{
    constexpr int a = 16 * XMOE_FLAT_TK * 2;
    constexpr int b = (XMOE_FLAT_TK / 16) * (MOE_TILESIZE_N / 16) * 256 / 16 * t_bits * 2;
    constexpr int frags_n = 2 * (MOE_TILESIZE_N / 16) / (EXL3_GEMM_BASE_THREADS / 32);
    constexpr int cbuf = 4 * EXL3_GEMM_BASE_THREADS * frags_n * 4;
    constexpr int gemm = XMOE_FLAT_SH_STAGES * (a + b) + cbuf;
    constexpr int had = ((N40_COMPUTE_THREADS + 32 * N40_PRODUCER_WARPS) / 32) * 128 * 4;
    return gemm > had ? gemm : had;
}
