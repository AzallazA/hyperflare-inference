#pragma once
// prefillbw (2026-10-20): prefill (bsz > XMOE_MT_MIN_BSZ) group kernel for host-RAM (cold) experts, xmoe_ds41pb.
//
// = exl3_moe_kernel<t_bits, N, cb, USE_MT = true> (exl3_moe_kernel.cuh, unchanged there) with two value-neutral changes to the
//   g/u/d GEMM calls; every other phase (gather + hadamard, activation, scatter, scheduler, barriers) is a verbatim copy:
//   * PB_ST: shared-memory pipeline depth of the 32-row (MT) GEMM loop (ds41a: MOE_SH_STAGES = 3, i.e. 2 x 1.5 KB B tiles in
//     flight per SM; here 3..8). The <= 16-row tail chunk keeps the stock 3-stage exl3_gemm_kernel_inner.
//   * PB_STAGE: an expert with more than XMOE_MT * 16 rows needs ceil(rows / 32) passes over each weight matrix. ds41a reads B
//     from host RAM in every pass (zero-copy reads are not cached on the GPU). Here the first pass also writes each landed B
//     tile to a VRAM stage copy and the later passes (incl. the <= 16-row tail) read B from that copy: PCIe bytes per expert
//     drop from passes x 6.6 MB to 6.6 MB (TP2 rank of DS4.1-1017). The stage copy lives in this group's own temp_state_u
//     slice, in the rows above the expert's token rows (temp_state_u is only read by the up GEMM, rows < token_count; the
//     stage copy starts at row roundup(token_count, 16)), so it needs no extra VRAM and is CUDA-graph safe. Skipped (plain
//     reads) when the matrix does not fit there, i.e. token_count > max_tokens_per_expert - ~216 for K=3 at 5120 x 1152.
//   Within a group every block re-reads only the tiles it wrote itself (same slice of the same matrix shape); gate -> up reuse
//   the same block-private positions, and up -> down are separated by group barriers.
// Bit-exactness: B bytes are identical in every pass, and the MMA / k-reduction / split-K lock order does not depend on PB_ST.

#include "exl3_moe_kernel.cuh"
#include "exl3_gemm_inner_mt_pb.cuh"

template<int t_bits, int MOE_TILESIZE_N, int cb, int PB_ST, bool PB_STAGE>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)
void exl3_moe_kernel_pb(EXL3_MOE_KERNEL_ARGS)
{
    static_assert(t_bits >= 1 && t_bits <= 8, "xmoe_ds41pb: fixed-K instances only");
    static_assert(XMOE_MT > 1, "xmoe_ds41pb: needs the multi-M GEMM (XMOE_MT >= 2)");
    const int group_idx = blockIdx.z;
    const int block_idx = blockIdx.x;
    const int group_size = gridDim.x;  // SMs per expert, set at launch
    const int num_groups = gridDim.z;
    const int block_threads = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;  // blockDim.x
    const int group_threads = group_size * block_threads;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_group = group_threads / 32;
    const int warps_per_block = block_threads / 32;
    const int warp_idx0 = block_idx * warps_per_block + warp_id;

    // Buffers for group
    temp_state_g += group_idx * max_tokens_per_expert * hidden_dim;
    temp_state_u += group_idx * max_tokens_per_expert * hidden_dim;
    temp_intermediate_g += group_idx * max_tokens_per_expert * intermediate_dim;
    temp_intermediate_u += group_idx * max_tokens_per_expert * intermediate_dim;

    // Barriers for group sync
    int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;

    // Expert scheduler state, self-resetting: [0] next ticket, [1] retired groups, [2 + g] ticket for group g
    int* sched = locks + MOE_SCHED_OFFSET;

    // Individual GEMM barriers per group
    locks += group_idx * MAX(hidden_dim, intermediate_dim) / 128;

    // Stage copy capacity (uint16 elements) per weight matrix: K x N x bits / 16 (trellis [K/16, N/16, 16 * bits])
    const int64_t stage_need = (int64_t) hidden_dim * intermediate_dim * t_bits / 16;

    int ticket = group_idx;

    // Loop over experts
    int start = 0;
    int end = 0;
    int expert_idx = 0;
    int expert_idx_assign = 0;
    for (; expert_idx < num_experts; ++expert_idx)
    {
        // Token span for current expert
        start = end;
        int64_t cnt = expert_count[expert_idx];
        bool skip_split = cnt < 0;   // split launch: negative count = offsets only, expert handled by the other launch
        if (skip_split) cnt = -cnt;
        end += cnt;
        int token_count = end - start;

        // Skip if no tokens or too many tokens for fused kernel (batch is handled by reconstruct path outside kernel)
        if (token_count == 0 || skip_split) continue;
        if (token_count > max_tokens_per_expert) continue;

        // Skip if expert is claimed by a different group
        if (expert_idx_assign++ != ticket) continue;

        // EXL3 weights for g, u, d
        const uint16_t* exp_gate_trellis = gate_trellis[expert_idx];
        const half* exp_gate_suh = gate_suh[expert_idx];
        const half* exp_gate_svh = gate_svh[expert_idx];
        const uint16_t* exp_up_trellis = up_trellis[expert_idx];
        const half* exp_up_suh = up_suh[expert_idx];
        const half* exp_up_svh = up_svh[expert_idx];
        const uint16_t* exp_down_trellis = down_trellis[expert_idx];
        const half* exp_down_suh = down_suh[expert_idx];
        const half* exp_down_svh = down_svh[expert_idx];

        // prefillbw: stage copy for this expert (uniform per group: every block sees the same token_count)
        const int stage_row0 = (token_count + 15) & ~15;
        const bool stage_on = PB_STAGE && token_count > XMOE_MT * 16 &&
                              (int64_t) (max_tokens_per_expert - stage_row0) * hidden_dim >= stage_need;
        uint16_t* stage_buf = stage_on ? (uint16_t*) (temp_state_u + (int64_t) stage_row0 * hidden_dim) : nullptr;

        // Gather + input hadamard for g, u. Non-gated mode skips the g staging (and the g GEMM
        // below); the activation synthesizes the gate lane from u
        const bool gated = act_function != MOE_ACT_RELU2_NOGATE;
        auto had_gather_gu_in = [&]()
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token];
                int token_off = warp_idx % warps_per_token;
                const half* in_ptr = hidden_state + token_idx * hidden_dim + token_off * 128;
                if (gated)
                    had_hf_r_128_inner<true, false>
                    (
                        in_ptr,
                        temp_state_g + 128 * warp_idx,
                        exp_gate_suh + 128 * token_off,
                        0.088388347648f
                    );
                had_hf_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );
            }
            group_barrier(group_idx, group_size, barrier_counters_sense);
        };

        had_gather_gu_in();

        // One GEMM over all row chunks of the expert: MT (32-row) chunks with PB_ST stages, <= 16-row tail with the stock
        // inner. With stage_on, chunk 0 writes the stage copy and every later chunk reads B from it.
        auto gemm_rows = [&](const half* in_addr, half* out_addr, const uint16_t* trellis, const int size_k, const int size_n)
        {
            int size_m = token_count;
            const uint16_t* b_src = trellis;
            bool first = true;
            while (size_m > 0)
            {
                if (size_m > 16)
                {
                    const int mc = MIN(size_m, XMOE_MT * 16);
                    // one inlined MT body per GEMM (two template copies made ptxas drop to 64 registers + 544 B of stack)
                    uint16_t* st_out = (stage_on && first) ? stage_buf : nullptr;
                    exl3_gemm_kernel_inner_mt_pb<t_bits, false, cb, MOE_TILESIZE_M, MOE_TILESIZE_K, MOE_TILESIZE_N,
                                                 PB_ST, XMOE_MT_FRAG_STAGES, false, XMOE_MT>
                        (in_addr, b_src, out_addr, mc, size_k, size_n, locks, nullptr, 0, st_out);
                    if (st_out != nullptr)
                    {
                        __threadfence_block();   // stage writes before this block's own later cp.async reads of them
                        b_src = stage_buf;
                    }
                    first = false;
                    in_addr += mc * size_k;
                    out_addr += mc * size_n;
                    size_m -= mc;
                    continue;
                }
                exl3_gemm_kernel_inner<t_bits, false, cb, MOE_TILESIZE_M, MOE_TILESIZE_K, MOE_TILESIZE_N,
                                       MOE_SH_STAGES, MOE_FRAG_STAGES, false>
                    (in_addr, b_src, out_addr, MIN(size_m, 16), size_k, size_n, locks, nullptr);
                first = false;
                in_addr += 16 * size_k;
                out_addr += 16 * size_n;
                size_m -= 16;
            }
        };

        if (gated)
            gemm_rows(temp_state_g, temp_intermediate_g, exp_gate_trellis, hidden_dim, intermediate_dim);
        gemm_rows(temp_state_u, temp_intermediate_u, exp_up_trellis, hidden_dim, intermediate_dim);
        group_barrier(group_idx, group_size, barrier_counters_sense);

        // Output hadamard for g, u + activation+gate + input hadamard for d
        auto had_guad = [&]()
        {
            const int warps_per_token = intermediate_dim / 128;
            const int total_warps = token_count * warps_per_token;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_off = warp_idx % warps_per_token;
                had_hf_r_128_guad_inner
                (
                    temp_intermediate_g + 128 * warp_idx,
                    temp_intermediate_u + 128 * warp_idx,
                    temp_intermediate_g + 128 * warp_idx,
                    exp_gate_svh + 128 * token_off,
                    exp_up_svh + 128 * token_off,
                    exp_down_suh + 128 * token_off,
                    0.088388347648f,
                    act_limit,
                    act_function
                );
            }
            group_barrier(group_idx, group_size, barrier_counters_sense);
        };

        had_guad();

        // d GEMM
        gemm_rows(temp_intermediate_g, temp_state_g, exp_down_trellis, intermediate_dim, hidden_dim);
        group_barrier(group_idx, group_size, barrier_counters_sense);

        // Output hadamard for d + scatter add
        auto had_d_out = [&]()
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int64_t* top_x = token_sorted + start;
            const half* weights = weight_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = top_x[warp_idx / warps_per_token];
                half weight = weights[warp_idx / warps_per_token];
                int token_off = warp_idx % warps_per_token;
                float* out_ptr = output_state + token_idx * hidden_dim + token_off * 128;
                had_hf_r_128_d_inner
                (
                    temp_state_g + 128 * warp_idx,
                    out_ptr,
                    exp_down_svh + 128 * token_off,
                    0.088388347648f * __half2float(weight)
                );
            }
        };

        had_d_out();

        // Draw the next ticket and publish it to the group through the end-of-expert barrier, which also protects
        // the temp buffers (and the stage copy) for reuse. Grabbed tickets continue from num_groups
        if (block_idx == 0 && threadIdx.x == 0)
            sched[2 + group_idx] = num_groups + atomicAdd(&sched[0], 1);
        group_barrier(group_idx, group_size, barrier_counters_sense);
        ticket = sched[2 + group_idx];
    }

    // Retire group; last group out resets the scheduler for the next launch (see exl3_moe_kernel)
    if (block_idx == 0 && threadIdx.x == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> next_ticket(sched[0]);
        cuda::atomic_ref<int, cuda::thread_scope_device> retired_groups(sched[1]);
        int retired = retired_groups.fetch_add(1, cuda::memory_order_acq_rel);
        if (retired == num_groups - 1)
        {
            next_ticket.store(0, cuda::memory_order_relaxed);
            retired_groups.store(0, cuda::memory_order_relaxed);
        }
    }
}
