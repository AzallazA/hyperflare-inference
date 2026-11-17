// SPDX-License-Identifier: MIT
// Copied unchanged below this header from diffbot recipe/kernels/xmoe_bulk/quant/exl3_flat128_helpers.cuh (MIT, Copyright (c) 2025 Turboderp
// and Copyright (c) 2026 Diffbot: ../LICENSE.exllamav3, ../LICENSE.diffbot). Ported by exp/ds41-exl3-final-bulk (2026-11-17).
#pragma once

// xmoe_ds41t (2026-10-19): hot-trimmed stage loop of the flat MoE GEMM.
//
// Bit-identical to exl3_gemm_kernel_inner_flat (the ds41a loop; exl3_gemm_inner_flat.cuh is unchanged and still included
// for FlatGemmJobs and by the dense path, but the MoE flat kernel now calls this function instead):
// same slice split over the grid, same tile layout in shared memory, same decode (the dq8 / dq8_aligned_2bits integer
// sequence and decode_3inst_2), same HMMA operands and order per output, same sub_k reduction, same column-lock protocol
// and fp16 partial sums. Only the bookkeeping around the math changed. On exl3_moe_flat_kernel<3,128,2> (ds41a) one
// 128x128 stage costs ~247 warp-instructions per warp, ~25 % of it bookkeeping; here:
//  * shared-memory addresses are 32-bit values computed once per GEMM phase and used as [reg+imm] operands of
//    ld.shared / ldmatrix / cp.async (inline PTX); the stage buffer is a compile-time immediate because the loop body is
//    unrolled over the buffer ring. The per-stage S2UR/LEA/IMAD rebuild (setup block) and the per-k-block ldmatrix IMADs
//    are gone, so are the constant re-materializations (0x6400 / 0x1eee / 0x1010101 MOVs).
//  * no runtime-indexed FlatGemmJobs arrays: A[mat], B[mat], C[mat] are selects (removes the 168 B stack frame and the
//    LDL loads on every column change)
//  * column bookkeeping by run counters: per stage one decrement + test for the copy pipe and one for the MMA pipe; the
//    (n-tile, job) position advances incrementally, no integer division inside the loop
//  * per-thread global copy pointers advanced by one 64-bit add per stage
//  * the second token-half HMMA (jobs with > 8 rows) is a compile-time branch: the loop is instantiated for both cases
//    and re-dispatched at a segment boundary where the row count crosses 8. ds41a issued it predicated off in decode
//    (<= 8 rows per expert): 4 issue slots per stage, 3.3 % of the kernel's stall samples (ncu, T=6)
//  * the MUL1 decode constants and the B copy address are pinned in registers (see xt_pin)
//  * XMOE_T_NBUF stage buffers: 2 (default) = the ds41a pipeline, copies one stage ahead of the MMAs; 3 prefetches two
//    stages ahead (10 KB more shared memory). Both produce identical results (data movement only). One barrier per stage
//    either way: two stages per barrier was measured in SASS (+1.5 instructions/stage: the copy burst splits in two).
// The three "@!PT LDS RZ, [RZ]" in front of the copy burst stay: ptxas emits them before every LDGSTS burst on sm_80+
// (even before the first copy of a kernel, see hottrim/tools/exp_ldgsts.cu), so only a non-LDGSTS copy path removes them.

#include <utility>
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_flat128_jobs.cuh"   // FlatGemmJobs

#ifndef XMOE_T_NBUF
#define XMOE_T_NBUF 2
#endif
// XMOE_T_FOLD=1 (opt-in, NOT bit-identical; K=3 MUL1 only): the codebook scale/offset of decode_3inst_2<2> is folded out
// of the inner decode. The MMA runs on h = 1024 + n (exact in fp16, no per-weight HFMA2), one extra MMA per k-block with
// an all-ones A operand accumulates sum_k x_k, and each MMA segment is mapped back once before the reduce:
// acc = k_inv * acc_h + k_bias * sum_k x_k. -16 HFMA2 +4 HMMA per stage and warp. Numerics: the fp16 rounding of every
// decoded weight disappears, so results differ from ds41a by that rounding: ~2.1e-4 relative L2 (tools/fold_error.py:
// ds41a vs exact codebook math 2.1e-4, fold vs exact 2.5e-6), below the fp16 rounding of the GEMM outputs (4.9e-4).
#ifndef XMOE_T_FOLD
#define XMOE_T_FOLD 0
#endif

// ---- compile-time unrolling (immediate offsets need constant indices) -------------------------------------------------
template <typename F, int... I>
__device__ __forceinline__ void xt_unroll_seq(F& f, std::integer_sequence<int, I...>) { (f(std::integral_constant<int, I>{}), ...); }
template <int N, typename F>
__device__ __forceinline__ void xt_unroll(F&& f) { xt_unroll_seq(f, std::make_integer_sequence<int, N>{}); }

// ---- shared-memory access with 32-bit [reg+imm] addresses -------------------------------------------------------------
// volatile: keeps them ordered against the barriers / cp.async waits at the NVVM level (ptxas still schedules freely)
template <int OFF>
__device__ __forceinline__ uint32_t xt_lds32(uint32_t a)
{
    uint32_t v;
    asm volatile ("ld.shared.b32 %0, [%1+%2];\n" : "=r"(v) : "r"(a), "n"(OFF));
    return v;
}

template <int OFF>
__device__ __forceinline__ void xt_ldsm4(FragA& f, uint32_t a)
{
    uint32_t* r = reinterpret_cast<uint32_t*>(&f);
    asm volatile ("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4+%5];\n"
                  : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a), "n"(OFF));
}

template <int OFF>
__device__ __forceinline__ void xt_cp_async(uint32_t s, const void* g)
{
    asm volatile ("cp.async.cg.shared.global [%0+%2], [%1], 16;\n" :: "r"(s), "l"(g), "n"(OFF));
}

// Identity shuffle: ptxas does not re-materialize SHFL results, so values that would otherwise be rebuilt from special
// registers (SR_CgaCtaId -> shared window) or immediates inside the loop stay in registers. One SHFL per GEMM phase.
__device__ __forceinline__ uint32_t xt_pin(uint32_t v) { return __shfl_sync(0xffffffffu, v, threadIdx.x % 32); }

// ---- decode ------------------------------------------------------------------------------------------------------------
// decode_3inst_2<2> (codebook.cuh) with its constants in registers: x * 0x83DCD12D, dp4a byte sum + 0x6400, fp16 fma
// with k_inv 0x1eee / k_bias 0xc931. Same instructions and operands, so the same bits.
struct XtMul1K
{
    uint32_t k_ones, k_acc;
    half2 k_inv_h2, k_bias_h2;
    __device__ __forceinline__ void init()
    {
        k_ones = xt_pin(0x01010101u);
        k_acc = xt_pin(0x6400u);
        k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));   //  0.00677 = 1/147.7
        k_bias_h2 = __half2half2(__ushort_as_half(0xc931));  // -10.39 = (-1024.0 - 510.0) * k_inv_h
        half2_uint32 ki(k_inv_h2);
        k_inv_h2 = half2_uint32(xt_pin(ki.as_uint32)).as_half2;
    }
};

template <int cb, bool FOLD = false>
__device__ __forceinline__ half2 xt_decode2(uint32_t x0, uint32_t x1, const XtMul1K& K)
{
    if constexpr (cb == 2)
    {
        x0 *= 0x83DCD12Du;
        x1 *= 0x83DCD12Du;
        uint32_t sum0 = __dp4a(x0, K.k_ones, K.k_acc);
        uint32_t sum1 = __dp4a(x1, K.k_ones, K.k_acc);
        half_uint16 h0((uint16_t) sum0);
        half_uint16 h1((uint16_t) sum1);
        if constexpr (FOLD) return __halves2half2(h0.as_half, h1.as_half);   // h = 1024 + n; scale/offset after the MMA
        else return __hfma2(__halves2half2(h0.as_half, h1.as_half), K.k_inv_h2, K.k_bias_h2);
    }
    else
    {
        return decode_3inst_2<cb>(x0, x1);
    }
}

// Per-lane constants of dq_dispatch<bits, cb>(ptr, lane << 3, frag0, frag1): the two 32-bit words a lane reads from its
// 16x16 tile, and the decode from them, as in exl3_dq.cuh (same integer ops, same codebook calls -> identical fp16 weights).
template <int bits> struct XtDq;

template <> struct XtDq<3>   // dq8<3, cb, 4>
{
    int wi0, wi1, s, s4;
    __device__ __forceinline__ void init(int t_offset)
    {
        const int b1 = (t_offset + 257) * 3;
        const int b0 = b1 - 16;
        const int b2 = b1 + 3 * 7;
        const int i0 = b0 / 32;
        const int i2 = (b2 - 1) / 32;
        s = (i2 + 1) * 32 - b2;
        s4 = s + 3 * 4;
        wi0 = i0 % (3 * 256 / 32);
        wi1 = i2 % (3 * 256 / 32);
    }
    template <int cb, bool FOLD = false>
    __device__ __forceinline__ void decode(uint32_t a, uint32_t b, FragB& frag0, FragB& frag1, const XtMul1K& K) const
    {
        const uint32_t w7 = fshift(b, a, s);
        const uint32_t w6 = w7 >> 3;
        const uint32_t w5 = w6 >> 3;
        const uint32_t w4 = w5 >> 3;
        const uint32_t w3 = fshift(b, a, s4);
        const uint32_t w2 = w3 >> 3;
        const uint32_t w1 = w2 >> 3;
        const uint32_t w0 = w1 >> 3;
        frag0[0] = xt_decode2<cb, FOLD>(w0 & 0xffff, w1 & 0xffff, K);
        frag0[1] = xt_decode2<cb, FOLD>(w2 & 0xffff, w3 & 0xffff, K);
        frag1[0] = xt_decode2<cb, FOLD>(w4 & 0xffff, w5 & 0xffff, K);
        frag1[1] = xt_decode2<cb, FOLD>(w6 & 0xffff, w7 & 0xffff, K);
    }
};

template <> struct XtDq<2>   // dq8_aligned_2bits<cb>
{
    int wi0, wi1, s, s4;
    __device__ __forceinline__ void init(int t_offset)
    {
        wi1 = t_offset >> 4;
        wi0 = (wi1 + 15) & 15;
        s = ((~t_offset) & 8) << 1;
        s4 = 0;
    }
    template <int cb, bool FOLD = false>
    __device__ __forceinline__ void decode(uint32_t a, uint32_t b, FragB& frag0, FragB& frag1, const XtMul1K& K) const
    {
        static_assert(!FOLD, "xmoe ds41t: fold is K=3 MUL1 only");
        uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
        b = fshift(b, a, s);
        w7 = b & 0xffff;
        BFE16_IMM(w6, b, 2);
        BFE16_IMM(w5, b, 4);
        BFE16_IMM(w4, b, 6);
        BFE16_IMM(w3, b, 8);
        BFE16_IMM(w2, b, 10);
        BFE16_IMM(w1, b, 12);
        BFE16_IMM(w0, b, 14);
        frag0[0] = xt_decode2<cb>(w0, w1, K);
        frag0[1] = xt_decode2<cb>(w2, w3, K);
        frag1[0] = xt_decode2<cb>(w4, w5, K);
        frag1[1] = xt_decode2<cb>(w6, w7, K);
    }
};

