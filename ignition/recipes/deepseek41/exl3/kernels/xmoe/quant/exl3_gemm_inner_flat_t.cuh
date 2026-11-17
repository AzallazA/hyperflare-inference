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
#include "exl3_gemm_inner_flat.cuh"   // FlatGemmJobs

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

template<EXL3_GEMM_T_ARGS>
inline __device__
void exl3_gemm_kernel_inner_flat_t
(
    const FlatGemmJobs& jobs,
    const int size_k,
    const int size_n,
    int* __restrict__ locks
)
{
    static_assert(!c_fp32, "xmoe flat: fp16 output only");
    constexpr int NBUF = SH_STAGES;                                          // stage buffers (XMOE_T_NBUF)
    constexpr bool FOLD = XMOE_T_FOLD && bits == 3 && cb == 2;               // opt-in, not bit-identical (see top)
    static_assert(NBUF == 2 || NBUF == 3, "xmoe ds41t: 2 or 3 stage buffers");
    const int blocks_n_full = size_n / 16;
    constexpr int TILEBLOCKS_K = TILESIZE_K / 16;
    constexpr int TILEBLOCKS_N = TILESIZE_N / 16;
    constexpr int KPS = TILEBLOCKS_K / 2;                                    // k-blocks per sub_k group
    constexpr int NT = 2 * EXL3_GEMM_BASE_THREADS;                           // threads per block
    constexpr int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);

    constexpr int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                 // halfs
    constexpr int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 16 * bits; // uint16s
    constexpr int SA_BYTES = sh_a_stage_size * 2;
    constexpr int SB_BYTES = sh_b_stage_size * 2;

    constexpr int A_COLS = TILESIZE_K / 8;
    constexpr int A_SWIZZLE_MASK = A_COLS - 1;
    constexpr int A_SWIZZLE_SHIFT = A_COLS <= 2 ? 2 : (A_COLS == 4 ? 1 : 0);   // 128-byte rows: xor with the row itself

    static_assert(EXL3_GEMM_BASE_THREADS == 256);
    static_assert(TILESIZE_M == 16, "Invalid kernel params");
    static_assert(TILEBLOCKS_K % 2 == 0, "xmoe flat: two sub_k groups");
    static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
    static_assert(FRAG_STAGES == 1, "xmoe_k64: fragments are consumed in the stage that decodes them");
    static_assert(EXL3_GEMM_H_ACC == 0, "xmoe flat: fp16 accumulation not supported");
    static_assert(SMEM_MAX >= NBUF * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP * 4,
                  "Invalid kernel params (insufficient shared memory for shape)");

    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + NBUF * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * NBUF);
    const uint32_t sm_a = (uint32_t) __cvta_generic_to_shared(sh_a);
    const uint32_t sm_b = (uint32_t) __cvta_generic_to_shared(sh_b);

    const int T = threadIdx.x;
    const int t = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    const int sub_k = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    const int warp_id = t / 32;
    const int lane_id = t % 32;

    const int tiles_k = size_k / TILESIZE_K;
    const int tiles_n = size_n / TILESIZE_N;
    const int num_jobs = jobs.num_jobs;
    const int total_cols = num_jobs * jobs.num_mats * tiles_n;
    const int total_slices = tiles_k * total_cols;

    const int num_slices = gridDim.x;
    const int slice_beg = (int) ((int64_t) total_slices * blockIdx.x / num_slices);
    const int slice_end = (int) ((int64_t) total_slices * (blockIdx.x + 1) / num_slices);
    const int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    // First column of the block: col = q * tiles_n + n, q = job index over both matrices (q >= num_jobs: matrix 1)
    const int col_beg = slice_beg / tiles_k;
    const int k_beg = slice_beg % tiles_k;

    const int gl_a_stride_k = TILESIZE_K;
    const int gl_b_stride_k = blocks_n_full * TILEBLOCKS_K * 16 * bits;
    constexpr int gl_b_stride_n = TILEBLOCKS_N * 16 * bits;

    // Copy assignment as in ds41a (one 16-byte copy of B per thread j = T < NB4, one of A per thread j = NT - 1 - T < NA4)
    constexpr int NA4 = TILESIZE_M * TILESIZE_K / 8;
    constexpr int NB4 = sh_b_stage_size / 8;
    static_assert(NA4 <= NT && NB4 <= NT, "xmoe ds41t: at most one A and one B 16-byte copy per thread and stage");
    const int ja = NT - 1 - T;
    const int a_k = ja % A_COLS;
    const int a_m = ja / A_COLS;
    const bool pa_thread = ja < NA4;
    const int load_a_gl = a_m * size_k / 8 + a_k;
    const uint32_t cp_a_s = xt_pin(sm_a + 16 * (a_m * A_COLS + (a_k ^ ((a_m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK))));
    const int jb = T;
    const int b_n = jb % (gl_b_stride_n / 8);
    const int b_k = jb / (gl_b_stride_n / 8);
    const int load_b_gl = b_k * (blocks_n_full * 16 * bits / 8) + b_n;
    const bool pb_thread = jb < NB4;
    const uint32_t cp_b_s = xt_pin(sm_b + 16 * jb);
    const int gb_step = gl_b_stride_k / 8;   // int4 per k-tile
    constexpr int ga_step = TILESIZE_K / 8;

    // ---- pipe 0: global -> shared ----
    int ld_n = col_beg % tiles_n, ld_q = col_beg / tiles_n, ld_k = k_beg;
    int ld_left = slice_len;   // slices not yet assigned to a copy segment
    int ld_run;                // copies left in the current segment
    const int4* gA;
    const int4* gB;
    bool pA, pB;
    auto ld_set = [&] ()
    {
        const int mat = ld_q >= num_jobs ? 1 : 0;
        const int j = ld_q - (mat ? num_jobs : 0);
        const int rows0 = jobs.rows[j];
        const half* A = mat ? jobs.A[1] : jobs.A[0];
        const uint16_t* const* Bt = mat ? jobs.B[1] : jobs.B[0];
        const half* gl_a_ptr = A + (size_t) jobs.start[j] * size_k + ld_k * gl_a_stride_k;
        const uint16_t* gl_b_ptr = Bt[jobs.expert[j]] + ld_k * gl_b_stride_k + ld_n * gl_b_stride_n;
        gA = (const int4*) gl_a_ptr + load_a_gl;
        gB = (const int4*) gl_b_ptr + load_b_gl;
        pA = pa_thread && a_m < rows0;
        pB = pb_thread;
        const int seg = min(tiles_k - ld_k, ld_left);
        ld_run = seg;
        ld_left -= seg;
    };
    ld_set();

    auto ld_issue = [&] (auto lbuf)
    {
        constexpr int LB = decltype(lbuf)::value;
        if (pB) xt_cp_async<LB * SB_BYTES>(cp_b_s, gB);
        if (pA) xt_cp_async<LB * SA_BYTES>(cp_a_s, gA);
        gA += ga_step;
        gB += gb_step;
        if (--ld_run == 0)
        {
            if (ld_left == 0)
            {
                pA = false; pB = false;
                ld_run = 1 << 30;
            }
            else
            {
                ld_k = 0;
                if (++ld_n == tiles_n) { ld_n = 0; ++ld_q; }
                ld_set();
            }
        }
    };

    // ---- pipe 2: MMA and output ----
    int cp_col = col_beg, cp_n = col_beg % tiles_n, cp_q = col_beg / tiles_n, cp_k = k_beg;   // cp_k = first k of the segment
    int cp_left = slice_len;
    int cp_seg, cp_run, rows2;
    bool two_halves;
    half* gl_c_ptr_16;
    auto cp_set = [&] ()
    {
        const int mat = cp_q >= num_jobs ? 1 : 0;
        const int j = cp_q - (mat ? num_jobs : 0);
        rows2 = jobs.rows[j];
        two_halves = rows2 > 8;
        gl_c_ptr_16 = (mat ? jobs.C[1] : jobs.C[0]) + (size_t) jobs.start[j] * size_n + cp_n * TILESIZE_N;
        cp_seg = min(tiles_k - cp_k, cp_left);
        cp_run = cp_seg;
        cp_left -= cp_seg;
    };
    cp_set();

    // xmoe_swap: operands swapped (C'[col][token] = W^T X^T): the decoded 16x16 weight tile is the m16k16 A operand
    // and <= 8 tokens are the n8 B operand, so one HMMA covers 256 weights instead of 128 when a job has <= 8 rows.
    constexpr int NB16 = FRAGS_N_PER_WARP / 2;   // 16-column blocks per warp
    register FragC frag_c[2][NB16];               // [token half][column block]
    register FragC frag_x[2];                     // FOLD: sum_k x_k per token (ones-MMA), same C layout as frag_c[h][*]

    // ldmatrix x4 over the token tile: m0 = rows 0-7 k 0-7, m1 = rows 0-7 k 8-15, m2/m3 = rows 8-15 -> B' fragments for both halves
    const int x_r = (lane_id % 8) + 8 * (lane_id / 16);
    const int x_c = (lane_id / 8) % 2;
    const int x_sw = (x_r >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK;
    const int x_row_off = x_r * A_COLS;
    const int b_warp_off = warp_id * NB16;
    uint32_t x_addr[KPS];
    #pragma unroll
    for (int kb = 0; kb < KPS; ++kb)
        x_addr[kb] = xt_pin(sm_a + 16 * (x_row_off + ((x_c + (sub_k * KPS + kb) * 2) ^ x_sw)));
    XtDq<bits> dql;
    dql.init(lane_id << 3);
    dql.s = (int) xt_pin((uint32_t) dql.s);
    dql.s4 = (int) xt_pin((uint32_t) dql.s4);
    XtMul1K K;
    if constexpr (cb == 2) K.init();
    FragA ones;                                   // FOLD: all-ones A operand (fp16 1.0)
    if constexpr (FOLD)
    {
        #pragma unroll
        for (int i = 0; i < 4; ++i) ones[i] = half2_uint32(xt_pin(0x3c003c00u + 0u * i)).as_half2;   // 4 distinct registers
    }
    const uint32_t bw_base = sm_b + (sub_k * KPS * TILEBLOCKS_N + b_warp_off) * 32 * bits;
    const uint32_t bw0 = xt_pin(bw_base + 4 * dql.wi0);
    const uint32_t bw1 = xt_pin(bw_base + 4 * dql.wi1);

    auto compute = [&] (auto cbuf, auto hic)
    {
        constexpr int CB = decltype(cbuf)::value;
        constexpr bool HI = decltype(hic)::value;
        xt_unroll<KPS>([&] (auto kbc)
        {
            constexpr int kb = decltype(kbc)::value;
            FragA xs;
            xt_ldsm4<CB * SA_BYTES>(xs, x_addr[kb]);
            FragB x0, x1;
            x0[0] = xs[0]; x0[1] = xs[1]; x1[0] = xs[2]; x1[1] = xs[3];
            if constexpr (FOLD)
            {
                ptx_mma_m16n8k16(ones, x0, frag_x[0]);
                if constexpr (HI) ptx_mma_m16n8k16(ones, x1, frag_x[1]);
            }
            xt_unroll<NB16>([&] (auto cc)
            {
                constexpr int c = decltype(cc)::value;
                constexpr int OFF = CB * SB_BYTES + (kb * TILEBLOCKS_N + c) * 32 * bits;
                const uint32_t wa0 = xt_lds32<OFF>(bw0);
                const uint32_t wa1 = xt_lds32<OFF>(bw1);
                FragB w0, w1;
                dql.template decode<cb, FOLD>(wa0, wa1, w0, w1, K);
                FragA wa;   // B fragments of columns 0-7 / 8-15 are the row 0-7 / 8-15 halves of the transposed A fragment
                wa[0] = w0[0]; wa[1] = w1[0]; wa[2] = w0[1]; wa[3] = w1[1];
                ptx_mma_m16n8k16(wa, x0, frag_c[0][c]);
                if constexpr (HI) ptx_mma_m16n8k16(wa, x1, frag_c[1][c]);
            });
        });
    };

    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int c = 0; c < NB16; ++c)
                frag_c[h][c] = {};
        if constexpr (FOLD) { frag_x[0] = {}; frag_x[1] = {}; }
    };

    // ---- reduce: verbatim ds41a (threadblock_reduce, read_sum_gl, write_sum_gl, lock protocol) ----
    auto threadblock_reduce = [&] ()
    {
        const int nh = rows2 > 8 ? 2 : 1;
        float* sh_red = sh_c + (FRAGS_N_PER_WARP * 4) * t;
        if (sub_k == 1)
        {
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                if (h < nh)
                    #pragma unroll
                    for (int c = 0; c < NB16; ++c)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) sh_red[(h * NB16 + c) * 4 + j] = frag_c[h][c][j];
        }
        __syncthreads();
        if (sub_k == 0)
        {
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                if (h < nh)
                    #pragma unroll
                    for (int c = 0; c < NB16; ++c)
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) frag_c[h][c][j] += sh_red[(h * NB16 + c) * 4 + j];
        }
    };

    // C' fragment: c[2 i + tt] = out[token 8 h + 2 (lane % 4) + tt][column 16 block + lane / 4 + 8 i]
    auto read_sum_gl = [&] ()
    {
        const int q = lane_id % 4, r = lane_id / 4;
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int c = 0; c < NB16; ++c)
                #pragma unroll
                for (int i = 0; i < 2; ++i)
                    #pragma unroll
                    for (int tt = 0; tt < 2; ++tt)
                    {
                        const int tok = 8 * h + 2 * q + tt;
                        const int col = (b_warp_off + c) * 16 + r + 8 * i;
                        if (tok < rows2) frag_c[h][c][2 * i + tt] += __half2float(gl_c_ptr_16[tok * size_n + col]);
                    }
    };

    auto write_sum_gl = [&] ()
    {
        const int q = lane_id % 4, r = lane_id / 4;
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int c = 0; c < NB16; ++c)
                #pragma unroll
                for (int i = 0; i < 2; ++i)
                    #pragma unroll
                    for (int tt = 0; tt < 2; ++tt)
                    {
                        const int tok = 8 * h + 2 * q + tt;
                        const int col = (b_warp_off + c) * 16 + r + 8 * i;
                        if (tok < rows2) gl_c_ptr_16[tok * size_n + col] = __float2half_rn(frag_c[h][c][2 * i + tt]);
                    }
    };

    // End of an MMA segment (last k-tile of a column, or of the block's slice range): reduce as ds41a, then step to the
    // next column. Returns true when the block's slice range is done.
    auto reduce_advance = [&] () -> bool
    {
        if constexpr (FOLD)   // h-space -> w-space, per sub_k group, before the (unchanged) reduce
        {
            const float k_inv = __half2float(__ushort_as_half(0x1eee));
            const float k_bias = __half2float(__ushort_as_half(0xc931));
            #pragma unroll
            for (int h = 0; h < 2; ++h)
                #pragma unroll
                for (int c = 0; c < NB16; ++c)
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) frag_c[h][c][j] = fmaf(k_inv, frag_c[h][c][j], k_bias * frag_x[h][j]);
        }
        threadblock_reduce();

        // Partial slices within a column are reduced in reverse k order (see exl3_gemm_kernel_inner)
        const int slice2_k = cp_k + cp_seg - 1;
        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - cp_k + 1;
        int* lock = &locks[cp_col];

        barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles_k;

        if (!sub_k && !first) read_sum_gl();
        if (!sub_k) write_sum_gl();

        barrier_release(lock, lock_d, last);

        clear_frag_c();

        if (cp_left == 0) return true;
        ++cp_col;
        cp_k = 0;
        if (++cp_n == tiles_n) { cp_n = 0; ++cp_q; }
        cp_set();
        return false;
    };

    // ---- one stage on ring buffer CB: wait for its copies, barrier, refill the buffer computed in the previous stage
    // with the stage NBUF - 1 ahead, MMAs, segment end. Returns 0 = continue, 1 = slice range done, 2 = the next segment
    // needs the other HI instance.
    auto stage = [&] (auto cbuf, auto hic) -> int
    {
        constexpr int CB = decltype(cbuf)::value;
        constexpr bool HI = decltype(hic)::value;
        cp_async_wait<NBUF - 2>();
        __syncthreads();
        ld_issue(std::integral_constant<int, (CB + NBUF - 1) % NBUF>{});
        cp_async_fence();
        compute(cbuf, hic);
        if (--cp_run == 0)
        {
            if (reduce_advance()) return 1;
            if (two_halves != HI) return 2;
        }
        return 0;
    };

    // Runs the ring from buffer `slot` with a fixed HI until the range is done (-1) or HI changes (returns the buffer of
    // the next stage). Entry at slot > 0 runs the remaining buffers of the ring once before the loop (no jump into it).
    using B0 = std::integral_constant<int, 0>;
    using B1 = std::integral_constant<int, 1>;
    using B2 = std::integral_constant<int, 2>;
    auto run = [&] (auto hic, int slot) -> int
    {
        int r;
        if constexpr (NBUF == 2)
        {
            if (slot == 1) { if ((r = stage(B1{}, hic))) return r == 1 ? -1 : 0; }
            while (true)
            {
                if ((r = stage(B0{}, hic))) return r == 1 ? -1 : 1;
                if ((r = stage(B1{}, hic))) return r == 1 ? -1 : 0;
            }
        }
        else
        {
            if (slot == 1) { if ((r = stage(B1{}, hic))) return r == 1 ? -1 : 2; }
            if (slot >= 1) { if ((r = stage(B2{}, hic))) return r == 1 ? -1 : 0; }
            while (true)
            {
                if ((r = stage(B0{}, hic))) return r == 1 ? -1 : 1;
                if ((r = stage(B1{}, hic))) return r == 1 ? -1 : 2;
                if ((r = stage(B2{}, hic))) return r == 1 ? -1 : 0;
            }
        }
    };

    // ---- prologue: copies of stages 0 .. NBUF - 2 ----
    xt_unroll<NBUF - 1>([&] (auto bc)
    {
        ld_issue(bc);
        cp_async_fence();
    });

    clear_frag_c();

    int slot = 0;
    do slot = two_halves ? run(std::true_type{}, slot) : run(std::false_type{}, slot);
    while (slot >= 0);
}
