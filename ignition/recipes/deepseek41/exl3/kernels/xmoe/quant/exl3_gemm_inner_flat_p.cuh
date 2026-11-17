#pragma once

// xmoe_ds41p (2026-10-19): pipeline/synchronization rework of the hot-trimmed flat MoE stage loop.
//
// Bit-identical to exl3_gemm_kernel_inner_flat_t (xmoe_ds41t) and so to the ds41a loop: the decode is the t-tree's
// XtDq / xt_decode2 code unchanged (same integer ops, same codebook constants, same fp16 HFMA2), the HMMA operands and
// their order per output are unchanged (4 k-blocks per sub_k group, accumulated in k order), the sub_k reduce, the
// column-lock protocol and the fp16 partial sums are verbatim. Only data movement and synchronization changed, so the
// GEMM outputs are the same bits by construction.
//
// Decode instructions: with ALU/FMA instructions the MUL1 decode is at its exact floor on sm_120 SASS. Per lane and
// 16x16 tile (8 weights) it is 2 SHF.R.U64 + 6 SHF.R.U32.HI + 8 LOP3 (16 bit-window extractions: there is no
// single-instruction shifted bit-field extract on sm_120, bfe.u32 -> SHF + SGXT, no 16-bit IMAD operand selectors,
// LOP3/SHF on 64 bit split into 2 x 32), 8 IMAD (x * 0x83DCD12D needs a zero-extended 16-bit state: any garbage above
// bit 15 changes bytes 2-3 of the product), 8 IDP.4A (one byte sum per product), 4 PRMT + 4 HFMA2 (pack + scale of 2
// weights, already packed math; dropping the HFMA2 = the inexact codebook fold). 40 + 2 LDS, which ptxas already emits.
// The only exact cut found moves the byte sums to the tensor pipe: XMOE_P_IMMA (opt-in) replaces the 8 IDP.4A of a tile
// by 2 IMMA m16n8k32.u8 (34 per tile, -12 % instructions per stage, half the FMA-heavy work), at 4 tensor-pipe cycles
// per SM each, the cost of an HMMA.16816 (xmoe_ds41tf showed +4 HMMA per stage can cost more than it saves).
// Other ideas evaluated (SASS probes not included):
//  - fp16x2 byte sums (PRMT/LOP3 bytes into fp16 lanes + HADD2): >= 2 instructions per weight vs 1 IDP; sums above
//    2048 are not exact in fp16, the denormal encoding needs k_inv * 2^24 (> fp16 max) in the HFMA2.
//  - cp.async.bulk (TMA) for the weight stream: UBLKCP takes uniform operands, so ptxas wraps each per-thread call in
//    ELECT + R2UR.BROADCAST + BRA.U.ANY (~10 instructions per copy); B is 8 strided 768-byte rows per stage (a single
//    2-D copy needs a host-built CUtensorMap per expert matrix) and A per-row copies into an unswizzled layout: ~80-100
//    instructions per stage on the issuing warp against ~10 per warp now. Not done; the mbarrier ring (SYNC 1) keeps
//    LDGSTS and takes the decoupling part of the idea.
//  - 2 CTAs per SM: needs <= 64 registers and a grid of 2 x the SM budget, which changes the slice partition and so
//    the split-K fp16 partial sums (not bit-exact against ds41a at the same budget); the block size is host-side.
//  - hand scheduling: ptxas reschedules inline PTX; what source order can do is done (copies after the first loads,
//    fewer barriers). The per-stage segment-end branch still bounds ptxas's window to one stage.
//
// What changes in the default build (all exact):
//  * XMOE_P_SYNC 0 (default): one __syncthreads per GROUP of XMOE_P_G stages instead of per stage, ring of 2 * G stage
//    buffers. During group g every stage refills the buffer its counterpart in group g - 1 used (G stages ahead), so a
//    single cp.async.wait_group 0 + barrier at the start of each group covers both the RAW (copies landed and visible)
//    and the WAR (everybody finished reading the buffer) hazards. Inside a group there is no barrier: ptxas schedules
//    the G stages as one straight-line block (next stage's LDS / ldmatrix hoisted over this stage's HMMA tail), and
//    warps no longer realign every stage (ncu, ds41a T=6: ~10 % of the stage loop's warp samples are barrier stalls).
//  * XMOE_P_SYNC 1 (opt-in): no per-stage __syncthreads at all. mbarrier ring (full / empty per buffer, 512 arrivals
//    each): every thread refills the buffer D = NBUF - 2 stages ahead after waiting on its 'empty' barrier (all threads
//    finished the stage that used it), signals 'full' with cp.async.mbarrier.arrive.noinc (fires when its copies have
//    landed), waits on 'full' before reading. Warps may drift up to NBUF - 1 - D stages apart.
//  * the stage's global->shared copies are issued after the first k-block's shared loads (XMOE_P_CPOS 1), so the
//    decode's first LDS / ldmatrix are not queued behind the LDGSTS burst of all 16 warps at the stage start (ncu: the
//    copy block had 14.5 % of the samples for 10 % of the instructions, mio throttle).
//  * single-token-half path (jobs with <= 8 rows, all of decode): ldmatrix .x2 instead of .x4 for the activation
//    operand (rows 8-15 were loaded and never used): half the shared-memory read bandwidth of the x operand
//    (XMOE_P_X2 1). Same registers for rows 0-7, so the same HMMA operands.
// The three "@!PT LDS RZ, [RZ]" that ptxas puts in front of an LDGSTS burst stay (see the t header).

#include <utility>
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_gemm_inner_flat_t.cuh"   // FlatGemmJobs, xt_* helpers, XtMul1K, XtDq (decode, unchanged)

// every lambda of the loop is force-inlined: nvcc otherwise outlines some once the ring has > 4 stage bodies, and
// the by-reference captures (fragments, pointers) then go to the stack
#define XP_INL __attribute__((always_inline))

#ifndef XMOE_P_SYNC
#define XMOE_P_SYNC 0
#endif
#ifndef XMOE_P_G
#define XMOE_P_G 2
#endif
#if XMOE_P_SYNC == 0
#undef XMOE_P_NBUF
#define XMOE_P_NBUF (2 * XMOE_P_G)
#else
#ifndef XMOE_P_NBUF
#define XMOE_P_NBUF 4
#endif
#ifndef XMOE_P_D
#define XMOE_P_D (XMOE_P_NBUF - 2)
#endif
#endif
#ifndef XMOE_P_X2
#define XMOE_P_X2 1
#endif
#ifndef XMOE_P_CPOS
#define XMOE_P_CPOS 1
#endif
// XMOE_P_IMMA (opt-in, exact; K=3 MUL1 only): the 8 IDP.4A byte sums of a lane's 16x16 tile become 2 IMMA m16n8k32
// u8 (4 byte sums per lane each, see XpImmaSum). 0 = off, 1 = every k-block, 2 = even k-blocks only (half the IDP
// work stays on the FMA-heavy pipe, half goes to the tensor pipe).
#ifndef XMOE_P_IMMA
#define XMOE_P_IMMA 0
#endif

// ---- extra PTX helpers -------------------------------------------------------------------------------------------------
template <int OFF>
__device__ __forceinline__ void xp_ldsm2(uint32_t& r0, uint32_t& r1, uint32_t a)
{
    asm volatile ("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2+%3];\n" : "=r"(r0), "=r"(r1) : "r"(a), "n"(OFF));
}

__device__ __forceinline__ void xp_mbar_init(uint32_t bar, uint32_t count)
{
    asm volatile ("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void xp_mbar_inval(uint32_t bar)
{
    asm volatile ("mbarrier.inval.shared.b64 [%0];\n" :: "r"(bar) : "memory");
}

// arrive (release, CTA scope): this thread's earlier shared-memory reads of the buffer are done
__device__ __forceinline__ void xp_mbar_arrive(uint32_t bar)
{
    asm volatile ("mbarrier.arrive.shared.b64 _, [%0];\n" :: "r"(bar) : "memory");
}

// arrive-on that fires when all of this thread's earlier cp.async have landed (pending count not incremented: the
// barrier's init count includes these arrivals)
__device__ __forceinline__ void xp_cp_async_arrive(uint32_t bar)
{
    asm volatile ("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n" :: "r"(bar) : "memory");
}

// wait (acquire, CTA scope) until the phase with the given parity has completed. A fresh barrier is in phase 0, so
// waiting for parity 1 returns at once (the 'previous' phase counts as complete).
__device__ __forceinline__ void xp_mbar_wait(uint32_t bar, uint32_t parity)
{
    asm volatile ("{\n"
                  "  .reg .pred p;\n"
                  "XP_MBAR_WAIT:\n"
                  "  mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
                  "  @!p bra XP_MBAR_WAIT;\n"
                  "}\n" :: "r"(bar), "r"(parity) : "memory");
}

// ---- IMMA byte sums (XMOE_P_IMMA) -----------------------------------------------------------------------------------
// mma.m16n8k32.row.col.s32.u8.u8.s32 D = A B + C with a constant 0/1 block matrix B computes, for every lane, the byte
// sums of its own four A registers: lane (g = lane / 4, q = lane % 4) holds a0 = A[g][4q..4q+3], a1 = A[g+8][4q..],
// a2 = A[g][16+4q..], a3 = A[g+8][16+4q..] and receives d0 = D[g][2q], d1 = D[g][2q+1], d2 = D[g+8][2q],
// d3 = D[g+8][2q+1]. With B[k][j] = 1 iff k / 4 == pi(j), pi(2q) = q, pi(2q+1) = 4 + q:
//   d0 = sum bytes(a0), d1 = sum bytes(a2), d2 = sum bytes(a1), d3 = sum bytes(a3)   (+ c, here 0x6400 = IDP's acc).
// Exact integer arithmetic, so d is bit-identical to IDP.4A.U8.U8(a, 0x01010101, 0x6400). B fragment of lane (g, q):
// b0 = B[4q..4q+3][g] = 1 iff pi(g) == q, b1 = B[16+4q..][g] = 1 iff pi(g) == 4 + q.
struct XpImmaSum
{
    uint32_t b0, b1, c0, c1, c2, c3;
    __device__ __forceinline__ void init(int lane)
    {
        const int g = lane >> 2, q = lane & 3;
        const bool e = (g >> 1) == q;
        b0 = xt_pin(e && !(g & 1) ? 0x01010101u : 0u);
        b1 = xt_pin(e && (g & 1) ? 0x01010101u : 0u);
        c0 = xt_pin(0x6400u); c1 = xt_pin(0x6400u + 0u * 1); c2 = xt_pin(0x6400u + 0u * 2); c3 = xt_pin(0x6400u + 0u * 3);
    }
    // s[i] = 0x6400 + byte sum of p[i]
    __device__ __forceinline__ void sum4(const uint32_t (&p)[4], uint32_t (&s)[4]) const
    {
        asm ("mma.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
             : "=r"(s[0]), "=r"(s[2]), "=r"(s[1]), "=r"(s[3])
             : "r"(p[0]), "r"(p[1]), "r"(p[2]), "r"(p[3]), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2), "r"(c3));
        // a0 = p0 -> d0 = s0, a1 = p1 -> d2 = s1, a2 = p2 -> d1 = s2, a3 = p3 -> d3 = s3: output operand order d0, d1, d2, d3
        // is bound to s0, s2, s1, s3 above.
    }
};

// dq8<3, 2, 4> + decode_3inst_2<2> with the IDP byte sums done by two IMMAs: same window extraction (XtDq<3>), same
// products x * 0x83DCD12D, same PRMT pairing and HFMA2 as xt_decode2<2>, so the same fp16 weights.
template <class DQ>   // XtDq<3> (only selected for bits == 3, cb == 2; generic so the K=2 instance still parses)
__device__ __forceinline__ void xp_decode3_mul1_imma(const DQ& dq, uint32_t a, uint32_t b, FragB& frag0, FragB& frag1,
                                                     const XtMul1K& K, const XpImmaSum& M)
{
    const uint32_t w7 = fshift(b, a, dq.s);
    const uint32_t w6 = w7 >> 3;
    const uint32_t w5 = w6 >> 3;
    const uint32_t w4 = w5 >> 3;
    const uint32_t w3 = fshift(b, a, dq.s4);
    const uint32_t w2 = w3 >> 3;
    const uint32_t w1 = w2 >> 3;
    const uint32_t w0 = w1 >> 3;
    uint32_t p0[4] = { (w0 & 0xffff) * 0x83DCD12Du, (w1 & 0xffff) * 0x83DCD12Du, (w2 & 0xffff) * 0x83DCD12Du, (w3 & 0xffff) * 0x83DCD12Du };
    uint32_t p1[4] = { (w4 & 0xffff) * 0x83DCD12Du, (w5 & 0xffff) * 0x83DCD12Du, (w6 & 0xffff) * 0x83DCD12Du, (w7 & 0xffff) * 0x83DCD12Du };
    uint32_t s0[4], s1[4];
    M.sum4(p0, s0);
    M.sum4(p1, s1);
    auto pair = [&] (uint32_t lo, uint32_t hi) -> half2
    {
        half_uint16 h0((uint16_t) lo);
        half_uint16 h1((uint16_t) hi);
        return __hfma2(__halves2half2(h0.as_half, h1.as_half), K.k_inv_h2, K.k_bias_h2);
    };
    frag0[0] = pair(s0[0], s0[1]);
    frag0[1] = pair(s0[2], s0[3]);
    frag1[0] = pair(s1[0], s1[1]);
    frag1[1] = pair(s1[2], s1[3]);
}

// sequential composition with early exit: f(0) && f(1) && ... && f(N-1)
template <typename F, int... I>
__device__ __forceinline__ bool xp_and_seq(F& f, std::integer_sequence<int, I...>) { return (f(std::integral_constant<int, I>{}) && ...); }
template <int N, typename F>
__device__ __forceinline__ bool xp_and(F&& f) { return xp_and_seq(f, std::make_integer_sequence<int, N>{}); }

template<EXL3_GEMM_T_ARGS>
__forceinline__ __device__   // both GEMM phases inlined, as in ds41t (an outlined call puts FlatGemmJobs on the stack)
void exl3_gemm_kernel_inner_flat_p
(
    const FlatGemmJobs& jobs,
    const int size_k,
    const int size_n,
    int* __restrict__ locks
)
{
    static_assert(!c_fp32, "xmoe flat: fp16 output only");
    constexpr int NBUF = SH_STAGES;                                          // stage buffers (XMOE_P_NBUF)
    constexpr bool MBAR = XMOE_P_SYNC == 1;
#if XMOE_P_SYNC == 0
    constexpr int G = XMOE_P_G;                                              // stages per barrier group
    static_assert(NBUF == 2 * G && G >= 1 && G <= 4, "xmoe ds41p: ring of 2 * G buffers");
    constexpr int D = G;                                                     // copy distance (stages ahead)
#else
    constexpr int G = 1;
    constexpr int D = XMOE_P_D;
    static_assert(NBUF >= 3 && NBUF <= 8 && D >= 1 && D <= NBUF - 2, "xmoe ds41p: mbarrier ring needs slack >= 1");
#endif
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

    // ---- mbarrier ring (XMOE_P_SYNC 1): full[b] at +8b, empty[b] at +8 (NBUF + b) -------------------------------------
    __shared__ __align__(8) uint64_t xp_bars[MBAR ? 2 * NBUF : 1];
    const uint32_t bar_full = (uint32_t) __cvta_generic_to_shared(xp_bars);
    const uint32_t bar_empty = bar_full + 8 * NBUF;
    if constexpr (MBAR)
    {
        __syncthreads();                       // previous user of the barrier words (the other GEMM phase) is done
        if (T == 0)
        {
            #pragma unroll
            for (int b = 0; b < 2 * NBUF; ++b) xp_mbar_init(bar_full + 8 * b, NT);
        }
        __syncthreads();
    }

    // First column of the block: col = q * tiles_n + n, q = job index over both matrices (q >= num_jobs: matrix 1)
    const int col_beg = slice_beg / tiles_k;
    const int k_beg = slice_beg % tiles_k;

    const int gl_a_stride_k = TILESIZE_K;
    const int gl_b_stride_k = blocks_n_full * TILEBLOCKS_K * 16 * bits;
    constexpr int gl_b_stride_n = TILEBLOCKS_N * 16 * bits;

    // Copy assignment as in ds41a (one 16-byte copy of B per thread j = T < NB4, one of A per thread j = NT - 1 - T < NA4)
    constexpr int NA4 = TILESIZE_M * TILESIZE_K / 8;
    constexpr int NB4 = sh_b_stage_size / 8;
    static_assert(NA4 <= NT && NB4 <= NT, "xmoe ds41p: at most one A and one B 16-byte copy per thread and stage");
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

    // ---- pipe 0: global -> shared (verbatim t) ----
    int ld_n = col_beg % tiles_n, ld_q = col_beg / tiles_n, ld_k = k_beg;
    int ld_left = slice_len;   // slices not yet assigned to a copy segment
    int ld_run;                // copies left in the current segment
    const int4* gA;
    const int4* gB;
    bool pA, pB;
    auto ld_set = [&] () XP_INL
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

    auto ld_issue = [&] (auto lbuf) XP_INL
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

    // ---- pipe 2: MMA and output (verbatim t) ----
    int cp_col = col_beg, cp_n = col_beg % tiles_n, cp_q = col_beg / tiles_n, cp_k = k_beg;   // cp_k = first k of the segment
    int cp_left = slice_len;
    int cp_seg, cp_run, rows2;
    bool two_halves;
    half* gl_c_ptr_16;
    auto cp_set = [&] () XP_INL
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

    // ldmatrix x4 over the token tile: m0 = rows 0-7 k 0-7, m1 = rows 0-7 k 8-15, m2/m3 = rows 8-15 -> B' fragments for both
    // halves. ldmatrix x2 uses the addresses of lanes 0-15 only (m0, m1): the same x_addr, rows 0-7.
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
    constexpr bool IMMA = XMOE_P_IMMA && bits == 3 && cb == 2;
    XpImmaSum M;
    if constexpr (IMMA) M.init(lane_id);
    const uint32_t bw_base = sm_b + (sub_k * KPS * TILEBLOCKS_N + b_warp_off) * 32 * bits;
    const uint32_t bw0 = xt_pin(bw_base + 4 * dql.wi0);
    const uint32_t bw1 = xt_pin(bw_base + 4 * dql.wi1);

    // One stage's MMAs on ring buffer CB. hook() (the refill of another buffer) runs after the first k-block's shared
    // loads have been issued (XMOE_P_CPOS 1) or before everything (0).
    auto compute = [&] (auto cbuf, auto hic, auto& hook) XP_INL
    {
        constexpr int CB = decltype(cbuf)::value;
        constexpr bool HI = decltype(hic)::value;
        constexpr bool X2 = XMOE_P_X2 && !HI;
        if constexpr (!XMOE_P_CPOS) hook();
        xt_unroll<KPS>([&] (auto kbc) XP_INL
        {
            constexpr int kb = decltype(kbc)::value;
            FragB x0, x1;
            if constexpr (X2)
            {
                uint32_t* xr = reinterpret_cast<uint32_t*>(&x0);
                xp_ldsm2<CB * SA_BYTES>(xr[0], xr[1], x_addr[kb]);
            }
            else
            {
                FragA xs;
                xt_ldsm4<CB * SA_BYTES>(xs, x_addr[kb]);
                x0[0] = xs[0]; x0[1] = xs[1]; x1[0] = xs[2]; x1[1] = xs[3];
            }
            uint32_t wa0[NB16], wa1[NB16];
            xt_unroll<NB16>([&] (auto cc) XP_INL
            {
                constexpr int c = decltype(cc)::value;
                constexpr int OFF = CB * SB_BYTES + (kb * TILEBLOCKS_N + c) * 32 * bits;
                wa0[c] = xt_lds32<OFF>(bw0);
                wa1[c] = xt_lds32<OFF>(bw1);
            });
            if constexpr (XMOE_P_CPOS && kb == 0) hook();
            xt_unroll<NB16>([&] (auto cc) XP_INL
            {
                constexpr int c = decltype(cc)::value;
                FragB w0, w1;
                if constexpr (IMMA && (XMOE_P_IMMA == 1 || kb % 2 == 0)) xp_decode3_mul1_imma(dql, wa0[c], wa1[c], w0, w1, K, M);
                else dql.template decode<cb, false>(wa0[c], wa1[c], w0, w1, K);
                FragA wa;   // B fragments of columns 0-7 / 8-15 are the row 0-7 / 8-15 halves of the transposed A fragment
                wa[0] = w0[0]; wa[1] = w1[0]; wa[2] = w0[1]; wa[3] = w1[1];
                ptx_mma_m16n8k16(wa, x0, frag_c[0][c]);
                if constexpr (HI) ptx_mma_m16n8k16(wa, x1, frag_c[1][c]);
            });
        });
    };

    auto clear_frag_c = [&] () XP_INL
    {
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int c = 0; c < NB16; ++c)
                frag_c[h][c] = {};
    };

    // ---- reduce: verbatim ds41a (threadblock_reduce, read_sum_gl, write_sum_gl, lock protocol) ----
    auto threadblock_reduce = [&] () XP_INL
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
    auto read_sum_gl = [&] () XP_INL
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

    auto write_sum_gl = [&] () XP_INL
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
    auto reduce_advance = [&] () XP_INL -> bool
    {
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

    // mbarrier phase parity of the current ring trip: stage s on buffer s % NBUF is use s / NBUF of that buffer
    uint32_t rpar = 0;

    // ---- one stage on ring buffer CB (compile time; stage index s = CB mod NBUF always). Returns 0 = continue,
    // 1 = slice range done, 2 = the next segment needs the other HI instance.
    auto stage = [&] (auto cbuf, auto hic) XP_INL -> int
    {
        constexpr int CB = decltype(cbuf)::value;
        constexpr bool HI = decltype(hic)::value;
        constexpr int BC = (CB + D) % NBUF;                  // buffer refilled in this stage (stage s + D)
        if constexpr (!MBAR)
        {
            if constexpr (CB % G == 0)
            {
                cp_async_wait<0>();                          // this thread's copies for the whole group have landed
                __syncthreads();                             // everybody's have, and everybody finished group g - 1
            }
            auto hook = [&] () XP_INL { ld_issue(std::integral_constant<int, BC>{}); cp_async_fence(); };
            compute(cbuf, hic, hook);
        }
        else
        {
            // stage s + D is use (s + D) / NBUF = trip + CU of buffer BC; wait until use - 1 has been consumed by all
            constexpr uint32_t CU = (CB + D) / NBUF;
            auto hook = [&] () XP_INL
            {
                // past the end of the range ld_issue copies nothing (pA = pB = false) but the arrival still happens:
                // the GEMM end waits for those trailing phases before invalidating the barriers
                xp_mbar_wait(bar_empty + 8 * BC, rpar ^ ((CU + 1) & 1));
                ld_issue(std::integral_constant<int, BC>{});
                xp_cp_async_arrive(bar_full + 8 * BC);
            };
            xp_mbar_wait(bar_full + 8 * CB, rpar);           // everybody's copies for stage s have landed
            compute(cbuf, hic, hook);
            xp_mbar_arrive(bar_empty + 8 * CB);              // this thread is done reading buffer CB
            if constexpr (CB == NBUF - 1) rpar ^= 1;
        }
        if (--cp_run == 0)
        {
            if (reduce_advance()) return 1;
            if (two_halves != HI) return 2;
        }
        return 0;
    };

    // Runs the ring with a fixed HI from buffer `slot` until the range is done (-1) or HI changes (returns the buffer of
    // the next stage). Entry at slot > 0 (after a HI change) first runs buffers slot .. NBUF - 1 once (no jump into the
    // loop); the loop itself is the unrolled ring.
    auto run = [&] (auto hic, int slot) XP_INL -> int
    {
        int nxt = -1;
        auto st = [&] (auto bc) XP_INL -> bool
        {
            constexpr int B = decltype(bc)::value;
            const int r = stage(bc, hic);
            if (r) { nxt = r == 1 ? -1 : (B + 1) % NBUF; return false; }
            return true;
        };
        if (slot != 0)
        {
            const bool cont = xp_and<NBUF>([&] (auto bc) XP_INL -> bool
            {
                constexpr int B = decltype(bc)::value;
                if constexpr (B == 0) return true;
                else return slot > B || st(bc);
            });
            if (!cont) return nxt;
        }
        while (xp_and<NBUF>(st)) {}
        return nxt;
    };

    // ---- prologue: copies of stages 0 .. D - 1 ----
    xt_unroll<D>([&] (auto bc) XP_INL
    {
        if constexpr (!MBAR)
        {
            ld_issue(bc);
            cp_async_fence();
        }
        else
        {
            ld_issue(bc);                                    // first use of buffers 0 .. D - 1: no 'empty' wait
            xp_cp_async_arrive(bar_full + 8 * decltype(bc)::value);
        }
    });

    clear_frag_c();

    int slot = 0;
    do slot = two_halves ? run(std::true_type{}, slot) : run(std::false_type{}, slot);
    while (slot >= 0);

    if constexpr (MBAR)
    {
        // Drain: stages slice_len .. slice_len + D - 1 got (copy-less) arrivals on 'full' from the refill hooks of the last
        // D stages (and from the prologue when slice_len < D). Wait for those phases so that no cp.async arrive-on is
        // still pending, then everybody is past every wait and thread 0 invalidates the ring.
        for (int c = slice_len; c < slice_len + D; ++c) xp_mbar_wait(bar_full + 8 * (c % NBUF), (uint32_t) (c / NBUF) & 1);
        __syncthreads();
        if (T == 0)
        {
            #pragma unroll
            for (int b = 0; b < 2 * NBUF; ++b) xp_mbar_inval(bar_full + 8 * b);
        }
    }
}
