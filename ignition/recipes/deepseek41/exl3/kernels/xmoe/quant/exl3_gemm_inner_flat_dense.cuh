#pragma once
// Beam round 6 (2026-10-18): dense variant of the swapped-operand flat GEMM loop (exl3_gemm_inner_flat.cuh) for the decode-size
// EXL3 dense projections (attention q/k/v/gate fused, o_proj, dense MLP). Jobs = (column slice of a source matrix, <= 16-row chunk):
// each job has its own A (the source's input-Hadamard rows), trellis slice base, C (contiguous rows x slice width) and the source
// matrix width (trellis k-row stride). Same main loop, split-K lock protocol and swapped MMA operands as the MoE flat kernel.
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_gemm_inner_flat.cuh"

struct FlatDenseJobs
{
    int num_jobs;
    const int64_t* a_ptr;   // [num_jobs] fp16 A, row stride size_k
    const int64_t* b_ptr;   // [num_jobs] trellis of the slice's first n-block
    const int64_t* c_ptr;   // [num_jobs] fp16 C, rows x size_n (row stride size_n)
    const int* b_bnf;       // [num_jobs] source matrix width / 16
    const int* rows;        // [num_jobs] 1..16
};

template<EXL3_GEMM_T_ARGS>
inline __device__
void exl3_gemm_kernel_inner_flat_dense
(
    const FlatDenseJobs& jobs,
    const int size_k,
    const int size_n,
    int* __restrict__ locks
)
{
    static_assert(!c_fp32, "xmoe flat: fp16 output only");
    constexpr int TILEBLOCKS_K = TILESIZE_K / 16;
    constexpr int TILEBLOCKS_N = TILESIZE_N / 16;
    constexpr int KPS = TILEBLOCKS_K / 2;                                    // k-blocks per sub_k group
    constexpr int NT = 2 * EXL3_GEMM_BASE_THREADS;                           // threads per block
    constexpr int FRAGS_N_PER_WARP = 2 * TILEBLOCKS_N / (EXL3_GEMM_BASE_THREADS / 32);

    constexpr int sh_a_stage_size = TILESIZE_M * TILESIZE_K;                 // halfs
    constexpr int sh_b_stage_size = TILEBLOCKS_K * TILEBLOCKS_N * 16 * bits; // uint16s
    constexpr int sh_c_size = 4 * EXL3_GEMM_BASE_THREADS * FRAGS_N_PER_WARP; // floats

    constexpr int A_COLS = TILESIZE_K / 8;
    constexpr int A_SWIZZLE_MASK = A_COLS - 1;
    constexpr int A_SWIZZLE_SHIFT = A_COLS <= 2 ? 2 : (A_COLS == 4 ? 1 : 0);   // 128-byte rows: xor with the row itself

    static_assert(EXL3_GEMM_BASE_THREADS == 256);
    static_assert(TILESIZE_M == 16, "Invalid kernel params");
    static_assert(TILEBLOCKS_K % 2 == 0, "xmoe flat: two sub_k groups");
    static_assert(TILESIZE_N % 128 == 0, "Invalid kernel params");
    static_assert(FRAG_STAGES == 1, "xmoe_k64: fragments are consumed in the stage that decodes them");
    static_assert(EXL3_GEMM_H_ACC == 0, "xmoe flat: fp16 accumulation not supported");
    static_assert(SMEM_MAX >= SH_STAGES * (2 * sh_a_stage_size + 2 * sh_b_stage_size) + 4 * sh_c_size,
                  "Invalid kernel params (insufficient shared memory for shape)");

    extern __shared__ half shared[];
    half* sh_a = shared;
    uint16_t* sh_b = (uint16_t*) (sh_a + SH_STAGES * sh_a_stage_size);
    float* sh_c = (float*) (sh_b + sh_b_stage_size * SH_STAGES);

    const int T = threadIdx.x;
    int t = threadIdx.x % EXL3_GEMM_BASE_THREADS;
    int sub_k = threadIdx.x / EXL3_GEMM_BASE_THREADS;
    int warp_id = t / 32;
    int lane_id = t % 32;

    const int tiles_k = size_k / TILESIZE_K;
    const int tiles_n = size_n / TILESIZE_N;
    const int total_cols = jobs.num_jobs * tiles_n;
    const int total_slices = tiles_k * total_cols;

    const int num_slices = gridDim.x;
    const int slice_beg = (int) ((int64_t) total_slices * blockIdx.x / num_slices);
    const int slice_end = (int) ((int64_t) total_slices * (blockIdx.x + 1) / num_slices);
    const int slice_len = slice_end - slice_beg;
    if (slice_len < 1) return;

    auto job_of = [&] (int col, int& j, int& n)
    {
        n = col % tiles_n;
        j = col / tiles_n;
    };

    // Pipe 0: global -> shared
    int slice0_k = slice_beg % tiles_k;
    int col0 = slice_beg / tiles_k;
    int slice0_iters = slice_len;

    const int gl_a_stride_k = TILESIZE_K;
    int gl_b_stride_k = 0;   // per job: the source matrix width sets the trellis k-row stride
    constexpr int gl_b_stride_n = TILEBLOCKS_N * 16 * bits;

    const half* gl_a_ptr;
    const uint16_t* gl_b_ptr;
    int rows0;
    int bnf0 = 0;
    auto set_col0 = [&] ()
    {
        int j, n;
        job_of(col0, j, n);
        rows0 = jobs.rows[j];
        bnf0 = jobs.b_bnf[j];
        gl_b_stride_k = bnf0 * TILEBLOCKS_K * 16 * bits;
        gl_a_ptr = (const half*) jobs.a_ptr[j] + slice0_k * gl_a_stride_k;
        gl_b_ptr = (const uint16_t*) jobs.b_ptr[j] + (size_t) slice0_k * gl_b_stride_k + n * gl_b_stride_n;
    };
    set_col0();

    // Balanced copy assignment: B int4 j = T, A int4 j = NT - 1 - T (B and A land on disjoint threads when they fit)
    constexpr int NA4 = TILESIZE_M * TILESIZE_K / 8;
    constexpr int NB4 = sh_b_stage_size / 8;
    constexpr int load_a_iters = (NA4 + NT - 1) / NT;
    constexpr int load_b_iters = (NB4 + NT - 1) / NT;
    int load_a_m[load_a_iters], load_a_gl[load_a_iters], load_a_sh[load_a_iters];
    bool pred_a[load_a_iters];
    #pragma unroll
    for (int i = 0; i < load_a_iters; ++i)
    {
        int j = i * NT + (NT - 1 - T);
        int k = j % A_COLS;
        int m = j / A_COLS;
        pred_a[i] = j < NA4;
        load_a_m[i] = m;
        load_a_gl[i] = m * size_k / 8 + k;
        load_a_sh[i] = m * A_COLS + (k ^ ((m >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK));
    }
    int load_b_gl[load_b_iters], load_b_sh[load_b_iters], load_b_k[load_b_iters], load_b_n[load_b_iters];
    bool pred_b[load_b_iters];
    #pragma unroll
    for (int i = 0; i < load_b_iters; ++i)
    {
        int j = i * NT + T;
        int n = j % (gl_b_stride_n / 8);
        int k = j / (gl_b_stride_n / 8);
        load_b_k[i] = k; load_b_n[i] = n;
        load_b_gl[i] = k * (bnf0 * 16 * bits / 8) + n;
        load_b_sh[i] = j;
        pred_b[i] = j < NB4;
    }

    int ld_stage = 0;
    auto async_load_gl = [&] ()
    {
        if (slice0_iters)
        {
            int4* sha = (int4*) (sh_a + ld_stage * sh_a_stage_size);
            int4* shb = (int4*) (sh_b + ld_stage * sh_b_stage_size);
            const int4* gla = (const int4*) gl_a_ptr;
            const int4* glb = (const int4*) gl_b_ptr;
            #pragma unroll
            for (int i = 0; i < load_b_iters; ++i)
                if (pred_b[i]) cp_async(shb + load_b_sh[i], glb + load_b_gl[i]);
            #pragma unroll
            for (int i = 0; i < load_a_iters; ++i)
                if (pred_a[i] && load_a_m[i] < rows0) cp_async(sha + load_a_sh[i], gla + load_a_gl[i]);

            slice0_k++;
            slice0_iters--;
            if (slice0_k >= tiles_k)
            {
                slice0_k = 0;
                col0++;
                if (slice0_iters)
                {
                    set_col0();
                    #pragma unroll
                    for (int i = 0; i < load_b_iters; ++i) load_b_gl[i] = load_b_k[i] * (bnf0 * 16 * bits / 8) + load_b_n[i];
                }
            }
            else
            {
                gl_a_ptr += gl_a_stride_k;
                gl_b_ptr += gl_b_stride_k;
            }
        }
        if (++ld_stage == SH_STAGES) ld_stage = 0;
        cp_async_fence();
    };

    // Pipe 2: MMA and output
    int slice2_k = slice0_k;
    int slice2_k0 = slice0_k;
    int col2 = col0;
    int slice2_iters = slice0_iters;

    half* gl_c_ptr_16;
    int rows2;
    auto set_col2 = [&] ()
    {
        int j, n;
        job_of(col2, j, n);
        rows2 = jobs.rows[j];
        gl_c_ptr_16 = (half*) jobs.c_ptr[j] + n * TILESIZE_N;
    };
    set_col2();

    // xmoe_swap: operands swapped (C'[col][token] = W^T X^T): the decoded 16x16 weight tile is the m16k16 A operand
    // and <= 8 tokens are the n8 B operand, so one HMMA covers 256 weights instead of 128 when a job has <= 8 rows.
    constexpr int NB16 = FRAGS_N_PER_WARP / 2;   // 16-column blocks per warp
    register FragC frag_c[2][NB16];               // [token half][column block]

    auto advance2 = [&] ()
    {
        slice2_k++;
        slice2_iters--;
        if (slice2_k >= tiles_k)
        {
            slice2_k = 0;
            slice2_k0 = 0;
            col2++;
            if (slice2_iters) set_col2();
        }
    };

    // ldmatrix x4 over the token tile: m0 = rows 0-7 k 0-7, m1 = rows 0-7 k 8-15, m2/m3 = rows 8-15 -> B' fragments for both halves
    const int x_r = (lane_id % 8) + 8 * (lane_id / 16);
    const int x_c = (lane_id / 8) % 2;
    const int x_sw = (x_r >> A_SWIZZLE_SHIFT) & A_SWIZZLE_MASK;
    const int x_row_off = x_r * A_COLS;
    const int b_warp_off = warp_id * NB16;
    int cp_stage = 0;
    auto compute = [&] ()
    {
        const bool hi = rows2 > 8;
        const int4* sha = (const int4*) (sh_a + cp_stage * sh_a_stage_size);
        const uint16_t* shb0 = sh_b + cp_stage * sh_b_stage_size;
        #pragma unroll
        for (int kb = 0; kb < KPS; ++kb)
        {
            const int kbi = sub_k * KPS + kb;
            FragA xs;
            ldsm4(xs, sha + x_row_off + ((x_c + kbi * 2) ^ x_sw));
            FragB x0, x1;
            x0[0] = xs[0]; x0[1] = xs[1]; x1[0] = xs[2]; x1[1] = xs[3];
            #pragma unroll
            for (int c = 0; c < NB16; ++c)
            {
                const uint32_t* shb = (const uint32_t*) (shb0 + (kbi * TILEBLOCKS_N + b_warp_off + c) * 16 * bits);
                FragB w0, w1;
                dq_dispatch<bits, cb>(shb, lane_id << 3, w0, w1);
                FragA wa;   // B fragments of columns 0-7 / 8-15 are the row 0-7 / 8-15 halves of the transposed A fragment
                wa[0] = w0[0]; wa[1] = w1[0]; wa[2] = w0[1]; wa[3] = w1[1];
                ptx_mma_m16n8k16(wa, x0, frag_c[0][c]);
                if (hi) ptx_mma_m16n8k16(wa, x1, frag_c[1][c]);
            }
        }
        if (++cp_stage == SH_STAGES) cp_stage = 0;
    };

    auto clear_frag_c = [&] ()
    {
        #pragma unroll
        for (int h = 0; h < 2; ++h)
            #pragma unroll
            for (int c = 0; c < NB16; ++c)
                frag_c[h][c] = {};
    };

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

    auto reduce = [&] ()
    {
        threadblock_reduce();

        // Partial slices within a column are reduced in reverse k order (see exl3_gemm_kernel_inner)
        int lock_i = tiles_k - slice2_k - 1;
        int lock_d = slice2_k - slice2_k0 + 1;
        int* lock = &locks[col2];

        barrier_acquire(lock, lock_i);

        bool first = lock_i == 0;
        bool last = lock_i + lock_d == tiles_k;

        if (!sub_k && !first) read_sum_gl();
        if (!sub_k) write_sum_gl();

        barrier_release(lock, lock_d, last);

        clear_frag_c();
    };

    #pragma unroll
    for (int i = 0; i < SH_STAGES - 1; ++i)
        async_load_gl();

    clear_frag_c();
    while (true)
    {
        cp_async_wait<SH_STAGES - 2>();
        __syncthreads();
        async_load_gl();
        compute();
        if (slice2_k == tiles_k - 1 || slice2_iters == 1) { reduce(); slice2_k0 = slice2_k + 1; }
        advance2();
        if (!slice2_iters) break;
    }
}
