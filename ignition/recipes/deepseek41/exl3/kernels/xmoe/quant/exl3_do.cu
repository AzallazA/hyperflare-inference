// Copyright (c) 2026 Diffbot. MIT License (kernels/xmoe/LICENSE.diffbot). Uses device code from ExLlamaV3 (MIT, Copyright
// (c) 2025 Turboderp, kernels/xmoe/LICENSE.exllamav3): the EXL3 trellis decoder (dq_dispatch), mma/ldmatrix/cp.async
// helpers and the 128-point Hadamard device functions.
// Ported from diffbot recipe/kernels/xmoe_do/do_gemm.cu (2026-10-01, "decode-once grouped GEMM for EXL3 MoE experts
// with many rows") into Flare's xmoe extension by exp/ds41-exl3 (2026-11-16, Flare community contributors):
//   - capture-safe: the job table has a FIXED capacity (job_capacity entries; unused entries have 0 rows and exit at once)
//     and is built ON THE DEVICE from the routing counts (xmoe_do_build_jobs), so the launch grids never depend on data
//     and no host sync happens; the pointer tables are gathered on the device too;
//   - split launches: only experts with slot < hot_rows (VRAM rows of a parked bank) get jobs; rows whose expert is cold
//     carry row_exp = -1 and are skipped by every kernel here (the group kernel handles them in the second launch);
//   - counts may be negative (the split-launch convention of exl3_moe: |count| rows, expert handled elsewhere).
// do_gemm_kernel is diffbot's kernel with one change: an early return for empty jobs. The gather/guad/scatter kernels
// are theirs with the row_exp < 0 skip.
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_dq.cuh"
#include "exl3_moe_common.cuh"
#include "hadamard_inner.cuh"

#define DO_MC 128          // rows per job
#define DO_HSCALE 0.088388347648f   // 1 / sqrt(128)

template <int bits, int cb, int SK, int MC, bool OUT_HALF>
__global__ __launch_bounds__(256, 2)
void xmoe_do_gemm_kernel(const half* __restrict__ A, void* __restrict__ Cv,
                         const int* __restrict__ job_row0, const int* __restrict__ job_rows,
                         const int64_t* __restrict__ job_trellis, int K, int N)
{
    constexpr int KB = SK / 16;
    constexpr int MCH = MC / 16;
    constexpr int SKP = SK + 8;
    const int job = blockIdx.y;
    const int rows = job_rows[job];
    if (rows <= 0) return;                       // exp/ds41-exl3: fixed-capacity job table, empty entry
    const int n_tile = blockIdx.x;
    const int row0 = job_row0[job];
    const uint16_t* trellis = reinterpret_cast<const uint16_t*>(job_trellis[job]);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int blocks_n = N / 16;
    const int nb = n_tile * 8 + warp;
    const int mch = (rows + 15) / 16;
    const int slabs = K / SK;
    float* C = (float*) Cv; half* Ch = (half*) Cv;

    __shared__ __align__(16) half sa[2][16 * SKP];

    FragC c[MCH][2];
    #pragma unroll
    for (int i = 0; i < MCH; ++i) { c[i][0] = {}; c[i][1] = {}; }

    auto load_a = [&](int buf, int slab, int mc) {
        constexpr int N16 = 16 * SK / 8;
        #pragma unroll
        for (int i = threadIdx.x; i < N16; i += 256) {
            const int r = i / (SK / 8), c8 = i % (SK / 8);
            const int grow = mc * 16 + r;
            half* dst = &sa[buf][r * SKP + c8 * 8];
            if (grow < rows) cp_async(dst, A + (size_t)(row0 + grow) * K + (size_t)slab * SK + c8 * 8);
            else *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
        }
        cp_async_fence();
    };

    FragB fb[KB][2];
    int buf = 0;
    load_a(0, 0, 0);
    for (int s = 0; s < slabs; ++s)
    {
        #pragma unroll
        for (int kb = 0; kb < KB; ++kb)
        {
            const size_t kblk = (size_t)s * KB + kb;
            const uint32_t* tp = reinterpret_cast<const uint32_t*>(trellis + (kblk * blocks_n + nb) * 16 * bits);
            dq_dispatch<bits, cb>(tp, lane << 3, fb[kb][0], fb[kb][1]);
        }
        #pragma unroll
        for (int mc = 0; mc < MCH; ++mc)
        {
            if (mc >= mch) break;
            int nmc = mc + 1, ns = s;
            if (nmc >= mch) { nmc = 0; ns = s + 1; }
            if (ns < slabs) load_a(buf ^ 1, ns, nmc); else cp_async_fence();
            cp_async_wait<1>();
            __syncthreads();
            const int r = (lane % 8) + 8 * ((lane / 8) % 2);
            #pragma unroll
            for (int kb = 0; kb < KB; ++kb)
            {
                FragA fa;
                ldsm4(fa, &sa[buf][r * SKP + kb * 16 + (lane / 16) * 8]);
                ptx_mma_m16n8k16(fa, fb[kb][0], c[mc][0]);
                ptx_mma_m16n8k16(fa, fb[kb][1], c[mc][1]);
            }
            __syncthreads();
            buf ^= 1;
        }
    }

    const int col0 = nb * 16 + 2 * (lane % 4);
    #pragma unroll
    for (int mc = 0; mc < MCH; ++mc)
    {
        if (mc >= mch) break;
        #pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const float* v = reinterpret_cast<const float*>(&c[mc][h]);
            const int ra = mc * 16 + lane / 4, rb = ra + 8;
            const int col = col0 + h * 8;
            if constexpr (OUT_HALF)
            {
                if (ra < rows) *reinterpret_cast<half2*>(Ch + (size_t)(row0 + ra) * N + col) = __floats2half2_rn(v[0], v[1]);
                if (rb < rows) *reinterpret_cast<half2*>(Ch + (size_t)(row0 + rb) * N + col) = __floats2half2_rn(v[2], v[3]);
            }
            else
            {
                if (ra < rows) *reinterpret_cast<float2*>(C + (size_t)(row0 + ra) * N + col) = make_float2(v[0], v[1]);
                if (rb < rows) *reinterpret_cast<float2*>(C + (size_t)(row0 + rb) * N + col) = make_float2(v[2], v[3]);
            }
        }
    }
}

static void launch_do_gemm(const half* A, int K, void* C, bool out_half, int N, const int* job_row0, const int* job_rows,
                           const int64_t* job_trellis, int J, int bits, int cb, cudaStream_t stream)
{
    TORCH_CHECK(K % 128 == 0 && N % 128 == 0, "xmoe_do: K, N must be multiples of 128");
    dim3 grid(N / 128, J);
    #define LAUNCH(B, CB, OH) xmoe_do_gemm_kernel<B, CB, 128, DO_MC, OH><<<grid, 256, 0, stream>>>(A, C, job_row0, job_rows, job_trellis, K, N)
    if (bits == 3 && cb == 2) { if (out_half) LAUNCH(3, 2, true); else LAUNCH(3, 2, false); }
    else if (bits == 3 && cb == 1) { if (out_half) LAUNCH(3, 1, true); else LAUNCH(3, 1, false); }
    else TORCH_CHECK(false, "xmoe_do: unsupported bits/codebook (3-bit MUL1 or MCG only)");
    #undef LAUNCH
}

// One block: exclusive scans over the experts' |counts| (row offsets in the expert-sorted order) and over the jobs of
// the hot experts (slot < hot_rows, count > 0: ceil(count / 128) jobs each), then fills the fixed-capacity job table
// (row0, rows, expert) and the three trellis pointers per job; entries past the last job get 0 rows.
__global__ void xmoe_do_build_jobs(const int64_t* __restrict__ counts, int E, int hot_rows, int capacity,
                                   int* __restrict__ job_row0, int* __restrict__ job_rows, int* __restrict__ job_exp,
                                   const int64_t* __restrict__ g_tr, const int64_t* __restrict__ u_tr, const int64_t* __restrict__ d_tr,
                                   int64_t* __restrict__ jg, int64_t* __restrict__ ju, int64_t* __restrict__ jd,
                                   int* __restrict__ overflow)
{
    extern __shared__ int sh[];                  // [2 * blockDim.x]: row offsets, job offsets
    const int t = threadIdx.x;
    const int per = (E + blockDim.x - 1) / blockDim.x;    // experts per thread (contiguous)
    int rows_sum = 0, jobs_sum = 0;
    for (int i = 0; i < per; ++i)
    {
        const int e = t * per + i;
        if (e >= E) break;
        int64_t c = counts[e]; if (c < 0) c = -c;
        rows_sum += (int) c;
        if (e < hot_rows && counts[e] > 0) jobs_sum += (int) ((c + DO_MC - 1) / DO_MC);
    }
    sh[t] = rows_sum; sh[blockDim.x + t] = jobs_sum;
    __syncthreads();
    for (int off = 1; off < blockDim.x; off <<= 1)    // inclusive Hillis-Steele scan of both arrays
    {
        int a = t >= off ? sh[t - off] : 0, b = t >= off ? sh[blockDim.x + t - off] : 0;
        __syncthreads();
        sh[t] += a; sh[blockDim.x + t] += b;
        __syncthreads();
    }
    int row = sh[t] - rows_sum, job = sh[blockDim.x + t] - jobs_sum;
    const int total_jobs = sh[2 * blockDim.x - 1];
    for (int i = 0; i < per; ++i)
    {
        const int e = t * per + i;
        if (e >= E) break;
        int64_t c = counts[e]; const bool hot = e < hot_rows && c > 0; if (c < 0) c = -c;
        if (hot)
        {
            for (int r = 0; r < (int) c; r += DO_MC)
            {
                if (job < capacity)
                {
                    job_row0[job] = row + r; job_rows[job] = min(DO_MC, (int) c - r); job_exp[job] = e;
                    jg[job] = g_tr[e]; ju[job] = u_tr[e]; jd[job] = d_tr[e];
                }
                ++job;
            }
        }
        row += (int) c;
    }
    for (int j = total_jobs + t; j < capacity; j += blockDim.x)
    {
        job_row0[j] = 0; job_rows[j] = 0; job_exp[j] = 0; jg[j] = g_tr[0]; ju[j] = u_tr[0]; jd[j] = d_tr[0];
    }
    if (t == 0) overflow[0] = total_jobs > capacity ? total_jobs : 0;
}

__global__ void xmoe_do_gather_had(const half* __restrict__ x, const int64_t* __restrict__ tok, const int* __restrict__ row_exp,
                                   const int64_t* __restrict__ suh_g, const int64_t* __restrict__ suh_u, half* __restrict__ xg,
                                   half* __restrict__ xu, int R, int hidden, bool gated)
{
    const int wpt = hidden / 128;
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (w >= R * wpt) return;
    const int r = w / wpt, off = w % wpt, e = row_exp[r];
    if (e < 0) return;                           // cold row: the group kernel's launch
    const half* in = x + tok[r] * hidden + off * 128;
    if (gated) had_hf_r_128_inner<true, false>(in, xg + (size_t)r * hidden + off * 128, (const half*) suh_g[e] + off * 128, DO_HSCALE);
    had_hf_r_128_inner<true, false>(in, xu + (size_t)r * hidden + off * 128, (const half*) suh_u[e] + off * 128, DO_HSCALE);
}

__global__ void xmoe_do_guad(half* __restrict__ cg, const half* __restrict__ cu, const int* __restrict__ row_exp,
                             const int64_t* __restrict__ svh_g, const int64_t* __restrict__ svh_u, const int64_t* __restrict__ suh_d,
                             int R, int inter, float act_limit, int act_fn)
{
    const int wpt = inter / 128;
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (w >= R * wpt) return;
    const int r = w / wpt, off = w % wpt, e = row_exp[r];
    if (e < 0) return;
    const size_t o = (size_t)r * inter + off * 128;
    had_hf_r_128_guad_inner(cg + o, cu + o, cg + o, (const half*) svh_g[e] + off * 128, (const half*) svh_u[e] + off * 128,
                            (const half*) suh_d[e] + off * 128, DO_HSCALE, act_limit, act_fn);
}

__global__ void xmoe_do_out(const half* __restrict__ d, float* __restrict__ out, const int64_t* __restrict__ tok,
                            const half* __restrict__ wts, const int* __restrict__ row_exp, const int64_t* __restrict__ svh_d,
                            int R, int hidden)
{
    const int wpt = hidden / 128;
    const int w = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (w >= R * wpt) return;
    const int r = w / wpt, off = w % wpt, e = row_exp[r];
    if (e < 0) return;
    had_hf_r_128_d_inner(d + (size_t)r * hidden + off * 128, out + tok[r] * hidden + off * 128,
                         (const half*) svh_d[e] + off * 128, DO_HSCALE * __half2float(wts[r]));
}

// out [T, hidden] fp32 += the hot experts' MoE output for the expert-sorted routed rows (tok, wts, row_exp; R = T * top_k).
// counts [E(+1)] int64 (|count| rows per expert in sorted order; negative = handled by another launch), hot_rows = VRAM
// rows of the bank (E when resident). Job table: job_i32 int32 [3, capacity] (row0, rows, expert), job_ptr int64
// [3, capacity] (gate, up, down trellis pointers), overflow int32 [1] (set to the needed job count if capacity was short;
// the caller sizes capacity = ceil(R / 128) + E so it cannot be). Scratch: xg, xu fp16 [>= R, hidden], cg, cu fp16
// [>= R, inter]; the down output reuses xg. Every launch grid depends only on R, E, capacity and the shapes.
void xmoe_do_moe(at::Tensor x, at::Tensor out, at::Tensor tok, at::Tensor wts, at::Tensor row_exp, at::Tensor counts,
                 int64_t hot_rows, at::Tensor job_i32, at::Tensor job_ptr, at::Tensor overflow,
                 at::Tensor g_tr, at::Tensor g_suh, at::Tensor g_svh, at::Tensor u_tr, at::Tensor u_suh, at::Tensor u_svh,
                 at::Tensor d_tr, at::Tensor d_suh, at::Tensor d_svh,
                 at::Tensor xg, at::Tensor xu, at::Tensor cg, at::Tensor cu,
                 int64_t bits, int64_t cb, double act_limit, int64_t act_fn)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int R = tok.size(0), hidden = x.size(1), inter = cg.size(1), E = g_tr.size(0), J = job_i32.size(1);
    TORCH_CHECK(x.dtype() == torch::kHalf && out.dtype() == torch::kFloat && wts.dtype() == torch::kHalf, "xmoe_do: x fp16, out fp32, wts fp16");
    TORCH_CHECK(tok.dtype() == torch::kLong && row_exp.dtype() == torch::kInt && counts.dtype() == torch::kLong, "xmoe_do: tok int64, row_exp int32, counts int64");
    TORCH_CHECK(job_i32.dtype() == torch::kInt && job_ptr.dtype() == torch::kLong && job_i32.size(0) == 3 && job_ptr.size(0) == 3 && job_ptr.size(1) == J,
                "xmoe_do: job tables int32 [3, J] and int64 [3, J]");
    TORCH_CHECK(counts.size(0) >= E && row_exp.size(0) == R && wts.size(0) == R, "xmoe_do: counts/row_exp/wts sizes");
    TORCH_CHECK(xg.size(0) >= R && xu.size(0) >= R && cg.size(0) >= R && cu.size(0) >= R && xg.size(1) == hidden && cu.size(1) == inter,
                "xmoe_do: scratch too small");
    TORCH_CHECK(J >= (R + DO_MC - 1) / DO_MC + E, "xmoe_do: job capacity below ceil(R / 128) + E");
    if (R == 0) return;
    const bool gated = act_fn != MOE_ACT_RELU2_NOGATE;
    int* row0 = job_i32.data_ptr<int>(); int* rows = row0 + J; int* jexp = rows + J;
    int64_t* jg = job_ptr.data_ptr<int64_t>(); int64_t* ju = jg + J; int64_t* jd = ju + J;
    const int bt = 512;
    xmoe_do_build_jobs<<<1, bt, 2 * bt * sizeof(int), stream>>>(counts.data_ptr<int64_t>(), E, (int) hot_rows, J, row0, rows, jexp,
        g_tr.data_ptr<int64_t>(), u_tr.data_ptr<int64_t>(), d_tr.data_ptr<int64_t>(), jg, ju, jd, overflow.data_ptr<int>());
    const int T = 256;
    auto blocks = [&](int warps) { return (warps * 32 + T - 1) / T; };
    xmoe_do_gather_had<<<blocks(R * (hidden / 128)), T, 0, stream>>>((const half*) x.data_ptr(), tok.data_ptr<int64_t>(), row_exp.data_ptr<int>(),
        g_suh.data_ptr<int64_t>(), u_suh.data_ptr<int64_t>(), (half*) xg.data_ptr(), (half*) xu.data_ptr(), R, hidden, gated);
    if (gated) launch_do_gemm((const half*) xg.data_ptr(), hidden, cg.data_ptr(), true, inter, row0, rows, jg, J, bits, cb, stream);
    launch_do_gemm((const half*) xu.data_ptr(), hidden, cu.data_ptr(), true, inter, row0, rows, ju, J, bits, cb, stream);
    xmoe_do_guad<<<blocks(R * (inter / 128)), T, 0, stream>>>((half*) cg.data_ptr(), (const half*) cu.data_ptr(), row_exp.data_ptr<int>(),
        g_svh.data_ptr<int64_t>(), u_svh.data_ptr<int64_t>(), d_suh.data_ptr<int64_t>(), R, inter, (float) act_limit, (int) act_fn);
    launch_do_gemm((const half*) cg.data_ptr(), inter, xg.data_ptr(), true, hidden, row0, rows, jd, J, bits, cb, stream);
    xmoe_do_out<<<blocks(R * (hidden / 128)), T, (T / 32) * 128 * sizeof(float), stream>>>((const half*) xg.data_ptr(), out.data_ptr<float>(),
        tok.data_ptr<int64_t>(), (const half*) wts.data_ptr(), row_exp.data_ptr<int>(), d_svh.data_ptr<int64_t>(), R, hidden);
    cuda_check(cudaPeekAtLastError());
}
