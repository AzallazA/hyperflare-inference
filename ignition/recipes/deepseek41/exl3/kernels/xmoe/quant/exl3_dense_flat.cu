// Beam round 6 (2026-10-18): flat decode GEMM for the EXL3 dense projections (<= 16 rows per job). One persistent block per SM,
// three phases separated by grid barriers: (1) input Hadamard x * suh_src per source, (2) the swapped-operand flat GEMM over all
// (slice, n-tile, k-tile) of every source (exl3_gemm_inner_flat_dense.cuh), (3) output Hadamard * svh per slice, written straight
// into the caller's output layout (no concat). Replaces exllamav3's exl3_mgemm (fused q/k/v/gate) and exl3_gemm (o_proj, dense MLP)
// at decode sizes.
#include <cuda_fp16.h>
#include <type_traits>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include "../util.h"
#include "../util.cuh"
#include "../ptx.cuh"
#include "exl3_moe_common.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner_flat_dense.cuh"
#include "exl3_devctx.cuh"

#include <cuda_bf16.h>

// Same math as had_hf_r_128_inner<true, false> on an fp16 copy of the input: x is converted to fp16 first (as x.to(fp16) did).
template <typename InT>
__device__ __forceinline__ void xd_had_in(const InT* __restrict__ in, half* __restrict__ out, const half* __restrict__ scale, float r_scale)
{
    const int t = threadIdx.x & 31;
    half4 v;
    if constexpr (std::is_same<InT, half>::value) v = ((const half4*) in)[t];
    else
    {
        const __nv_bfloat162* b = (const __nv_bfloat162*) in + 2 * t;
        float2 f0 = __bfloat1622float2(b[0]), f1 = __bfloat1622float2(b[1]);
        v.x = __floats2half2_rn(f0.x, f0.y); v.y = __floats2half2_rn(f1.x, f1.y);
    }
    half4 sc = ((const half4*) scale)[t];
    v.x = __hmul2(v.x, sc.x); v.y = __hmul2(v.y, sc.y);
    float v0 = __low2float(v.x), v1 = __high2float(v.x), v2 = __low2float(v.y), v3 = __high2float(v.y);
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale); v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
    ((half4*) out)[t] = v;
}

// Same math as had_hf_r_128_inner<false, true> (fp16 result), then stored as OutT (bf16: = .to(bf16) of the fp16 result).
template <typename OutT>
__device__ __forceinline__ void xd_had_out(const half* __restrict__ in, OutT* __restrict__ out, const half* __restrict__ scale, float r_scale)
{
    const int t = threadIdx.x & 31;
    half4 v = ((const half4*) in)[t];
    float v0 = __low2float(v.x), v1 = __high2float(v.x), v2 = __low2float(v.y), v3 = __high2float(v.y);
    float s0 = v0 + v1, d0 = v0 - v1, s1 = v2 + v3, d1 = v2 - v3;
    float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;
    shuffle_had_f4x32(h0, h1, h2, h3, t);
    v.x = __floats2half2_rn(h0 * r_scale, h1 * r_scale); v.y = __floats2half2_rn(h2 * r_scale, h3 * r_scale);
    half4 sc = ((const half4*) scale)[t];
    v.x = __hmul2(v.x, sc.x); v.y = __hmul2(v.y, sc.y);
    if constexpr (std::is_same<OutT, half>::value) ((half4*) out)[t] = v;
    else
    {
        __nv_bfloat162* o = (__nv_bfloat162*) out + 2 * t;
        o[0] = __floats2bfloat162_rn(__low2float(v.x), __high2float(v.x));
        o[1] = __floats2bfloat162_rn(__low2float(v.y), __high2float(v.y));
    }
}

#define XDENSE_TK 128
#define XDENSE_SH 2
#define XDENSE_TN 256
#define XDENSE_BARRIER_ID 210

template <int bits, int cb, typename IOT>
__global__ __launch_bounds__(512)
void exl3_dense_flat_kernel(const IOT* __restrict__ x, int rows, int K, int n_src, const int64_t* __restrict__ suh,
                            half* __restrict__ temp, FlatDenseJobs jobs, int W, const int64_t* __restrict__ svh_ptr,
                            IOT* __restrict__ out, const int64_t* __restrict__ out_off, int ld_out, int* __restrict__ locks)
{
    int* barrier = locks + BARRIER_LOCKS_OFFSET;
    auto grid_sync = [&] () { group_barrier(XDENSE_BARRIER_ID, gridDim.x, barrier); };
    const int gwarp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, gwarps = gridDim.x * (blockDim.x / 32);

    const int segs = K / 128;
    for (int w = gwarp; w < n_src * rows * segs; w += gwarps)
    {
        const int s = w / (rows * segs), rem = w % (rows * segs), r = rem / segs, off = rem % segs;
        xd_had_in<IOT>(x + (size_t) r * K + off * 128, temp + ((size_t) s * 16 + r) * K + off * 128,   // fixed 16-row source stride
                       (const half*) suh[s] + off * 128, 0.088388347648f);
    }
    grid_sync();

    exl3_gemm_kernel_inner_flat_dense<bits, false, cb, 16, XDENSE_TK, XDENSE_TN, XDENSE_SH, 1>(jobs, K, W, locks);
    grid_sync();

    const int wsegs = W / 128;
    for (int w = gwarp; w < jobs.num_jobs * 16 * wsegs; w += gwarps)
    {
        const int j = w / (16 * wsegs), rem = w % (16 * wsegs), r = rem / wsegs, off = rem % wsegs;
        if (r >= jobs.rows[j]) continue;
        xd_had_out<IOT>((const half*) jobs.c_ptr[j] + (size_t) r * W + off * 128, out + out_off[j] + (size_t) r * ld_out + off * 128,
                        (const half*) svh_ptr[j] + off * 128, 0.088388347648f);
    }
}

template <int bits, int cb>
constexpr int xdense_smem()
{
    constexpr int a = 16 * XDENSE_TK * 2;
    constexpr int b = (XDENSE_TK / 16) * (XDENSE_TN / 16) * 16 * bits * 2;
    constexpr int frags_n = 2 * (XDENSE_TN / 16) / (EXL3_GEMM_BASE_THREADS / 32);
    constexpr int cbuf = 4 * EXL3_GEMM_BASE_THREADS * frags_n * 4;
    return XDENSE_SH * (a + b) + cbuf;
}

// x [rows, K] fp16; suh [n_src] int64 ptrs; temp [n_src, 16, K] fp16 scratch (job A = temp + src * 16 * K); job tables [J]; svh [J] ptrs (slice-offset);
// out base fp16 + out_off [J] element offsets (row 0 of the job, first column of the slice); ld_out = out row stride.
void xdense_forward(at::Tensor x, at::Tensor suh, at::Tensor temp, at::Tensor a_ptr, at::Tensor b_ptr, at::Tensor c_ptr,
                    at::Tensor b_bnf, at::Tensor rows_t, at::Tensor svh, at::Tensor out, at::Tensor out_off, int64_t ld_out,
                    int64_t W, int64_t bits, int64_t cb)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(x));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = x.device().index();
    const int rows = x.size(0), K = x.size(1), n_src = suh.size(0), J = a_ptr.size(0);
    TORCH_CHECK(rows >= 1 && rows <= 16 && K % XDENSE_TK == 0 && W % XDENSE_TN == 0 && W % 128 == 0, "xdense: shape");
    TORCH_CHECK(bits == 6 && cb == 1, "xdense: only 6-bit MCG instantiated");
    FlatDenseJobs jobs;
    jobs.num_jobs = J; jobs.a_ptr = a_ptr.data_ptr<int64_t>(); jobs.b_ptr = b_ptr.data_ptr<int64_t>(); jobs.c_ptr = c_ptr.data_ptr<int64_t>();
    jobs.b_bnf = b_bnf.data_ptr<int>(); jobs.rows = rows_t.data_ptr<int>();
    int* locks = DevCtx::instance().get_locks(device);
    int num_sms = DevCtx::instance().get_num_sms(device);
    constexpr int smem = xdense_smem<6, 1>();
    static bool attr[2][16] = {};
    TORCH_CHECK(x.dtype() == out.dtype() && (x.dtype() == torch::kHalf || x.dtype() == torch::kBFloat16), "xdense: x/out fp16 or bf16, same dtype");
    if (x.dtype() == torch::kBFloat16)
    {
        auto k = exl3_dense_flat_kernel<6, 1, __nv_bfloat16>;
        if (!attr[1][device]) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr[1][device] = true; }
        k<<<num_sms, 512, smem, stream>>>((const __nv_bfloat16*) x.data_ptr(), rows, K, n_src, suh.data_ptr<int64_t>(), (half*) temp.data_ptr(),
            jobs, (int) W, svh.data_ptr<int64_t>(), (__nv_bfloat16*) out.data_ptr(), out_off.data_ptr<int64_t>(), (int) ld_out, locks);
    }
    else
    {
        auto k = exl3_dense_flat_kernel<6, 1, half>;
        if (!attr[0][device]) { cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); attr[0][device] = true; }
        k<<<num_sms, 512, smem, stream>>>((const half*) x.data_ptr(), rows, K, n_src, suh.data_ptr<int64_t>(), (half*) temp.data_ptr(),
            jobs, (int) W, svh.data_ptr<int64_t>(), (half*) out.data_ptr(), out_off.data_ptr<int64_t>(), (int) ld_out, locks);
    }
    cuda_check(cudaPeekAtLastError());
}
