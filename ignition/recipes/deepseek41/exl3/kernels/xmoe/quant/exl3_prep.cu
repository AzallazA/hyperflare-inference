// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0
//
// exp/ds41-exl3-next (2026-11-16): fused decode routing prep for flare::ds41_exl3_moe (FLARE_DS41_EXL3_FUSED_PREP=1).
// One launch replaces the op's torch glue at decode (~18 launches, ~20 us per layer in the seq229 profile):
//   flat = ids.reshape(-1); order = argsort(flat, stable=True); token_sorted = order // top_k
//   weight_sorted = weights.reshape(-1)[order].half(); counts = bincount(flat, minlength=E1)
//   hot = where(slot < hot_rows, counts, -counts); cold = -hot; x16 = hidden.half(); acc = 0 (fp32)
// Block 0 does the routing (counting sort; a route's position = its expert's offset + the number of earlier routes to
// the same expert, which is exactly the stable argsort order); blocks 1.. convert the activations and zero the
// accumulator. Every output is integer or a single round-to-nearest conversion, so it is bit-identical to the torch path.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_scan.cuh>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int PREP_THREADS = 512;

// explicit conversions (the extension builds with __CUDA_NO_HALF_CONVERSIONS__ / __CUDA_NO_BFLOAT16_CONVERSIONS__)
__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }     // max routes (rows * top_k) and max bins (num_experts + 1) per launch

template <typename IdT, typename WT, typename HT>
__global__ __launch_bounds__(PREP_THREADS)
void xmoe_decode_prep_kernel(const IdT* __restrict__ ids, const WT* __restrict__ weights, const HT* __restrict__ hidden,
                             int routes, int top_k, int bins, int hot_rows, int64_t elements,
                             int64_t* __restrict__ token_sorted, half* __restrict__ weight_sorted,
                             int64_t* __restrict__ counts_hot, int64_t* __restrict__ counts_cold,
                             half* __restrict__ x16, float* __restrict__ acc)
{
    if (blockIdx.x > 0)
    {
        const int64_t stride = (int64_t) (gridDim.x - 1) * blockDim.x;
        for (int64_t k = (int64_t) (blockIdx.x - 1) * blockDim.x + threadIdx.x; k < elements; k += stride)
        {
            x16[k] = __float2half_rn(to_f(hidden[k]));
            acc[k] = 0.0f;
        }
        return;
    }
    using Scan = cub::BlockScan<int, PREP_THREADS>;
    __shared__ typename Scan::TempStorage scan_tmp;
    __shared__ int s_hist[PREP_THREADS];
    __shared__ int s_off[PREP_THREADS];
    __shared__ int s_flat[PREP_THREADS];
    const int t = threadIdx.x;
    s_hist[t] = 0;
    __syncthreads();
    int e = -1;
    if (t < routes)
    {
        e = (int) ids[t];
        if (e < 0 || e >= bins) __trap();       // the torch path's scatter_add device-asserts on the same input
        s_flat[t] = e;
        atomicAdd(&s_hist[e], 1);
    }
    __syncthreads();
    const int c = s_hist[t];
    int off;
    Scan(scan_tmp).ExclusiveSum(c, off);
    s_off[t] = off;
    if (t < bins)
    {
        counts_hot[t] = t < hot_rows ? c : -c;
        counts_cold[t] = t < hot_rows ? -c : c;
    }
    __syncthreads();
    if (t < routes)
    {
        int rank = 0;
        for (int j = 0; j < t; ++j) rank += s_flat[j] == e;
        const int pos = s_off[e] + rank;
        token_sorted[pos] = t / top_k;
        weight_sorted[pos] = __float2half_rn(to_f(weights[t]));
    }
}

}  // namespace

// ids [R, K] int32/int64 in [0, bins); weights [R, K] fp32/bf16; hidden [R, H] bf16/fp16 contiguous. hot_rows = the
// bank's VRAM rows (bins for an unparked bank, so hot == counts). Outputs as the torch path: token_sorted [R*K] int64,
// weight_sorted [R*K] fp16, counts_hot/counts_cold [bins] int64, x16 [R, H] fp16, acc [R, H] fp32 (zeroed).
void xmoe_decode_prep(at::Tensor ids, at::Tensor weights, at::Tensor hidden, int64_t hot_rows,
                      at::Tensor token_sorted, at::Tensor weight_sorted, at::Tensor counts_hot, at::Tensor counts_cold,
                      at::Tensor x16, at::Tensor acc)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(hidden));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(ids.dim() == 2 && weights.sizes() == ids.sizes() && hidden.dim() == 2 && hidden.size(0) == ids.size(0),
                "xmoe_decode_prep: ids/weights [R, K], hidden [R, H]");
    TORCH_CHECK(ids.is_contiguous() && weights.is_contiguous() && hidden.is_contiguous(), "xmoe_decode_prep: contiguous inputs");
    const int routes = (int) ids.numel(), top_k = (int) ids.size(1), bins = (int) counts_hot.numel();
    TORCH_CHECK(routes <= PREP_THREADS && bins <= PREP_THREADS, "xmoe_decode_prep: at most 512 routes and 512 bins");
    TORCH_CHECK(token_sorted.numel() == routes && token_sorted.scalar_type() == at::kLong
                && weight_sorted.numel() == routes && weight_sorted.scalar_type() == at::kHalf
                && counts_cold.numel() == bins && counts_hot.scalar_type() == at::kLong && counts_cold.scalar_type() == at::kLong,
                "xmoe_decode_prep: token_sorted int64 [R*K], weight_sorted fp16 [R*K], counts int64 [bins]");
    TORCH_CHECK(x16.sizes() == hidden.sizes() && x16.scalar_type() == at::kHalf && acc.sizes() == hidden.sizes()
                && acc.scalar_type() == at::kFloat && x16.is_contiguous() && acc.is_contiguous(),
                "xmoe_decode_prep: x16 fp16 and acc fp32 shaped like hidden");
    const int64_t elements = hidden.numel();
    const int extra = (int) std::min<int64_t>(64, std::max<int64_t>(1, (elements + PREP_THREADS * 8 - 1) / (PREP_THREADS * 8)));
    const int hot = (int) std::min<int64_t>(hot_rows, bins);

#define PREP_LAUNCH(IDT, WT, HT)                                                                                        \
    xmoe_decode_prep_kernel<IDT, WT, HT><<<1 + extra, PREP_THREADS, 0, stream>>>(                                       \
        ids.data_ptr<IDT>(), reinterpret_cast<const WT*>(weights.data_ptr()), reinterpret_cast<const HT*>(hidden.data_ptr()), \
        routes, top_k, bins, hot, elements, token_sorted.data_ptr<int64_t>(),                                          \
        reinterpret_cast<half*>(weight_sorted.data_ptr()), counts_hot.data_ptr<int64_t>(), counts_cold.data_ptr<int64_t>(), \
        reinterpret_cast<half*>(x16.data_ptr()), acc.data_ptr<float>())

#define PREP_H(IDT, WT)                                                                                                 \
    if (hidden.scalar_type() == at::kBFloat16) PREP_LAUNCH(IDT, WT, __nv_bfloat16);                                     \
    else if (hidden.scalar_type() == at::kHalf) PREP_LAUNCH(IDT, WT, half);                                             \
    else TORCH_CHECK(false, "xmoe_decode_prep: hidden must be bf16 or fp16")

#define PREP_W(IDT)                                                                                                     \
    if (weights.scalar_type() == at::kFloat) { PREP_H(IDT, float); }                                                    \
    else if (weights.scalar_type() == at::kBFloat16) { PREP_H(IDT, __nv_bfloat16); }                                   \
    else TORCH_CHECK(false, "xmoe_decode_prep: weights must be fp32 or bf16")

    if (ids.scalar_type() == at::kLong) { PREP_W(int64_t); }
    else if (ids.scalar_type() == at::kInt) { PREP_W(int32_t); }
    else TORCH_CHECK(false, "xmoe_decode_prep: ids must be int32 or int64");
#undef PREP_W
#undef PREP_H
#undef PREP_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
