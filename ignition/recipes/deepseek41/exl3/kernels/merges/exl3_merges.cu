// SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
// SPDX-License-Identifier: Apache-2.0
//
// exp/ds41-exl3-final-merges (2026-11-17): small bit-exact merges around flare::ds41_exl3_moe (exl3/merges.py). Its own
// extension, so the xmoe extension (kernels/xmoe, digest-named) is unchanged. Built WITHOUT --use_fast_math: the
// epilogue's fp32 add must not flush denormals (the graph's add does not).
//
// xmoe_decode_prep_mapped (FLARE_DS41_EXL3_MERGE_TOPK): quant/exl3_prep.cu's fused decode routing prep with the merged
//   bank's hot_map lookup folded in. Replaces decoder.py's ``ids = index_select(hot_map, 0, ids)`` + the op's prep:
//   e = map[ids[t]] (map == nullptr: e = ids[t]); everything else is exl3_prep.cu line for line (counting sort =
//   stable argsort order, counts, hot/cold split, fp16 activations, zeroed fp32 accumulator).
// xmoe_epilogue_add (FLARE_DS41_EXL3_MERGE_EPILOGUE): out = bf16(float(bf16(acc)) + float(add)), the op's
//   ``out.copy_(acc)`` followed by the graph's ``routed + shared`` (bf16 + bf16: fp32 add, one round to nearest).
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_scan.cuh>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr int PREP_THREADS = 512;
constexpr int EPI_THREADS = 256;

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename IdT, typename MapT, typename WT, typename HT>
__global__ __launch_bounds__(PREP_THREADS)
void xmoe_decode_prep_mapped_kernel(const IdT* __restrict__ ids, const MapT* __restrict__ map, int map_len,
                                    const WT* __restrict__ weights, const HT* __restrict__ hidden,
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
        const int64_t raw = (int64_t) ids[t];
        if (map != nullptr)
        {
            if (raw < 0 || raw >= map_len) __trap();      // index_select device-asserts on the same input
            e = (int) map[raw];
        }
        else
            e = (int) raw;
        if (e < 0 || e >= bins) __trap();                 // the torch path's scatter_add device-asserts on the same input
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

// 8 elements per thread: two float4 of acc, one 16-byte vector of add, one of out.
__global__ __launch_bounds__(EPI_THREADS)
void xmoe_epilogue_add_kernel(const float4* __restrict__ acc, const uint4* __restrict__ add, uint4* __restrict__ out,
                              int64_t vectors)
{
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vectors) return;
    const float4 a0 = acc[2 * i], a1 = acc[2 * i + 1];
    const uint4 s = add[i];
    const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
    const __nv_bfloat16* sv = reinterpret_cast<const __nv_bfloat16*>(&s);
    uint4 r;
    __nv_bfloat16* rv = reinterpret_cast<__nv_bfloat16*>(&r);
#pragma unroll
    for (int k = 0; k < 8; ++k)
    {
        const __nv_bfloat16 routed = __float2bfloat16_rn(av[k]);                     // out.copy_(acc)
        rv[k] = __float2bfloat16_rn(__fadd_rn(__bfloat162float(routed), __bfloat162float(sv[k])));   // routed + shared
    }
    out[i] = r;
}

}  // namespace

// As xmoe_decode_prep (quant/exl3_prep.cu) plus ``map``: an int32/int64 [M] router-ID -> bank-slot table, or an empty
// tensor for none. ids [R, K] int32/int64; weights [R, K] fp32/bf16; hidden [R, H] bf16/fp16.
void xmoe_decode_prep_mapped(at::Tensor ids, at::Tensor map, at::Tensor weights, at::Tensor hidden, int64_t hot_rows,
                             at::Tensor token_sorted, at::Tensor weight_sorted, at::Tensor counts_hot,
                             at::Tensor counts_cold, at::Tensor x16, at::Tensor acc)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(hidden));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(ids.dim() == 2 && weights.sizes() == ids.sizes() && hidden.dim() == 2 && hidden.size(0) == ids.size(0),
                "xmoe_decode_prep_mapped: ids/weights [R, K], hidden [R, H]");
    TORCH_CHECK(ids.is_contiguous() && weights.is_contiguous() && hidden.is_contiguous() && map.is_contiguous(),
                "xmoe_decode_prep_mapped: contiguous inputs");
    TORCH_CHECK(map.dim() == 1, "xmoe_decode_prep_mapped: map [M] (empty = none)");
    const int routes = (int) ids.numel(), top_k = (int) ids.size(1), bins = (int) counts_hot.numel();
    TORCH_CHECK(routes <= PREP_THREADS && bins <= PREP_THREADS, "xmoe_decode_prep_mapped: at most 512 routes and 512 bins");
    TORCH_CHECK(token_sorted.numel() == routes && token_sorted.scalar_type() == at::kLong
                && weight_sorted.numel() == routes && weight_sorted.scalar_type() == at::kHalf
                && counts_cold.numel() == bins && counts_hot.scalar_type() == at::kLong && counts_cold.scalar_type() == at::kLong,
                "xmoe_decode_prep_mapped: token_sorted int64 [R*K], weight_sorted fp16 [R*K], counts int64 [bins]");
    TORCH_CHECK(x16.sizes() == hidden.sizes() && x16.scalar_type() == at::kHalf && acc.sizes() == hidden.sizes()
                && acc.scalar_type() == at::kFloat && x16.is_contiguous() && acc.is_contiguous(),
                "xmoe_decode_prep_mapped: x16 fp16 and acc fp32 shaped like hidden");
    const int64_t elements = hidden.numel();
    const int extra = (int) std::min<int64_t>(64, std::max<int64_t>(1, (elements + PREP_THREADS * 8 - 1) / (PREP_THREADS * 8)));
    const int hot = (int) std::min<int64_t>(hot_rows, bins);
    const int map_len = (int) map.numel();

#define PREP_LAUNCH(IDT, MT, WT, HT)                                                                                    \
    xmoe_decode_prep_mapped_kernel<IDT, MT, WT, HT><<<1 + extra, PREP_THREADS, 0, stream>>>(                           \
        ids.data_ptr<IDT>(), map_len ? reinterpret_cast<const MT*>(map.data_ptr()) : nullptr, map_len,                 \
        reinterpret_cast<const WT*>(weights.data_ptr()), reinterpret_cast<const HT*>(hidden.data_ptr()),               \
        routes, top_k, bins, hot, elements, token_sorted.data_ptr<int64_t>(),                                          \
        reinterpret_cast<half*>(weight_sorted.data_ptr()), counts_hot.data_ptr<int64_t>(), counts_cold.data_ptr<int64_t>(), \
        reinterpret_cast<half*>(x16.data_ptr()), acc.data_ptr<float>())

#define PREP_H(IDT, MT, WT)                                                                                             \
    if (hidden.scalar_type() == at::kBFloat16) PREP_LAUNCH(IDT, MT, WT, __nv_bfloat16);                                 \
    else if (hidden.scalar_type() == at::kHalf) PREP_LAUNCH(IDT, MT, WT, half);                                         \
    else TORCH_CHECK(false, "xmoe_decode_prep_mapped: hidden must be bf16 or fp16")

#define PREP_W(IDT, MT)                                                                                                 \
    if (weights.scalar_type() == at::kFloat) { PREP_H(IDT, MT, float); }                                                \
    else if (weights.scalar_type() == at::kBFloat16) { PREP_H(IDT, MT, __nv_bfloat16); }                               \
    else TORCH_CHECK(false, "xmoe_decode_prep_mapped: weights must be fp32 or bf16")

#define PREP_M(IDT)                                                                                                     \
    if (map.scalar_type() == at::kInt) { PREP_W(IDT, int32_t); }                                                        \
    else if (map.scalar_type() == at::kLong) { PREP_W(IDT, int64_t); }                                                  \
    else TORCH_CHECK(false, "xmoe_decode_prep_mapped: map must be int32 or int64")

    if (ids.scalar_type() == at::kLong) { PREP_M(int64_t); }
    else if (ids.scalar_type() == at::kInt) { PREP_M(int32_t); }
    else TORCH_CHECK(false, "xmoe_decode_prep_mapped: ids must be int32 or int64");
#undef PREP_M
#undef PREP_W
#undef PREP_H
#undef PREP_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// acc fp32 [R, H], add bf16 [R, H], out bf16 [R, H], all contiguous, R * H a multiple of 8, 16-byte aligned.
void xmoe_epilogue_add(at::Tensor acc, at::Tensor add, at::Tensor out)
{
    const at::cuda::OptionalCUDAGuard guard(device_of(acc));
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(acc.scalar_type() == at::kFloat && add.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
                "xmoe_epilogue_add: acc fp32, add and out bf16");
    TORCH_CHECK(acc.sizes() == add.sizes() && acc.sizes() == out.sizes(), "xmoe_epilogue_add: equal shapes");
    TORCH_CHECK(acc.is_contiguous() && add.is_contiguous() && out.is_contiguous(), "xmoe_epilogue_add: contiguous");
    const int64_t n = acc.numel();
    TORCH_CHECK(n % 8 == 0, "xmoe_epilogue_add: element count must be a multiple of 8");
    for (const at::Tensor* t : {&acc, &add, &out})
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0, "xmoe_epilogue_add: 16-byte aligned storage");
    const int64_t vectors = n / 8;
    if (!vectors) return;
    const int64_t blocks = (vectors + EPI_THREADS - 1) / EPI_THREADS;
    xmoe_epilogue_add_kernel<<<(unsigned) blocks, EPI_THREADS, 0, stream>>>(
        reinterpret_cast<const float4*>(acc.data_ptr()), reinterpret_cast<const uint4*>(add.data_ptr()),
        reinterpret_cast<uint4*>(out.data_ptr()), vectors);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("xmoe_decode_prep_mapped", &xmoe_decode_prep_mapped,
          "FLARE_DS41_EXL3_MERGE_TOPK: fused decode routing prep with the hot_map lookup folded in");
    m.def("xmoe_epilogue_add", &xmoe_epilogue_add,
          "FLARE_DS41_EXL3_MERGE_EPILOGUE: out = bf16(bf16(acc) + add) (op copy + graph routed+shared add)");
}
