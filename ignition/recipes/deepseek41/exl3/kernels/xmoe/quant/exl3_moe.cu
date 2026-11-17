#include <cuda_fp16.h>
#include "exl3_gemm.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#include "../util.h"
#include "../util.cuh"
#include "comp_units/exl3_moe_instances.cuh"
#include "exl3_devctx.cuh"
#include "exl3_moe_flat_kernel.cuh"
#include <set>
#include <vector>

int exl3_moe_max_concurrency(int device)
{
    int num_sms = DevCtx::instance().get_num_sms(device);
    return num_sms / MOE_SMS_PER_EXPERT;
}

std::set<void*> moe_kernel_attr_set[MAX_DEVICES] = {};

fp_exl3_moe_kernel exl3_moe_kernel_instances[] =
{
    // xmoe scaffold: K=2 and K=3 MCG (cb1) are built (2026-09-28); [K][cb-1][N_off] indexing kept
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    exl3_moe_kernel_k2_n128_cb1(), exl3_moe_kernel_k2_n256_cb1(), nullptr, nullptr,
    exl3_moe_kernel_k3_n128_cb1(), exl3_moe_kernel_k3_n256_cb1(), exl3_moe_kernel_k3_n128_cb2(), exl3_moe_kernel_k3_n256_cb2(),
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr
};

// xmoe: multi-M (prefill) instances, chosen when bsz > XMOE_MT_MIN_BSZ (env override)
fp_exl3_moe_kernel exl3_moe_kernel_instances_mt[] =
{
    // xmoe scaffold: K=2 and K=3 MCG (cb1) are built (2026-09-28); [K][cb-1][N_off] indexing kept
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    exl3_moe_kernel_k2_n128_cb1_mt(), exl3_moe_kernel_k2_n256_cb1_mt(), nullptr, nullptr,
    exl3_moe_kernel_k3_n128_cb1_mt(), exl3_moe_kernel_k3_n256_cb1_mt(), exl3_moe_kernel_k3_n128_cb2_mt(), exl3_moe_kernel_k3_n256_cb2_mt(),
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr
};
#include <cstdlib>
static int xmoe_env_int(const char* name, int dflt) { const char* e = getenv(name); return e ? atoi(e) : dflt; }
// Beam (2026-10-17): the flat kernel is the n128 instance; it also covers 256-divisible shapes (Beam: hidden 5120,
// intermediate/TP 1280), which the group dispatch would give the n256 instance. XMOE_FLAT_N256=0 restores the old gate.
static int xmoe_flat_n256_ok() { static int v = xmoe_env_int("XMOE_FLAT_N256", 1); return v; }
static int xmoe_flat_max_bsz() { static int v = xmoe_env_int("XMOE_FLAT_MAX_BSZ", 64); return v; }
static int xmoe_flat_grid_env() { static int v = xmoe_env_int("XMOE_FLAT_GRID", 0); return v; }
static int xmoe_force_group = 0;  // split launch: 1 = never take the flat path (cold/host experts)
void xmoe_set_force_group(int v) { xmoe_force_group = v; }
// overlap (hot and cold launches on two streams): the side launch takes the alternate lock buffer (barrier counters,
// tile locks and expert scheduler are per-buffer) and each launch is held to its SM budget so both grids stay co-resident
static int xmoe_lock_slot = 0;
void xmoe_set_lock_slot(int v) { xmoe_lock_slot = v; }
static int xmoe_sm_budget = 0;  // 0 = all SMs
void xmoe_set_sm_budget(int v) { xmoe_sm_budget = v; }
static int xmoe_group_width = 0;  // group kernel: SMs per expert group when the active count is unknown (0 = MOE_SMS_PER_EXPERT)
void xmoe_set_group_width(int v) { xmoe_group_width = v; }
static int xmoe_flat_ok_k[2][4][MAX_DEVICES] = {};  // [cb_idx][K][device] (3f: K=3 MUL1 = cb_idx 1): 0 unknown, 1 ok, -1 unusable (occupancy); K=2,3 built
static int xmoe_mt_min_bsz() { static int v = -1; if (v < 0) { const char* e = getenv("XMOE_MT_MIN_BSZ"); v = e ? atoi(e) : 64; } return v; }

// prefillbw (xmoe_ds41pb, 2026-10-20): prefill (bsz > XMOE_MT_MIN_BSZ) group launches of host-RAM experts -- the split
// launch's cold side, recognized by xmoe_set_force_group(1) -- take exl3_moe_kernel_pb (exl3_moe_kernel_pb.cuh): deeper cp.async
// pipeline (XMOE_PB_STAGES = 3/4/6/8, default 4; 0 = off -> the ds41a MT kernel) and the read-once stage copy of multi-chunk experts
// (XMOE_PB_STAGE = 1 default / 0). XMOE_PB_HOT = 1 also gives the hot (VRAM) prefill group launches the deeper pipeline (never the
// stage copy). Only K=3, MUL1, N=128 (DS4.1-1017) has instances; anything else keeps the ds41a kernels. xmoe_pb_set() overrides
// the env (tests); every other launch is byte-for-byte the ds41a path.
static int xmoe_pb_cfg[3] = {-2, -2, -2};   // stages, stage copy, hot; -2 = not read yet
static void xmoe_pb_init()
{
    if (xmoe_pb_cfg[0] == -2) xmoe_pb_cfg[0] = xmoe_env_int("XMOE_PB_STAGES", 4);
    if (xmoe_pb_cfg[1] == -2) xmoe_pb_cfg[1] = xmoe_env_int("XMOE_PB_STAGE", 1);
    if (xmoe_pb_cfg[2] == -2) xmoe_pb_cfg[2] = xmoe_env_int("XMOE_PB_HOT", 0);
}
void xmoe_pb_set(int stages, int stage, int hot)
{
    xmoe_pb_init();
    if (stages >= 0) xmoe_pb_cfg[0] = stages;
    if (stage >= 0) xmoe_pb_cfg[1] = stage;
    if (hot >= 0) xmoe_pb_cfg[2] = hot;
}
std::vector<int64_t> xmoe_pb_get()
{
    xmoe_pb_init();
    return {xmoe_pb_cfg[0], xmoe_pb_cfg[1], xmoe_pb_cfg[2]};
}
static fp_exl3_moe_kernel xmoe_pb_kernel(int stages, bool stage)
{
    switch (stages)
    {
        case 3: return stage ? exl3_moe_kernel_pb_k3_n128_cb2_s3_st1() : exl3_moe_kernel_pb_k3_n128_cb2_s3_st0();
        case 4: return stage ? exl3_moe_kernel_pb_k3_n128_cb2_s4_st1() : exl3_moe_kernel_pb_k3_n128_cb2_s4_st0();
        case 6: return stage ? exl3_moe_kernel_pb_k3_n128_cb2_s6_st1() : exl3_moe_kernel_pb_k3_n128_cb2_s6_st0();
        case 8: return stage ? exl3_moe_kernel_pb_k3_n128_cb2_s8_st1() : exl3_moe_kernel_pb_k3_n128_cb2_s8_st0();
        default: return nullptr;
    }
}


/*
Fused mixture-of-experts MLP operation for EXL3 weights

inputs:
    hidden_state:
        input hidden state - shape (bsz, hidden_dim) - fp16

    output_state:
        output hidden state - shape (bsz, hidden_dim) - fp32
        zero-initialized

    expert_count:
        bincount of expert indices across all tokens in batch - shape (num_experts + 1,) - int64
        last item is ignored, used for the case where some tokens may activate less than num_experts_per_token
        experts (specifically in expert split mode)

    token_sorted:
        token indices, sorted by expert - shape (bsz * num_experts_per_tok,)  - int64

    weight_sorted:
        routing weight per token, sorted by expert - shape (bsz * num_experts_per_tok,) - fp16

    temp_state_g:
    temp_state_u:
        temp state storage - shape (concurrency, max_tokens_per_expert, hidden_dim), fp16

    temp_intermediate_g
    temp_intermediate_u:
        temp intermediate storage - shape (concurrency, max_tokens_per_expert, intermediate_dim), fp16

    act_function:
        int, see exl3_moe.cuh

    K_gate
    K_up
    K_down:
        int, bitrates for gate, up, down tensors

    gate_ptrs_trellis
    gate_ptrs_suh
    gate_ptrs_svh
    up_ptrs_trellis
    up_ptrs_suh
    up_ptrs_svh
    down_ptrs_trellis
    down_ptrs_suh
    down_ptrs_svh:
        tensors of data_ptrs to quantized tensor data - each shape (num_experts,) - void*

    gate_mcg
    gate_mul1
    up_mcg
    up_mul1
    down_mcg
    down_mul1:
        bool, codebook flags

    num_active:
        number of experts with 0 < token count <= max_tokens_per_expert, i.e. the number of experts this kernel
        will process. Used to size the launch: fewer, wider expert groups when few experts are active. Pass -1 if
        unknown (defaults to MOE_SMS_PER_EXPERT-wide groups at max concurrency)
*/

void exl3_moe
(
    const at::Tensor& hidden_state,
    const at::Tensor& output_state,
    const at::Tensor& expert_count,
    const at::Tensor& token_sorted,
    const at::Tensor& weight_sorted,

    const at::Tensor& temp_state_g,
    const at::Tensor& temp_state_u,
    const at::Tensor& temp_intermediate_g,
    const at::Tensor& temp_intermediate_u,

    const int act_function,

    const int K_gate,
    const int K_up,
    const int K_down,

    const at::Tensor& gate_ptrs_trellis,
    const at::Tensor& gate_ptrs_suh,
    const at::Tensor& gate_ptrs_svh,
    const at::Tensor& up_ptrs_trellis,
    const at::Tensor& up_ptrs_suh,
    const at::Tensor& up_ptrs_svh,
    const at::Tensor& down_ptrs_trellis,
    const at::Tensor& down_ptrs_suh,
    const at::Tensor& down_ptrs_svh,

    const bool gate_mcg,
    const bool gate_mul1,
    const bool up_mcg,
    const bool up_mul1,
    const bool down_mcg,
    const bool down_mul1,

    const float act_limit,
    const int num_active
)
{
    const at::cuda::OptionalCUDAGuard device_guard(hidden_state.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Nothing for the fused kernel to do
    if (num_active == 0) return;

    // Validate args
    TORCH_CHECK_DTYPE(hidden_state, kHalf);
    TORCH_CHECK_DIM(hidden_state, 2);
    size_t bsz = hidden_state.size(0);
    size_t hidden_dim = hidden_state.size(1);

    TORCH_CHECK_DTYPE(output_state, kFloat);
    TORCH_CHECK_SHAPES_FULL(output_state, hidden_state);

    TORCH_CHECK_DTYPE(expert_count, kLong);
    TORCH_CHECK_DIM(expert_count, 1);
    size_t num_experts = expert_count.size(0) - 1;

    TORCH_CHECK_DTYPE(token_sorted, kLong);
    TORCH_CHECK_DIM(token_sorted, 1);
    TORCH_CHECK_SHAPES_FULL(token_sorted, weight_sorted);
    size_t num_experts_per_tok = token_sorted.size(0) / bsz;

    TORCH_CHECK_DTYPE(temp_state_g, kHalf);
    TORCH_CHECK_DTYPE(temp_state_u, kHalf);
    TORCH_CHECK_DIM(temp_state_g, 3);
    TORCH_CHECK_SHAPES(temp_state_g, 2, hidden_state, 1, 1);
    TORCH_CHECK_SHAPES_FULL(temp_state_g, temp_state_u);
    size_t max_tokens_per_expert = temp_state_g.size(1);
    size_t concurrency = temp_state_g.size(0);

    TORCH_CHECK_DTYPE(temp_intermediate_g, kHalf);
    TORCH_CHECK_DTYPE(temp_intermediate_u, kHalf);
    TORCH_CHECK_DIM(temp_intermediate_g, 3);
    TORCH_CHECK_DIM(temp_intermediate_u, 3);
    TORCH_CHECK_SHAPES_FULL(temp_intermediate_g, temp_intermediate_u);
    TORCH_CHECK_SHAPES(temp_intermediate_g, 1, temp_state_g, 1, 1);
    size_t intermediate_dim = temp_intermediate_g.size(2);

    // TORCH_CHECK(!(gate_mcg && gate_mul1), "Specified both mcg and mul1 (gate)");
    // TORCH_CHECK(!(up_mcg && up_mul1), "Specified both mcg and mul1 (up)");
    // TORCH_CHECK(!(down_mcg && down_mul1), "Specified both mcg and mul1 (down)");
    TORCH_CHECK(gate_mcg == up_mcg && up_mcg == down_mcg && gate_mul1 == up_mul1 && up_mul1 == down_mul1,
                "MoE kernel: gate/up/down must share the same codebook");
    TORCH_CHECK(gate_mcg != gate_mul1, "MoE kernel: Only mcg and mul1 codebooks are supported");
    const int cb_idx = gate_mul1 ? 1 : 0;

    // TORCH_CHECK(act_function == MOE_ACT_SILU, "MoE kernel: Only SiLU is currently supported");

    int K = 0;
    if (K_gate == K_up && K_up == K_down) K = K_gate;

    TORCH_CHECK_DIM(gate_ptrs_trellis, 1);
    TORCH_CHECK(gate_ptrs_trellis.size(0) == num_experts, "Number of gate tensors doesn't match num_experts");
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, gate_ptrs_suh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, gate_ptrs_svh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, up_ptrs_trellis);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, up_ptrs_suh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, up_ptrs_svh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, down_ptrs_trellis);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, down_ptrs_suh);
    TORCH_CHECK_SHAPES_FULL(gate_ptrs_trellis, down_ptrs_svh);

    // Device properties
    int device;
    cudaGetDevice(&device);
    int num_sms = DevCtx::instance().get_num_sms(device);
    int cc = DevCtx::instance().get_cc(device);
    int* locks = xmoe_lock_slot ? DevCtx::instance().get_locks_alt(device) : DevCtx::instance().get_locks(device);
    int budget = xmoe_sm_budget > 0 ? MIN(xmoe_sm_budget, num_sms) : num_sms;

    // Launch. All blocks of the grid must be co-resident for the group barriers, so groups * width <= num_sms.
    // With a known number of active experts, launch only as many groups as there are experts and widen them to
    // use the freed SMs, up to MOE_MAX_SMS_PER_EXPERT
    int block_dim = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;
    TORCH_CHECK(concurrency * MOE_SMS_PER_EXPERT <= num_sms, "Concurrency too high for device num_sms");
    int num_groups = MIN((int) concurrency, MOE_MAX_GROUPS);
    int group_size = xmoe_group_width > 0 ? MAX(MOE_SMS_PER_EXPERT, MIN(xmoe_group_width, MOE_MAX_SMS_PER_EXPERT)) : MOE_SMS_PER_EXPERT;
    group_size = MAX(MOE_SMS_PER_EXPERT, MIN(group_size, budget));  // a group never outgrows the launch's SM budget
    num_groups = MIN(num_groups, MAX(1, budget / group_size));
    if (num_active > 0)
    {
        num_groups = MIN(num_groups, num_active);
        group_size = MIN(budget / num_groups, MOE_MAX_SMS_PER_EXPERT);
    }
    dim3 grid_dim(group_size, 1, num_groups);

    int N_off = 0;
    if (hidden_dim % 256 == 0 && intermediate_dim % 256 == 0) N_off = 1;
    fp_exl3_moe_kernel kernel = ((int) bsz > xmoe_mt_min_bsz() ? exl3_moe_kernel_instances_mt : exl3_moe_kernel_instances)[4 * K + 2 * cb_idx + N_off];
    TORCH_CHECK(kernel != nullptr, "xmoe scaffold: kernel variant not built (K=2/3 MCG, K=3 MUL1)");
    // prefillbw: prefill cold launch (or hot with XMOE_PB_HOT) -> exl3_moe_kernel_pb
    if ((int) bsz > xmoe_mt_min_bsz() && K == 3 && cb_idx == 1 && N_off == 0 && act_function != MOE_ACT_RELU2_NOGATE)
    {
        xmoe_pb_init();
        if (xmoe_pb_cfg[0] > 0 && (xmoe_force_group || xmoe_pb_cfg[2] > 0))
        {
            fp_exl3_moe_kernel pk = xmoe_pb_kernel(xmoe_pb_cfg[0], xmoe_force_group && xmoe_pb_cfg[1] > 0);
            if (pk != nullptr)
            {
                static bool announced_pb = false;
                if (!announced_pb)
                {
                    announced_pb = true;
                    fprintf(stderr, "xmoe pb: prefill group launches -> exl3_moe_kernel_pb (stages %d, stage copy %s, hot launches %s)\n",
                            xmoe_pb_cfg[0], xmoe_pb_cfg[1] > 0 ? "on" : "off", xmoe_pb_cfg[2] > 0 ? "too" : "no");
                }
                kernel = pk;
            }
        }
    }

    if (moe_kernel_attr_set[device].find((void*) kernel) == moe_kernel_attr_set[device].end())
    {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_MAX);
        moe_kernel_attr_set[device].insert((void*) kernel);
        cuda_check(cudaPeekAtLastError());
    }

    void* _hidden_state = hidden_state.data_ptr();
    void* _temp_state_g = temp_state_g.data_ptr();
    void* _temp_state_u = temp_state_u.data_ptr();
    void* _temp_intermediate_g = temp_intermediate_g.data_ptr();
    void* _temp_intermediate_u = temp_intermediate_u.data_ptr();
    void* _output_state = output_state.data_ptr();

    void* _gate_ptrs_trellis = gate_ptrs_trellis.data_ptr();
    void* _gate_ptrs_suh = gate_ptrs_suh.data_ptr();
    void* _gate_ptrs_svh = gate_ptrs_svh.data_ptr();
    void* _up_ptrs_trellis = up_ptrs_trellis.data_ptr();
    void* _up_ptrs_suh = up_ptrs_suh.data_ptr();
    void* _up_ptrs_svh = up_ptrs_svh.data_ptr();
    void* _down_ptrs_trellis = down_ptrs_trellis.data_ptr();
    void* _down_ptrs_suh = down_ptrs_suh.data_ptr();
    void* _down_ptrs_svh = down_ptrs_svh.data_ptr();

    void* _expert_count = expert_count.data_ptr();
    void* _token_sorted = token_sorted.data_ptr();
    void* _weight_sorted = weight_sorted.data_ptr();

    void* kernelArgs[] =
    {
        &_hidden_state,
        &_temp_state_g,
        &_temp_state_u,
        &_temp_intermediate_g,
        &_temp_intermediate_u,
        &_output_state,
        &_gate_ptrs_trellis,
        &_gate_ptrs_suh,
        &_gate_ptrs_svh,
        &_up_ptrs_trellis,
        &_up_ptrs_suh,
        &_up_ptrs_svh,
        &_down_ptrs_trellis,
        &_down_ptrs_suh,
        &_down_ptrs_svh,
        &_expert_count,
        &_token_sorted,
        &_weight_sorted,
        (void*) &hidden_dim,
        (void*) &intermediate_dim,
        (void*) &num_experts,
        (void*) &num_experts_per_tok,
        (void*) &max_tokens_per_expert,
        (void*) &num_groups,
        (void*) &act_limit,
        (void*) &act_function,
        (void*) &K_gate,
        (void*) &K_up,
        (void*) &K_down,
        (void*) &locks
    };

    // xmoe flat scheduler for decode-sized batches (one block per SM, all experts per phase, see exl3_moe_flat_kernel.cuh)
    if ((int) bsz <= xmoe_flat_max_bsz() && bsz * num_experts_per_tok <= XMOE_FLAT_MAX_PAIRS &&
        bsz * num_experts_per_tok <= concurrency * max_tokens_per_expert && num_experts <= XMOE_FLAT_MAX_EXPERTS &&
        act_function != MOE_ACT_RELU2_NOGATE && !xmoe_force_group && (K == 2 || K == 3) && (cb_idx == 0 || K == 3) && (N_off == 0 || xmoe_flat_n256_ok()) && xmoe_flat_ok_k[cb_idx][K][device] >= 0)
    {
        // 2026-09-28: K=3 flat instance (3 bpw host-expert pack)
        int* xmoe_flat_ok = xmoe_flat_ok_k[cb_idx][K];
        fp_exl3_moe_kernel fk = K == 3 ? (cb_idx ? exl3_moe_flat_kernel_k3_n128_cb2() : exl3_moe_flat_kernel_k3_n128_cb1())
                                       : exl3_moe_flat_kernel_k2_n128_cb1();
        int fsmem = K == 3 ? exl3_moe_flat_smem_k3_n128() : exl3_moe_flat_smem_k2_n128();
        if (xmoe_flat_ok[device] == 0)
        {
            cudaFuncSetAttribute(fk, cudaFuncAttributeMaxDynamicSharedMemorySize, fsmem);
            int per_sm = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fk, block_dim, fsmem);
            cuda_check(cudaPeekAtLastError());
            xmoe_flat_ok[device] = per_sm >= 1 ? 1 : -1;
            if (per_sm < 1) fprintf(stderr, "xmoe flat: kernel does not fit one block per SM, using group kernel\n");
        }
        if (xmoe_flat_ok[device] > 0)
        {
            int fgrid = xmoe_flat_grid_env() > 0 ? MIN(xmoe_flat_grid_env(), budget) : budget;
            static bool announced = false;
            if (!announced) { announced = true; fprintf(stderr, "xmoe flat: active (K %d, grid %d, smem %d, max bsz %d)\n", K, fgrid, fsmem, xmoe_flat_max_bsz()); }
            cudaLaunchKernel((void*) fk, dim3(fgrid, 1, 1), block_dim, kernelArgs, fsmem, stream);
            cuda_check(cudaPeekAtLastError());
            return;
        }
    }

    cudaLaunchKernel
    (
        (void*) kernel,
        grid_dim,
        block_dim,
        kernelArgs,
        SMEM_MAX,
        stream
    );

    cuda_check(cudaPeekAtLastError());
}
