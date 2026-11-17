#include <torch/extension.h>
#include "quant/exl3_moe.cuh"
#include "quant/exl3_devctx.cuh"
void xdense_forward(at::Tensor x, at::Tensor suh, at::Tensor temp, at::Tensor a_ptr, at::Tensor b_ptr, at::Tensor c_ptr,
                    at::Tensor b_bnf, at::Tensor rows_t, at::Tensor svh, at::Tensor out, at::Tensor out_off, int64_t ld_out,
                    int64_t W, int64_t bits, int64_t cb);
// exp/ds41-exl3 (2026-11-16): decode-once prefill MoE, ported from diffbot's xmoe_do (quant/exl3_do.cu)
// exp/ds41-exl3-next (2026-11-16): fused decode routing prep (quant/exl3_prep.cu)
void xmoe_decode_prep(at::Tensor ids, at::Tensor weights, at::Tensor hidden, int64_t hot_rows,
                      at::Tensor token_sorted, at::Tensor weight_sorted, at::Tensor counts_hot, at::Tensor counts_cold,
                      at::Tensor x16, at::Tensor acc);
void xmoe_do_moe(at::Tensor x, at::Tensor out, at::Tensor tok, at::Tensor wts, at::Tensor row_exp, at::Tensor counts,
                 int64_t hot_rows, at::Tensor job_i32, at::Tensor job_ptr, at::Tensor overflow,
                 at::Tensor g_tr, at::Tensor g_suh, at::Tensor g_svh, at::Tensor u_tr, at::Tensor u_suh, at::Tensor u_svh,
                 at::Tensor d_tr, at::Tensor d_suh, at::Tensor d_svh,
                 at::Tensor xg, at::Tensor xu, at::Tensor cg, at::Tensor cu,
                 int64_t bits, int64_t cb, double act_limit, int64_t act_fn);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("xmoe_decode_prep", &xmoe_decode_prep, "exp/ds41-exl3-next: fused decode routing prep (sort, counts, hot/cold split, fp16 x, zeroed acc)");
    m.def("xmoe_do_moe", &xmoe_do_moe, "exp/ds41-exl3: decode-once prefill MoE (hot experts; fixed-capacity device job table)");
    m.def("xdense_forward", &xdense_forward, "flat decode GEMM for EXL3 dense projections (<=16 rows)");
    m.def("exl3_moe_max_concurrency", &exl3_moe_max_concurrency, "exl3_moe_max_concurrency");
    m.def("exl3_moe", &exl3_moe, "exl3_moe");
    m.def("xmoe_set_lock_slot", &xmoe_set_lock_slot, "1 = alternate lock buffer (side-stream launch)");
    m.def("xmoe_set_sm_budget", &xmoe_set_sm_budget, "cap the SMs a launch may occupy (0 = all)");
    m.def("xmoe_set_group_width", &xmoe_set_group_width, "group-kernel SMs per expert when the active count is unknown (0 = 8)");
    m.def("xmoe_set_flat_bulk", &xmoe_set_flat_bulk, "exp/ds41-exl3-final-bulk: 1 = K3 MUL1 flat launches (rows x top-k <= 384) take diffbot's xmoe_bulk hot loop");
    m.def("xmoe_set_force_group", &xmoe_set_force_group, "force the group (non-flat) kernel for subsequent exl3_moe calls");
    m.def("xmoe_pb_set", &xmoe_pb_set, "prefillbw: (stages 0=off/3/4/6/8, stage copy 0/1, hot 0/1); -1 keeps a field");
    m.def("xmoe_set_lock_buffers", &xmoe_set_lock_buffers, "exp/ds41-exl3: use caller-owned zeroed lock buffers (device, main, alt)");
    m.def("xmoe_lock_buffer_bytes", &xmoe_lock_buffer_bytes, "bytes of one lock buffer");
    m.def("xmoe_pb_get", &xmoe_pb_get, "prefillbw: [stages, stage copy, hot]");
}
