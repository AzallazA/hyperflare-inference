#pragma once

#include "../exl3_moe_common.cuh"

typedef void (*fp_exl3_moe_kernel) (EXL3_MOE_KERNEL_ARGS);

#define EXL3_MOE_DECLARE_GETTERS(K) \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb1(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb1(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n128_cb2(); \
    fp_exl3_moe_kernel exl3_moe_kernel_k##K##_n256_cb2(); \

EXL3_MOE_DECLARE_GETTERS(0);
EXL3_MOE_DECLARE_GETTERS(1);
EXL3_MOE_DECLARE_GETTERS(2);
EXL3_MOE_DECLARE_GETTERS(3);
EXL3_MOE_DECLARE_GETTERS(4);
EXL3_MOE_DECLARE_GETTERS(5);
EXL3_MOE_DECLARE_GETTERS(6);
EXL3_MOE_DECLARE_GETTERS(7);
EXL3_MOE_DECLARE_GETTERS(8);

#undef EXL3_MOE_DECLARE_GETTERS

extern fp_exl3_moe_kernel exl3_moe_kernel_instances[];
fp_exl3_moe_kernel exl3_moe_kernel_k2_n128_cb1_mt();
fp_exl3_moe_kernel exl3_moe_kernel_k2_n256_cb1_mt();
fp_exl3_moe_kernel exl3_moe_kernel_k3_n128_cb1_mt();
fp_exl3_moe_kernel exl3_moe_kernel_k3_n256_cb1_mt();
extern fp_exl3_moe_kernel exl3_moe_kernel_instances_mt[];
fp_exl3_moe_kernel exl3_moe_flat_kernel_k2_n128_cb1();
int exl3_moe_flat_smem_k2_n128();
fp_exl3_moe_kernel exl3_moe_flat_kernel_k3_n128_cb1();
int exl3_moe_flat_smem_k3_n128();
fp_exl3_moe_kernel exl3_moe_kernel_k3_n128_cb2_mt();
fp_exl3_moe_kernel exl3_moe_kernel_k3_n256_cb2_mt();
fp_exl3_moe_kernel exl3_moe_flat_kernel_k3_n128_cb2();
// prefillbw (xmoe_ds41pb): prefill cold-path group kernel, pipeline depth s, stage copy st (exl3_moe_kernel_pb.cuh)
#define XMOE_PB_DECLARE(S) \
    fp_exl3_moe_kernel exl3_moe_kernel_pb_k3_n128_cb2_s##S##_st0(); \
    fp_exl3_moe_kernel exl3_moe_kernel_pb_k3_n128_cb2_s##S##_st1();
XMOE_PB_DECLARE(3)
XMOE_PB_DECLARE(4)
XMOE_PB_DECLARE(6)
XMOE_PB_DECLARE(8)
#undef XMOE_PB_DECLARE
