#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel.cuh"
#include "../exl3_moe_flat_kernel.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_k3_n128_cb1() { return exl3_moe_kernel<3, 128, 1>; }
fp_exl3_moe_kernel exl3_moe_kernel_k3_n256_cb1() { return exl3_moe_kernel<3, 256, 1>; }
fp_exl3_moe_kernel exl3_moe_kernel_k3_n128_cb1_mt() { return exl3_moe_kernel<3, 128, 1, true>; }
fp_exl3_moe_kernel exl3_moe_kernel_k3_n256_cb1_mt() { return exl3_moe_kernel<3, 256, 1, true>; }
fp_exl3_moe_kernel exl3_moe_flat_kernel_k3_n128_cb1() { return exl3_moe_flat_kernel<3, 128, 1>; }
int exl3_moe_flat_smem_k3_n128() { return exl3_moe_flat_dyn_smem<3, 128>(); }
