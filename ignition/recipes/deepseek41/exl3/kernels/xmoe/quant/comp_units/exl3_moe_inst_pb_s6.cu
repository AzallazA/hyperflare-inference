// prefillbw: xmoe_ds41pb prefill cold-path instances, pipeline depth 6 (K=3, MUL1 codebook, N=128), stage copy off / on
#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel_pb.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_pb_k3_n128_cb2_s6_st0() { return exl3_moe_kernel_pb<3, 128, 2, 6, false>; }
fp_exl3_moe_kernel exl3_moe_kernel_pb_k3_n128_cb2_s6_st1() { return exl3_moe_kernel_pb<3, 128, 2, 6, true>; }
