// prefillbw: xmoe_ds41pb prefill cold-path instances, pipeline depth 3 (K=3, MUL1 codebook, N=128), stage copy off / on
#include "exl3_moe_instances.cuh"
#include "../exl3_moe_kernel_pb.cuh"

fp_exl3_moe_kernel exl3_moe_kernel_pb_k3_n128_cb2_s3_st0() { return exl3_moe_kernel_pb<3, 128, 2, 3, false>; }
fp_exl3_moe_kernel exl3_moe_kernel_pb_k3_n128_cb2_s3_st1() { return exl3_moe_kernel_pb<3, 128, 2, 3, true>; }
