# Third-party code in `ignition/recipes/deepseek41/exl3/`

| Path | Origin | Licence | Our changes |
|---|---|---|---|
| `kernels/xmoe/` (all files: `bindings.cpp`, `compat.cuh`, `graph.cuh`, `ptx.cuh`, `util.cuh`, `util.h`, `quant/*.cu`, `quant/*.cuh`, `quant/comp_units/*`) | diffbot, `DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000`, `recipe/kernels/xmoe/` (Hugging Face, fetched 2026-11-16): their sm_120 rewrite of ExLlamaV3's `exl3_moe` extension (turboderp-org/exllamav3 @ 5be88657) | MIT, Copyright (c) 2025 Turboderp (`kernels/xmoe/LICENSE.exllamav3`) and MIT, Copyright (c) 2026 Diffbot (`kernels/xmoe/LICENSE.diffbot`) | `bindings.cpp`, `quant/exl3_devctx.cu`, `quant/exl3_devctx.cuh`: added `xmoe_set_lock_buffers` / `xmoe_lock_buffer_bytes` so Flare supplies the lock/barrier/scheduler buffers from its core storage (captured kernels then point at package-owned memory), marked `exp/ds41-exl3`; `bindings.cpp` also binds `xmoe_do_moe` (see the next row); every other file byte-identical except the new `quant/exl3_do.cu`; built through `brrr/cuext.build` as `ds41_xmoe_<digest>` for sm_120a with `--use_fast_math -lineinfo` |
| `kernels/xmoe/quant/exl3_do.cu` | diffbot, `recipe/kernels/xmoe_do/do_gemm.cu` (the fp16 decode-once prefill kernel, "xmoe_do", 2026-10-01; Hugging Face, fetched 2026-11-16) | MIT, Copyright (c) 2026 Diffbot (`kernels/xmoe/LICENSE.diffbot`); uses ExLlamaV3 device code, MIT, Copyright (c) 2025 Turboderp (`kernels/xmoe/LICENSE.exllamav3`) | ported by `exp/ds41-exl3` (2026-11-16): their `do_gemm_kernel` kept with one change (empty jobs return at once) and renamed `xmoe_do_gemm_kernel`; their gather/Hadamard, SwiGLU and scatter kernels kept with a skip for cold rows (`row_exp < 0`); NEW and ours: `xmoe_do_build_jobs` (the job table built on the device from the routing counts into a fixed-capacity table, so grids never depend on data and the op is CUDA-graph capturable; split-launch counts honoured, only VRAM slots get jobs) and the `xmoe_do_moe` entry point; only 3-bit MUL1/MCG instances; their Python job builder and their int8 build (`xmoe_do_i8`) not used. Bound in `bindings.cpp`, compiled into `ds41_xmoe_<digest>` |
| `kernels/xmoe/UPSTREAM-exl3_moe.patch` | diffbot, `recipe/kernels/exllamav3-5be8865-exl3_moe.patch` | MIT (as above) | none; kept as the record of their changes against upstream ExLlamaV3 |

Not used, by decision: diffbot's `recipe/patches/vllm-exl3/` (AGPL-3.0-only). Nothing in this folder copies or
paraphrases it; host-resident experts will use our own parking mechanism (`FLARE_DS41_HOST_EXPERT_LAYERS`).

`reference.py`, `pack_reader.py`, `do_reference.py` and `xmoe.py` are ours (Apache-2.0). `reference.py` implements the EXL3 format from
its definition in ExLlamaV3 (MIT); no source is copied.

Known behaviour difference kept from the kernel: the SwiGLU limit is applied after SiLU on the gate
(`min(silu(g), L) * clamp(u, -L, L)`, `quant/hadamard_inner.cuh`), the DeepSeek reference clamps before SiLU
(`silu(min(g, L))`). They differ only where g > ~10; `reference.swiglu(order=...)` models both.
