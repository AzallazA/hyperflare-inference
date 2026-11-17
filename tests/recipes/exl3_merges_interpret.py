# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0
"""Run under TRITON_INTERPRET=1 (CPU) by test_deepseek41_exl3_merges.py: the merged inverse-RoPE quantizer
(exl3/merges.py) against the op sequence it replaces, rotary.py ``_rotate`` (inverse, in place) then b12x
``_quantize_grouped_tgd_to_tdg_kernel`` with the arguments b12x's ``_WOQuantizers.quantize_a`` passes. Prints
``ROPEQ INTERPRET OK <cases>`` when every output byte (values, both scale layouts, rotated source) matches."""
import sys

import torch
from b12x.gemm._shared import wo_mxfp8 as w

from ignition.components import rotary
from ignition.recipes.deepseek41.exl3 import merges


def reference(source, positions, freq, chunks):
    tokens, heads, head_dim = source.shape
    rotary._rotate[(tokens, heads)](source, positions, freq, source.stride(0), source.stride(1), head_dim - 64, -1,
                                    None, False)
    tgd = source.view(tokens, -1, 4096)
    out = w.empty_mxfp8_rows_for_dense_gemm(tokens, 4096, num_groups=tgd.shape[1], device="cpu")
    out.values.zero_(); out.scale_rows.view(torch.uint8).zero_(); out.scale_mma.view(torch.uint8).zero_()
    w._quantize_grouped_tgd_to_tdg_kernel[(tokens, tgd.shape[1], 4096 // 32 // chunks)](
        tgd, out.values, out.scale_rows.view(torch.uint8), out.scale_mma.view(torch.uint8), tokens, tgd.shape[1], 4096,
        tgd.stride(0), tgd.stride(1), tgd.stride(2), out.values.stride(0), out.values.stride(1),
        out.values.stride(2) if out.values.ndim == 3 else 0, *out.scale_mma.stride(), chunks)
    return out


def merged(source, positions, freq, chunks, write_back=True):
    tokens = source.shape[0]
    tgd = source.view(tokens, -1, 4096)
    out = w.empty_mxfp8_rows_for_dense_gemm(tokens, 4096, num_groups=tgd.shape[1], device="cpu")
    out.values.zero_(); out.scale_rows.view(torch.uint8).zero_(); out.scale_mma.view(torch.uint8).zero_()
    merges.inv_rope_quantize(tgd, positions, freq, out, chunks=chunks, head_dim=source.shape[-1],
                             nope_dim=source.shape[-1] - 64, write_back=write_back)
    return out


def main():
    cases = 0
    for tokens, groups, chunks, scale in ((1, 4, 16, 0.3), (3, 4, 16, 40.0), (2, 1, 8, 1e-3), (5, 2, 4, 0.05)):
        g = torch.Generator().manual_seed(tokens * 100 + groups)
        source = (torch.randn(tokens, groups * 8, 512, generator=g) * scale).to(torch.bfloat16)
        source[0, 0, :32] = 0                      # an all-zero 32-block (scale 1 path)
        source[-1, -1, 448:480] = 1e-39            # bf16 denormals in a rope block
        positions = torch.randint(0, 1 << 20, (tokens,), generator=g, dtype=torch.int64)
        positions[0] = 0
        freq = (1 / 10000 ** (torch.arange(32, dtype=torch.float32) / 32)).contiguous()
        a, b = source.clone(), source.clone()
        ra, rb = reference(a, positions, freq, chunks), merged(b, positions, freq, chunks)
        for name, x, y in (("values", ra.values.view(torch.uint8), rb.values.view(torch.uint8)),
                           ("scale_rows", ra.scale_rows.view(torch.uint8), rb.scale_rows.view(torch.uint8)),
                           ("scale_mma", ra.scale_mma.view(torch.uint8), rb.scale_mma.view(torch.uint8)),
                           ("source", a.view(torch.int16), b.view(torch.int16))):
            if not torch.equal(x, y):
                print(f"ROPEQ INTERPRET MISMATCH {name} tokens={tokens} groups={groups} chunks={chunks}: "
                      f"{int((x != y).sum())} of {x.numel()}", flush=True)
                return 1
        c = source.clone()                         # the decoder's launch: no write-back, same bytes, source untouched
        rc = merged(c, positions, freq, chunks, write_back=False)
        if not (torch.equal(rc.values.view(torch.uint8), rb.values.view(torch.uint8))
                and torch.equal(rc.scale_mma.view(torch.uint8), rb.scale_mma.view(torch.uint8))
                and torch.equal(c.view(torch.int16), source.view(torch.int16))):
            print(f"ROPEQ INTERPRET MISMATCH no-write-back tokens={tokens}", flush=True)
            return 1
        cases += 1
    print(f"ROPEQ INTERPRET OK {cases}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
