# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Functional MLA norm/quant outputs need no zero fill except the MMA-scale padding rows.

Runs b12x's ``mla_norm_mxfp8`` Triton kernel in Triton's CPU interpreter through
``PreparedMLANormQuant.__call__`` with every ``torch.empty`` poisoned (0x7F, 0xFF) and compares the
query, KV and encoded bases with a zero-filled run bit for bit.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch

ROWS = (1, 5, 128, 130)


def test_functional_norm_quant_outputs_match_zero_filled_outputs():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH")))))
    completed = subprocess.run([sys.executable, "-m", "tests.components.test_mla_norm_quant_poison_interpret"],
                               cwd=root, env=env, capture_output=True, text=True, timeout=1800)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "norm quant ok" in completed.stdout


def _call(rows, poison):
    from b12x.quantization.mla_norm_mxfp8._kernels import mla_norm_mxfp8_kernel
    from ignition.components import mla_norm_quant

    shapes = (((rows, 1280), torch.bfloat16), ((rows, 512), torch.bfloat16),
              ((rows, 1280), torch.float8_e4m3fn), ((1, rows, 40), torch.uint8),
              ((1, (rows + 127) // 128, 10, 32, 4, 4), torch.uint8))
    owner = object.__new__(mla_norm_quant.PreparedMLANormQuant)
    owner.query, owner.layouts, owner.plan = None, shapes, None
    generator = torch.Generator().manual_seed(rows)
    owner.weights = (torch.rand(1280, generator=generator) + .5, torch.rand(512, generator=generator) + .5)
    projected = torch.randn((rows, 1792), generator=generator).to(torch.bfloat16)

    def run(source, qw, kw, query, kv, values, scale_rows, scale_mma, *, plan):
        mla_norm_mxfp8_kernel[(rows, 2, 1)](source, qw, kw, query, kv, values, scale_rows, scale_mma,
                                             1280, 512, 1e-20, 1e-4, False, False)

    empty = torch.empty

    def poisoned(*args, **kwargs):
        tensor = empty(*args, **kwargs)
        if poison is not None and tensor.numel():
            tensor.view(torch.uint8).fill_(poison)
        return tensor

    saved = mla_norm_quant.mla_norm_mxfp8.run, torch.empty
    mla_norm_quant.mla_norm_mxfp8.run = run
    torch.empty = poisoned
    if poison is None:  # the reference: every output zero-filled, as before
        torch.empty = lambda *args, **kwargs: torch.zeros(*args, **kwargs)
    try:
        query, kv, bases = owner(projected)
    finally:
        mla_norm_quant.mla_norm_mxfp8.run, torch.empty = saved
    return [t.view(torch.uint8) for t in (query, kv, *bases)]


def _main():
    assert os.environ.get("TRITON_INTERPRET") == "1"
    for rows in ROWS:
        expected = _call(rows, None)
        for poison in (0x7F, 0xFF):
            actual = _call(rows, poison)
            for name, a, e in zip(("query", "kv", "values", "scale_rows", "scale_mma"), actual, expected,
                                  strict=True):
                assert torch.equal(a, e), f"rows {rows}: {name} differs under 0x{poison:02X}"
    print("norm quant ok")


if __name__ == "__main__":
    _main()
