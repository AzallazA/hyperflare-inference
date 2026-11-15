# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""gather_rows_three stores every element of its destinations (DS41 compressor pending state needs no fill).

Runs the Triton kernel in Triton's CPU interpreter on destinations poisoned with 0x7F/0xFF bytes and on
zeroed destinations; the gathered rows must be identical.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch


def test_gathered_rows_cover_every_destination_element():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH")))))
    completed = subprocess.run([sys.executable, "-m", "tests.components.test_gather_rows_three_interpret"],
                               cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "gather ok" in completed.stdout


def _gather(fill):
    import triton

    from ignition.components.state import _gather_rows_three

    generator = torch.Generator().manual_seed(5)
    sources = (torch.randn((9, 512), generator=generator), torch.randn((9, 512), generator=generator),
               torch.randint(-5, 50, (9, 1), generator=generator))
    rows = torch.tensor([3, 0, 8, 3, 1], dtype=torch.int64)
    outputs = tuple(torch.empty((rows.numel(), s.shape[1]), dtype=s.dtype) for s in sources)
    for output in outputs:
        output.view(torch.uint8).fill_(fill)
    width = max(o.shape[1] for o in outputs)
    _gather_rows_three[(rows.numel(), triton.cdiv(width, 1024))](
        *sources, rows, *outputs, *(s.stride(0) for s in sources), *(o.stride(0) for o in outputs),
        rows.stride(0), *(o.shape[1] for o in outputs), 1024)
    return outputs


def _main():
    expected = _gather(0)
    for fill in (0x7F, 0xFF):
        for actual, reference in zip(_gather(fill), expected, strict=True):
            assert torch.equal(actual, reference)
    print("gather ok")


if __name__ == "__main__":
    _main()
