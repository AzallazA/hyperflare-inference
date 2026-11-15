# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The functional decode context-parallel combine hands its combiner an unfilled output.

b12x's PCIe reduce-scatter stores every element (checked on GPUs by test_glm53_dcp3_collectives_cuda, which
pre-fills the output with 7 and requires unattended rows and padded heads to come back zero). The RoCE merge is
run here in Triton's CPU interpreter on poisoned and zeroed outputs, including a head no rank attended; the
results must be identical.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch

from ignition.components import collectives


def test_functional_combine_does_not_fill_its_output(monkeypatch):
    owner = object.__new__(collectives.PreparedLseReduceScatter)
    owner.output, owner.roce, owner.handle = None, False, 11
    owner.output_layout = (2, 3, 8), (24, 8, 1), torch.bfloat16
    empty_strided, seen = torch.empty_strided, {}

    def poisoned(*args, **kwargs):
        tensor = empty_strided(*args, **kwargs)
        tensor.view(torch.uint8).fill_(0xFF)
        return tensor

    def combine(partial, lse, output, handle):
        seen["output"] = output.view(torch.uint8).clone()

    monkeypatch.setattr(torch, "empty_strided", poisoned)
    monkeypatch.setattr(collectives, "_lse_reduce_scatter", combine)
    output = owner(torch.zeros(2, 9, 8, dtype=torch.bfloat16), torch.zeros(2, 9))
    assert output.shape == (2, 3, 8) and (seen["output"] == 0xFF).all()


def test_roce_merge_stores_every_output_element():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH")))))
    completed = subprocess.run([sys.executable, "-m", "tests.components.test_lse_combine_unfilled_interpret"],
                               cwd=root, env=env, capture_output=True, text=True, timeout=600)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "merge ok" in completed.stdout


def _merge(fill, rank, world=2, rows=3, heads=4, dim=16):
    from ignition.components.head_exchange import merge, pack, pack_width

    generator = torch.Generator().manual_seed(17)
    partials = [torch.randn((rows, heads, dim), generator=generator).bfloat16() for _ in range(world)]
    lses = [torch.randn((rows, heads), generator=generator) for _ in range(world)]
    for lse in lses:
        lse[2] = -float("inf")  # A row no rank attended.
    sends = []
    for source in range(world):
        send = torch.zeros((rows, pack_width(world, heads, dim)), dtype=torch.bfloat16)
        pack(partials[source], lses[source], send, torch.tensor([source]), world)
        sends.append(send)
    output = torch.empty((rows, heads // world, dim), dtype=torch.bfloat16)
    output.view(torch.uint8).fill_(fill)
    merge(torch.cat(sends, dim=1), partials[rank], lses[rank], output, torch.tensor([rank]), world)
    return output


def _main():
    assert os.environ.get("TRITON_INTERPRET") == "1"
    for rank in range(2):
        expected = _merge(0, rank)
        assert not expected[2].count_nonzero()
        for fill in (0x7F, 0xFF):
            assert torch.equal(_merge(fill, rank).view(torch.int16), expected.view(torch.int16))
    print("merge ok")


if __name__ == "__main__":
    _main()
