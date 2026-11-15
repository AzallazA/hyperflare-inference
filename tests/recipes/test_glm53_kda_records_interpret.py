# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""GLM KDA verification records need no zero fill: the commit reads only what verification wrote.

``SpeculativePreparation.records`` hands out uninitialised record planes. This runs the Triton
recording verification and accepted-prefix commit (``kda_recovery``) in Triton's CPU interpreter on
planes poisoned with 0x7F and 0xFF bytes (recycled graph-pool memory) and on zeroed planes, over
GLM's verification metadata conventions: active requests with full and short query lengths, an
inactive request on the null row, a dead padded request past ``num_seqs``, and counts bounded by
query lengths. The committed states and outputs must match bit for bit.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch

LAYERS, SLOTS, HEADS, WINDOW = 2, 6, 2, 4
# (slot, query length, committed count) per request; slot 0 is the null row.
REQUESTS = ((3, 4, 3), (0, 0, 1), (5, 2, 2), (4, 3, 0), (0, 0, 0))
LIVE = 4  # the last request is a padded row past num_seqs (target_metadata gives it the null row)


def test_uninitialised_kda_records_commit_like_zeroed_records():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH")))))
    completed = subprocess.run([sys.executable, "-m", "tests.recipes.test_glm53_kda_records_interpret"],
                               cwd=root, env=env, capture_output=True, text=True, timeout=1800)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "records ok" in completed.stdout


def _run(records_fill, requests=REQUESTS):
    from ignition.recipes.glm53 import kda_recovery as k

    generator = torch.Generator().manual_seed(7)
    lengths = torch.tensor([length for _, length, _ in requests])
    starts = torch.cat((torch.zeros(1, dtype=torch.int64), lengths.cumsum(0))).to(torch.int32)
    tokens = int(lengths.sum())
    slots = torch.tensor([[slot, 0] for slot, _, _ in requests], dtype=torch.int32)
    counts = torch.tensor([count for _, _, count in requests], dtype=torch.int32)
    num_seqs = torch.tensor([LIVE], dtype=torch.int32)
    row_floats = LAYERS * HEADS * 128 * 128
    backing = torch.randn((SLOTS, row_floats), generator=generator) * 0.1
    corrections = torch.empty((LAYERS, SLOTS, HEADS, WINDOW, 128), dtype=torch.float32)
    keys_gates = torch.empty((LAYERS, SLOTS, HEADS, WINDOW, 256), dtype=torch.bfloat16)
    for plane in (corrections, keys_gates):
        if records_fill is None:
            plane.zero_()
        else:
            plane.view(torch.uint8).fill_(records_fill)
    a_log = torch.randn((LAYERS, HEADS), generator=generator) * 0.1
    dt_bias = torch.randn((LAYERS, HEADS, 128), generator=generator) * 0.1
    outputs = []
    for layer in range(LAYERS):
        mixed = (torch.randn((tokens, 3 * HEADS * 128), generator=generator)).to(torch.bfloat16)
        gate = (torch.randn((tokens, HEADS * 128), generator=generator)).to(torch.bfloat16)
        beta = (torch.randn((tokens, HEADS), generator=generator)).to(torch.bfloat16)
        state = backing[:, layer * HEADS * 128 * 128:(layer + 1) * HEADS * 128 * 128].view(SLOTS, HEADS, 128, 128)
        output = torch.full((tokens, HEADS * 128), 3.0, dtype=torch.bfloat16)
        k._verify[(128 // k.ROWS, len(requests) * HEADS, 1)](
            mixed, gate, beta, a_log[layer], dt_bias[layer], state, starts, slots[:, 0], num_seqs, output,
            corrections[layer], keys_gates[layer], mixed.stride(0), gate.stride(0), beta.stride(0),
            state.stride(0), slots.stride(0), corrections.stride(1), keys_gates.stride(1),
            HEADS=HEADS, SCALE=128 ** -0.5, LOWER=-5.0, WINDOW=WINDOW, ROWS=k.ROWS, NULL_SLOT=0)
        outputs.append(output)
    offsets = torch.tensor([layer * HEADS * 128 * 128 for layer in range(LAYERS)], dtype=torch.int64)
    k._commit[(128 // k.ROWS, len(requests), LAYERS * HEADS)](
        backing, offsets, corrections, keys_gates, a_log, dt_bias, slots[:, 0], counts, num_seqs,
        row_floats, slots.stride(0), SLOTS=SLOTS, HEADS=HEADS, LOWER=-5.0, WINDOW=WINDOW, ROWS=k.ROWS,
        NULL_SLOT=0)
    return backing, outputs


def _main():
    assert os.environ.get("TRITON_INTERPRET") == "1"
    # Counts never exceed the verified query length, and dead/null rows are skipped by both kernels.
    for slot, length, count in REQUESTS[:LIVE]:
        assert slot == 0 or count <= length
    expected_backing, expected_outputs = _run(None)
    for fill in (0x7F, 0xFF):
        backing, outputs = _run(fill)
        assert torch.equal(backing, expected_backing), f"committed state differs with 0x{fill:02X} records"
        for actual, expected in zip(outputs, expected_outputs, strict=True):
            assert torch.equal(actual, expected)
    # The check is sensitive: committing past a verified prefix reads unwritten records.
    overrun = tuple((slot, length, length + 1) if slot == 5 else (slot, length, count)
                    for slot, length, count in REQUESTS)
    assert not torch.equal(_run(0x7F, overrun)[0], _run(None, overrun)[0])
    print("records ok")


if __name__ == "__main__":
    _main()
