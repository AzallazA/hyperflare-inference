# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 fork (DS41 tp3.rotary_once): rotations from the per-forward shared (cos, sin) table are bit-identical to the
per-launch rotations they replace.

Runs in Triton's CPU interpreter (a subprocess, TRITON_INTERPRET=1): builds the DS41 ``AttentionIndices`` with a
``Rotary`` on metadata whose positions span 0, 30K, the sinf/cosf range-reduction switch (105,000 / 105,615 /
105,616 / 106,000), 130K, 520K, 1M-1 and padding rows (-1), takes its ``angles``, and rotates query-shaped
(rows x 24 x 512), KV-window-shaped (rows x 512) and index-query-shaped (rows x 32 x 128) BF16 tensors both ways
(also the inverse rotation), under both frequency sets; every output must be bit-identical. A negative control shows
the table must not serve the ratio-2 compressor positions (-2 at position 0 and on padding rows): the clamped table
rotates those rows differently. The interpreter evaluates sin/cos with NumPy in both paths, so this proves the
plumbing (arguments, signs, strides, offsets, clamping, keys); the GPU proof of libdevice identity is the B1 chain's
microbench (same expression in both kernels, analysis/work/decode-small-levers-20261118/IMPL.md).
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

POSITIONS = [0, 1, 30000, 105000, 105615, 105616, 106000, 130000, 520000, 1048575, -1, -1]
TEXT = {"rope_theta": 10000, "compress_rope_theta": 160000,
        "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
                         "original_max_position_embeddings": 65536}}


def test_shared_angles_rotate_bit_identically_in_the_cpu_interpreter():
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH")))))
    completed = subprocess.run([sys.executable, "-m", "tests.components.test_rotary_shared_angles_interpret"],
                               cwd=root, env=env, capture_output=True, text=True, timeout=900)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "rotary shared ok 12" in completed.stdout, completed.stdout + completed.stderr
    assert "negative control differs" in completed.stdout, completed.stdout


def test_switch_is_off_by_default_parsed_and_recorded():
    import pytest

    from ignition.recipes.deepseek41.tp3_config import TP3Config

    assert TP3Config().rotary_once is False and "rotary_once" not in TP3Config().to_metadata()
    on = TP3Config.from_environment({"FLARE_DS41_ROTARY_ONCE": "1"})
    assert on.rotary_once is True and on.to_metadata()["rotary_once"] is True and "rotary_once=True" in on.describe()
    with pytest.raises(TypeError):
        TP3Config(rotary_once=1)


def test_indices_without_rotary_share_nothing():
    from ignition.components.paged_batch import Metadata
    from ignition.recipes.deepseek41.attention import AttentionIndices

    m = Metadata.allocate(6, 1, 8, torch.device("cpu"), max_query_tokens=6)
    indices = AttentionIndices(m, mode="verify", tokens=6, page_size=134, width=128, ratios=(2,))
    assert indices.shared_angles(True) is None and indices.shared_angles(False) is None
    assert indices.angle_operations == {}


def _metadata(tokens):
    from ignition.components.paged_batch import Metadata

    m = Metadata.allocate(tokens, 1, 8, torch.device("cpu"), max_query_tokens=tokens)
    m.positions.copy_(torch.tensor(POSITIONS))
    m.requests.zero_()
    m.starts.zero_()
    m.lengths.fill_(tokens)
    m.state_rows.fill_(1)
    return m


def _main():
    from ignition.components.rotary import CompressedRotary, PreparedRotaryAngles, apply_shared_angles
    from ignition.recipes.deepseek41.attention import AttentionIndices

    torch.manual_seed(7)
    tokens = len(POSITIONS)
    rotary = CompressedRotary(SimpleNamespace(text=TEXT), torch.device("cpu"))
    m = _metadata(tokens)
    indices = AttentionIndices(m, mode="verify", tokens=tokens, page_size=128 + tokens, width=128, ratios=(1, 2),
                               rotary=rotary)
    keys = {indices.shared_angles(False), indices.shared_angles(True)}
    assert keys == {False, True} and indices.shared_angles(True) is True  # one table per frequency set
    geometry = indices(m)
    assert set(geometry.angles) == {False, True}
    position = geometry.position
    assert position.tolist() == [max(p, 0) for p in POSITIONS]
    checked = 0
    for compressed in (False, True):
        table = geometry.angles[compressed]
        assert table.shape == (tokens, 32, 2) and table.dtype == torch.float32
        for shape in ((tokens, 24, 512), (tokens, 512), (tokens, 32, 128)):
            for inverse in (False, True):
                values = (torch.randn(shape) * 4).to(torch.bfloat16)
                expected, actual = values.clone(), values.clone()
                rotary.apply(expected, position, compressed=compressed, inverse=inverse)
                apply_shared_angles(actual, table, rotary.frequencies[compressed], inverse=inverse)
                assert torch.equal(expected.view(torch.int16), actual.view(torch.int16)), (compressed, shape, inverse)
                assert not torch.equal(expected, values)  # the rotation did something
                checked += 1
    # Negative control: the ratio-2 compressor positions ((p + 1) // 2 * 2 - 2) are -2 at p = 0 and on padding rows;
    # the clamped table rotates those rows by position 0, so the compressor rotations must stay per launch.
    compressor = geometry.ratios[2].position
    assert (compressor < 0).any()
    values = (torch.randn((tokens, 512)) * 4).to(torch.bfloat16)
    expected, actual = values.clone(), values.clone()
    rotary.apply(expected, compressor, compressed=True)
    clamped = PreparedRotaryAngles(compressor, rotary.frequencies[True])()  # _angles clamps at 0
    apply_shared_angles(actual, clamped, rotary.frequencies[True])
    negative = compressor < 0
    assert torch.equal(expected[~negative], actual[~negative])  # equal wherever the position is >= 0
    assert not torch.equal(expected[negative], actual[negative])
    print("negative control differs")
    print(f"rotary shared ok {checked}")


if __name__ == "__main__":
    _main()
