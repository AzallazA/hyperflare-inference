# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Verification validity checks used columns only, exactly as the dummy-substituted check did."""

import pytest
import torch

from ignition.components.speculation import _distributions_valid, _sliced


def _dummy_substituted(probabilities, used):
    dummy = (torch.arange(probabilities.shape[-1]) == 0).float()
    if used is not None:
        probabilities = torch.where(used[:, :, None], probabilities, dummy)
    slices = _sliced(probabilities)
    entries = (torch.isfinite(slices) & (slices >= 0)).all(-1).all(-1)
    mass = slices.sum(-1).sum(-1)
    return (entries & torch.isfinite(mass) & ((mass - 1).abs() <= 1e-4)).all(-1)


@pytest.mark.parametrize("vocab", [256, 300])  # sliced and unsliced vocabularies
@pytest.mark.parametrize("width", [1, 3, 5])
def test_used_column_validity_matches_dummy_substitution(vocab, width):
    generator = torch.Generator().manual_seed(vocab * 10 + width)
    batch = 16
    probabilities = torch.rand((batch, width, vocab), generator=generator).softmax(-1)
    defects = (float("nan"), float("inf"), -1e-3, 0.5)
    for row in range(batch):
        column, kind = row % width, row % (len(defects) + 1)
        if kind < 2:
            probabilities[row, column, row % vocab] = defects[kind]
        elif kind == 2:
            probabilities[row, column, 1] = defects[2]
        elif kind == 3:
            probabilities[row, column] *= 1 + defects[3]
    limits = torch.randint(0, width + 2, (batch,), generator=generator)
    used = torch.arange(width)[None, :] < limits[:, None]
    for mask in (None, used, torch.zeros_like(used), torch.ones_like(used)):
        expected = _dummy_substituted(probabilities, mask)
        assert torch.equal(_distributions_valid(probabilities, mask), expected)
    # A defect invalidates its row exactly when its column is used.
    defective = ~_distributions_valid(probabilities, None)
    unused_only = defective & ~(used & ~_column_valid(probabilities)).any(-1)
    assert _distributions_valid(probabilities, used)[unused_only].all()
    assert not _distributions_valid(probabilities, used)[(used & ~_column_valid(probabilities)).any(-1)].any()


def _column_valid(probabilities):
    return torch.stack([_distributions_valid(probabilities[:, column:column + 1], None)
                        for column in range(probabilities.shape[1])], -1)
