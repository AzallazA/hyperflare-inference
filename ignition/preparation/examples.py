# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Share preparation operands while functional forwards allocate their own values."""

from math import prod

import torch


def declaration_tensor(examples, role, shape, dtype, device):
    device = torch.device(device)
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if examples is None:
        return torch.zeros(shape, dtype=dtype, device=device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = role, shape, dtype, device
    if key not in examples:
        storage_key = role, dtype, device
        storage = examples.get(storage_key)
        elements = prod(shape)
        if storage is None or storage.numel() < elements:
            storage = examples[storage_key] = torch.zeros(shape, dtype=dtype, device=device)
            # A superseded storage stays alive through the views earlier declarations hold (until the examples are
            # released), so callers build the largest shape of a role first; declaration_summary reports misses.
            stats = examples.setdefault(_ALLOCATIONS, {})
            stats.setdefault(storage_key, []).append(storage.nbytes)
        examples[key] = storage.view(-1)[:elements].view(shape)
    return examples[key]


_ALLOCATIONS = ("declaration_tensor.allocations",)


def declaration_summary(examples):
    """Bytes behind the shared declaration examples: ``allocated`` (every storage ever made), ``largest`` (the
    largest request per role, what one storage per role would hold) and ``superseded`` = allocated - largest, with
    the roles that grew after a smaller declaration ({role: superseded bytes})."""
    stats = {} if examples is None else examples.get(_ALLOCATIONS, {})
    allocated = sum(sum(sizes) for sizes in stats.values())
    largest = sum(max(sizes) for sizes in stats.values())
    grown = {key[0]: sum(sizes) - max(sizes) for key, sizes in stats.items() if len(sizes) > 1}
    return dict(roles=len(stats), allocated=allocated, largest=largest, superseded=allocated - largest, grown=grown)
