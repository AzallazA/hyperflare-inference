# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Reuse computed opaque operands before enforcing their exact layouts.

TP3 fork: upstream a386501f, behind ``FLARE_EXPORT_SHARE_COMPUTED_OPERANDS`` (``1`` = on, unset/``0`` = off = the
lowering as before the pick). The release export sets it; the switch is part of the compiler cache key
(``runtime_patches.identity``), so packages compiled with and without it never share Inductor results. Drop the
switch at the next upstream sync (upstream has the behaviour unconditionally).
"""

import os

from . import _replace

SWITCH = "FLARE_EXPORT_SHARE_COMPUTED_OPERANDS"


def enabled() -> bool:
    value = os.environ.get(SWITCH, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{SWITCH} must be 0 or 1, got {value!r}")
    return value == "1"


def replacements():
    from torch._inductor import lowering

    lowering._flare_share_computed_operands = enabled
    constrain = _replace(lowering.constrain_to_fake_tensor, ((
        "    if isinstance(arg, ir.IRNode):\n"
        "        return ir.ExternKernel.require_exact_strides(arg, fake_arg.stride())\n",
        "    if isinstance(arg, ir.IRNode):\n"
        "        if (_flare_share_computed_operands() and isinstance(arg, ir.TensorBox)\n"
        "                and isinstance(arg.data, ir.StorageBox) and isinstance(arg.data.data, ir.Pointwise)):\n"
        "            node = V.graph.current_node\n"
        "            if (node is not None and isinstance(node.target, torch._ops.OpOverload)\n"
        "                    and node.target.name() == 'flare::storage_call'):\n"
        "                arg.realize()\n"
        "        return ir.ExternKernel.require_exact_strides(arg, fake_arg.stride())\n",
    ),), allow_applied=True)
    return [(lowering, 'constrain_to_fake_tensor', constrain)]
