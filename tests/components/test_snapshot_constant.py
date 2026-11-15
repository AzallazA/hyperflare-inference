# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""flare.snapshot realises a constant once for every opaque reader (CPU Inductor).

Without it Inductor re-materialises a constant metadata tensor (GLM verification's ``fresh``) before every
mutating custom op that reads it: one fill launch per KDA layer.
"""

import re

import torch
from torch._inductor.lowering import register_lowering
from torch._inductor.utils import run_and_get_code

from ignition.preparation import compilation


@torch.library.custom_op("flare_test::fresh_reader", mutates_args=("out",))
def _reader(x: torch.Tensor, fresh: torch.Tensor, out: torch.Tensor) -> None:
    out.copy_(x + fresh.to(x.dtype))


@_reader.register_fake
def _(x, fresh, out):
    pass


def _forward(x, snapshot):
    fresh = torch.zeros(1, dtype=torch.bool)
    if snapshot:
        fresh = torch.ops.flare.snapshot(fresh)
    for _ in range(6):
        out = torch.empty_like(x)
        torch.ops.flare_test.fresh_reader(x, fresh, out)
        x = out
    return x * 1


def _kernels(snapshot):
    register_lowering(torch.ops.flare.snapshot.default)(compilation._snapshot_lowering)
    torch._dynamo.reset()
    x = torch.randn(4)
    result, (code, *_) = run_and_get_code(
        torch.compile(lambda value: _forward(value, snapshot), backend="inductor", fullgraph=True), x)
    torch.testing.assert_close(result, x, rtol=0, atol=0)
    return re.findall(r"(cpp_fused_\w+)\(", code[code.index("def call("):])


def test_snapshot_realises_a_constant_once_for_every_reader():
    assert len(_kernels(False)) >= 6
    assert len([k for k in _kernels(True) if "zeros" in k]) == 1
