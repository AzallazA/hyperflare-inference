# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""The export progress line listing every compiled forward's Inductor graph-pool requests, largest first."""

from types import SimpleNamespace

from ignition.preparation.capture import pool_request_report


class Eager:
    @property
    def capture_allocation_sizes(self):
        raise AttributeError("no Inductor memory plan")


def test_pool_requests_are_listed_largest_first_in_mib():
    mib = 1 << 20
    compiled = {"decode1": SimpleNamespace(capture_allocation_sizes=(3 * mib,)),
                "prefill4096": SimpleNamespace(capture_allocation_sizes=(1024 * mib, 512 * mib)),
                "vision256": SimpleNamespace(capture_allocation_sizes=(1500 * mib,)),
                "resume8": SimpleNamespace(capture_allocation_sizes=()),
                "eager": Eager()}
    line = pool_request_report(compiled)
    assert line.startswith("Graph pool requests (MiB, 4 compiled forwards, largest first; max 1500.0 in vision256): ")
    entries = line.split(": ", 1)[1].split(", ")
    assert entries == ["vision256 1500.0", "prefill4096 1536.0 (1024.0+512.0)", "decode1 3.0", "resume8 0.0"]


def test_no_memory_plan_is_reported_plainly():
    assert pool_request_report({"eager": Eager()}) == "Graph pool requests: no compiled forward reports a memory plan"
