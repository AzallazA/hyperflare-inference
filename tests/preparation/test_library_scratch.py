# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from ignition.preparation import library_scratch
from ignition.preparation.memory_inventory import Claim, attribute, cublas_workspace_claims

MiB = 1 << 20


def test_new_library_allocations_keep_only_large_new_device_allocations_outside_torch():
    before = {0x1000: (64 * MiB, "device")}
    after = {
        0x1000: (64 * MiB, "device"),             # existed before priming
        0x7000_0000_0000: (64 * MiB, "device"),   # cuBLAS's pool
        0x7100_0000_0000: (2 * MiB, "device"),    # small library state: kept exportable
        0x7200_0000_0000: (32 * MiB, "device"),   # a PyTorch segment still live
        0x7300_0000_0000: (16 * MiB, "mapped_host"),
    }
    torch_segments = [(0x7200_0000_0000 + MiB, 2 * MiB)]
    assert library_scratch.new_library_allocations(before, after, torch_segments) == [(0x7000_0000_0000, 64 * MiB)]


def test_live_inactive_drops_allocations_freed_or_resized_since_priming():
    scratch = [(0x10_0000, 64 * MiB), (0x900_0000, 16 * MiB)]
    recorded = {0x10_0000: (64 * MiB, "device"), 0x900_0000: (8 * MiB, "device")}
    assert library_scratch.live_inactive(scratch, recorded) == [(0x10_0000, 0x10_0000 + 64 * MiB)]
    assert library_scratch.live_inactive((), recorded) == []


def test_prime_is_a_no_op_without_a_recorder_or_when_disabled(monkeypatch):
    assert library_scratch.prime(None, 0) == []
    monkeypatch.setenv("FLARE_EXPORT_LIBRARY_SCRATCH", "0")
    images = SimpleNamespace(allocations={}, library_scratch=[])
    assert library_scratch.prime(images, 0) == []
    assert images.library_scratch == []


def test_cublas_workspaces_are_owned_outside_the_graph_pool():
    def block(address, requested, state="active_allocated"):
        return dict(address=address, size=requested, requested_size=requested, state=state)
    segments = [
        dict(segment_pool_id=(0, 1), blocks=[block(0x1000_0000, 32 * MiB), block(0x3000_0000, 8 * MiB),
                                              block(0x4000_0000, 32 * MiB, "inactive")]),
        dict(segment_pool_id=(0, 2), blocks=[block(0x5000_0000, 32 * MiB)]),   # graph pool
    ]
    claims = cublas_workspace_claims(segments, {32 * MiB}, (0, 2))
    assert [(c.start, c.end, c.category) for c in claims] == [(0x1000_0000, 0x1000_0000 + 32 * MiB, "cuBLAS workspaces")]
    rows = attribute([(0x1000_0000, 0x1000_0000 + 40 * MiB)], claims)
    by_category = {}
    for row in rows:
        by_category[row["category"]] = by_category.get(row["category"], 0) + row["nbytes"]
    assert by_category == {"cuBLAS workspaces": 32 * MiB, "Unknown storage": 8 * MiB}
    assert isinstance(claims[0], Claim)


def test_unknown_spans_are_the_retained_ranges_no_claim_covers():
    from ignition.preparation.memory_inventory import unknown_spans
    claims = [Claim(0x1000_0000, 0x1000_0000 + 32 * MiB, "cuBLAS workspaces", "w"),
              Claim(0x2000_0000, 0x2000_0000 + 4 * MiB, "Weights / numeric constants", "x")]
    retained = [(0x1000_0000, 0x1000_0000 + 64 * MiB), (0x2000_0000, 0x2000_0000 + 4 * MiB + 4096)]
    assert unknown_spans(retained, claims) == [(0x1000_0000 + 32 * MiB, 32 * MiB)]
    assert unknown_spans(retained, claims, minimum=1) == [(0x1000_0000 + 32 * MiB, 32 * MiB),
                                                          (0x2000_0000 + 4 * MiB, 4096)]


def _fake_cuda(monkeypatch, calls, recorded, *, fail=False):
    """Replace every torch call ``prime`` makes with recorders (CPU only); GEMMs add a cuBLAS pool to ``recorded``."""
    import contextlib
    import torch

    class Stream:
        def __init__(self, device=None):
            calls.append(("stream", device))

        def wait_stream(self, other):
            pass

    real_empty = torch.empty

    def factory(name):
        def make(*shape, dtype=None, device=None):
            calls.append((name, device))
            return real_empty(*shape, dtype=dtype, device="meta")
        return make

    def gemm(name):
        def run(*args, out=None, **kwargs):
            if fail:
                raise RuntimeError("no GEMM")
            calls.append((name, out.device, kwargs.get("out_dtype")))
            recorded.setdefault(0x7000_0000_0000, (64 * MiB, "device"))
        return run

    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: Stream(device))
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append(("empty_cache",)))
    monkeypatch.setattr(torch.cuda, "memory_snapshot", lambda: [])
    monkeypatch.setattr(torch._C, "_cuda_clearCublasWorkspaces", lambda: calls.append(("clear",)), raising=False)
    monkeypatch.setattr(torch, "zeros", factory("zeros"))
    monkeypatch.setattr(torch, "empty", factory("empty"))
    monkeypatch.setattr(torch, "mm", gemm("mm"))
    monkeypatch.setattr(torch, "addmm", gemm("addmm"))


def test_prime_builds_every_operand_on_the_ranks_cuda_device(monkeypatch):
    import torch

    calls, recorded = [], {}
    _fake_cuda(monkeypatch, calls, recorded)
    images = SimpleNamespace(allocations=recorded, library_scratch=[])
    assert library_scratch.prime(images, 2) == [(0x7000_0000_0000, 64 * MiB)]
    assert images.library_scratch == [(0x7000_0000_0000, 64 * MiB)]
    factories = [c for c in calls if c[0] in ("zeros", "empty")]
    assert factories and all(c[1] == torch.device("cuda", 2) for c in factories)
    assert ("mm", torch.device("meta"), torch.float32) in calls and ("addmm", torch.device("meta"), None) in calls
    assert calls[-2:] == [("clear",), ("empty_cache",)]


def test_prime_failure_is_logged_and_never_raises(monkeypatch, capsys):
    calls, recorded = [], {}
    _fake_cuda(monkeypatch, calls, recorded, fail=True)
    images = SimpleNamespace(allocations=recorded, library_scratch=[])
    assert library_scratch.prime(images, 0) == []
    assert images.library_scratch == []
    assert "library scratch priming failed: RuntimeError: no GEMM" in capsys.readouterr().err


def test_prime_with_nothing_new_logs_no_exclusion(monkeypatch, capsys):
    calls, recorded = [], {0x7000_0000_0000: (64 * MiB, "device")}   # cuBLAS's pool predates the recorder bracket
    _fake_cuda(monkeypatch, calls, recorded)
    images = SimpleNamespace(allocations=recorded, library_scratch=[])
    assert library_scratch.prime(images, 0) == []
    assert "nothing new" in capsys.readouterr().err
