# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 fork: export-speed switches (forward compiler workers after tuning, tune-only exports)."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from ignition.preparation import capture, export_speed
from ignition.preparation.execution import GraphSignature

_ENV = (export_speed.FORWARD_WORKERS_ENV, export_speed.FORWARD_WORKER_GIB_ENV,
        export_speed.FORWARD_WORKER_VRAM_MIB_ENV, export_speed.TUNE_ONLY_ENV)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def test_unset_switches_keep_upstream_behaviour():
    assert export_speed.forward_workers() is None
    assert export_speed.forward_worker_bytes() is None
    assert export_speed.forward_worker_device_bytes() is None
    assert export_speed.tune_only() is False


def test_switches_parse_and_reject_bad_values(monkeypatch):
    monkeypatch.setenv(export_speed.FORWARD_WORKERS_ENV, "9")
    monkeypatch.setenv(export_speed.FORWARD_WORKER_GIB_ENV, "16")
    monkeypatch.setenv(export_speed.TUNE_ONLY_ENV, "1")
    assert export_speed.forward_workers() == 9
    assert export_speed.forward_worker_bytes() == 16 << 30
    assert export_speed.tune_only() is True
    monkeypatch.setenv(export_speed.TUNE_ONLY_ENV, "0")
    assert export_speed.tune_only() is False
    for name, value in ((export_speed.FORWARD_WORKERS_ENV, "0"), (export_speed.FORWARD_WORKERS_ENV, "-3"),
                        (export_speed.FORWARD_WORKERS_ENV, "auto"), (export_speed.FORWARD_WORKER_GIB_ENV, "1.5"),
                        (export_speed.FORWARD_WORKER_VRAM_MIB_ENV, "0.75"),
                        (export_speed.TUNE_ONLY_ENV, "yes")):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match=name):
            {export_speed.TUNE_ONLY_ENV: export_speed.tune_only,
             export_speed.FORWARD_WORKERS_ENV: export_speed.forward_workers,
             export_speed.FORWARD_WORKER_GIB_ENV: export_speed.forward_worker_bytes,
             export_speed.FORWARD_WORKER_VRAM_MIB_ENV: export_speed.forward_worker_device_bytes}[name]()
        monkeypatch.delenv(name)


@pytest.mark.parametrize("status", [None, "collecting", "tuning", "contextual", "finalizing", "failed"])
def test_overrides_never_apply_before_tuning_completes(monkeypatch, status):
    monkeypatch.setenv(export_speed.FORWARD_WORKERS_ENV, "9")
    monkeypatch.setenv(export_speed.FORWARD_WORKER_GIB_ENV, "16")
    monkeypatch.setenv(export_speed.FORWARD_WORKER_VRAM_MIB_ENV, "768")
    session = SimpleNamespace(requested_compile_workers=3)
    if status is not None:
        session._tune_status = status
    assert export_speed.forward_compile_settings(session) == (3, None, None)


def test_overrides_apply_after_tuning_and_default_to_the_tuning_budget(monkeypatch):
    session = SimpleNamespace(requested_compile_workers=3, _tune_status="ready")
    assert export_speed.forward_compile_settings(session) == (3, None, None)
    monkeypatch.setenv(export_speed.FORWARD_WORKER_GIB_ENV, "16")
    assert export_speed.forward_compile_settings(session) == (3, 16 << 30, None)
    monkeypatch.setenv(export_speed.FORWARD_WORKERS_ENV, "9")
    monkeypatch.setenv(export_speed.FORWARD_WORKER_VRAM_MIB_ENV, "768")
    assert export_speed.forward_compile_settings(session) == (9, 16 << 30, 768 << 20)
    session.requested_compile_workers = None
    assert export_speed.forward_compile_settings(session) == (9, 16 << 30, 768 << 20)


def _tp3_rank(forward_worker_bytes=None, *, available=197 << 30, free=5 << 30, workers=9, device_bytes=None,
              integrated=False):
    """One TP3 rank as DS41 HOT 342 sees it after tuning: ~197 GiB host RAM available (sar, 11-16 02:50-03:00),
    ~5 GiB free per GPU (capture of 160815 started at 4,954 MiB free)."""
    from ignition.preparation import compile_pool, host_memory

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "mem_get_info", lambda device: (free, 32 << 30))
        patch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(is_integrated=integrated))
        patch.setattr(host_memory, "available_bytes", lambda: available)
        return compile_pool.compiler_resources("cuda:0", headroom_bytes=1 << 30, compile_workers=workers,
                                               forward_worker_bytes=forward_worker_bytes,
                                               forward_worker_device_bytes=device_bytes)


def test_forward_worker_estimate_override_admits_more_compilers_per_host():
    from ignition.preparation.compile_pool import allocate_workers

    default = _tp3_rank()
    assert (default.worker_bytes, default.forward_bytes, default.capacity, default.limit) == (32 << 30, None, 4, 9)
    # The upstream 32 GiB estimate admits (197 - 8) // 32 = 5 workers on this host (the port's automatic DS41 count).
    assert allocate_workers([default] * 3, [10] * 3, ["host"] * 3) == [2, 2, 1]
    smaller = _tp3_rank(16 << 30)
    assert (smaller.worker_bytes, smaller.forward_bytes) == (16 << 30, 16 << 30)
    assert allocate_workers([smaller] * 3, [10] * 3, ["host"] * 3) == [3, 3, 3]
    # Device admission still applies: 2 GiB free per GPU leaves room for one 1 GiB worker beside the 1 GiB headroom.
    tight = _tp3_rank(16 << 30, free=2 << 30)
    assert allocate_workers([tight] * 3, [10] * 3, ["host"] * 3) == [1, 1, 1]
    with pytest.raises(ValueError, match="forward_worker_bytes"):
        _tp3_rank(0)


def test_minimum_worker_allowance_uses_the_overridden_estimate():
    from ignition.preparation import host_memory
    from ignition.preparation.compile_pool import CompilerResources, allocate_workers

    # Below one 64 GiB estimate, the one-worker-per-host floor needs only the 2 GiB minimum beside the headroom.
    resources = [CompilerResources(8, 20 << 30, 16, 64 << 30, 8 << 30, 64 << 30)] * 2
    assert allocate_workers(resources, [10] * 2, ["a"] * 2) == [1, 0]
    resources = [CompilerResources(8, (8 << 30) + host_memory._WORKER_BYTES - 1, 16, 64 << 30, 8 << 30, 64 << 30)]
    with pytest.raises(MemoryError, match="no capacity"):
        allocate_workers(resources, [1], ["a"])


@pytest.mark.parametrize("status,expected", [("ready", (9, 16 << 30, 768 << 20)), ("contextual", (3, None, None))])
def test_capture_passes_forward_settings_to_the_forward_cache(monkeypatch, status, expected):
    stream = SimpleNamespace(wait_stream=lambda other: None, synchronize=lambda: None)
    for name, value in {"current_stream": lambda device: stream, "stream": lambda stream: nullcontext(),
                        "CUDAGraph": lambda **options: object(), "graph": lambda *args, **options: nullcontext(),
                        "synchronize": lambda device: None, "mem_get_info": lambda device: (1024, 2048),
                        "memory_allocated": lambda device: 0, "memory_reserved": lambda device: 0}.items():
        monkeypatch.setattr(capture.torch.cuda, name, value)
    monkeypatch.setattr(capture.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv(export_speed.FORWARD_WORKERS_ENV, "9")
    monkeypatch.setenv(export_speed.FORWARD_WORKER_GIB_ENV, "16")
    monkeypatch.setenv(export_speed.FORWARD_WORKER_VRAM_MIB_ENV, "768")
    seen = {}

    class Compiled:
        cache_hit = True
        capture_allocation_sizes = ()

        def prepare_capture(self):
            return 0

        def retire_after_capture(self):
            pass

        def __call__(self):
            pass

    class Cache:
        def __init__(self, **options):
            seen.update(options)

        def prepare_many(self, forwards, signatures, **options):
            return {"decode": Compiled()}

    monkeypatch.setattr(capture, "ForwardCache", Cache)
    session = SimpleNamespace(namespace=SimpleNamespace(to_dict=lambda: {}), capture=nullcontext,
                              requested_compile_workers=3, _tune_status=status)
    capture.capture_forwards({"decode": lambda: None}, {"decode": GraphSignature("decode", 1, 1)},
                             session=session, device="cpu", stream=stream, pool=None, report_catalog=False)
    assert (seen["compile_workers"], seen["forward_worker_bytes"], seen["forward_worker_device_bytes"]) == expected


def test_forward_cache_key_ignores_worker_settings():
    """The compiled-forward cache directory (and so every restored artifact) is independent of worker counts."""
    import inspect
    from ignition.preparation.aot import ForwardCache

    source = inspect.getsource(ForwardCache.__init__)
    key = source[source.index("options = {name: value"):source.index("root = DEFAULT_CACHE_ROOT")]
    assert "key = hashlib.sha256" in key and "!= 'compile_threads'" in key
    assert "compile_workers" not in key and "forward_worker_bytes" not in key


@pytest.mark.parametrize("free_mib,device_mib,per_rank", [
    # seq322 (DS41 TP3, 2026-11-17): 2,520 MiB free with one 700 MiB worker -> ~3.2 GiB before workers.
    (3220, None, 2), (2900, None, 1), (2900, 768, 2), (3220, 768, 2), (3328, 768, 3), (1800, 768, 1)])
def test_device_estimate_override_keeps_the_1_gib_headroom(free_mib, device_mib, per_rank):
    from ignition.preparation.compile_pool import allocate_workers

    rank = _tp3_rank(16 << 30, free=free_mib << 20, device_bytes=None if device_mib is None else device_mib << 20)
    assert rank.capacity == ((free_mib - 1024) // (device_mib or 1024))
    assert rank.capacity == per_rank and rank.free == free_mib << 20
    assert allocate_workers([rank] * 3, [10] * 3, ["host"] * 3) == [per_rank] * 3


def test_device_estimate_override_leaves_integrated_gpus_and_host_estimate_alone():
    shared = _tp3_rank(16 << 30, free=64 << 30, device_bytes=768 << 20, integrated=True)
    assert shared.device_bytes == 2 << 30 and shared.worker_bytes == (16 << 30) + (2 << 30)
    discrete = _tp3_rank(16 << 30, device_bytes=768 << 20)
    assert discrete.device_bytes == 768 << 20 and discrete.worker_bytes == 16 << 30
    with pytest.raises(ValueError, match="forward_worker_device_bytes"):
        _tp3_rank(device_bytes=0)


def test_admission_report_names_free_memory_and_admitted_workers():
    from ignition.preparation.compile_pool import admission_report, allocate_workers

    ranks = [_tp3_rank(16 << 30, free=3220 << 20, device_bytes=768 << 20)] * 3
    workers = allocate_workers(ranks, [10, 10, 10], ["host"] * 3)
    line = admission_report(ranks, [10, 10, 10], workers)
    assert line.startswith("Forward compiler admission: 6 workers (per worker 768 MiB device, 16384 MiB host;")
    assert "rank 2 free 3220 MiB, capacity 2, missing 10, admitted 2" in line
