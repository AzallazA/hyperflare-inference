# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Compile prepared model forwards and capture CUDA graphs for their shapes."""

import os
from contextlib import nullcontext
from time import monotonic

import torch

from ignition.progress import graph_catalog, phase

from .aot import ForwardCache
from .collectives import warmup_wait
from .compilation import _defer_full_collection
from .export_speed import forward_compile_settings
from .startup_tuning import capture_consumers
from .runtime_core import capture_graph, instantiate_graph, record_export_graph
from .host_memory import COMPILE_HEADROOM_BYTES


def _capture_order(names, compiled, coordinator):
    sizes = {name: forward.capture_allocation_sizes for name, forward in compiled.items()}
    ranks = [(names, sizes)]
    if coordinator is not None:
        ranks = [None] * len(coordinator.ranks)
        torch.distributed.all_gather_object(ranks, (names, sizes), group=coordinator.group)
        if any(selected != names or requests.keys() != sizes.keys() for selected, requests in ranks):
            raise ValueError("tensor-parallel capture requires matching forward catalogs")

    def priority(name):
        requests = [rank_sizes[name] for _, rank_sizes in ranks if name in rank_sizes]
        return (bool(requests), max((max(sizes, default=0) for sizes in requests), default=0),
                max((sum(sizes) for sizes in requests), default=0))

    return tuple(sorted(names, key=priority, reverse=True))


def pool_request_report(compiled):
    """One line: every compiled forward's Inductor graph-pool requests (``capture_allocation_sizes``), largest first.

    TP3 fork diagnostic for the Inductor-planned memory scheme: all serving graphs allocate their combined pools from
    one shared graph pool, so the largest request bounds the "Graph capture pool" and the KV that remains."""
    rows = []
    for name, forward in compiled.items():
        try:
            sizes = tuple(forward.capture_allocation_sizes)
        except AttributeError:  # an eager or guarded forward without an Inductor memory plan
            continue
        rows.append((max(sizes, default=0), sum(sizes), name, sizes))
    if not rows:
        return "Graph pool requests: no compiled forward reports a memory plan"
    rows.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)

    def mib(value):
        return f"{value / (1 << 20):.1f}"

    entries = ", ".join(f"{name} {mib(total)}" + ("" if len(sizes) <= 1 else f" ({'+'.join(mib(s) for s in sizes)})")
                        for _, total, name, sizes in rows)
    return (f"Graph pool requests (MiB, {len(rows)} compiled forwards, largest first; max {mib(rows[0][0])} "
            f"in {rows[0][2]}): {entries}")


@_defer_full_collection()
def capture_forwards(functions, signatures, *, session, device, stream, pool,
                     collective=None, coordinator=None, compile_decoder=True, materialize=None,
                     progress=None, label="Prepare serving graphs", cache_root=None, retain_graphs=False,
                     warmup_storage=None, capture_names=None, before_run=None, compile_group=None,
                     report_catalog=True, retain_forwards=False,
                     compile_kinds=("prefill", "scored_prefill", "decode", "resume", "speculative", "mixed", "vision",
                                    "encoder_prefill")):
    """Retain compiled resources and capture with an optional b12x IPC runtime.

    Persistent state and graph outputs must live in caller-owned storage.
    Only graph-local intermediates may use the shared graph memory pool;
    serving replays variants serially and may choose them in any order.
    All eager warmups precede capture so their transient workspaces do not
    coexist with retained graph intermediates.
    ``retain_graphs`` keeps templates and executables for export when
    ``warmup_storage`` supplies external CUDA pool owners.
    ``before_run`` restores caller-owned trial inputs outside compiled warmup
    and graph capture; it is unused for the ordinary serving catalog.
    ``compile_group`` compiles one representative for same-signature trial
    forwards, then checks every owner's guards and storage aliases before
    capturing their distinct CUDA graphs.
    ``retain_forwards`` preserves guarded Python execution for eager diagnostics.
    """
    if (report_catalog and progress is not None and not getattr(session, "_graph_catalog_reported", False)
            and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0)):
        graph_catalog(progress, (signatures[name] for name in functions))
        session._graph_catalog_reported = True
    names = tuple(functions) if capture_names is None else tuple(capture_names)
    if len(set(names)) != len(names) or not set(names) <= functions.keys():
        raise ValueError("capture selection must name distinct prepared forwards")
    if compile_group is not None and (not isinstance(compile_group, str) or not compile_group
                                      or compile_group in functions or not compile_decoder or not names):
        raise ValueError("grouped compilation requires a distinct name and selected compiled forwards")
    graphs, compiled = {}, {}
    warmup = nullcontext if warmup_storage is None else warmup_storage.transient
    if compile_decoder:
        # TP3 fork: FLARE_EXPORT_FORWARD_WORKERS / _WORKER_GIB / _WORKER_VRAM_MIB after tuning (export_speed.py); unset = tuning budget.
        compile_workers, forward_worker_bytes, forward_worker_device_bytes = forward_compile_settings(session)
        cache = ForwardCache(namespace=session.namespace.to_dict(), device=device, root=cache_root,
                             headroom_bytes=getattr(session, "headroom_bytes", 0),
                             compile_workers=compile_workers, forward_worker_bytes=forward_worker_bytes,
                             forward_worker_device_bytes=forward_worker_device_bytes,
                             lock_gpu_clocks=getattr(session, "lock_gpu_clocks", False) and session.autotune,
                             compile_headroom_bytes=getattr(session, "compile_headroom_bytes", COMPILE_HEADROOM_BYTES))
        forwards = {name: function for name, function in functions.items()
                    if name in names and signatures[name].kind in compile_kinds}
        if compile_group is not None:
            if len(forwards) != len(names):
                raise ValueError("grouped compilation requires every selected forward to be compiled")
            first = names[0]
            if any(type(forward) is not type(forwards[first]) or signatures[name] != signatures[first]
                   for name, forward in forwards.items()):
                raise ValueError("grouped forwards require one type and graph signature")
        stream.wait_stream(torch.cuda.current_stream(device))
        retention = nullcontext() if warmup_storage is None else warmup_storage.retain_transient_backing()
        with retention, warmup_wait(collective, coordinator):
            with torch.cuda.stream(stream):
                if compile_group is None:
                    compiled = cache.prepare_many(forwards, signatures, progress=progress,
                                                  coordinator=coordinator, materialize=materialize, warmup=warmup,
                                                  before_run=before_run)
                else:
                    representative = {compile_group: forwards[first]}
                    template = cache.prepare_many(representative, {compile_group: signatures[first]},
                        progress=progress, coordinator=coordinator, materialize=materialize,
                        warmup=warmup, before_run=None if before_run is None else lambda _: before_run(first))
            stream.synchronize()
        if compile_group is not None:
            compiled = cache.bind_group(template[compile_group], forwards,
                                        signature=signatures[first], coordinator=coordinator)
            del template
        with phase(progress, "Release serialized compiler state", total=len(compiled)) as stage:
            released, counted = 0, set()
            for forward in compiled.values():
                artifact = id(getattr(forward, "artifact", forward))
                size = forward.prepare_capture()
                if artifact not in counted:
                    released += size
                    counted.add(artifact)
                stage.update(advance=1)
            stage.update(f"{released >> 20} MiB; guards and kernel owners retained")
        from .host_memory import release_unused

        release_unused()
    elif materialize is not None:
        materialize()
    # TP3 profiling (opt-in, FLARE_DS41_EAGER_PROFILE=<substring>): run matching forwards eagerly (compiled code,
    # no graph capture) inside a cudaProfiler range for nsys --capture-range=cudaProfilerApi, then exit.
    profile_match = os.getenv("FLARE_DS41_EAGER_PROFILE")
    if profile_match and label != "Prepare and tune kernels":
        iters = int(os.getenv("FLARE_DS41_EAGER_ITERS", "20"))
        targets = [name for name in names if profile_match in name]
        if progress is not None:
            progress(f"eager-profile targets ({profile_match!r}): {targets} of {list(names)}")
        if targets:
            for name in targets:
                fn = compiled.get(name, functions[name])
                if coordinator is not None:
                    coordinator.barrier(f"profile {name}", coordinator.ranks)
                stream.wait_stream(torch.cuda.current_stream(device))
                with warmup(), torch.cuda.stream(stream):
                    for _ in range(3):
                        fn()
                    stream.synchronize()
                    torch.cuda.nvtx.range_push(f"ds41_profile_{name}")
                    torch.cuda.profiler.start()
                    for _ in range(iters):
                        torch.cuda.nvtx.range_push("step")
                        fn()
                        torch.cuda.nvtx.range_pop()
                    stream.synchronize()
                    torch.cuda.profiler.stop()
                    torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize(device)
            import sys
            sys.stdout.flush()
            sys.exit(0)
    with phase(progress, "Warm serving graphs", total=len(names)) as stage:
        for name in names:
            function = functions[name]
            if coordinator is not None:
                coordinator.barrier(f"capture warmup {name}", coordinator.ranks)
            stream.wait_stream(torch.cuda.current_stream(device))
            with warmup(), torch.cuda.stream(stream):
                if before_run is not None:
                    before_run(name)
                if name in compiled:
                    function = compiled[name]
                    if compile_group is not None:
                        function()
                else:
                    function()
            stream.synchronize()
            stage.update(advance=1)
    ordered_names = _capture_order(names, compiled, coordinator)
    if progress is not None and compiled:
        progress(pool_request_report({name: compiled[name] for name in ordered_names if name in compiled}))
    with phase(progress, label, total=len(names)) as stage:
        for name in ordered_names:
            function = compiled.get(name, functions[name])
            if coordinator is not None:
                coordinator.barrier(f"capture {name}", coordinator.ranks)
            started = monotonic()
            stream.wait_stream(torch.cuda.current_stream(device))
            exporting = retain_graphs and warmup_storage is not None
            graph = torch.cuda.CUDAGraph(keep_graph=True) if exporting else capture_graph(retain_graphs=retain_graphs)
            communication = nullcontext() if collective is None else collective.capture(stream=stream)
            def memory_detail():
                free, total = torch.cuda.mem_get_info(device)
                allocated = torch.cuda.memory_allocated(device)
                reserved = torch.cuda.memory_reserved(device)
                return (f"{free / (1 << 20):.0f} MiB free, "
                        f"{allocated / (1 << 20):.0f} MiB allocated, "
                        f"{reserved / (1 << 20):.0f} MiB reserved, "
                        f"{(total - free - reserved) / (1 << 20):.0f} MiB outside PyTorch")

            try:
                if progress is not None:
                    progress(f"capturing {name}: {memory_detail()}")
                if before_run is not None:
                    with torch.cuda.stream(stream):
                        before_run(name)
                    stream.synchronize()
                with capture_consumers(session, graph), communication, session.capture(), torch.cuda.graph(graph, pool=pool, stream=stream):
                    function()
                if exporting:
                    recorded = record_export_graph(graph, warmup_storage, owners=(function,))
                    registry = getattr(session, "_startup_collectives", None)
                    if registry is not None:
                        registry.captures[id(recorded)] = registry.captures.pop(id(graph))
                    graph = recorded
                elif retain_graphs:
                    if progress is not None:
                        progress(f"instantiating {name}: {memory_detail()}")
                    instantiate_graph(graph)
            except Exception as error:
                error.add_note(f"Graph {name!r} on {device}: {memory_detail()}")
                raise
            if name in compiled and not retain_forwards:
                function.retire_after_capture()
            graphs[name] = graph
            stage.update(advance=1)
            if progress is not None:
                cached = " (compiled cache hit)" if name in compiled and compiled[name].cache_hit else ""
                progress(f"captured {name} in {monotonic() - started:.2f}s{cached}")
    torch.cuda.synchronize(device)
    return graphs, compiled
