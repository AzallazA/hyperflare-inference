# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Preserve temporary lifetimes, optional flat 1D pool packing, and retain compiler pool sizes for capture."""

import os
import sympy

from . import _replace

FLAT_POOL_SWITCH = "FLARE_FLAT_MEMORY_POOL"


def flat_pool_enabled() -> bool:
    value = os.environ.get(FLAT_POOL_SWITCH, "").strip()
    if value not in ("", "0", "1"):
        raise ValueError(f"{FLAT_POOL_SWITCH}={value!r}: expected 0 or 1")
    return value == "1"


class _FlatAllocationRoot:
    """Flat 1D offset-packed root for an Inductor AllocationPool (zero guillotine fragmentation)."""

    def __init__(self, total_bytes: int, assignments: list):
        self._total_bytes = int(total_bytes)
        self._assignments = assignments
        self.allocations = ()
        self._live_ranges = None

    def get_symbolic_size(self):
        return sympy.Integer(self._total_bytes)

    def get_live_ranges(self):
        if self._live_ranges is None:
            from torch._inductor.codegen.memory_planning import LiveRange, LiveRanges

            ranges = [block.live_range for block, _ in self._assignments]
            self._live_ranges = LiveRanges(ranges if ranges else [LiveRange(0, 0)])
        return self._live_ranges

    def finalize(self, pool, offset):
        base = int(offset)
        for block, block_offset in self._assignments:
            block.finalize(pool, sympy.Integer(base + block_offset))
        return self


def _pack_flat_pool(blocks, align_bytes: int = 256):
    """Greedy 1D offset packing over exact half-open LiveRange [begin, end) intervals.

    Returns (total_bytes, [(block, offset), ...]) or None if any block has a symbolic or deferred size.
    """
    sized = []
    for block in blocks:
        if block.get_earliest_available() is not None:
            return None
        sym = block.symbolic_size
        if not isinstance(sym, (int, sympy.Integer)):
            return None
        nbytes = int(sym)
        if nbytes < 0:
            return None
        sized.append((block, nbytes, block.live_range.begin, block.live_range.end))
    # Largest first, then longest live range, then earliest begin.
    sized.sort(key=lambda item: (-item[1], -(item[3] - item[2]), item[2]))
    placed = []  # (begin, end, off, off_end)
    assignments = []
    total = 0
    for block, nbytes, begin, end in sized:
        if nbytes == 0:
            assignments.append((block, 0))
            continue
        # Inductor LiveRange is half-open [begin, end): two ranges overlap iff a.begin < b.end and b.begin < a.end.
        occupied = [(po, poe) for pb, pe, po, poe in placed if pb < end and begin < pe]
        occupied.sort()
        cand = 0
        for po, poe in occupied:
            if cand + nbytes <= po:
                break
            if poe > cand:
                cand = (poe + align_bytes - 1) & -align_bytes
        placed.append((begin, end, cand, cand + nbytes))
        assignments.append((block, cand))
        if cand + nbytes > total:
            total = cand + nbytes
    total = (total + align_bytes - 1) & -align_bytes if total else 0
    # Verify non-overlap and bounds invariant before committing.
    for i, (b1, e1, o1, oe1) in enumerate(placed):
        if o1 % align_bytes != 0 or oe1 > total:
            raise AssertionError(f"flat pool alignment/bounds violated: [{o1}, {oe1}) vs {total}")
        for b2, e2, o2, oe2 in placed[i + 1:]:
            if b1 < e2 and b2 < e1 and o1 < oe2 and o2 < oe1:
                raise AssertionError(f"flat pool overlap: [{b1},{e1})@[{o1},{oe1}) vs [{b2},{e2})@[{o2},{oe2})")
    return total, assignments


def _flat_allocate_groups(planner, fallback):
    """Optional flat 1D offset packer for combined memory pools (FLARE_FLAT_MEMORY_POOL=1)."""
    from torch._inductor import config
    from torch._inductor.codegen.memory_planning import AllocationPool

    if not flat_pool_enabled() or config.memory_pool != "combined" or not planner.buffer_groups:
        return fallback(planner)
    for group in planner.buffer_groups:
        group.make_allocation()
    by_device: dict = {}
    for group in planner.buffer_groups:
        if not group.allocation:
            raise AssertionError("group has no allocation")
        by_device.setdefault(group.allocation.device, []).append(group.allocation)
    for device, blocks in by_device.items():
        packed = _pack_flat_pool(blocks, align_bytes=256)
        if packed is None:
            for block in sorted(blocks, key=lambda x: (-x.size_hint, -len(x.live_range))):
                planner.pools.allocate(block)
            continue
        total, assignments = packed
        for block, _ in assignments:
            block.mark_allocated()
        pool = AllocationPool(device, _FlatAllocationRoot(total, assignments), can_expand=False)
        planner.pools.device_to_pools.setdefault(device, []).append(pool)
    planner.pools.finalize()


def replacements():
    from torch._inductor import memory
    from torch._inductor.codegen.memory_planning import MemoryPlanner
    from torch._inductor.output_code import CompiledFxGraph

    orig_allocate_groups = getattr(MemoryPlanner.allocate_groups, "_flare_orig", MemoryPlanner.allocate_groups)

    def allocate_groups(self):
        return _flat_allocate_groups(self, orig_allocate_groups)

    allocate_groups._flare_orig = orig_allocate_groups

    plan = _replace(MemoryPlanner.plan, ((
        "    self.allocate_groups()\n    self.mark_first_last_usage(lines)\n",
        "    self.allocate_groups()\n"
        "    V.graph._flare_pool_sizes = tuple(\n"
        "        (device.type, pool.root.get_symbolic_size())\n"
        "        for device, pools in self.pools.device_to_pools.items()\n"
        "        for pool in pools\n"
        "    )\n"
        "    self.mark_first_last_usage(lines)\n",
    ),), allow_applied=True)
    compiled_graph = _replace(CompiledFxGraph.__init__, ((
        "    self.metrics_deltas = metrics_deltas\n    self.counter_deltas = counter_deltas\n",
        "    self.metrics_deltas = metrics_deltas\n"
        "    self._flare_pool_sizes = getattr(graph, '_flare_pool_sizes', ())\n"
        "    self.counter_deltas = counter_deltas\n",
    ),), allow_applied=True)

    compute_live_ranges = _replace(MemoryPlanner.compute_live_ranges, (
        (
            "    for group in self.buffer_groups:\n",
            "    from torch._inductor.codegen.common import WorkspaceArg, WorkspaceZeroMode\n"
            "    for group in self.buffer_groups:\n",
        ),
        (
            "        if group.is_output:\n            group.update_usage(timestep)\n",
            "        if group.is_output or (\n"
            "            isinstance(group.node, WorkspaceArg)\n"
            "            and group.node.zero_mode == WorkspaceZeroMode.ZERO_PER_GRAPH\n"
            "        ):\n"
            "            group.update_usage(timestep)\n",
        ),
    ), allow_applied=True)
    assign_memory_planning_info = _replace(memory.assign_memory_planning_info_for_scheduler_nodes, (
        (
            "        for buffer in node.get_outputs():\n"
            "            for succ_node in buffer.mpi_buffer.succ_nodes:\n",
            "        for buffer in node.get_outputs():\n"
            "            if not config.reorder_for_peak_memory_debug and (\n"
            "                buffer.mpi_buffer.size_alloc == 0 and buffer.mpi_buffer.size_free == 0\n"
            "            ):\n"
            "                continue\n"
            "            for succ_node in buffer.mpi_buffer.succ_nodes:\n",
        ),
        (
            "    for freeable_buffer in name_to_freeable_input_buf.values():\n"
            "        for succ_node in freeable_buffer.mpi_buffer.succ_nodes:\n",
            "    for freeable_buffer in name_to_freeable_input_buf.values():\n"
            "        if not config.reorder_for_peak_memory_debug and (\n"
            "            freeable_buffer.mpi_buffer.size_alloc == 0\n"
            "            and freeable_buffer.mpi_buffer.size_free == 0\n"
            "        ):\n"
            "            continue\n"
            "        for succ_node in freeable_buffer.mpi_buffer.succ_nodes:\n",
        ),
    ), allow_applied=True)
    return [(MemoryPlanner, "plan", plan),
            (MemoryPlanner, "allocate_groups", allocate_groups),
            (CompiledFxGraph, "__init__", compiled_graph),
            (MemoryPlanner, "compute_live_ranges", compute_live_ranges),
            (memory, "assign_memory_planning_info_for_scheduler_nodes", assign_memory_planning_info)]
