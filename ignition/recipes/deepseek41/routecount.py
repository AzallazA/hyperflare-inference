# SPDX-License-Identifier: Apache-2.0
"""Debug-serve-only routed-expert counters for expert-order calibration (exp/ds41-exl3, m7).

FLARE_DS41_ROUTE_COUNTS=<directory> (unset/empty = off; never export with it: tp3-tools/ds41-exl3-export-tp2.sh
refuses it for exports): every target decoder layer adds its router's top-k expert IDs (router ID space, before any
hot-first relabel) of every LIVE row (metadata position >= 0; graph padding rows carry -1) into a persistent int64
table [2 groups x 40 layers x experts]: group 0 = prefill graphs, group 1 = decode + verify graphs. DSpark draft
layers (mode "draft") are not counted. When off, nothing is traced (no op, no allocation).

Readout runs in every rank process (Python threads beside native replay, as probe.py): a thread polls the directory
every 0.2 s; for each ``request-<tag>`` file it has not answered yet it writes
``<dir>/<proc>/snap-<tag>.json`` = {"tag", "time", "layers", "experts", "groups": {"prefill": [[...]], "step":
[[...]]}} and then ``<dir>/<proc>/done-<tag>``, where <proc> = dev<cuda index>-pid<pid>. Routing is replicated over
the tensor-parallel ranks, so every rank's tables must agree (the reader checks). Snapshots are cumulative; phase
counts are differences of snapshots taken while the server is idle.
"""

import json
import os
import threading
import time

import torch

ENV = "FLARE_DS41_ROUTE_COUNTS"
LAYERS, GROUPS = 40, 2
_STATE = {}
MISMATCHES = set()  # (trace time) layers whose routed rows did not match their metadata rows: counted unmasked


def directory():
    value = os.environ.get(ENV, "")
    return value or None


def _state(device, experts):
    key = torch.device(device).index
    if key not in _STATE:
        # Persistent storage: the table must outlive the transient pools that preparation constructs in.
        from ignition.preparation.persistent import persistent_zeros as zeros
        table = zeros((GROUPS * LAYERS, experts), dtype=torch.int64, device=device)
        _STATE[key] = table
        out = directory()
        if out is not None:
            threading.Thread(target=_dumper, args=(table, key, out), daemon=True, name="ds41-route-counts").start()
    return _STATE[key]


def snapshot(table):
    rows = table.detach().cpu().tolist()
    return {"prefill": rows[:LAYERS], "step": rows[LAYERS:]}


def _dumper(table, index, out):
    proc = os.path.join(out, f"dev{index}-pid{os.getpid()}")
    os.makedirs(proc, exist_ok=True)
    answered = set()
    while True:
        try:
            for name in sorted(os.listdir(out)):
                if not name.startswith("request-") or name in answered:
                    continue
                tag = name[len("request-"):]
                payload = {"tag": tag, "time": time.time(), "layers": LAYERS, "experts": table.shape[1],
                           "unmasked": sorted(MISMATCHES), "groups": snapshot(table)}
                tmp = os.path.join(proc, f".snap-{tag}.json.tmp")
                with open(tmp, "w") as handle:
                    json.dump(payload, handle)
                os.replace(tmp, os.path.join(proc, f"snap-{tag}.json"))
                open(os.path.join(proc, f"done-{tag}"), "w").close()
                answered.add(name)
        except Exception as error:  # never let a readout problem touch serving; report it next to the snapshots
            try:
                with open(os.path.join(proc, "errors.log"), "a") as handle:
                    handle.write(f"{time.time():.0f} {error!r}\n")
            except OSError:
                pass
        time.sleep(0.2)


@torch.library.custom_op("flare_debug::route_count", mutates_args=("table",))
def _route_count(ids: torch.Tensor, positions: torch.Tensor, table: torch.Tensor, slot: int) -> None:
    flat = ids.reshape(ids.shape[0], -1)
    live = (positions.reshape(-1) >= 0).to(torch.int64)
    weights = live[:, None].expand(-1, flat.shape[1]).reshape(-1)
    table[slot].index_add_(0, flat.reshape(-1).long(), weights)


@_route_count.register_fake
def _route_count_fake(ids, positions, table, slot):
    pass


class RouteCounter:
    """One per prepared layer; ``record`` is a no-op unless FLARE_DS41_ROUTE_COUNTS names a directory."""

    def __init__(self, enabled=None):
        self.enabled = directory() is not None if enabled is None else bool(enabled)
        self.table = None

    def allocate(self, device, experts):
        """Allocate the table (and start the reader thread) eagerly, outside traced forwards, and keep it on the
        object: forwards are traced in compile-worker processes that receive the pickled forward (tensors -> fake)
        but not this module's _STATE, so a lookup there would allocate and start a thread inside the trace."""
        if self.enabled:
            self.table = _state(device, experts)

    def record(self, ids, positions, layer, mode, experts):
        if not self.enabled or mode not in ("prefill", "decode", "verify") or not 0 <= layer < LAYERS:
            return
        if positions.numel() != ids.shape[0]:
            # Never fail a preparation over a debug counter: count every row and name the case in each snapshot.
            MISMATCHES.add(f"layer {layer} {mode}: {ids.shape[0]} rows vs {positions.numel()} positions")
            positions = torch.zeros(ids.shape[0], dtype=torch.int64, device=ids.device)
        if self.table is None:
            raise RuntimeError("route counter: record() before allocate() (the table must exist before tracing)")
        _route_count(ids, positions, self.table, (0 if mode == "prefill" else 1) * LAYERS + layer)
