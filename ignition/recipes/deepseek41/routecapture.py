# SPDX-License-Identifier: Apache-2.0
"""Debug-serve-only per-round route capture for the cold-expert prefetch study (exp/ds41-exl3-s1).

FLARE_DS41_EXL3_ROUTE_CAPTURE=<directory> (unset/empty = off: nothing traced, nothing allocated; never export with it,
tp3-tools/ds41-exl3-export-tp2.sh refuses it unless DEBUG_SERVE=1). In every decode and verification graph of the
target decoder (DSpark draft layers and prefill graphs are not captured) each forward takes one slot of a device ring:

  meta  [RING, 4] int64      forward counter, graph rows, mode (1 decode, 2 verify), live rows
  mask  [RING, RMAX] int8    row validity (metadata position >= 0; graph padding rows are 0)
  ids   [RING, L, RMAX, 6]   the router's top-6 expert IDs per layer and row (router ID space, before the hot-first
                             relabel; padding rows keep whatever the router chose for them: the cold launch sees them)
  nxt   [RING, L, RMAX, 12]  top-12 of layer L+1's router applied to layer L's routed-MoE input (prefetch predictor
                             with a one-layer lead); -1 for the last layer
  early [RING, L, RMAX, 12]  top-12 of layer L's own router applied to layer L's attention input (attention_hc
                             normalized: known at the start of layer L, lead = attention + tail)

Predictor scores use the router's selection law (sqrtsoftplus or softmax of BF16 logits, plus the correction bias);
the GEMVs run in BF16 (the production router accumulates the same products in FP32; near-ties may order differently,
which only matters for the predictor's 12th place). The ops only READ their inputs and write their own persistent
buffers, so outputs are unchanged (tests/recipes/test_deepseek41_route_capture.py checks inputs bit-identical before /
after, eager and compiled). Rows beyond RMAX are not stored (meta keeps the full row count).

Readout runs in every rank process (Python thread beside native replay, like probe.py / routecount.py). It makes no
CUDA call until ``<dir>/go`` (or a ``mark-`` file) exists: the driver touches it once /v1/models answers, because a
synchronize from this thread during preparation invalidates other threads' graph captures. Then every 0.2 s it reads the counter on its own stream and copies the finished slots (counter - 1 is still being written) to
``<dir>/dev<cuda index>-pid<pid>/chunk-<n>.npz`` (arrays trimmed to the largest row count of the chunk), checks each
slot's stored counter (a lapped slot is counted in ``lost``), and keeps ``layout.json`` current (per layer: hot_map =
router ID -> bank slot, hot_rows). For each ``mark-<tag>`` file in <dir> it appends {"tag", "time", "counter"} to
``<proc>/marks.jsonl`` (phase boundaries for the analysis). Routing is replicated over tensor-parallel ranks, so
every rank's chunks must agree (the analysis checks a sample).
"""

import json
import os
import threading
import time

import torch

ENV = "FLARE_DS41_EXL3_ROUTE_CAPTURE"
RING, RMAX, LAYERS, TOPK, PRED = 512, 48, 40, 6, 12
MODES = {"decode": 1, "verify": 2}
GO = "go"  # <dir>/go: the driver touches it once serving is up; the dumper makes no CUDA call before it
_STATE = {}
_LAYOUT = {}  # layer -> {"hot_map": [...], "hot_rows": n}; filled at preparation (eager), written once by the dumper


def directory():
    value = os.environ.get(ENV, "")
    return value or None


def _state(device):
    key = torch.device(device).index
    if key not in _STATE:
        # Persistent storage: the rings must outlive the transient pools that preparation constructs in.
        from ignition.preparation.persistent import persistent_zeros as zeros
        state = {
            "counter": zeros((1,), torch.int64, device),
            "meta": zeros((RING, 4), torch.int64, device),
            "mask": zeros((RING, RMAX), torch.int8, device),
            # -1 = not written: a layer the model does not have (mini views) never reads as expert 0
            "ids": zeros((RING, LAYERS, RMAX, TOPK), torch.int16, device).fill_(-1),
            "nxt": zeros((RING, LAYERS, RMAX, PRED), torch.int16, device).fill_(-1),
            "early": zeros((RING, LAYERS, RMAX, PRED), torch.int16, device).fill_(-1),
        }
        _STATE[key] = state
        out = directory()
        if out is not None:
            threading.Thread(target=_dumper, args=(state, key, out), daemon=True, name="ds41-route-capture").start()
    return _STATE[key]


def _score(logits, bias, score_func):
    logits = logits.float()
    if score_func == "sqrtsoftplus":
        scores = torch.sqrt(torch.nn.functional.softplus(logits))
    elif score_func == "softmax":
        scores = torch.softmax(logits, dim=-1)
    elif score_func == "sigmoid":
        scores = torch.sigmoid(logits)
    else:
        raise ValueError(f"route capture: unknown scoring function {score_func!r}")
    return scores + bias.float()[None, :] if bias is not None else scores


def predict(hidden, weight, bias, score_func, k=PRED):
    """Top-k expert IDs of a router (weight [experts, hidden]) applied to ``hidden`` [rows, hidden]."""
    logits = torch.matmul(hidden.to(weight.dtype), weight.t())
    return torch.topk(_score(logits, bias, score_func), k, dim=-1).indices


# (argument names must not shadow torch's dispatch keywords: an int named ``mode`` breaks inductor's lowering)
@torch.library.custom_op("flare_debug::route_capture_step", mutates_args=("counter", "meta", "mask"))
def _capture_step(positions: torch.Tensor, counter: torch.Tensor, meta: torch.Tensor, mask: torch.Tensor,
                  mode_code: int) -> None:
    counter.add_(1)
    slot = torch.remainder(counter - 1, RING)
    flat = positions.reshape(-1)
    rows = flat.shape[0]
    live = (flat >= 0).to(torch.int8)
    row = torch.zeros((1, RMAX), dtype=torch.int8, device=mask.device)
    n = min(rows, RMAX)
    row[0, :n] = live[:n]
    mask.index_copy_(0, slot, row)
    info = torch.stack((counter[0], torch.full_like(counter[0], rows), torch.full_like(counter[0], mode_code),
                        live.to(torch.int64).sum()))
    meta.index_copy_(0, slot, info[None])


@_capture_step.register_fake
def _capture_step_fake(positions, counter, meta, mask, mode_code):
    pass


@torch.library.custom_op("flare_debug::route_capture_layer", mutates_args=("ids_ring", "nxt_ring", "early_ring"))
def _capture_layer(ids: torch.Tensor, moe_input: torch.Tensor, attn_input: torch.Tensor, own_weight: torch.Tensor,
                   own_bias: torch.Tensor, next_weight: torch.Tensor | None, next_bias: torch.Tensor | None,
                   counter: torch.Tensor, ids_ring: torch.Tensor, nxt_ring: torch.Tensor, early_ring: torch.Tensor,
                   layer: int, score_func: str) -> None:
    slot = torch.remainder(counter - 1, RING)
    n = min(ids.shape[0], RMAX)
    dev = ids_ring.device
    got = torch.full((1, RMAX, TOPK), -1, dtype=torch.int16, device=dev)
    got[0, :n] = ids[:n].to(torch.int16)
    ids_ring[:, layer].index_copy_(0, slot, got)
    early = torch.full((1, RMAX, PRED), -1, dtype=torch.int16, device=dev)
    early[0, :n] = predict(attn_input[:n], own_weight, own_bias, score_func).to(torch.int16)
    early_ring[:, layer].index_copy_(0, slot, early)
    nxt = torch.full((1, RMAX, PRED), -1, dtype=torch.int16, device=dev)
    if next_weight is not None:
        nxt[0, :n] = predict(moe_input[:n], next_weight, next_bias, score_func).to(torch.int16)
    nxt_ring[:, layer].index_copy_(0, slot, nxt)


@_capture_layer.register_fake
def _capture_layer_fake(ids, moe_input, attn_input, own_weight, own_bias, next_weight, next_bias, counter, ids_ring,
                        nxt_ring, early_ring, layer, score_func):
    pass


def _dumper(state, index, out, interval=0.2):
    proc = os.path.join(out, f"dev{index}-pid{os.getpid()}")
    os.makedirs(proc, exist_ok=True)
    stream = None
    done, chunk, lost, marks, layout_written, live = 0, 0, 0, set(), -1, False
    while True:
        try:
            # Layers register their bank layout one by one while preparation constructs them (seq298: written once
            # after the first layer, the file named layer 0 only): rewrite whenever the set grew, and once at go.
            if _LAYOUT and len(_LAYOUT) != layout_written:
                layout_written = write_layout(proc)
            if not live:
                # No CUDA call of any kind before serving starts: preparation captures graphs (b12x races, model
                # graphs) in other threads under the global capture mode, and a synchronize from this thread
                # invalidates them (seq296: comm.pcie race on rank 1, cudaErrorStreamCaptureInvalidated).
                live = _serving(out)
                if not live:
                    time.sleep(interval)
                    continue
                layout_written = write_layout(proc)   # preparation is over: every layer has registered
                stream = torch.cuda.Stream(device=index) if state["counter"].is_cuda else None
            done, chunk, lost = drain(state, proc, done, chunk, lost, stream)
            for name in sorted(os.listdir(out)):
                if name.startswith("mark-") and name not in marks:
                    done, chunk, lost = drain(state, proc, done, chunk, lost, stream)
                    with open(os.path.join(proc, "marks.jsonl"), "a") as handle:
                        handle.write(json.dumps({"tag": name[5:], "time": time.time(), "counter": done,
                                                 "lost": lost}) + "\n")
                    marks.add(name)
        except Exception as error:  # never let a readout problem touch serving; report it next to the chunks
            try:
                with open(os.path.join(proc, "errors.log"), "a") as handle:
                    handle.write(f"{time.time():.0f} {error!r}\n")
            except OSError:
                pass
        time.sleep(interval)


def write_layout(proc):
    """Write <proc>/layout.json (atomic) from the layers registered so far; returns how many."""
    layout = dict(_LAYOUT)
    tmp = os.path.join(proc, ".layout.json.tmp")
    with open(tmp, "w") as handle:
        json.dump({"ring": RING, "rmax": RMAX, "layers": LAYERS, "hot": layout}, handle)
    os.replace(tmp, os.path.join(proc, "layout.json"))
    return len(layout)


def _serving(out):
    """True once the driver says serving runs: a ``go`` file (touched after /v1/models answers) or any ``mark-``."""
    names = os.listdir(out)
    return GO in names or any(name.startswith("mark-") for name in names)


def drain(state, proc, done, chunk, lost, stream=None):
    """Copy finished slots (done, counter - 1] into the next chunk file; returns (done, chunk, lost)."""
    import numpy as np
    from contextlib import nullcontext

    with torch.cuda.stream(stream) if stream is not None else nullcontext():
        counter = int(state["counter"].to("cpu")[0])
        last = counter - 1                       # the forward with number ``counter`` may still be writing
        if last <= done:
            return done, chunk, lost
        first = max(done + 1, last - RING + 2)   # a lapped slot is gone; keep one slot of margin
        lost += first - (done + 1)
        numbers = torch.arange(first, last + 1, dtype=torch.int64)
        slots = torch.remainder(numbers - 1, RING).to(state["meta"].device)
        host = {name: state[name].index_select(0, slots).to("cpu") for name in ("meta", "mask", "ids", "nxt", "early")}
    meta = host["meta"].numpy()
    ok = meta[:, 0] == numbers.numpy()
    lost += int((~ok).sum())
    if ok.any():
        rows = int(min(RMAX, meta[ok, 1].max()))
        arrays = {"meta": meta[ok], "mask": host["mask"].numpy()[ok, :rows]}
        for name in ("ids", "nxt", "early"):
            arrays[name] = host[name].numpy()[ok][:, :, :rows]
        np.savez(os.path.join(proc, f"chunk-{chunk:06d}.npz"), time=np.float64(time.time()), lost=np.int64(lost), **arrays)
        chunk += 1
    return last, chunk, lost


class RouteCapture:
    """One per prepared layer and one per decoder; every call is a no-op unless FLARE_DS41_EXL3_ROUTE_CAPTURE is set."""

    def __init__(self, enabled=None):
        self.enabled = directory() is not None if enabled is None else bool(enabled)
        self.state = None

    def active(self, mode):
        return self.enabled and mode in MODES

    def allocate(self, device, mode, layer=None, hot_map=None, hot_rows=None):
        """Allocate the rings eagerly (outside traced forwards); record the layer's bank layout once."""
        if not self.active(mode):
            return
        # Kept on the object: compile workers trace the pickled forward without this module's _STATE (see routecount).
        self.state = _state(device)
        if layer is not None and layer not in _LAYOUT:
            _LAYOUT[layer] = {"hot_map": None if hot_map is None else [int(v) for v in hot_map.cpu().tolist()],
                              "hot_rows": None if hot_rows is None else int(hot_rows)}

    def _allocated(self):
        if self.state is None:
            raise RuntimeError("route capture: step()/record() before allocate() (rings must exist before tracing)")
        return self.state

    def step(self, positions, mode):
        if self.active(mode):
            state = self._allocated()
            _capture_step(positions, state["counter"], state["meta"], state["mask"], MODES[mode])

    def record(self, ids, moe_input, attn_input, layer, mode, own, nxt, score_func):
        """``own`` / ``nxt`` = (router weight, correction bias) of this layer / the next one (``nxt`` None: last)."""
        if not self.active(mode) or not 0 <= layer < LAYERS:
            return
        state = self._allocated()
        next_weight, next_bias = (None, None) if nxt is None else nxt
        _capture_layer(ids, moe_input, attn_input, own[0], own[1], next_weight, next_bias, state["counter"],
                       state["ids"], state["nxt"], state["early"], layer, score_func)
