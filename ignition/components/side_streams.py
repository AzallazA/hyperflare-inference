# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Run independent prepared decode work on a dedicated side stream inside captured graphs (TP3 DS41).

A *task* is a Python callable registered once at preparation (``register``) that launches prepared kernels on the
current stream, reading ``inputs`` and writing only ``outputs`` (every buffer it touches is allocated by the traced
forward on the main stream and passed in, never allocated inside the task). The traced forward calls ``launch``: the
side stream waits for the main stream (fork), the task runs there, and an event is recorded after it. ``join`` makes
the main stream wait for that event only (not the side stream's tail, so later tasks on the same stream keep
running). Both operators take every input and output, so Inductor keeps all of them alive until the join and orders
every reader of an output after it; a tensor listed in ``join``'s outputs that the task never touched (an anchor) is
only declared written, which orders its later readers after the join too.

Same kernels, plans and arguments as the serial path, disjoint outputs, join before the first reader: bit-identical.
The capture records the fork and join as graph edges (the side stream joins the capture through the event), so a
switch that uses this takes effect at export; serving replays the recorded graph.

Streams are created with ``cuStreamCreate`` (non-blocking), one per device and purpose, never from PyTorch's
round-robin pool (it can alias a later capture stream; same reason as ``ScratchArena.auxiliary_stream``).
"""

import itertools

import torch

_STREAMS = {}  # (device index, purpose) -> torch.cuda.ExternalStream
_TASKS = {}    # handle -> callable(inputs, outputs)
_EVENTS = {}   # handle -> torch.cuda.Event recorded after the task's last launch
_HANDLES = itertools.count(1)
# Test hook: launch order as ("launch" | "join", handle, purpose) tuples; None = not recorded.
TRACE = None


def side_stream(device, purpose):
    """The side stream of ``purpose`` on ``device`` (created on first use)."""
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, purpose)
    stream = _STREAMS.get(key)
    if stream is None:
        from cuda.bindings import driver as cuda

        with torch.cuda.device(index):
            status, handle = cuda.cuStreamCreate(cuda.CUstream_flags.CU_STREAM_NON_BLOCKING)
            if status != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cannot create the {purpose} side stream: {status}")
            stream = torch.cuda.ExternalStream(int(handle), device=torch.device("cuda", index))
        _STREAMS[key] = stream
    return stream


class Task:
    """A registered side-stream task: ``run(inputs, outputs)`` on the stream named ``purpose`` of ``device``.

    Registered at preparation in the process that captures and replays: the stream handle and the task handle are
    plain integers baked into the traced graph, so compile workers (which only trace, through the fake kernels) see
    the preparing process's stream, as ``ConcurrentProjectionChain`` does with the arena's auxiliary stream."""

    def __init__(self, run, purpose, device):
        if not callable(run) or not isinstance(purpose, str) or not purpose:
            raise ValueError("a side-stream task needs a callable and a purpose name")
        self.run, self.purpose = run, purpose
        self.stream_handle = int(side_stream(device, purpose).cuda_stream)
        self.handle = next(_HANDLES)
        _TASKS[self.handle] = self

    def launch(self, anchor, inputs, outputs):
        """Traced: fork after ``anchor`` (read only) and the inputs' producers; run the task on the side stream."""
        _check_disjoint(anchor, inputs, outputs)
        _launch(anchor, list(inputs), list(outputs), self.handle, self.stream_handle)
        return outputs

    def join(self, dependency, inputs, outputs):
        """Traced: the main stream waits for this task; ``dependency`` (read only) pins the join after its producer."""
        _check_disjoint(dependency, inputs, outputs)
        _join(dependency, list(inputs), list(outputs), self.handle)
        return outputs


def register(run, purpose, device):
    return Task(run, purpose, device)


def _check_disjoint(anchor, inputs, outputs):
    # A mutated operand may not alias another operand of the same call (functionalization); fail at trace time.
    seen = {}
    for role, group in (("anchor", (anchor,)), ("input", inputs), ("output", outputs)):
        for tensor in group:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"side-stream {role} must be a tensor")
            key = id(tensor)
            if key in seen and "output" in (seen[key], role):
                raise ValueError("a side-stream output may not also be passed as another operand")
            seen[key] = role


def _record(kind, handle):
    if TRACE is not None:
        TRACE.append((kind, handle, _TASKS[handle].purpose))


@torch.library.custom_op("flare::side_stream_launch", mutates_args=("outputs",))
def _launch(anchor: torch.Tensor, inputs: list[torch.Tensor], outputs: list[torch.Tensor],
            handle: int, stream: int) -> None:
    device = anchor.device
    task = _TASKS[handle]
    side = torch.cuda.ExternalStream(stream, device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    _record("launch", handle)
    with torch.cuda.stream(side):
        task.run(inputs, outputs)
        event = torch.cuda.Event()
        event.record(side)
    _EVENTS[handle] = event


@_launch.register_fake
def _fake_launch(anchor, inputs, outputs, handle, stream):
    pass


@torch.library.custom_op("flare::side_stream_join", mutates_args=("outputs",))
def _join(dependency: torch.Tensor, inputs: list[torch.Tensor], outputs: list[torch.Tensor], handle: int) -> None:
    event = _EVENTS.pop(handle, None)
    if event is None:
        raise RuntimeError("side-stream join without a preceding launch of the same task")
    _record("join", handle)
    torch.cuda.current_stream(dependency.device).wait_event(event)


@_join.register_fake
def _fake_join(dependency, inputs, outputs, handle):
    pass
