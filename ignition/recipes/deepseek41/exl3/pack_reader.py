# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Read EXL3 expert tensors from a safetensors pack without torch (header parse + NumPy views of byte ranges).

The pack is untrusted input: headers are size-checked, dtypes are whitelisted, and tensor byte ranges must lie inside
the file. Files are opened read-only and mapped, never written.
"""

from __future__ import annotations

import json
import mmap
import re
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_DTYPES = {"I16": np.int16, "I32": np.int32, "F16": np.float16, "F32": np.float32, "U8": np.uint8, "I8": np.int8,
           "BF16": np.uint16, "F8_E4M3": np.uint8, "F8_E8M0": np.uint8}
_MAX_HEADER = 64 << 20
_EXPERT = re.compile(r"^(?P<prefix>(layers|mtp)\.\d+\.ffn\.experts\.\d+)\.(?P<matrix>w[123])\.(?P<part>trellis|suh|svh|mul1|mcg)$")


@dataclass(frozen=True)
class TensorInfo:
    file: Path
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


class SafetensorsFile:
    def __init__(self, path):
        self.path = Path(path)
        size = self.path.stat().st_size
        with self.path.open("rb") as handle:
            (length,) = struct.unpack("<Q", handle.read(8))
            if not 0 < length <= min(_MAX_HEADER, size - 8):
                raise ValueError(f"{self.path.name}: bad safetensors header length {length}")
            header = json.loads(handle.read(length))
        base = 8 + length
        self.metadata = header.pop("__metadata__", None)
        self.tensors = {}
        for name, entry in header.items():
            begin, end = (int(v) for v in entry["data_offsets"])
            dtype = entry["dtype"]
            shape = tuple(int(v) for v in entry["shape"])
            if dtype not in _DTYPES:
                raise ValueError(f"{self.path.name}: {name} has unsupported dtype {dtype}")
            count = int(np.prod(shape, dtype=np.int64)) if shape else 1
            if not 0 <= begin <= end or base + end > size or end - begin != count * np.dtype(_DTYPES[dtype]).itemsize:
                raise ValueError(f"{self.path.name}: {name} byte range {begin}..{end} inconsistent with {dtype}{shape}")
            self.tensors[name] = TensorInfo(self.path, dtype, shape, base + begin, base + end)

    def read(self, name) -> np.ndarray:
        info = self.tensors[name]
        with info.file.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as view:
            data = np.frombuffer(view[info.begin:info.end], dtype=_DTYPES[info.dtype]).copy()
        return data.reshape(info.shape)


def expert_matrices(index_path) -> dict[str, dict[str, tuple[str, TensorInfo | None]]]:
    """Group a pack's expert tensors: ``{"layers.0.ffn.experts.0.w1": {"trellis": shard, ...}}`` from the index."""
    index_path = Path(index_path)
    weight_map = json.loads(index_path.read_text())["weight_map"]
    groups: dict[str, dict[str, str]] = {}
    for name, shard in weight_map.items():
        match = _EXPERT.match(name)
        if match:
            groups.setdefault(f"{match['prefix']}.{match['matrix']}", {})[match["part"]] = shard
    return groups


def read_matrix(files: dict[str, SafetensorsFile], prefix: str) -> dict[str, np.ndarray]:
    """``trellis``/``suh``/``svh`` and the codebook marker of one matrix from already-opened shard files."""
    out = {}
    for part in ("trellis", "suh", "svh", "mul1", "mcg"):
        name = f"{prefix}.{part}"
        for handle in files.values():
            if name in handle.tensors:
                out[part] = handle.read(name)
                break
    missing = {"trellis", "suh", "svh"} - out.keys()
    if missing:
        raise KeyError(f"{prefix}: missing {sorted(missing)}")
    return out
