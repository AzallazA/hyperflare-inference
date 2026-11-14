# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 Engram tables on CPU: MXFP4 rows and the shared/partitioned layout switch (TP3Config.engram_tables).

* Shared tables (the TP3 default) publish exactly the pre-#220 layout every TP3 package has: one whole-table span per
  plane, canonical hash offsets, no ``shards``, the same resident catalog on every rank, no request row remapping.
  The expected records are written out literally from tp3-main's ``Tables.export_metadata``/``resident_catalog``.
* Partitioned tables (upstream #220) carry 128-byte weight rows when the checkpoint stores MXFP4 tables.
* ``Tables.prepare`` detects MXFP4 rows, stages 128-byte rows and applies the layout switch.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from b12x.sequence import engram

from ignition.components.parallel import TensorParallelShard
from ignition.preparation.runtime_core import FileSource
from ignition.recipes.deepseek41 import engram as recipe_engram
from ignition.recipes.deepseek41.engram import Tables

ENCODINGS = {False: "FP8_E4M3_E8M0", True: "MXFP4_E2M1_E8M0"}


def _tables(tmp_path, *, world, rank=0, packed, shared):
    geometry = engram.build_geometry(layer_ids=(1,), base_table_size=101, compressed_vocab_size=32)
    rows = geometry.num_embeddings[0]
    sources, contents = [], []
    for width in (128 if packed else 256, 8):
        data = bytes((row * 13 + column) % 251 for row in range(rows) for column in range(width))
        path = tmp_path / f"{width}.bin"
        if not path.exists():
            path.write_bytes(bytes(37) + data)
        sources.append(FileSource(path, 37, len(data)))
        contents.append(data)
    tables = Tables(geometry, tuple(range(32)), {1: tuple(sources)}, {}, {}, None, TensorParallelShard(rank, world),
                    {}, None, packed_fp4=packed, shared=shared)
    return tables, geometry, rows, sources, contents


def _read(chunks):
    return b"".join(Path(chunk["path"]).read_bytes()[chunk["offset"]:chunk["offset"] + chunk["nbytes"]]
                    for chunk in chunks)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("world", [1, 3, 4])
def test_shared_tables_publish_the_pre_partition_layout(tmp_path, packed, world):
    tables, geometry, rows, sources, contents = _tables(tmp_path, world=world, packed=packed, shared=True)
    widths = (128 if packed else 256, 8)
    # tp3-main 27ac90fa Tables.export_metadata, verbatim (whole-table spans, canonical offsets, no shards).
    expected = dict(layer=1, rows=rows, embedding_width=256, scale_block=32, encoding=ENCODINGS[packed],
        weight_buffer="engram_1_weights", scale_buffer="engram_1_scales", hash_buffer="engram_1_hashes",
        ready_event="engram_1_ready", request_buffer="engram_1_requests",
        weights=[dict(path=str(sources[0].path), offset=sources[0].offset, nbytes=sources[0].nbytes)],
        scales=[dict(path=str(sources[1].path), offset=sources[1].offset, nbytes=sources[1].nbytes)],
        hash=dict(method="XOR_MODULO", orders=[order for order in (2, 3, 4) for _ in range(8)],
                  multipliers=list(geometry.multipliers[0]), primes=list(geometry.primes[0]),
                  offsets=list(geometry.offsets[0])))
    for rank in range(world):
        rank_tables, *_ = _tables(tmp_path, world=world, rank=rank, packed=packed, shared=True)
        metadata = rank_tables.export_metadata()
        assert metadata["tables"] == [expected]
        assert metadata["token_map"] == list(range(32)) and metadata["pad_token_id"] == 2
        # tp3-main resident_catalog: one segment per plane with the stored row width, the same on every rank.
        planes = rank_tables.resident_catalog()["tables"][0]["planes"]
        assert planes == [{"segments": [{"path": str(source.path), "offset": source.offset, "nbytes": source.nbytes}],
                           "row_bytes": width} for source, width in zip(sources, widths)]
        assert rank_tables.rank_bytes(rank) == rows * sum(widths)
    for chunks, data in zip((expected["weights"], expected["scales"]), contents):
        assert _read(chunks) == data


@pytest.mark.parametrize("world", [2, 3, 4, 6])
def test_partitioned_mxfp4_tables_use_128_byte_rows(tmp_path, world):
    tables, geometry, rows, _, contents = _tables(tmp_path, world=world, packed=True, shared=False)
    table = tables.export_metadata()["tables"][0]
    assert table["encoding"] == "MXFP4_E2M1_E8M0" and "weights" not in table
    first = 0
    for rank in range(world):
        rank_tables, *_ = _tables(tmp_path, world=world, rank=rank, packed=True, shared=False)
        partition = rank_tables.partition(0, rank)
        stored = table["shards"][rank]
        assert (stored["rank"], stored["first_row"], stored["rows"]) == (rank, first, partition.rows)
        for head in partition.heads:
            assert table["hash"]["offsets"][head] == first + partition.row_mapping[head][2]
        resident = rank_tables.resident_catalog()["tables"][0]["planes"]
        for plane, name, width, data in zip(resident, ("weights", "scales"), (128, 8), contents):
            expected = b"".join(data[a * width:b * width] for a, b in partition.source_rows)
            assert _read(stored[name]) == expected and len(expected) == partition.rows * width
            assert plane["segments"] == stored[name] and plane["row_bytes"] == width
        assert rank_tables.rank_bytes(rank) == partition.rows * 136
        first += partition.rows
    assert first == rows


def test_shared_layout_negative_control(tmp_path):
    # The same tables partitioned at TP3 do not publish the shared record (each rank stores a third of the rows).
    shared, *_ = _tables(tmp_path, world=3, packed=True, shared=True)
    partitioned, *_ = _tables(tmp_path, world=3, packed=True, shared=False)
    assert shared.export_metadata() != partitioned.export_metadata()
    assert partitioned.rank_bytes(0) < shared.rank_bytes(0)


class _Inputs:
    def __init__(self, packed_fp4):
        self.packed_fp4 = packed_fp4


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("shared", [None, True, False])
def test_prepare_detects_mxfp4_rows_and_applies_the_layout_switch(monkeypatch, tmp_path, packed, shared):
    geometry = engram.build_geometry(layer_ids=(1, 2), base_table_size=101, compressed_vocab_size=32)
    text = dict(engram_compressed_vocab_size=32, engram_layer_ids=[1, 2], engram_vocab_size=101,
                engram_num_embeddings=list(geometry.num_embeddings))
    tensors = {}
    for layer, rows in zip((1, 2), geometry.num_embeddings):
        tensors[f"layers.{layer}.engram.embed.weight"] = SimpleNamespace(
            shape=(rows, 128 if packed else 256), dtype="U8" if packed else "F8_E4M3", shard="w", offset=0)
        tensors[f"layers.{layer}.engram.embed.scale"] = SimpleNamespace(shape=(rows, 8), dtype="F8_E8M0",
                                                                         shard="s", offset=0)
    for name in ("w", "s"):
        (tmp_path / name).write_bytes(bytes(sum(geometry.num_embeddings) * 256))
    manifest = SimpleNamespace(metadata_path=tmp_path, snapshot=tmp_path, tensors=tensors)
    import transformers

    monkeypatch.setattr(transformers, "PreTrainedTokenizerFast", lambda tokenizer_file: None)
    monkeypatch.setattr(recipe_engram.engram, "build_compressed_token_map", lambda tokenizer: (list(range(32)), 32))
    monkeypatch.setattr(recipe_engram.engram, "build_geometry", lambda **kwargs: geometry)
    monkeypatch.setattr(recipe_engram.ReadinessTimings, "allocate", lambda frame, count: None)
    monkeypatch.setattr(recipe_engram.EngramInputs, "allocate",
                        lambda tokens, device, *, timings, interval, packed_fp4=False: _Inputs(packed_fp4))
    host = lambda name, shape, dtype: SimpleNamespace(host_view=torch.zeros(shape, dtype=dtype))
    tables = Tables.prepare(manifest, SimpleNamespace(text=text), tokens=4, device="cpu", sequence=None, frame=None,
                            host_allocator=host, shard=TensorParallelShard(1, 3), shared=shared)
    assert tables.packed_fp4 is packed and tables.row_bytes == ((128 if packed else 256), 8)
    assert all(inputs.packed_fp4 is packed for inputs in tables.inputs.values())
    expect_shared = bool(shared)  # None keeps upstream: partitions for any multi-rank export
    assert tables.shared is expect_shared
    assert all((mapping is None) == expect_shared for mapping in tables.row_mapping.values())
    assert tables.columns.tolist() == [head * 256 + column for head in range(1, 24, 3) for column in range(256)]
    # Mixed FP8/MXFP4 tables are refused.
    tensors["layers.2.engram.embed.weight"] = SimpleNamespace(
        shape=(geometry.num_embeddings[1], 256 if packed else 128), dtype="F8_E4M3" if packed else "U8",
        shard="w", offset=0)
    with pytest.raises(ValueError, match="mix FP8 and MXFP4"):
        Tables.prepare(manifest, SimpleNamespace(text=text), tokens=4, device="cpu", sequence=None, frame=None,
                       host_allocator=host, shard=TensorParallelShard(1, 3), shared=shared)
