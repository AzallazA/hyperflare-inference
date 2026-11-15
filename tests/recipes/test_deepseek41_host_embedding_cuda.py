# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""TP3 tp3.embed_rows on one GPU: the vectorised host-row gather equals b12x's lookup bit for bit, and its speed.

The table has DS41's geometry (129280 x 5120 BF16, 1.23 GiB) in page-locked host memory read through a device
pointer, as ``embed_host`` parks it. Both paths run as the serving graphs do: compiled with ``compile_forward`` (the
export's tracer and Inductor), captured in a CUDA graph, replayed after the ids and the live count change in place.
Reference = ``PreparedEmbedding`` (zero fill + b12x ``embedding_out``). Shapes are the DSpark draft lookups (block 5:
accepted token + 4 noise rows per request, c1/c4/c8 = 5/20/40 rows) and the target lookups (6/12/24/48 rows).

    FLARE_RUN_CUDA_TESTS=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -p no:cacheprovider -s -q \\
        tests/recipes/test_deepseek41_host_embedding_cuda.py

``EMBED_TIMING`` lines (-s) give microseconds per lookup inside a graph of 20 back-to-back lookups for both paths.
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                                reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1")

ROWS, WIDTH, NOISE = 129280, 5120, 128799  # NOISE: any id; the kernel takes it as an argument
REPEAT = 20


@pytest.fixture(scope="module")
def table():
    from b12x.sequence._shared.disk_table import _tensor_from_pointer

    device = torch.device("cuda", torch.cuda.current_device())
    host = torch.empty((ROWS, WIDTH), dtype=torch.bfloat16, pin_memory=True)
    generator = torch.Generator(device=device).manual_seed(0)
    for start in range(0, ROWS, 8192):  # fill in slabs (no 1.2 GiB device temporary)
        stop = min(start + 8192, ROWS)
        host[start:stop].copy_(torch.randn((stop - start, WIDTH), generator=generator, device=device)
                               .to(torch.bfloat16))
    view = _tensor_from_pointer(host.data_ptr(), shape=(ROWS, WIDTH), dtype=torch.bfloat16, device=device,
                                nbytes=host.nbytes)
    yield view
    torch.cuda.synchronize()
    del view, host


def _ids(kind, rows, generator, device):
    ids = torch.randint(0, ROWS, (rows,), generator=generator, device="cpu")
    if kind == "draft":
        ids.view(-1, 5)[:, 1:] = NOISE
    return ids.to(device)


def _forwards(table, ids, count, *, noise, session):
    from ignition.components.embedding import PreparedEmbedding
    from ignition.preparation.compilation import compile_forward
    from ignition.recipes.deepseek41.host_embedding import HostRowGather

    rows = ids.numel()
    scratch = torch.empty((rows, WIDTH), dtype=torch.bfloat16, device=ids.device)
    base = PreparedEmbedding(table, ids, scratch, num_rows=count, session=session, name=f"emb{rows}") \
        .functional(session=session, dynamic_inputs=True)
    candidate = HostRowGather(table, rows=rows, noise_token=NOISE if noise else None)
    one = (lambda: base(token_ids=ids, num_rows=count), lambda: candidate(ids, count))
    many = (lambda: tuple(base(token_ids=ids, num_rows=count) for _ in range(REPEAT)),
            lambda: tuple(candidate(ids, count) for _ in range(REPEAT)))
    return [compile_forward(f) for f in one], [compile_forward(f) for f in many]


def _graph(session, compiled):
    """Capture one compiled forward; returns the graph and its (first) output tensor."""
    graph = torch.cuda.CUDAGraph()
    compiled()
    torch.cuda.synchronize()
    with session.capture(), torch.cuda.graph(graph):
        outputs = compiled()
    return graph, outputs if isinstance(outputs, torch.Tensor) else outputs[0]


def _time(graph, replays=50):
    for _ in range(5):
        graph.replay()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(replays):
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        times.append(start.elapsed_time(stop) * 1000 / REPEAT)
    return sorted(times)[len(times) // 2]


@torch.no_grad()
@pytest.mark.parametrize("kind,rows", [("draft", 5), ("draft", 20), ("draft", 40),
                                       ("target", 6), ("target", 12), ("target", 24), ("target", 48)])
def test_host_rows_equal_b12x_under_replay_and_timing(table, kind, rows):
    from b12x.preparation import PreparationSession

    device = table.device
    generator = torch.Generator().manual_seed(rows)
    ids = _ids(kind, rows, generator, device)
    count = torch.tensor([rows], dtype=torch.int32, device=device)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        (base_one, rows_one), (base_many, rows_many) = _forwards(table, ids, count, noise=kind == "draft",
                                                                 session=session)
        graphs = [_graph(session, f) for f in (base_one, rows_one)]
        for live in (rows, rows - 3, 1, 0, rows):
            ids.copy_(_ids(kind, rows, generator, device))
            ids[live:] = torch.where(torch.arange(rows - live, device=device) % 2 == 0, NOISE, 3)  # dead rows
            count.fill_(live)
            for graph, _ in graphs:
                graph.replay()
            torch.cuda.synchronize()
            (_, expected), (_, actual) = graphs
            assert torch.equal(actual, expected), (kind, rows, live)
            reference = torch.zeros_like(expected)
            reference[:live] = table[ids[:live].long()]
            assert torch.equal(expected, reference)
            for _, output in graphs:  # dirty the graph outputs: the next replay must store every element again
                output.view(torch.uint8).fill_(0x7F)
        count.fill_(rows)
        ids.copy_(_ids(kind, rows, generator, device))
        timing = [_graph(session, f)[0] for f in (base_many, rows_many)]
        base_us, rows_us = (_time(graph) for graph in timing)
        print(f"EMBED_TIMING kind={kind} rows={rows} b12x_us={base_us:.2f} rows_us={rows_us:.2f} "
              f"saved_us={base_us - rows_us:.2f}")
        for graph, _ in graphs:
            graph.reset()
        for graph in timing:
            graph.reset()
