"""PreparedChunkedAllReduce part cutting on CPU: halving (base) vs max_part_rows (TP3Config.ar_part_rows)."""

import pytest
import torch

from ignition.components import collectives


class _Channel:
    def __init__(self, capacity):
        self.capacity = capacity

    def should_allreduce(self, tensor):
        return tensor.numel() * tensor.element_size() <= self.capacity


class _Runtime:
    algorithm = "oneshot"

    def __init__(self, rows):
        self.channel = _Channel(rows * 5120 * 2)

    def for_stream(self):
        return self.channel


@pytest.fixture
def parts(monkeypatch):
    made = []

    class _Part:
        def __init__(self, source, output, **kwargs):
            made.append((source.shape[0], kwargs["name"]))

    monkeypatch.setattr(collectives, "PreparedAllReduce", _Part)
    monkeypatch.setattr(collectives, "_roce", lambda runtime: False)

    def build(rows, capacity_rows, **options):
        made.clear()
        source = torch.empty(rows, 5120, dtype=torch.bfloat16)
        chunked = collectives.PreparedChunkedAllReduce(source, torch.empty_like(source), runtime=_Runtime(capacity_rows),
                                                       session=None, coordinator=None, name="r", **options)
        return [end - begin for begin, end in chunked.bounds], [rows for rows, _ in made]
    return build


@pytest.mark.parametrize("rows,capacity,expected", [
    (4096, 1024, [1024] * 4), (3072, 1024, [768] * 4), (2560, 1024, [640] * 4), (4096, 768, [512] * 8), (24, 1024, [24])])
def test_halving_is_the_base(parts, rows, capacity, expected):
    bounds, made = parts(rows, capacity)
    assert bounds == made == expected


@pytest.mark.parametrize("rows,capacity,cap,expected", [
    (4096, 1024, 768, [768] * 5 + [256]), (3072, 1024, 768, [768] * 4), (2560, 1024, 768, [768] * 3 + [256]),
    (4096, 768, 768, [768] * 5 + [256]), (512, 1024, 768, [512]), (24, 1024, 768, [24]),
    (4096, 512, 768, [384] * 10 + [256])])   # cap above capacity: halving resumes from the cap
def test_max_part_rows_cuts_at_the_cap(parts, rows, capacity, cap, expected):
    bounds, made = parts(rows, capacity, max_part_rows=cap)
    assert bounds == made == expected


def test_max_part_rows_rejects_negative(parts):
    with pytest.raises(ValueError):
        parts(4096, 1024, max_part_rows=-1)
