"""b12x overlay patch flashinfer-pcie-twoshot-pull-world3 on CPU: the two-shot pull alternative's launch sweep,
equivalence keys and family bucket mirror the upstream two-shot entry; the switch defaults off.
Skipped unless the patched overlay is first on PYTHONPATH."""

import pytest

tuning = pytest.importorskip("b12x.comm.pcie._collective_tuning")
if not hasattr(tuning, "TWOSHOT_PULL"):
    pytest.skip("b12x without the two-shot pull overlay patch", allow_module_level=True)

from b12x.preparation import FrozenMapping  # noqa: E402


def _native(transport, rows):
    return {"surface": "PCIeTwoShotBF16.all_reduce", "world_size": 3, "rank": 0, "topology": "pcie_ipc",
            "call": {"operation": "all_reduce", "dtype": "bfloat16", "shape": (rows, 5120), "transport": transport,
                     "threads": 512, "block_limit": 64},
            "setup": {"max_rows": 655360, "pack_stride": 218464, "slot_bytes": 1, "slots": 2}}


def _query(names, rows=1024):
    alternatives = {name: _native("pull" if name == tuning.TWOSHOT_PULL else "push", rows) for name in names}
    default = tuning.CollectiveConfig(implementation=names[0], threads=512, blocks=64)
    return tuning.CollectiveQuery(operation="all_reduce", world_size=3, rank=0,
                                  alternatives=FrozenMapping(alternatives), default=default)


def test_switch_defaults_off(monkeypatch):
    monkeypatch.delenv("B12X_PCIE_TWOSHOT_PULL", raising=False)
    assert tuning.twoshot_pull_enabled() is False
    for value, expected in (("0", False), ("", False), ("1", True), ("on", True)):
        monkeypatch.setenv("B12X_PCIE_TWOSHOT_PULL", value)
        assert tuning.twoshot_pull_enabled() is expected
    monkeypatch.setenv("B12X_PCIE_TWOSHOT_PULL", "maybe")
    with pytest.raises(ValueError):
        tuning.twoshot_pull_enabled()


@pytest.mark.parametrize("rows", [24, 640, 768, 1024])
def test_pull_sweep_and_keys_mirror_push(rows):
    query = _query(("twoshot", tuning.TWOSHOT_PULL), rows)
    configs = list(tuning._configs(query))
    push = [(c.threads, c.blocks) for c in configs if c.implementation == "twoshot"]
    pull = [(c.threads, c.blocks) for c in configs if c.implementation == tuning.TWOSHOT_PULL]
    assert push == pull and len(push) >= 5
    for c in configs:
        key = tuning._equivalence_key(query, None, c)
        twin = tuning.CollectiveConfig(implementation="twoshot", threads=c.threads, blocks=c.blocks)
        assert key[1:] == tuning._equivalence_key(query, None, twin)[1:]
        assert key[0] == c.implementation        # pull and push never collapse into one candidate


def test_family_bucket_accepts_pull_and_differs_from_upstream():
    upstream = tuning._family_bucket(_query(("twoshot",)))
    patched = tuning._family_bucket(_query(("twoshot", tuning.TWOSHOT_PULL)))
    assert upstream[2] == patched[2] == 1024 * 5120 * 2
    assert tuning.TWOSHOT_PULL in patched[1]["alternatives"] and upstream[1] != patched[1]
