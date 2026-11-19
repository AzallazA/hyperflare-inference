"""Keys b12x's persistent program discovery to the DSV41 decode hooks that change which programs a decode plan launches.

b12x stores each candidate set's discovered programs in the preparation progress database under
``b12x.preparation._discovery.discovery_key``: the b12x source closure of the factories, the toolchain and the
candidate pickles. Our hooks (``FLARE_DS41_MLA_DECODE_SPLITS``: other split counts; ``FLARE_DS41_MLA_H24``: our own
24-head program) change the programs ``compile_compressed_sparse_mla`` launches without changing any of that, so a
record written by a run without them (or with another policy, or another H24 source) is restored as the planned set.
Materialisation then launches a program that was never planned and b12x's no-compilation phase rejects it (E2
export seq346: ``flare.attention.mla.sm120.decode_h24`` unplanned at ``SparseMlaConfig(max_chunks_per_row=1)``).

The wrapper mixes the installed hook identity into the key, only for candidate sets that contain the compressed MLA
factory and only while a hook is installed, so the default (no hook) and every other operation keep b12x's keys.
No b12x file changes."""

from __future__ import annotations

import hashlib
import json

FACTORY_MARK = "compressed_sparse_mla"
_ORIGINAL = "_flare_mla_discovery_original"


def hook_identity():
    """JSON-ready identity of the installed DSV41 decode hooks in this process, ``None`` when none is installed."""
    from . import mla_decode_splits, mla_h24_decode

    identity = {}
    splits = mla_decode_splits.installed_policy()
    if splits:
        identity["mla_decode_splits"] = mla_decode_splits.format_policy(splits)
    h24 = mla_h24_decode.installed_policy()
    if h24:
        own, base = mla_h24_decode.source_identity()
        identity["mla_h24"] = [mla_h24_decode.format_policy(h24), mla_h24_decode.KERNEL_VERSION, own, base]
    return identity or None


def _has_mla_factory(candidates):
    return any(FACTORY_MARK in str(job.factory) for _, _, jobs in candidates for job in jobs)


class _HookedDiscoveryKey:
    def __init__(self, original):
        self.original = original

    def __call__(self, candidates, device):
        key = self.original(candidates, device)
        if key is None or not _has_mla_factory(candidates):
            return key
        identity = hook_identity()
        if identity is None:
            return key
        return hashlib.sha256(json.dumps({"b12x": key, "flare_mla_hooks": identity},
                                         sort_keys=True).encode()).hexdigest()


def install(session=None):
    """Wrap the discovery key b12x's preparation session uses (idempotent). Returns the wrapper."""
    if session is None:
        from b12x.preparation import session
    current = session.discovery_key
    if isinstance(current, _HookedDiscoveryKey):
        return current
    wrapper = _HookedDiscoveryKey(current)
    setattr(session, _ORIGINAL, current)
    session.discovery_key = wrapper
    return wrapper


def uninstall(session=None):
    if session is None:
        from b12x.preparation import session
    original = getattr(session, _ORIGINAL, None)
    if original is not None:
        session.discovery_key = original
        delattr(session, _ORIGINAL)
