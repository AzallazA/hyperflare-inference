# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Per-row-count split policy for DeepSeek V4.1 BF16 MLA decode (TP3 decode lever 4a, 2026-11-18).

b12x's DSV41 decode forces ``num_splits = workspace.num_chunks_value`` (prepare_unified_decode_launch), so the split
count follows the tuned split geometry: 10 splits (one 64-candidate window per CTA) or 1 split for the 640-wide
SWA + indexed decode. ``B12X_MLA_SM120_NUM_SPLITS`` cannot change it (the explicit/forced value wins over that
environment variable in plan_unified_decode_splits). At TP3, 24 heads run as a 16-head and an 8-head grid of
rows x 1 x splits CTAs each (one CTA per SM), so the decode costs whole waves of 2 x rows x splits CTAs on 188 SMs;
fewer, longer splits can save a wave at some row counts (two-point model in decode-trace REVIEW-FABLE.md section 5).

``FLARE_DS41_MLA_DECODE_SPLITS="9:10,18:5,28:10,37:5"`` wraps ``prepare_unified_decode_launch`` at runtime and passes
``forced_num_splits`` for decode launches that attend an indexed cache: the first ``ROWS:SPLITS`` pair with prepared
rows <= ROWS applies; rows above the last pair, SWA-only launches (drafts) and non-V4.1 models keep b12x's choice.
b12x clamps the value to [1, ceil(width / 64)] and to the workspace's split capacity; splits stay on 64-candidate
window boundaries, so the result differs from 10 splits only by the merge's FP32 rounding, as the tuner's 1-split
candidate does today. No b12x source changes (b12x keys programs and tuning races by source fingerprints; a runtime
attribute does not enter them). Export-time: graphs capture the prepared launch. Unset/empty leaves b12x untouched.
"""

import inspect
import os

ENV = "FLARE_DS41_MLA_DECODE_SPLITS"
# 6-9 splits of 10 windows give chunks_per_split 2 with empty splits that still occupy SMs (always worse than 5).
ALLOWED_SPLITS = (1, 2, 3, 4, 5, 10)
_TARGET = "prepare_unified_decode_launch"
_ORIGINAL = "_flare_decode_splits_original"


def parse_policy(text):
    """``"9:10,18:5"`` -> ((9, 10), (18, 5)); empty/None -> (). Row bounds strictly increasing, values >= 1."""
    if text is None or not text.strip():
        return ()
    pairs = []
    for item in text.split(","):
        rows, sep, splits = item.strip().partition(":")
        if not sep:
            raise ValueError(f"{ENV}: expected ROWS:SPLITS pairs, got {item!r}")
        rows, splits = int(rows), int(splits)
        if rows < 1 or splits not in ALLOWED_SPLITS:
            raise ValueError(f"{ENV}: rows must be positive and splits one of {ALLOWED_SPLITS}, got {item!r}")
        if pairs and rows <= pairs[-1][0]:
            raise ValueError(f"{ENV}: row bounds must increase, got {text!r}")
        pairs.append((rows, splits))
    return tuple(pairs)


def format_policy(policy):
    return ",".join(f"{rows}:{splits}" for rows, splits in policy)


def policy_from_environment(env=None):
    return parse_policy((os.environ if env is None else env).get(ENV))


def splits_for(policy, rows):
    for bound, splits in policy:
        if rows <= bound:
            return splits
    return None


def _kernel_module():
    from b12x.attention._shared.mla import kernel
    return kernel


def _is_dsv41(traits):
    if traits is None:
        return False
    from b12x.attention._shared.mla.traits import ModelType
    return int(traits.model_type) == int(ModelType.DSV41)


class _DecodeSplitPolicy:
    """Replacement for kernel.prepare_unified_decode_launch; delegates every call, adding forced_num_splits."""

    def __init__(self, original, policy, is_dsv41=_is_dsv41):
        self.original = original
        self.policy = policy
        self.is_dsv41 = is_dsv41
        self.signature = inspect.signature(original)
        self.applied = []  # (rows, requested, planned num_splits) per forced preparation, for logs and tests

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        arguments = bound.arguments
        if (arguments.get("forced_num_splits") is None and arguments.get("indexed_k_cache") is not None
                and self.is_dsv41(arguments.get("traits_override"))):
            rows = int(arguments["q_all"].shape[0])
            splits = splits_for(self.policy, rows)
            if splits is not None:
                arguments["forced_num_splits"] = splits
                launch = self.original(*bound.args, **bound.kwargs)
                self.applied.append((rows, splits, getattr(launch, "num_splits", None)))
                return launch
        return self.original(*bound.args, **bound.kwargs)


def install(policy, kernel=None, is_dsv41=_is_dsv41):
    """Wrap b12x's decode preparation with ``policy`` (replaces an earlier policy). Returns the wrapper."""
    kernel = _kernel_module() if kernel is None else kernel
    current = getattr(kernel, _TARGET)
    original = current.original if isinstance(current, _DecodeSplitPolicy) else current
    wrapper = _DecodeSplitPolicy(original, tuple(policy), is_dsv41)
    setattr(kernel, _ORIGINAL, original)
    setattr(kernel, _TARGET, wrapper)
    return wrapper


def uninstall(kernel=None):
    kernel = _kernel_module() if kernel is None else kernel
    original = getattr(kernel, _ORIGINAL, None)
    if original is not None:
        setattr(kernel, _TARGET, original)
        delattr(kernel, _ORIGINAL)


def install_from_environment():
    """Install only when a policy is set, so the default leaves b12x exactly as shipped."""
    policy = policy_from_environment()
    if policy:
        wrapper = install(policy)
        print(f"[flare] {ENV}={format_policy(policy)}: DSV41 decode split policy installed (pid {os.getpid()})",
              flush=True)
        return wrapper
    return None


def installed_policy(kernel=None):
    """The policy the b12x decode preparation runs with in this process (``()`` when the hook is absent)."""
    kernel = _kernel_module() if kernel is None else kernel
    current = getattr(kernel, _TARGET)
    return current.policy if isinstance(current, _DecodeSplitPolicy) else ()
