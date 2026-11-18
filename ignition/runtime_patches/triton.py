# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Identify compiled helpers by host architecture without invoking a process."""

from functools import lru_cache


@lru_cache
def platform_key() -> str:
    from platform import machine, system
    from struct import calcsize

    return ",".join([machine(), system(), f"{calcsize('P') * 8}bit"])


def replacements():
    from triton.runtime import build

    return ((build, "platform_key", platform_key),)


def initialize_b12x_worker(*args):
    from b12x._lib.compile_pool import _initialize_worker

    from . import apply_triton

    # b12x uses spawn, so this process imports its original initializer.
    apply_triton()
    # TP3 fork: FLARE_DS41_MLA_DECODE_SPLITS (inherited) must shape the decode programs workers discover and compile
    # exactly as in the main process, or the main process asks for programs no worker compiled.
    from ignition.components import mla_decode_splits

    mla_decode_splits.install_from_environment()
    _initialize_worker(*args)
