import hashlib
import os
from functools import cache
from pathlib import Path
from types import ModuleType

import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.utils.cpp_extension import load

KERNELS = Path(__file__).resolve().parent / "kernels"
SOURCES = {path.stem: path for path in sorted((*KERNELS.glob("*.cu"), *KERNELS.glob("*.cpp")))}
LIBRARIES = {"engram_nvme": ["-luring"]}


CUDA_FLAGS = ("-O3", "-gencode=arch=compute_120a,code=sm_120a", "-Xcompiler", "-fno-gnu-unique")


def build(name: str, sources, digest_files, *, defines: tuple[str, ...] = (), cuda: bool = True,
          extra_cuda_cflags: tuple[str, ...] = (), extra_cflags: tuple[str, ...] = ("-O3", "-std=c++17"),
          ldflags=None) -> ModuleType:
    """Shared builder: load `sources` as module `<name>_<digest of digest_files>[_<defines hash>]` (sm_120a).

    Used by `extension` (single-file BRRR kernels) and by recipe-local multi-file extensions (DS41 EXL3 experts)."""
    digest = hashlib.sha1(b"".join(Path(path).read_bytes() for path in digest_files)).hexdigest()[:10]
    defines = tuple(defines)
    if defines and not cuda:
        raise ValueError(f"compile-time defines are only supported for CUDA builds, not {name}")
    variant = "" if not defines else "_" + hashlib.sha1("\n".join(defines).encode()).hexdigest()[:8]
    return load(
        name=f"{name}_{digest}{variant}",
        sources=[str(source) for source in sources],
        extra_cflags=list(extra_cflags) if extra_cflags else None,
        # -fno-gnu-unique: static locals of templates such as host.cuh opt_in_smem<layer_kernel> are otherwise GNU-unique
        # symbols shared by every loaded variant, so a second variant in one process skipped its smem opt-in ("fits 0 CTAs").
        extra_cuda_cflags=[*CUDA_FLAGS, *(["-lineinfo"] if os.environ.get("BRRR_LINEINFO") else []), *extra_cuda_cflags,
                           *(f"-D{d}" for d in defines)] if cuda else None,
        extra_ldflags=ldflags,
        verbose=False,
    )


@cache
def extension(name: str, defines: tuple[str, ...] = ()) -> ModuleType:
    """Build/load kernel source `name`; `defines` ("NAME=VALUE" strings, passed as -D) select a compile-time variant,
    built and cached under its own module name (no defines = the plain build, same name scheme as before)."""
    source = SOURCES[name]
    cuda = source.suffix == ".cu"
    return build(name, [source], (source, *sorted(KERNELS.glob("*.cuh"))), defines=defines, cuda=cuda,
                 extra_cflags=() if cuda else ("-O3", "-std=c++17"), ldflags=LIBRARIES.get(name))


def exchange_ipc(ext: ModuleType, nbytes: int, rank: int, group: ProcessGroup) -> list[int]:
    mine = ext.alloc_ipc(nbytes)
    handles: list[list[int]] = [[] for _ in range(ext.RANKS)]
    dist.all_gather_object(handles, mine[1:], group=group)
    return [mine[0] if peer == rank else ext.open_ipc(handles[peer]) for peer in range(ext.RANKS)]
