# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""One-launch 24-head DSV41 BF16 MLA decode for TP3 (decode lever 4b, 2026-11-18). Default off.

At TP3 a DeepSeek V4.1 rank holds 24 query heads. b12x tiles BF16 MLA decode 16 heads per CTA, so
``run_unified_decode`` launches a 16-head grid and an 8-head remainder grid (zero-padded to 16 rows), rows x 1 x
splits CTAs each, one CTA per SM: 2 x rows x splits CTAs, and each remainder CTA gathers and dequantizes the same
KV again. ``H24DecodeKernel`` is our own CuTe kernel (no b12x file changes): one CTA per (row, split) stages Q for all
24 heads, gathers each 64-candidate KV window once (b12x's IO producer, unchanged), dequantizes every K/V fragment
once and feeds it to two MMAs, one for heads 0-15 and one for heads 16-23 (rows 8-15 of that tile repeat heads
16-23 instead of b12x's zero padding; MMA rows are independent and those rows are never written). Each 16-row tile
keeps its own online-softmax state and accumulators and runs b12x's own S3/S4/S7 stages, so every written value is
computed exactly as in the 16 + 8 grids at the same split count (bit-identical by construction; the GPU test checks
it). The CTA count halves: rows x splits.

Selection: ``FLARE_DS41_MLA_H24="9:0,80:1"`` (ROWS:0|1 pairs, first ``rows <= ROWS`` wins, rows above the last
bound keep 16 + 8) wraps b12x's module-level ``_sparse_mla_decode_grid_flat_launch``. For a selected DSV41 BF16
fused-mapping decode with 24 heads, the 16-head grid call launches the H24 kernel (same runtime argument tuple, so a
prepared launch replays it through b12x's own ``run_compiled``) and the remainder call returns ``REMAINDER_DONE``, a
launcher that launches nothing. The split count comes from the prepared launch (FLARE_DS41_MLA_DECODE_SPLITS
applies unchanged). b12x's compile cache keys skip sources outside b12x, so the compile spec carries this file's
SHA-256 and b12x's MLA source fingerprint explicitly.

Shared memory (sm_120 opt-in limit 101376 B incl. the 1 KB the DSL reserves): Q 24 x 520 BF16 (24960), KV 2 x 64 x
544 (69632), mbarriers, one softmax reduce scratch shared by both tiles (an extra named barrier separates them),
token indices, P 16 x 72 + 8 x 72 BF16. b12x's FP8-W, W-scale, Q-scale, KV-scale and RoPE regions are unused on
this path and dropped. 288 threads (8 math warps + 1 IO warp), one CTA per SM as before.
"""

import functools
import hashlib
import inspect
import os
from pathlib import Path

ENV = "FLARE_DS41_MLA_H24"
_TARGET = "_sparse_mla_decode_grid_flat_launch"
_ORIGINAL = "_flare_h24_original"
KERNEL_ID = "flare.attention.mla.sm120.decode_h24"
KERNEL_VERSION = 1
HEADS = 24
SMEM_LIMIT = 101376 - 1024  # sm_120 opt-in per-block limit minus the DSL's static reservation

# DSV41 BF16 geometry (b12x make_unified_traits(DSV41, BF16, NVFP4_E4M3)); asserted against b12x in check_traits.
BI = 64
KV_SMEM_STRIDE = 544
Q_NOPE_STRIDE = 520
D_NOPE = 512
SM_P_STRIDE = 72
N_V_CHUNKS = 8
MATH_THREADS = 256
BLOCK_THREADS = 288
CAND_WINDOW = 64


# ---------------------------------------------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------------------------------------------
def parse_policy(text):
    """``"9:0,80:1"`` -> ((9, False), (80, True)); empty/None -> (). Row bounds strictly increasing."""
    if text is None or not text.strip():
        return ()
    pairs = []
    for item in text.split(","):
        rows, sep, value = item.strip().partition(":")
        if not sep or value.strip() not in ("0", "1"):
            raise ValueError(f"{ENV}: expected ROWS:0|1 pairs, got {item!r}")
        rows = int(rows)
        if rows < 1:
            raise ValueError(f"{ENV}: rows must be positive, got {item!r}")
        if pairs and rows <= pairs[-1][0]:
            raise ValueError(f"{ENV}: row bounds must increase, got {text!r}")
        pairs.append((rows, value.strip() == "1"))
    if not any(on for _, on in pairs):
        raise ValueError(f"{ENV}: {text!r} selects no row count; leave it unset instead")
    return tuple(pairs)


def format_policy(policy):
    return ",".join(f"{rows}:{int(bool(on))}" for rows, on in policy)


def policy_from_environment(env=None):
    return parse_policy((os.environ if env is None else env).get(ENV))


def selected(policy, rows):
    for bound, on in policy:
        if rows <= bound:
            return on
    return False


# ---------------------------------------------------------------------------------------------------------------
# Compile identity: b12x's cache key covers b12x sources only (source_dependencies skips other modules).
# ---------------------------------------------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def source_identity():
    here = Path(__file__).resolve().parent
    own = hashlib.sha256(b"".join((here / name).read_bytes() for name in ("mla_h24_decode.py", "mla_h24_kernel.py")))
    own = own.hexdigest()
    from b12x.attention._shared.mla import decode_math, io, kernel, smem, traits
    base = hashlib.sha256()
    for module in (kernel, decode_math, io, smem, traits):
        base.update(Path(inspect.getsourcefile(module)).read_bytes())
    return own, base.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------------------------------------------
def smem_offsets():
    """(name, offset, bytes) of the H24 layout, mirroring mla_h24_kernel.storage_cls alignments (checked against it in tests)."""
    def align(value, to):
        return (value + to - 1) // to * to
    regions = (("q", HEADS * Q_NOPE_STRIDE * 2, 128), ("kv", 2 * BI * KV_SMEM_STRIDE, 128), ("mbar", 32, 16),
               ("reduce", 2 * 8 * 16 * 4, 4), ("token_idx", 2 * BI * 4, 16), ("sm_p", 16 * SM_P_STRIDE * 2, 128),
               ("sm_p2", 8 * SM_P_STRIDE * 2, 16))
    out, offset = [], 0
    for name, size, alignment in regions:
        offset = align(offset, alignment)
        out.append((name, offset, size))
        offset += size
    return tuple(out), align(offset, 128)


def check_traits(traits):
    from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat
    want = dict(model_type=ModelType.DSV41, compute_mode=ComputeMode.BF16, scale_format=ScaleFormat.NVFP4_E4M3,
                d_nope=D_NOPE, d_rope=0, d_v=512, quant_tile=64, num_scales=8, n_v_chunks=N_V_CHUNKS,
                nt_per_warp_xv=1, kv_smem_stride=KV_SMEM_STRIDE, q_nope_stride=Q_NOPE_STRIDE, bi=BI, hpb=16,
                block_threads=BLOCK_THREADS, math_threads=MATH_THREADS, v_has_rope=False, has_extra_cache=True,
                fp8_rope=False, latent_scale_per_token=False, fp8_internal=False)
    have = {name: getattr(traits, name) for name in want}
    if have != want:
        raise ValueError(f"H24 decode needs DSV41 BF16 traits {want}, got {have}")
    from b12x.attention._shared.mla.smem import make_smem_layout
    layout = make_smem_layout(traits)
    if (layout.sm_p_full_stride, layout.kv_bufs, layout.token_idx_buf_bytes) != (SM_P_STRIDE, 2, BI * 4):
        raise ValueError("H24 decode: b12x DSV41 shared-memory strides changed")


def launch_h24(q_all, kv_flat, swa_indices, mid_out, mid_lse, swa_len_t, extra_kv_flat, extra_indices_t,
               extra_len_t, sm_scale, latent_scale, swa_page_size, topk, extra_topk, num_main_chunks, num_splits,
               chunks_per_split, stride_kv_block, pbs_extra, stride_extra_kv_block, indexed_page_table,
               indexed_num_pages):
    """Compile (or, in a compile-only scope, plan) and launch the H24 program. The runtime argument tuple is b12x's
    fused-mapping tuple (``_sparse_mla_decode_grid_flat_launch``), so a prepared H24 program replays through b12x."""
    import cuda.bindings.driver as cuda
    import cutlass
    import torch
    from cutlass import Float32, Int32, Int64
    from b12x._lib.compiler import DimKey, KernelCompileSpec, key_field, launch, tensor_key
    from b12x.attention._shared.mla.kernel import LOG2_E, _to_cute
    from b12x.attention._shared.mla.traits import ComputeMode, ModelType, ScaleFormat, make_unified_traits

    traits = make_unified_traits(ModelType.DSV41, ComputeMode.BF16, ScaleFormat.NVFP4_E4M3, fp8_rope=False)
    rows = int(q_all.shape[0])
    if tuple(q_all.shape[1:]) != (HEADS, D_NOPE) or int(q_all.stride(2)) != 1:
        raise ValueError(f"H24 decode needs q (rows, 24, 512) with unit dim stride, got {tuple(q_all.shape)}")
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    if indexed_page_table is None:
        table_flat, table_width, table_stride = extra_indices_t.view(-1), -1, 0
    else:
        table_width, table_stride = int(indexed_page_table.shape[1]), int(indexed_page_table.stride(0))
        table_flat = (extra_indices_t.view(-1) if table_width == 0
                      else indexed_page_table.as_strided((table_width,), (1,)) if table_stride == 0
                      else indexed_page_table.view(-1))
    args = (_to_cute(q_all, cutlass.BFloat16, dynamic_layout=True), _to_cute(kv_flat, cutlass.Uint8, align=16),
            _to_cute(swa_indices, cutlass.Int32, align=4, dynamic_layout=True),
            _to_cute(mid_out, cutlass.BFloat16, align=16, dynamic_layout=True),
            _to_cute(mid_lse, cutlass.Float32, align=4, dynamic_layout=True),
            Float32(float(sm_scale) * LOG2_E), Float32(float(latent_scale)),
            _to_cute(swa_len_t, cutlass.Int32, align=4, dynamic_layout=True), Int64(stride_kv_block),
            _to_cute(extra_kv_flat, cutlass.Uint8, align=16),
            _to_cute(extra_indices_t, cutlass.Int32, align=4, dynamic_layout=True),
            _to_cute(extra_len_t, cutlass.Int32, align=4, dynamic_layout=True),
            Int32(num_main_chunks), Int64(stride_extra_kv_block),
            _to_cute(table_flat, cutlass.Int32, align=4, dynamic_layout=True), Int32(table_width),
            Int64(table_stride), Int64(indexed_num_pages), Int32(rows), stream)
    from ignition.components.mla_h24_kernel import H24DecodeKernel
    kernel = H24DecodeKernel(traits, int(swa_page_size), int(chunks_per_split), int(num_splits), int(topk),
                             int(extra_topk), tuple(q_all.stride()), int(swa_indices.stride(0)),
                             int(extra_indices_t.stride(0)), tuple(mid_out.stride()), tuple(mid_lse.stride()),
                             int(pbs_extra))
    own, base = source_identity()
    from b12x.attention._shared.mla.kernel import _topk_bucket
    spec = KernelCompileSpec.from_fields(
        KERNEL_ID, KERNEL_VERSION,
        key_field("flare_source", own), key_field("b12x_mla_source", base),
        key_field("num_splits", int(num_splits)), key_field("chunks_per_split", int(chunks_per_split)),
        key_field("page_block_size", int(swa_page_size)), key_field("pbs_extra", int(pbs_extra)),
        key_field("topk", int(topk)), key_field("extra_topk", int(extra_topk)),
        key_field("topk_bucket", _topk_bucket(topk)),
        key_field("latent_scale_identity", int(float(latent_scale) == 1.0)),
        tensor_key("q_all", q_all, dims=(DimKey.dynamic(), DimKey.exact(HEADS), DimKey.exact(D_NOPE))),
        tensor_key("swa_indices", swa_indices, dims=(DimKey.dynamic(), DimKey.bucket(topk))),
        tensor_key("mid_out", mid_out, dims=(DimKey.dynamic(), DimKey.exact(HEADS), DimKey.bucket(num_splits),
                                             DimKey.exact(512))),
        tensor_key("mid_lse", mid_lse, dims=(DimKey.dynamic(), DimKey.exact(HEADS), DimKey.bucket(num_splits))),
        tensor_key("extra_indices", extra_indices_t, dims=(DimKey.dynamic(), DimKey.bucket(max(extra_topk, 1)))),
        tensor_key("topk_length", swa_len_t, dims=(DimKey.dynamic(),)),
        tensor_key("extra_topk_length", extra_len_t, dims=(DimKey.dynamic(),)),
    )
    return launch(kernel.call_mapped_extra_pertok, compile_spec=spec, compile_args=args, runtime_args=args)


# ---------------------------------------------------------------------------------------------------------------
# Hook around b12x's grid launcher
# ---------------------------------------------------------------------------------------------------------------
class _RemainderDone:
    """Prepared launcher for the remainder grid an H24 launch already covered: launches nothing. Callable (b12x's
    run_compiled calls it with the runtime arguments when this hook is absent at replay) and carries no programs."""

    __b12x_programs__ = ()
    __b12x_dependencies__ = ()

    def __call__(self, *args, **kwargs):
        return None

    def __repr__(self):
        return "REMAINDER_DONE"


REMAINDER_DONE = _RemainderDone()


def _is_h24_program(prepared):
    return any(getattr(key, "name", "") == KERNEL_ID for key in getattr(prepared, "__b12x_programs__", ()) or ())


class _H24Launch:
    """Replacement for kernel._sparse_mla_decode_grid_flat_launch; delegates every call it does not take over."""

    def __init__(self, inner, real, policy, launcher=launch_h24):
        self.inner = inner          # what was installed before (e.g. the remainder-stream wrapper)
        self.real = real            # b12x's own launcher (runs a prepared H24 program without a stream fork)
        self.policy = tuple(policy)
        self.launcher = launcher
        self.signature = inspect.signature(real)
        self.pending = {}           # device index -> (q ptr, mid_out ptr) of an H24 launch awaiting its remainder call
        self.applied = []           # (rows, num_splits) per H24 launch, for logs and tests
        self.replays = 0            # prepared H24 programs run (eager or captured), for the bench's self-check

    def _eligible(self, a):
        q = a["q_all"]
        return (int(q.shape[1]) == HEADS and int(a["model_type"]) == 3 and int(a["compute_mode"]) == 1
                and int(a["scale_format"]) == 2 and not a["fp8_rope"] and a["has_extra"] and a["per_token_len"]
                and not a["latent_scale_per_token"] and not a["v41_fp8_internal"]
                and int(a["v41_heads_per_block"]) == 16 and int(a["topk"]) > 0 and int(a["extra_topk"]) > 0
                and selected(self.policy, int(q.shape[0])))

    @staticmethod
    def _identity(a):
        def pointer(tensor):
            try:
                return int(tensor.data_ptr())
            except Exception:  # meta/fake tensors during planning
                return id(tensor)
        q = a["q_all"]
        index = getattr(getattr(q, "device", None), "index", None)
        return index, (pointer(q), pointer(a["mid_out"]))

    def __call__(self, *args, **kwargs):
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        prepared = a.get("_prepared")
        if prepared is REMAINDER_DONE or isinstance(prepared, _RemainderDone):
            return None
        if prepared is not None and _is_h24_program(prepared):
            self.replays += 1
            return self.real(*args, **kwargs)  # b12x builds the identical runtime tuple and runs our program
        if not self._eligible(a):
            return self.inner(*args, **kwargs)
        offset, blocks, valid = int(a["head_block_offset"]), int(a["grid_h_blocks"]), int(a["valid_hpb"])
        index, identity = self._identity(a)
        if (offset, blocks, valid) == (0, 1, 16):
            launched = self.launcher(
                a["q_all"], a["kv_flat"], a["swa_indices"], a["mid_out"], a["mid_lse"], a["swa_len_t"],
                a["extra_kv_flat"], a["extra_indices_t"], a["extra_len_t"], a["sm_scale"], a["latent_scale"],
                a["swa_page_size"], a["topk"], a["extra_topk"], a["num_main_chunks"], a["num_splits"],
                a["chunks_per_split"], a["stride_kv_block"], a["pbs_extra"], a["stride_extra_kv_block"],
                a["indexed_page_table"], a["indexed_num_pages"])
            self.pending[index] = identity
            self.applied.append((int(a["q_all"].shape[0]), int(a["num_splits"])))
            return launched
        if (offset, blocks, valid) == (1, 1, 8) and self.pending.get(index) == identity:
            del self.pending[index]
            return REMAINDER_DONE
        raise RuntimeError(f"H24 decode: unexpected grid call (offset {offset}, blocks {blocks}, valid {valid}) "
                           "for a selected 24-head decode")


def _kernel_module():
    from b12x.attention._shared.mla import kernel
    return kernel


def install(policy, kernel=None, launcher=launch_h24):
    """Wrap b12x's grid launcher with ``policy`` (outermost; replaces an earlier H24 wrapper). Returns the wrapper."""
    kernel = _kernel_module() if kernel is None else kernel
    current = getattr(kernel, _TARGET)
    inner = current.inner if isinstance(current, _H24Launch) else current
    real = getattr(kernel, "_flare_remainder_stream_original", None) or inner
    wrapper = _H24Launch(inner, real, policy, launcher)
    setattr(kernel, _ORIGINAL, inner)
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
        print(f"[flare] {ENV}={format_policy(policy)}: DSV41 24-head one-launch decode installed (pid {os.getpid()})",
              flush=True)
        return wrapper
    return None


def installed_policy(kernel=None):
    kernel = _kernel_module() if kernel is None else kernel
    current = getattr(kernel, _TARGET)
    return current.policy if isinstance(current, _H24Launch) else ()
