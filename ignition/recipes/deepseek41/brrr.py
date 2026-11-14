# SPDX-License-Identifier: Apache-2.0
"""TP3: BRRR's fused DS41 layer kernel (o-projection through MLP all-reduce) inside Flare forwards.

``TP3Config.brrr_layers`` selects layers. For those layers the decode and verify forwards replace inverse RoPE, the
grouped o-projection, the attention all-reduce, the FFN mHC, the router, routed and shared experts and the MLP
all-reduce with one kernel launch. Prefill keeps Flare's path.

Functional forwards (upstream's Inductor-planned memory, 2026-11-14 port): the tail takes the attention output and the
attention mHC values as arguments and returns the FFN mHC values and the MLP all-reduce output as forward-local
tensors (``PreparedBRRRTail.__call__``); ``flare::brrr_layer_tail`` declares every tensor it writes in
``mutates_args``, so Inductor keeps the launch and places its outputs in the graph's combined pool. Everything the
kernel holds raw pointers to beyond Flare's own weights (its checkpoint copies of block scales, mHC parameters and
staged dense copies) is packed into the persistent arena at registration (``persistent_copy``), never left in the
catalog's transient pool.

The kernel (brrr/kernels/decoder.cu, Apache-2.0, fraserprice/DeepSeek-4.1-Flash-1017-BRRR, TP3 port) reads:
- routed experts from b12x's prepared W4A8 planes, scales decoded in-kernel from b12x's CSF payload
  (``brrr_native_scales`` loads these layers without CSF instead),
- dense weights as Flare's own row-major values (``brrr_dense_native``, default) or as its own staged copies read
  from the checkpoint view ``brrr_ckpt``; block scales always come from that view,
- peer flags/inboxes in Flare PeerStorage, so export resets them (and the epoch kept in the flag page) to zero,
- with CSF experts, b12x's compact scale payload directly (the expanded scale planes are b12x preparation temporaries
  since FlashInfer aa6c4221, so the kernel is given a small placeholder for them instead).

Every switch arrives through the ``TP3Config`` the engine and the prepared tails hold; nothing here reads the
environment (``TORCH_EXTENSIONS_DIR`` is set for the extension build only).
"""

import ctypes as C
import json
import os
import sys
from functools import cache
from pathlib import Path

import torch
import torch.distributed as dist

from ignition.preparation.persistent import persistent_copy, persistent_zeros

from .tp3_config import TP3Config

_ROOT = Path(__file__).resolve().parents[3]


def _cuext():
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(_ROOT / ".brrr-ext"))
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from brrr import cuext

    return cuext


# Compile-time kernel switches: (TP3Config field, compile-time define, extension attribute). Only the test build's spin
# trap is left (the item-6 split tail / fast route / counter parity experiments were not ported, 2026-11-14).
SWITCHES = (("brrr_spin_trap", "BRRR_SPIN_TRAP", "SPIN_TRAP"),)


def kernel_defines(tp3: TP3Config) -> tuple[str, ...]:
    """Compile-time defines of the layer kernel for this config (all off = the serving build)."""
    return tuple(f"{define}=1" for field, define, _ in SWITCHES if getattr(tp3, field))


def engine_key(tp3: TP3Config):
    """What makes two engines differ: the kernel build."""
    return kernel_defines(tp3)


@cache
def extension(defines: tuple[str, ...] = ()):
    """The layer kernel extension; ``defines`` (from kernel_defines) selects a compile-time variant."""
    return _cuext().extension("decoder", tuple(defines))


def check_switches(ext, tp3: TP3Config):
    """Contract check: the loaded build's compile-time switches are the ones the config asked for."""
    want = {attr: int(bool(getattr(tp3, field))) for field, _, attr in SWITCHES}
    got = {attr: int(getattr(ext, attr, -1)) for attr in want}
    if got != want:
        raise RuntimeError(f"BRRR kernel build switches {got} do not match the TP3 config {want}")


# Engines by global handle (the custom op's `handle`): (extension module, the module's own engine handle). Several
# engines (one per compile-time variant) can coexist in one process (the layer test's A/B pass); serving has one.
_RUNNERS = []


def _stages(w):
    w = w.contiguous().view(torch.uint8)
    w = torch.nn.functional.pad(w, (0, -w.shape[1] % 128))
    return w.reshape(w.shape[0] // 64, 4, 2, 8, -1, 4, 4, 8).permute(0, 4, 5, 1, 3, 6, 2, 7).contiguous().view(-1)


def _interleave(w, block):
    return w.reshape(2, -1, block, *w.shape[1:]).transpose(0, 1).reshape(w.shape).contiguous()


def _rows2d(weight):
    """Flare's row-major E4M3 values ([L*M, K]) from an MXFP8LinearWeight or MXFP8Rows, without copying."""
    rows = getattr(weight, "weight", weight)
    v = rows.values
    if v.ndim == 3:  # logical [M, K, L] view over physical [L, M, K]
        v = v.permute(2, 0, 1).reshape(-1, v.shape[1])
    if not v.is_contiguous() or v.data_ptr() != rows.values.data_ptr():
        raise RuntimeError("BRRR dense-native needs contiguous row-major values")
    return v


def _shard(rank, size):
    return slice(rank * size, (rank + 1) * size)


def check_constants(ext, config, weights):
    """Constants contract: the kernel's compile-time model constants against Flare's config and the mHC options
    decoder.py hands to PreparedMHC (rms_eps = norm_eps = rms_norm_eps, hc_eps, hc_sinkhorn_iters)."""
    import math

    text = config.text
    ints = {"HC": text["hc_mult"], "TOPK": text["num_experts_per_tok"], "ROPE": text["qk_rope_head_dim"],
            "EXPERTS": text["n_routed_experts"], "HID": text["hidden_size"], "SINKHORN": text["hc_sinkhorn_iters"]}
    floats = {"LIMIT": text["swiglu_limit"], "ROUTE_SCALE": text["routed_scaling_factor"],
              "HC_EPS": text["hc_eps"], "NORM_EPS": text["rms_norm_eps"]}
    bad = [f"{k}={getattr(ext, k)} (config {v})" for k, v in ints.items() if getattr(ext, k) != int(v)]
    bad += [f"{k}={getattr(ext, k)!r} (config {v!r})" for k, v in floats.items()
            if not math.isclose(getattr(ext, k), float(v), rel_tol=1e-6, abs_tol=0.0)]  # kernel floats are fp32
    if text.get("scoring_func") != "sqrtsoftplus" or not text.get("norm_topk_prob", False):
        bad.append(f"router scoring {text.get('scoring_func')!r}/norm_topk_prob={text.get('norm_topk_prob')!r} (kernel: sqrtsoftplus, normalized)")
    res = weights.residual
    if tuple(res["hc_ffn_fn"].shape) != (ext.MIX, ext.HC * ext.HID) or res["hc_ffn_base"].numel() != ext.MIX or res["hc_ffn_scale"].numel() != 3:
        bad.append(f"mHC fn {tuple(res['hc_ffn_fn'].shape)} base {res['hc_ffn_base'].numel()} scale {res['hc_ffn_scale'].numel()} "
                   f"(kernel MIX={ext.MIX}, HC*HID={ext.HC * ext.HID}, 3)")
    if bad:
        raise RuntimeError("BRRR kernel constants do not match the model: " + "; ".join(bad))


class Checkpoint:
    def __init__(self, root):
        from safetensors import safe_open

        self.root = Path(root)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self._open = cache(lambda name: safe_open(self.root / name, framework="pt"))

    def shape(self, name):
        return self._open(self.index[name]).get_slice(name).get_shape()

    def get(self, name, rows=slice(None), cols=slice(None)):
        tensor = self._open(self.index[name]).get_slice(name)
        picked = tensor[rows, cols] if len(tensor.get_shape()) == 2 else tensor[rows]
        return picked.contiguous()


class Engine:
    """One per rank: the kernel's peer buffers, scratch and registered layers, configured by one TP3Config."""

    def __init__(self, group, device, ipc, tp3: TP3Config):
        if tp3.brrr_ckpt is None:
            raise ValueError("TP3 config: BRRR needs brrr_ckpt (the checkpoint view it reads dense copies from)")
        self.ext = ext = extension(kernel_defines(tp3))
        check_switches(ext, tp3)
        if ext.MAXM != MAX_ROWS:
            raise RuntimeError(f"BRRR kernel MAXM {ext.MAXM} differs from brrr.MAX_ROWS {MAX_ROWS}")
        self.group, self.device, self.ipc, self.tp3 = group, device, ipc, tp3
        self.rank, world = dist.get_rank(group=group), dist.get_world_size(group=group)
        if world != ext.RANKS:
            raise ValueError(f"BRRR kernel is built for {ext.RANKS} ranks, group has {world}")
        # Same kernel build on every rank (handbook D.3.4): a mixed build is a configuration error.
        builds = [None] * world
        dist.all_gather_object(builds, engine_key(tp3), group=group)
        if any(b != builds[0] for b in builds):
            raise RuntimeError(f"BRRR kernel builds differ between ranks: {builds}")
        nbytes = ext.flag_bytes() + ext.inbox_bytes()
        own = ipc.cudaMalloc(nbytes)
        ipc.storage.cuda.call("cuMemsetD8_v2", C.c_uint64(own), C.c_ubyte(0), C.c_size_t(nbytes))
        torch.cuda.synchronize(device)
        handles = [None] * world
        dist.all_gather_object(handles, ipc.cudaIpcGetMemHandleBytes(own), group=group)
        buffers = [own if r == self.rank else ipc.cudaIpcOpenMemHandleBytes(handles[r]) for r in range(world)]
        # The epoch counter sits in the unused tail of this rank's flag page (flags use 2*2*RANKS*CHUNKS words).
        # fused_attn_per_chunk is a per-engine kernel setting (every rank must agree: it shapes the flag protocol).
        local = ext.create(buffers, self.rank, epoch_addr=own + ext.flag_bytes() - 64,
                           attn_per_chunk_flags=int(tp3.fused_attn_per_chunk))
        self.handle = len(_RUNNERS)  # global handle (the custom op argument); serving has one engine, handle 0
        _RUNNERS.append((ext, local))
        self.local_handle = local
        dist.barrier(group=group)
        self.checkpoint = Checkpoint(tp3.brrr_ckpt)
        self.frequencies = None
        self.slots = {}
        self.keep = []
        # Stand-in for the expanded scale planes of CSF layers: add_layer requires the tensors, the kernel reads the
        # compact payload instead (issue_csf), and b12x releases its expanded planes after preparation.
        self.unused_scales = persistent_zeros((1, 16), torch.uint8, device)

    def raw(self, name, rows=slice(None), cols=slice(None)):
        return self.checkpoint.get(name, rows, cols).to(self.device)

    def raw_u8(self, name, rows=slice(None), cols=slice(None)):
        return self.raw(name, rows, cols).view(torch.uint8)

    def raw_padded(self, name, rows=slice(None), cols=slice(None)):
        """Rank shards may run past the 8 real o-groups (TP3 pads to 9); zero-fill like Flare's padded loader."""
        shape = self.checkpoint.shape(name)
        r0, r1 = rows.start or 0, rows.stop if rows.stop is not None else shape[0]
        if len(shape) == 2:
            c0, c1 = cols.start or 0, cols.stop if cols.stop is not None else shape[1]
            got = self.raw_u8(name, slice(r0, max(min(r1, shape[0]), r0)), slice(c0, max(min(c1, shape[1]), c0)))
            out = torch.zeros((r1 - r0, c1 - c0), dtype=torch.uint8, device=self.device)
            out[: got.shape[0], : got.shape[1]] = got
        else:
            got = self.raw_u8(name, slice(r0, max(min(r1, shape[0]), r0)))
            out = torch.zeros((r1 - r0,), dtype=torch.uint8, device=self.device)
            out[: got.shape[0]] = got
        return out

    def register(self, index, weights, config=None):
        if index in self.slots:
            return self.slots[index]
        ext, r, p, tp3 = self.ext, self.rank, f"layers.{index}", self.tp3
        if config is not None:
            check_constants(ext, config, weights)
        inter, oa = ext.INTER, ext.OA
        experts = weights.experts.experts
        impl = experts._impl
        rep = impl.representation_for("w4a8_mx")
        if rep is None or rep.n64_repack or rep.intermediate_size != inter or rep.num_experts != ext.EXPERTS:
            raise RuntimeError(f"BRRR layer {index}: unexpected expert representation {rep}")
        u8 = lambda t: t.contiguous().view(torch.uint8)
        rows = _shard(r, inter)
        gate_up = None
        if not tp3.brrr_dense_native or tp3.brrr_csf_check:
            gate_up = torch.cat((self.raw(f"{p}.ffn.shared_experts.w1.weight", rows),
                                 self.raw(f"{p}.ffn.shared_experts.w3.weight", rows)))
        blocks = _shard(r, inter // 32)
        scales = (self.raw_u8(f"{p}.ffn.shared_experts.w1.scale", blocks), self.raw_u8(f"{p}.ffn.shared_experts.w3.scale", blocks))
        kwargs = dict(
            woa_s=u8(self.raw_padded(f"{p}.attn.wo_a.scale", _shard(r, oa // 32))),
            wob_s=u8(self.raw_padded(f"{p}.attn.wo_b.scale", cols=_shard(r, oa // 32))),
            gate_b=self.raw(f"{p}.ffn.gate.bias").float().contiguous(),
            bias_vl=None,
            sh13_s=_interleave(torch.cat(scales), 1),
            sh2_s=torch.nn.functional.pad(self.raw_u8(f"{p}.ffn.shared_experts.w2.scale", cols=blocks),
                                          (0, ext.SH2_LD - inter // 32)).contiguous(),
            w13=rep.w13_rp.contiguous(), s13=rep.w13_sfb.contiguous(),
            w2=rep.w2_rp.contiguous(), s2=rep.w2_sfb.contiguous(),
            fn=self.raw(f"{p}.hc_ffn_fn").float().contiguous(),
            hc_scale=self.raw(f"{p}.hc_ffn_scale").float().contiguous(),
            hc_base=self.raw(f"{p}.hc_ffn_base").float().contiguous(),
            norm=self.raw(f"{p}.ffn_norm.weight").contiguous(),
            image_lo=0, dense_native=False,
            # Merged hot/cold bank: experts are stored hot-first; the kernel routes by global ID and reads the slot.
            # Already a persistent int32 forward constant (weights.py); passed as is, never copied.
            hot_map=weights.experts.hot_map,
            # Cold-first job ordering: slots >= hot_rows are host-RAM experts. Default EXPERTS (all resident) keeps
            # the arrival order; a merged bank without a known hot count also stays unordered (safe).
            hot_rows=ext.EXPERTS if weights.experts.hot_map is None or weights.experts.hot_rows is None
                     else int(weights.experts.hot_rows))
        if tp3.brrr_dense_native:
            # Read Flare's own row-major E4M3 dense values (same bytes as the checkpoint shards; block scales stay as
            # small checkpoint copies) instead of staged copies: ~35-40 MB per layer back to the KV pool.
            out, shared = weights.attention["output"], weights.experts.shared
            native = dict(woa=_rows2d(out.wo_a), wob=_rows2d(out.wo_b), gate=weights.experts.router,
                          sh13=_rows2d(shared.gate_up), sh2=_rows2d(shared.down))
            if tp3.brrr_csf_check:
                ref = dict(woa=self.raw_padded(f"{p}.attn.wo_a.weight", _shard(r, oa)),
                           wob=self.raw_padded(f"{p}.attn.wo_b.weight", cols=_shard(r, oa)),
                           gate=self.raw_u8(f"{p}.ffn.gate.weight"), sh13=u8(gate_up),
                           sh2=self.raw_u8(f"{p}.ffn.shared_experts.w2.weight", cols=rows))
                for name, t in native.items():
                    a, b = t.contiguous().view(torch.uint8), ref[name]
                    same = a.shape == b.shape and bool(torch.equal(a, b))
                    print(f"[brrr] rank {r} layer {index} native {name} {tuple(t.shape)} {t.dtype} "
                          f"stride {tuple(t.stride())}: {'identical' if same else f'MISMATCH vs {tuple(b.shape)}'}", flush=True)
                    if not same:
                        raise RuntimeError(f"BRRR layer {index}: Flare's {name} differs from the checkpoint bytes")
            kwargs.update(native, dense_native=True)
        else:
            kwargs.update(woa=_stages(self.raw_padded(f"{p}.attn.wo_a.weight", _shard(r, oa))),
                          wob=_stages(self.raw_padded(f"{p}.attn.wo_b.weight", cols=_shard(r, oa))),
                          gate=_stages(self.raw(f"{p}.ffn.gate.weight")), sh13=_stages(_interleave(gate_up, 32)),
                          sh2=_stages(self.raw(f"{p}.ffn.shared_experts.w2.weight", cols=rows)))
        decoder = impl.nvfp4_csf
        if decoder is not None:
            # Same b12x CSF payload as the vLLM build BRRR's in-kernel reader was qualified on.
            for tag, plane in (("csf13", decoder.first), ("csf2", decoder.second)):
                a = _csf_args(plane.scales)
                kwargs[f"{tag}_storage"], kwargs[f"{tag}_gdesc"], kwargs[f"{tag}_meta"] = a["storage"], a["gdesc"], a["meta"]
            if tp3.brrr_csf_check:
                decoder.decode_all(impl.w1_blockscale, impl.w2_blockscale)
                for tag, ref in (("csf13", rep.w13_sfb), ("csf2", rep.w2_sfb)):
                    want = ref.contiguous().view(torch.int32).reshape(-1)
                    got = torch.empty_like(want)
                    ext.csf_expand_all(kwargs[f"{tag}_storage"], kwargs[f"{tag}_gdesc"], kwargs[f"{tag}_meta"], got)
                    torch.cuda.synchronize(self.device)
                    print(f"[brrr] rank {r} layer {index} {tag} CSF self-check: {int((got != want).sum())} of "
                          f"{want.numel()} words differ", flush=True)
        hot_map = kwargs["hot_map"]
        if hot_map is not None and (hot_map.dtype != torch.int32 or not hot_map.is_contiguous()):
            raise RuntimeError(f"BRRR layer {index}: the merged bank's hot_map must be contiguous int32")
        planes = ("w13", "w2") if decoder is not None else ("w13", "s13", "w2", "s2")
        for name in planes:
            if kwargs[name].data_ptr() != getattr(rep, {"w13": "w13_rp", "s13": "w13_sfb", "w2": "w2_rp", "s2": "w2_sfb"}[name]).data_ptr():
                raise RuntimeError(f"BRRR layer {index}: {name} is not b12x's resident plane")
        if decoder is not None:
            # The kernel reads the CSF payload (issue_csf), never s13/s2; holding b12x's expanded planes here would pin
            # a per-layer preparation temporary (FlashInfer aa6c4221 releases them after preparation).
            kwargs.update(s13=self.unused_scales, s2=self.unused_scales)
        # The kernel holds raw pointers to these tensors: the copies made here are forward constants, packed into the
        # persistent arena with the final weights (registration runs inside the catalog's transient pool).
        for name in PERSISTENT_COPIES:
            if kwargs.get(name) is not None and not (tp3.brrr_dense_native and name in NATIVE_DENSE):
                kwargs[name] = persistent_copy(kwargs[name])
        self.keep.append(kwargs)
        self.slots[index] = ext.add_layer(self.local_handle, **kwargs)
        if tp3.brrr_log:
            print(f"[brrr] rank {r} layer {index} -> slot {self.slots[index]} (w13 {tuple(rep.w13_rp.shape)}, "
                  f"sfb {tuple(rep.w13_sfb.shape)}, layout {rep.w13_layout})", flush=True)
        return self.slots[index]


# add_layer tensors built by register() from checkpoint reads (copies, not views of Flare's weights or b12x's planes).
NATIVE_DENSE = ("woa", "wob", "gate", "sh13", "sh2")
PERSISTENT_COPIES = ("woa_s", "wob_s", "gate_b", "sh13_s", "sh2_s", "fn", "hc_scale", "hc_base", "norm",
                     "csf13_gdesc", "csf2_gdesc", *NATIVE_DENSE)


def _csf_args(scales):
    """Describe a b12x CsfBlockPayload (destination 2, mxfp4, 3-byte group records) for the fused kernel: per-(payload
    expert, group) record descriptors first | count << 22, mirroring BlockStorageLayout (from BRRR's TP3 decoder.py)."""
    st = scales.storage
    R, C, Es = int(scales.rows), int(scales.columns), int(scales.num_experts)
    if not (scales.destination == 2 and scales.native_codes and scales.group_record_bytes == 3 and scales.quantization == "mxfp4"):
        raise RuntimeError(f"BRRR kernel needs CSF destination 2 / mxfp4 / 3-byte records; got {scales}")
    a16 = lambda v: (v + 15) // 16 * 16
    cg, groups = (C + 3) // 4, (R + 127) // 128 * ((C + 3) // 4)
    pages = (groups + 31) // 32
    prefixes, code_bytes = groups + pages, R * C // 8
    bases = 1024
    codes = a16(bases + Es * R)
    page_bases = a16(codes + Es * code_bytes)
    offsets = a16(page_bases + Es * pages * 4)
    recpos = a16(offsets + Es * prefixes * 2)
    value_off = int(st[:8].view(torch.int64).item())
    pb = st[page_bases:page_bases + Es * pages * 4].view(torch.int32).view(Es, pages)
    off = st[offsets:offsets + Es * prefixes * 2].view(torch.int16).view(Es, prefixes).to(torch.int32) & 0xFFFF
    sg = torch.arange(groups, device=st.device)
    page = sg // 32
    first = pb[:, page] + off[:, sg + page]
    count = pb[:, page] + off[:, sg + page + 1] - first
    if int(first.max()) >= 1 << 22 or int(count.max()) >= 1 << 10 or int(count.min()) < 0:
        raise RuntimeError(f"CSF record descriptor out of range: first<{int(first.max())} count<{int(count.max())}")
    gdesc = (first | (count << 22)).to(torch.int32).contiguous()
    return dict(storage=st, gdesc=gdesc, meta=[bases, codes, code_bytes, value_off, recpos, R, cg, groups, int(scales.sections)])


_ENGINE = []


def create_engine(group, device, ipc, tp3: TP3Config):
    """Create the rank's engine once; None (no-op) unless ``tp3.brrr_layers`` selects layers."""
    if not tp3.brrr_layers:
        return None
    if not _ENGINE:
        _ENGINE.append(Engine(group, device, ipc, tp3))
    return _ENGINE[0]


def engine():
    return _ENGINE[0] if _ENGINE else None


# Rows one launch accepts (decoder.cu MAXM, checked against the built extension in Engine.__init__): PreparedLayer
# refuses a BRRR layer in a wider graph at construction, before tuning, instead of at the first launch.
MAX_ROWS = 128


# Writes only the five FFN-side outputs; attn_post/attn_comb/attn_pre are read (__ldg in the kernel's residual mix), so
# they are not declared mutated: the compiler then needs no copies of them around the launch (40 per decode step).
@torch.library.custom_op("flare::brrr_layer_tail", mutates_args=("y", "resid1", "post1", "comb1", "pre1"))
def brrr_layer_tail(handle: int, slot: int, o: torch.Tensor, positions: torch.Tensor, frequencies: torch.Tensor,
                    resid0: torch.Tensor, attn_post: torch.Tensor, attn_comb: torch.Tensor, attn_pre: torch.Tensor,
                    y: torch.Tensor, resid1: torch.Tensor, post1: torch.Tensor, comb1: torch.Tensor,
                    pre1: torch.Tensor) -> None:
    ext, local = _RUNNERS[handle]
    ext.run(local, slot, o, positions, None, frequencies, resid0, attn_post, attn_comb, attn_pre,
            y, resid1, post1, comb1, pre1)


@brrr_layer_tail.register_fake
def _fake_brrr_layer_tail(handle, slot, o, positions, frequencies, resid0, attn_post, attn_comb, attn_pre,
                          y, resid1, post1, comb1, pre1):
    pass


class PreparedBRRRTail:
    """Layer tail through the BRRR kernel, from the pre-inverse-RoPE attention output to the MLP all-reduce.

    ``__call__(attended, residual, post, comb, attn_mix, positions)`` takes the attention output, the attention mHC
    values (residual, post, comb and its pre-mix output ``attn_mix``) and the token positions, and returns
    ``(mlp_output, (ffn_residual, ffn_post, ffn_comb), pre_mix)``: what Flare's path gets from the MLP all-reduce, the
    FFN mHC and its pre-mix output. The outputs are forward-local (``torch.empty``; the kernel writes every row) unless
    ``out`` names destinations (the layer test). The kernel also writes the attention mHC post/comb/pre-mix
    (``mutates_args``); the caller must not read them afterwards.
    """

    def __init__(self, index, weights, positions_dtype, frequencies, config=None, tp3=None, engine_=None):
        e = engine() if engine_ is None else engine_  # engine_: an explicit engine (test A/B of kernel variants)
        if e is None:
            raise RuntimeError("BRRR engine was not created")
        tp3 = e.tp3 if tp3 is None else tp3
        self.handle, self.slot = int(e.handle), int(e.register(index, weights, config))
        self.frequencies = frequencies
        if tp3.brrr_break == "rope":
            # Negative control for test tooling: reversed RoPE frequencies (wrong inverse rotation of the attention output).
            self.frequencies = persistent_copy(frequencies.flip(0).contiguous())
        if positions_dtype != torch.int64:
            raise ValueError("BRRR tail expects int64 positions")

    def __call__(self, attended, residual, post, comb, attn_mix, positions, *, out=None):
        if out is None:
            mlp_output = torch.empty((residual.shape[0], residual.shape[2]), dtype=residual.dtype, device=residual.device)
            out = (mlp_output, torch.empty_like(residual), torch.empty_like(post), torch.empty_like(comb),
                   torch.empty_like(attn_mix))
        mlp_output, ffn_residual, ffn_post, ffn_comb, pre_mix = out
        brrr_layer_tail(self.handle, self.slot, attended, positions, self.frequencies, residual, post, comb, attn_mix,
                        mlp_output, ffn_residual, ffn_post, ffn_comb, pre_mix)
        return mlp_output, (ffn_residual, ffn_post, ffn_comb), pre_mix
