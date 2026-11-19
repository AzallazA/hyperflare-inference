# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CuTe kernel of the one-launch 24-head DSV41 BF16 MLA decode (see mla_h24_decode.py for the design and hook).

Specialised copy of b12x ``UnifiedDecodeKernel._kernel_body`` for the path a TP3 DS41 rank runs (DSV41, BF16
arithmetic, NVFP4/MXFP8 544-byte staging, SWA + indexed sections, per-token lengths, fused page mapping, generic
8-math-warp arm). The IO producer, S3 mask, S4 online softmax and S7 epilogue are b12x's own functions; the QK, P
store and PV stages are b12x's with each dequantized K/V fragment fed to two MMAs (heads 0-15 and 16-23).
"""

import functools

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.intrinsics.atomics import ld_shared_u32, st_shared_bf16_from_f32
from b12x._lib.intrinsics.constants import mma_m16n8k16_f32_bf16
from b12x._lib.intrinsics.memory import ldmatrix_m8n8x4_b16, shared_ptr_to_u32
from b12x._lib.intrinsics.packed import byte_perm
from b12x.attention._shared.mla import decode_math as dm
from b12x.attention._shared.mla.io import io_issue_gather
from b12x.attention._shared.mla.kernel import _exit_thread

from ignition.components.mla_h24_decode import (BI, BLOCK_THREADS, CAND_WINDOW, D_NOPE, HEADS, KV_SMEM_STRIDE,
                                                MATH_THREADS, N_V_CHUNKS, Q_NOPE_STRIDE, SM_P_STRIDE, check_traits,
                                                smem_offsets, SMEM_LIMIT)


@functools.lru_cache(maxsize=1)
def storage_cls():
    class SharedStorage:
        pass

    SharedStorage.__annotations__ = {
        "q": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, HEADS * Q_NOPE_STRIDE], 128],
        "kv": cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 2 * BI * KV_SMEM_STRIDE], 128],
        "mbar": cute.struct.Align[cute.struct.MemRange[cutlass.Uint64, 4], 16],
        "reduce": cute.struct.MemRange[cutlass.Float32, 2 * 8 * 16],
        "token_idx": cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 2 * BI], 16],
        "sm_p": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, 16 * SM_P_STRIDE], 128],
        "sm_p2": cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, 8 * SM_P_STRIDE], 16],
    }
    storage = cute.struct(SharedStorage)
    expected, total = smem_offsets()
    if dict(storage._offsets) != {name: offset for name, offset, _ in expected}:
        raise ValueError(f"H24 SharedStorage offsets {dict(storage._offsets)} diverge from {expected}")
    if int(storage.size_in_bytes()) != total or total > SMEM_LIMIT:
        raise ValueError(f"H24 SharedStorage is {int(storage.size_in_bytes())} B (layout {total}, limit {SMEM_LIMIT})")
    return storage


@cute.jit
def s0_load_q24(q_token: cute.Tensor, q_addr: Int32, tid: Int32):
    """b12x s0_load_q_bf16_to_smem for all 24 heads (no padding rows; d_rope = 0)."""
    i = tid
    while i < Int32(HEADS * D_NOPE):
        h = i // Int32(D_NOPE)
        d = i - h * Int32(D_NOPE)
        st_shared_bf16_from_f32(q_addr + (h * Int32(Q_NOPE_STRIDE) + d) * Int32(2), Float32(q_token[h, d]))
        i += Int32(MATH_THREADS)
    cute.arch.barrier(barrier_id=2, number_of_threads=MATH_THREADS)


@cute.jit
def _qk_source_pair(qk, qk2, q_addr: Int32, kv_addr: Int32, warp_first_cand: Int32, lane: Int32,
                    valid: cutlass.Boolean, *, swa: cutlass.Constexpr):
    """b12x _s1_qk_dsv41_source with each decoded K pair fed to both head tiles."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_row2 = Int32(16) + (lane & Int32(7))
    a_col = (lane >> Int32(4)) * Int32(8)
    kv_gid_row = warp_first_cand + gid
    for blk in cutlass.range_constexpr(8):
        swa_scales = []
        if cutlass.const_expr(swa):
            for sg in cutlass.range_constexpr(2):
                scale = Uint32(0)
                if valid:
                    scale = dm._ld_u8_zext(kv_addr + kv_gid_row * Int32(544), Int32(512 + blk * 2 + sg))
                swa_scales.append(scale)
        for ks in cutlass.range_constexpr(4):
            ko = Int32(blk) * Int32(64) + Int32(ks) * Int32(16)
            b0, b1 = Uint32(0), Uint32(0)
            if valid:
                if cutlass.const_expr(swa):
                    b0, b1 = dm._dsv41_qk_bfloat2x2_source(kv_addr, kv_gid_row, ko + tid * Int32(2), swa=swa,
                                                           scale=swa_scales[ks // 2])
                else:
                    b0, b1 = dm._dsv41_qk_bfloat2x2_source(kv_addr, kv_gid_row, ko + tid * Int32(2), swa=swa)
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                dm._smem_byte(q_addr, a_row * Int32(Q_NOPE_STRIDE * 2) + (ko + a_col) * Int32(2)))
            qk[0], qk[1], qk[2], qk[3] = mma_m16n8k16_f32_bf16(qk[0], qk[1], qk[2], qk[3], a0, a1, a2, a3, b0, b1)
            c0, c1, c2, c3 = ldmatrix_m8n8x4_b16(
                dm._smem_byte(q_addr, a_row2 * Int32(Q_NOPE_STRIDE * 2) + (ko + a_col) * Int32(2)))
            qk2[0], qk2[1], qk2[2], qk2[3] = mma_m16n8k16_f32_bf16(qk2[0], qk2[1], qk2[2], qk2[3],
                                                                   c0, c1, c2, c3, b0, b1)
    return qk[0], qk[1], qk[2], qk[3], qk2[0], qk2[1], qk2[2], qk2[3]


@cute.jit
def s1_pair(qk, qk2, q_addr: Int32, kv_addr: Int32, warp_first_cand: Int32, lane: Int32):
    """b12x s1_qk_nope_nvfp4_bf16 (544-byte DSV41 staging) for both head tiles."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_row2 = Int32(16) + (lane & Int32(7))
    a_col = (lane >> Int32(4)) * Int32(8)
    kv_gid_row = warp_first_cand + gid
    tag = ld_shared_u32(kv_addr + kv_gid_row * Int32(544) + Int32(528)).to(Int32)
    q0, q1, q2, q3 = qk[0], qk[1], qk[2], qk[3]
    r0, r1, r2, r3 = qk2[0], qk2[1], qk2[2], qk2[3]
    swa_warp = cute.arch.vote_all_sync((tag == Int32(1)) | (tag == Int32(2)))
    indexed_warp = cute.arch.vote_all_sync((tag == Int32(0)) | (tag == Int32(2)))
    if swa_warp:
        q0, q1, q2, q3, r0, r1, r2, r3 = _qk_source_pair([q0, q1, q2, q3], [r0, r1, r2, r3], q_addr, kv_addr,
                                                         warp_first_cand, lane, tag != Int32(2), swa=True)
    elif indexed_warp:
        q0, q1, q2, q3, r0, r1, r2, r3 = _qk_source_pair([q0, q1, q2, q3], [r0, r1, r2, r3], q_addr, kv_addr,
                                                         warp_first_cand, lane, tag != Int32(2), swa=False)
    else:
        for blk in cutlass.range_constexpr(8):
            for ks in cutlass.range_constexpr(4):
                ko = Int32(blk) * Int32(64) + Int32(ks) * Int32(16)
                b0, b1 = dm._dsv41_qk_bfloat2x2(kv_addr, kv_gid_row, ko + tid * Int32(2))
                a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                    dm._smem_byte(q_addr, a_row * Int32(Q_NOPE_STRIDE * 2) + (ko + a_col) * Int32(2)))
                q0, q1, q2, q3 = mma_m16n8k16_f32_bf16(q0, q1, q2, q3, a0, a1, a2, a3, b0, b1)
                c0, c1, c2, c3 = ldmatrix_m8n8x4_b16(
                    dm._smem_byte(q_addr, a_row2 * Int32(Q_NOPE_STRIDE * 2) + (ko + a_col) * Int32(2)))
                r0, r1, r2, r3 = mma_m16n8k16_f32_bf16(r0, r1, r2, r3, c0, c1, c2, c3, b0, b1)
    qk[0], qk[1], qk[2], qk[3] = q0, q1, q2, q3
    qk2[0], qk2[1], qk2[2], qk2[3] = r0, r1, r2, r3
    return qk, qk2


@cute.jit
def s5_pair(w_pre, w_pre2, sm_p_addr: Int32, sm_p2_addr: Int32, warp_id: Int32, lane: Int32):
    """b12x s5_fill_sm_p_full's P stores (its W-scale zeroing feeds the FP8 PV path only); tile 2 rows 0-7."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    c0 = warp_id * Int32(8) + tid * Int32(2)
    c1 = c0 + Int32(1)
    st_shared_bf16_from_f32(sm_p_addr + (gid * Int32(SM_P_STRIDE) + c0) * Int32(2), w_pre[0])
    st_shared_bf16_from_f32(sm_p_addr + (gid * Int32(SM_P_STRIDE) + c1) * Int32(2), w_pre[1])
    st_shared_bf16_from_f32(sm_p_addr + ((gid + Int32(8)) * Int32(SM_P_STRIDE) + c0) * Int32(2), w_pre[2])
    st_shared_bf16_from_f32(sm_p_addr + ((gid + Int32(8)) * Int32(SM_P_STRIDE) + c1) * Int32(2), w_pre[3])
    st_shared_bf16_from_f32(sm_p2_addr + (gid * Int32(SM_P_STRIDE) + c0) * Int32(2), w_pre2[0])
    st_shared_bf16_from_f32(sm_p2_addr + (gid * Int32(SM_P_STRIDE) + c1) * Int32(2), w_pre2[1])


@cute.jit
def _pv_pair(acc, acc2, sm_p_addr: Int32, sm_p2_addr: Int32, kv_addr: Int32, warp_id: Int32, lane: Int32,
             latent_scale: Float32, *, known_swa: cutlass.Constexpr):
    """b12x s6_xv_nope_nvfp4_bf16 (n_v_chunks 8, nt 1, 8 warps) with each decoded V pair fed to both tiles."""
    gid = lane >> Int32(2)
    tid = lane & Int32(3)
    entry_parity = gid & Int32(1)
    pair_selector = Int32(0x5410 if entry_parity == Int32(0) else 0x3276)
    a_row = (lane & Int32(7)) + ((lane >> Int32(3)) & Int32(1)) * Int32(8)
    a_row2 = lane & Int32(7)
    a_col = (lane >> Int32(4)) * Int32(8)
    for vc in cutlass.range_constexpr(N_V_CHUNKS):
        dim_base = Int32(vc) * Int32(64) + warp_id * Int32(8)
        col = dim_base + gid
        xv0, xv1, xv2, xv3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
        yv0, yv1, yv2, yv3 = Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
        for ks in cutlass.range_constexpr(BI // 16):
            k_base = Int32(ks) * Int32(16)
            a0, a1, a2, a3 = ldmatrix_m8n8x4_b16(
                sm_p_addr + (a_row * Int32(SM_P_STRIDE) + (k_base + a_col)) * Int32(2))
            c0, c1, c2, c3 = ldmatrix_m8n8x4_b16(
                sm_p2_addr + (a_row2 * Int32(SM_P_STRIDE) + (k_base + a_col)) * Int32(2))
            ent0 = k_base + tid * Int32(2)
            pair0 = dm._nvfp4_pair_bfloat2(kv_addr, ent0 + entry_parity, col & ~Int32(1), latent_scale,
                                           kv_smem_stride=KV_SMEM_STRIDE, known_swa=known_swa)
            pair8 = dm._nvfp4_pair_bfloat2(kv_addr, ent0 + Int32(8) + entry_parity, col & ~Int32(1),
                                           latent_scale, kv_smem_stride=KV_SMEM_STRIDE, known_swa=known_swa)
            peer0 = cute.arch.shuffle_sync_bfly(pair0, offset=4)
            peer8 = cute.arch.shuffle_sync_bfly(pair8, offset=4)
            b0 = byte_perm(pair0, peer0, pair_selector)
            b1 = byte_perm(pair8, peer8, pair_selector)
            xv0, xv1, xv2, xv3 = mma_m16n8k16_f32_bf16(xv0, xv1, xv2, xv3, a0, a1, a2, a3, b0, b1)
            yv0, yv1, yv2, yv3 = mma_m16n8k16_f32_bf16(yv0, yv1, yv2, yv3, c0, c1, c2, c3, b0, b1)
        acc[vc][0] = acc[vc][0] + xv0
        acc[vc][1] = acc[vc][1] + xv1
        acc[vc][2] = acc[vc][2] + xv2
        acc[vc][3] = acc[vc][3] + xv3
        acc2[vc][0] = acc2[vc][0] + yv0
        acc2[vc][1] = acc2[vc][1] + yv1
        acc2[vc][2] = acc2[vc][2] + yv2
        acc2[vc][3] = acc2[vc][3] + yv3
    return acc, acc2


@cute.jit
def s6_pair(acc, acc2, sm_p_addr: Int32, sm_p2_addr: Int32, kv_addr: Int32, warp_id: Int32, lane: Int32,
            latent_scale: Float32, known_swa: Int32):
    """b12x s6_xv_nope's known-source staging (one specialization per source, results through registers)."""
    staged = cute.make_rmem_tensor(N_V_CHUNKS * 4, Float32)
    staged2 = cute.make_rmem_tensor(N_V_CHUNKS * 4, Float32)
    if known_swa:
        a = [[value for value in row] for row in acc]
        b = [[value for value in row] for row in acc2]
        a, b = _pv_pair(a, b, sm_p_addr, sm_p2_addr, kv_addr, warp_id, lane, latent_scale, known_swa=True)
        for at in cutlass.range_constexpr(N_V_CHUNKS):
            for r in cutlass.range_constexpr(4):
                staged[at * 4 + r] = a[at][r]
                staged2[at * 4 + r] = b[at][r]
    else:
        a = [[value for value in row] for row in acc]
        b = [[value for value in row] for row in acc2]
        a, b = _pv_pair(a, b, sm_p_addr, sm_p2_addr, kv_addr, warp_id, lane, latent_scale, known_swa=False)
        for at in cutlass.range_constexpr(N_V_CHUNKS):
            for r in cutlass.range_constexpr(4):
                staged[at * 4 + r] = a[at][r]
                staged2[at * 4 + r] = b[at][r]
    return ([[staged[at * 4 + r] for r in range(4)] for at in range(N_V_CHUNKS)],
            [[staged2[at * 4 + r] for r in range(4)] for at in range(N_V_CHUNKS)])


class H24DecodeKernel:
    """DSV41 BF16 fused-mapping decode for 24 heads in one CTA per (row, split); see the module docstring."""

    def __init__(self, traits, page_block_size, chunks_per_split, num_splits, topk, extra_topk, q_stride,
                 swa_indices_stride0, extra_indices_stride0, mid_out_stride, mid_lse_stride, pbs_extra):
        check_traits(traits)
        self.traits = traits
        self.page_block_size = int(page_block_size)
        self.chunks_per_split = int(chunks_per_split)
        self.num_splits = int(num_splits)
        self.topk = int(topk)
        self.extra_topk = int(extra_topk)
        if self.topk <= 0 or self.extra_topk <= 0:
            raise ValueError("H24 decode needs both SWA and indexed sections")
        self.q_stride_row, self.q_stride_head, self.q_stride_dim = (int(v) for v in q_stride)
        self.swa_indices_stride_row = int(swa_indices_stride0)
        self.extra_indices_stride_row = int(extra_indices_stride0)
        (self.mid_out_stride_row, self.mid_out_stride_head, self.mid_out_stride_split,
         self.mid_out_stride_dim) = (int(v) for v in mid_out_stride)
        self.mid_lse_stride_row, self.mid_lse_stride_head, self.mid_lse_stride_split = (
            int(v) for v in mid_lse_stride)
        self.pbs_extra = int(pbs_extra)

    @cute.jit
    def call_mapped_extra_pertok(self, q_all: cute.Tensor, kv_cache_u8: cute.Tensor, swa_indices: cute.Tensor,
                                 mid_out: cute.Tensor, mid_lse: cute.Tensor, sm_scale_log2: Float32,
                                 latent_scale: Float32, topk_length: cute.Tensor, stride_kv_block: Int64,
                                 extra_kv_cache_u8: cute.Tensor, extra_indices: cute.Tensor,
                                 extra_topk_length: cute.Tensor, num_main_chunks: Int32,
                                 stride_extra_kv_block: Int64, indexed_page_table: cute.Tensor,
                                 table_width: Int32, table_stride: Int64, indexed_num_pages: Int64,
                                 num_tokens: Int32, stream):
        # Same runtime argument tuple as b12x UnifiedDecodeKernel.call_mapped_extra_pertok.
        self.kernel_h24(q_all, kv_cache_u8, swa_indices, mid_out, mid_lse, sm_scale_log2, latent_scale,
                        topk_length, stride_kv_block, extra_kv_cache_u8, extra_indices, extra_topk_length,
                        num_main_chunks, stride_extra_kv_block, indexed_page_table, table_width, table_stride,
                        indexed_num_pages).launch(grid=(num_tokens, 1, self.num_splits),
                                                  block=[BLOCK_THREADS, 1, 1], min_blocks_per_mp=1,
                                                  stream=stream)

    @cute.kernel
    def kernel_h24(self, q_all: cute.Tensor, kv_cache_u8: cute.Tensor, swa_indices: cute.Tensor,
                   mid_out: cute.Tensor, mid_lse: cute.Tensor, sm_scale_log2: Float32, latent_scale: Float32,
                   topk_length: cute.Tensor, stride_kv_block: Int64, extra_kv_cache_u8: cute.Tensor,
                   extra_indices: cute.Tensor, extra_topk_length: cute.Tensor, num_main_chunks: Int32,
                   stride_extra_kv_block: Int64, mapping_page_table: cute.Tensor, mapping_table_width: Int32,
                   mapping_table_stride: Int64, mapping_num_pages: Int64):
        t = self.traits
        tid = Int32(cute.arch.thread_idx()[0])
        lane = cute.arch.lane_idx()
        warp_id = tid >> Int32(5)
        token_idx, _head_block, split_idx = cute.arch.block_idx()
        token_idx = Int32(token_idx)
        split_idx = Int32(split_idx)

        # Per-token section lengths (b12x _kernel_body, per_token_len + has_extra).
        section_len = Int32(topk_length[token_idx])
        if section_len < Int32(0):
            section_len = Int32(0)
        if section_len > Int32(self.topk):
            section_len = Int32(self.topk)
        extra_section_len = Int32(extra_topk_length[token_idx])
        if extra_section_len < Int32(0):
            extra_section_len = Int32(0)
        if extra_section_len > Int32(self.extra_topk):
            extra_section_len = Int32(self.extra_topk)

        smem = cutlass_utils.SmemAllocator()
        st = smem.allocate(storage_cls())
        q_addr = shared_ptr_to_u32(st.q.data_ptr())
        kv_addr = shared_ptr_to_u32(st.kv.data_ptr())
        reduce_addr = shared_ptr_to_u32(st.reduce.data_ptr())
        reduce_max_addr = reduce_addr
        reduce_sum_addr = reduce_addr + Int32(8 * 16 * 4)
        sm_p_addr = shared_ptr_to_u32(st.sm_p.data_ptr())
        sm_p2_addr = shared_ptr_to_u32(st.sm_p2.data_ptr())
        token_idx_view = st.token_idx.get_tensor(cute.make_layout(2 * BI))
        is_io = warp_id >= Int32(MATH_THREADS // 32)
        kv_buf = Int32(BI * KV_SMEM_STRIDE)
        tok_buf_elems = Int32(BI)
        mbar_base = st.mbar.data_ptr()
        n_buf = 2

        topk_row = cute.make_tensor(
            swa_indices.iterator + token_idx.to(Int64) * Int64(self.swa_indices_stride_row),
            cute.make_layout(self.topk))
        extra_row = cute.make_tensor(
            extra_indices.iterator + token_idx.to(Int64) * Int64(self.extra_indices_stride_row),
            cute.make_layout(self.extra_topk))
        q_token = cute.make_tensor(
            q_all.iterator + token_idx.to(Int64) * Int64(self.q_stride_row),
            cute.make_layout((HEADS, D_NOPE), stride=(self.q_stride_head, self.q_stride_dim)))
        warp_first_cand = warp_id * Int32(8)

        if tid == Int32(0):
            for s in cutlass.range_constexpr(n_buf):
                cute.arch.mbarrier_init(mbar_base + s, Int32(1))
                cute.arch.mbarrier_init(mbar_base + n_buf + s, Int32(1))
        cute.arch.barrier()

        # Chunk bookkeeping (b12x _kernel_body, has_extra).
        cps = Int32(self.chunks_per_split)
        split_first_chunk = split_idx * cps
        split_last_chunk = split_first_chunk + cps
        main_valid_chunks = (section_len + Int32(CAND_WINDOW - 1)) // Int32(CAND_WINDOW)
        if main_valid_chunks < Int32(0):
            main_valid_chunks = Int32(0)
        max_main_chunks = Int32((self.topk + CAND_WINDOW - 1) // CAND_WINDOW)
        if main_valid_chunks > max_main_chunks:
            main_valid_chunks = max_main_chunks
        main_chunk_end = split_last_chunk
        if main_chunk_end > main_valid_chunks:
            main_chunk_end = main_valid_chunks
        main_chunk_count = main_chunk_end - split_first_chunk
        if main_chunk_count < Int32(0):
            main_chunk_count = Int32(0)
        extra_first_chunk = split_first_chunk
        extra_valid_chunks = (extra_section_len + Int32(CAND_WINDOW - 1)) // Int32(CAND_WINDOW)
        if extra_valid_chunks < Int32(0):
            extra_valid_chunks = Int32(0)
        max_extra_chunks = Int32((self.extra_topk + CAND_WINDOW - 1) // CAND_WINDOW)
        if extra_valid_chunks > max_extra_chunks:
            extra_valid_chunks = max_extra_chunks
        if extra_first_chunk < num_main_chunks:
            extra_first_chunk = num_main_chunks
        extra_chunk_end = split_last_chunk
        extra_valid_end = num_main_chunks + extra_valid_chunks
        if extra_chunk_end > extra_valid_end:
            extra_chunk_end = extra_valid_end
        extra_chunk_count = extra_chunk_end - extra_first_chunk
        if extra_chunk_count < Int32(0):
            extra_chunk_count = Int32(0)
        active_chunks = main_chunk_count + extra_chunk_count
        if active_chunks == Int32(0):
            if tid < Int32(HEADS):
                mid_lse[token_idx, tid, split_idx] = Float32(-Float32.inf)
            _exit_thread()

        if is_io:
            # IO producer: b12x io_issue_gather, unchanged arguments.
            io_lane = tid - Int32(MATH_THREADS)
            prod_phase = Int32(1)
            prod_idx = Int32(0)
            for lc in cutlass.range(active_chunks, unroll=1):
                active_idx = Int32(lc)
                ci = split_first_chunk + active_idx
                if active_idx >= main_chunk_count:
                    ci = extra_first_chunk + active_idx - main_chunk_count
                buf = Int32(lc) & Int32(1)
                cute.arch.mbarrier_wait(mbar_base + n_buf + prod_idx, phase=prod_phase)
                tok_buf_view = cute.make_tensor(token_idx_view.iterator + buf * tok_buf_elems,
                                                cute.make_layout(BI))
                io_kw = dict(bi=BI, kv_smem_stride=KV_SMEM_STRIDE, rope_smem_stride=0, scale_bytes_per_token=8,
                             bulk_tx_bytes=t.bulk_tx_bytes, scale_format=t.scale_format,
                             io_threads=BLOCK_THREADS - MATH_THREADS, split_mbar_arrival=False,
                             fp8_rope=False, packed_glm=False, packed_dsv4=False, overlap_footer_gather=False,
                             per_token_latent_scale=False, dsv41=True, dsv41_fp8=False)
                if ci >= num_main_chunks:
                    cis = ci - num_main_chunks
                    g_start = cis * Int32(CAND_WINDOW)
                    g_end = g_start + Int32(CAND_WINDOW)
                    if g_end > extra_section_len:
                        g_end = extra_section_len
                    io_issue_gather(extra_kv_cache_u8, extra_row, kv_addr + buf * kv_buf, kv_addr, kv_addr,
                                    tok_buf_view, mbar_base + buf, g_start, g_end, Int32(self.pbs_extra),
                                    stride_extra_kv_block, io_lane, dsv41_swa=False,
                                    mapping_page_table=mapping_page_table,
                                    mapping_table_width=mapping_table_width,
                                    mapping_table_offset=Int64(token_idx) * mapping_table_stride,
                                    mapping_num_pages=mapping_num_pages, mapping_page_size=self.pbs_extra,
                                    **io_kw)
                else:
                    g_start = ci * Int32(CAND_WINDOW)
                    g_end = g_start + Int32(CAND_WINDOW)
                    if g_end > section_len:
                        g_end = section_len
                    io_issue_gather(kv_cache_u8, topk_row, kv_addr + buf * kv_buf, kv_addr, kv_addr,
                                    tok_buf_view, mbar_base + buf, g_start, g_end, Int32(self.page_block_size),
                                    stride_kv_block, io_lane, **io_kw)
                prod_idx += Int32(1)
                if prod_idx == Int32(n_buf):
                    prod_idx = Int32(0)
                    prod_phase ^= Int32(1)
        else:
            s0_load_q24(q_token, q_addr, tid)
            accn = cute.make_rmem_tensor(N_V_CHUNKS * 4, Float32)
            accn2 = cute.make_rmem_tensor(N_V_CHUNKS * 4, Float32)
            accr = cute.make_rmem_tensor(4, Float32)
            accr2 = cute.make_rmem_tensor(4, Float32)
            gstate = cute.make_rmem_tensor(8, Float32)  # max0 max1 sum0 sum1 (tile 1), then tile 2
            for k in cutlass.range_constexpr(N_V_CHUNKS * 4):
                accn[k] = Float32(0.0)
                accn2[k] = Float32(0.0)
            for k in cutlass.range_constexpr(4):
                accr[k] = Float32(0.0)
                accr2[k] = Float32(0.0)
            for g in cutlass.range_constexpr(2):
                gstate[g * 4 + 0] = Float32(-1e30)
                gstate[g * 4 + 1] = Float32(-1e30)
                gstate[g * 4 + 2] = Float32(0.0)
                gstate[g * 4 + 3] = Float32(0.0)
            cons_phase = Int32(0)
            cons_idx = Int32(0)
            for lc in cutlass.range(active_chunks, unroll=1):
                active_idx = Int32(lc)
                ci = split_first_chunk + active_idx
                if active_idx >= main_chunk_count:
                    ci = extra_first_chunk + active_idx - main_chunk_count
                split_cand_start = ci * Int32(CAND_WINDOW)
                buf = Int32(lc) & Int32(1)
                kv_b = kv_addr + buf * kv_buf
                tok_buf_view = cute.make_tensor(token_idx_view.iterator + buf * tok_buf_elems,
                                                cute.make_layout(BI))
                acc_nope = [[accn[at * 4 + r] for r in range(4)] for at in range(N_V_CHUNKS)]
                acc_nope2 = [[accn2[at * 4 + r] for r in range(4)] for at in range(N_V_CHUNKS)]
                acc_rope = [accr[k] for k in range(4)]
                acc_rope2 = [accr2[k] for k in range(4)]
                global_max = [gstate[0], gstate[1]]
                global_sum = [gstate[2], gstate[3]]
                global_max2 = [gstate[4], gstate[5]]
                global_sum2 = [gstate[6], gstate[7]]

                cute.arch.mbarrier_wait(mbar_base + cons_idx, phase=cons_phase)
                cute.arch.barrier(barrier_id=3, number_of_threads=MATH_THREADS)

                qk = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                qk2 = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                qk, qk2 = s1_pair(qk, qk2, q_addr, kv_b, warp_first_cand, lane)
                sc_start = split_cand_start
                sec_len = section_len
                if ci >= num_main_chunks:
                    sc_start = (ci - num_main_chunks) * Int32(CAND_WINDOW)
                    sec_len = extra_section_len
                sc_end = sc_start + Int32(CAND_WINDOW)
                if sc_end > sec_len:
                    sc_end = sec_len
                qk = dm.s3_mask_and_scale(qk, tok_buf_view, warp_first_cand, sc_start, sc_end, sec_len,
                                          sm_scale_log2, lane)
                qk2 = dm.s3_mask_and_scale(qk2, tok_buf_view, warp_first_cand, sc_start, sc_end, sec_len,
                                           sm_scale_log2, lane)
                p = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                p, wr0, wr1 = dm.s4_online_softmax(
                    qk, p, acc_nope, acc_rope, global_max, global_sum, reduce_max_addr, reduce_sum_addr, False,
                    warp_id, lane, tid, n_v_chunks=N_V_CHUNKS, hpb=16, n_warps=8, valid_hpb=16,
                    num_threads=MATH_THREADS, barrier_id=3, n_acc_tiles=N_V_CHUNKS)
                # Every warp has read tile 1's per-warp maxima/sums before tile 2 overwrites the scratch.
                cute.arch.barrier(barrier_id=3, number_of_threads=MATH_THREADS)
                p2 = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                p2, vr0, vr1 = dm.s4_online_softmax(
                    qk2, p2, acc_nope2, acc_rope2, global_max2, global_sum2, reduce_max_addr, reduce_sum_addr,
                    False, warp_id, lane, tid, n_v_chunks=N_V_CHUNKS, hpb=16, n_warps=8, valid_hpb=8,
                    num_threads=MATH_THREADS, barrier_id=3, n_acc_tiles=N_V_CHUNKS)
                w_pre = [p[0] * wr0, p[1] * wr0, p[2] * wr1, p[3] * wr1]
                w_pre2 = [p2[0] * vr0, p2[1] * vr0, p2[2] * vr1, p2[3] * vr1]
                s5_pair(w_pre, w_pre2, sm_p_addr, sm_p2_addr, warp_id, lane)
                cute.arch.barrier(barrier_id=3, number_of_threads=MATH_THREADS)
                acc_nope, acc_nope2 = s6_pair(acc_nope, acc_nope2, sm_p_addr, sm_p2_addr, kv_b, warp_id, lane,
                                              latent_scale, ci < num_main_chunks)

                for at in cutlass.range_constexpr(N_V_CHUNKS):
                    for r in cutlass.range_constexpr(4):
                        accn[at * 4 + r] = acc_nope[at][r]
                        accn2[at * 4 + r] = acc_nope2[at][r]
                for k in cutlass.range_constexpr(4):
                    accr[k] = acc_rope[k]
                    accr2[k] = acc_rope2[k]
                gstate[0] = global_max[0]
                gstate[1] = global_max[1]
                gstate[2] = global_sum[0]
                gstate[3] = global_sum[1]
                gstate[4] = global_max2[0]
                gstate[5] = global_max2[1]
                gstate[6] = global_sum2[0]
                gstate[7] = global_sum2[1]

                cute.arch.barrier(barrier_id=3, number_of_threads=MATH_THREADS)
                if tid == Int32(0):
                    cute.arch.mbarrier_arrive(mbar_base + n_buf + cons_idx)
                cons_idx += Int32(1)
                if cons_idx == Int32(n_buf):
                    cons_idx = Int32(0)
                    cons_phase ^= Int32(1)

            # S7 per tile: b12x s7_epilogue into mid_out[token, head_base:, split] / mid_lse.
            for tile in cutlass.range_constexpr(2):
                head_base = Int32(16 * tile)
                src = accn if tile == 0 else accn2
                rope = accr if tile == 0 else accr2
                fin_acc = [[src[at * 4 + r] for r in range(4)] for at in range(N_V_CHUNKS)]
                fin_rope = [rope[k] for k in range(4)]
                fin_gmax = [gstate[tile * 4 + 0], gstate[tile * 4 + 1]]
                fin_gsum = [gstate[tile * 4 + 2], gstate[tile * 4 + 3]]
                out_o = cute.make_tensor(
                    mid_out.iterator + token_idx.to(Int64) * Int64(self.mid_out_stride_row)
                    + head_base.to(Int64) * Int64(self.mid_out_stride_head)
                    + split_idx.to(Int64) * Int64(self.mid_out_stride_split),
                    cute.make_layout((16, t.d_v), stride=(self.mid_out_stride_head, self.mid_out_stride_dim)))
                out_lse = cute.make_tensor(
                    mid_lse.iterator + token_idx.to(Int64) * Int64(self.mid_lse_stride_row)
                    + head_base.to(Int64) * Int64(self.mid_lse_stride_head)
                    + split_idx.to(Int64) * Int64(self.mid_lse_stride_split),
                    cute.make_layout((16,), stride=(self.mid_lse_stride_head,)))
                dm.s7_epilogue(fin_acc, fin_rope, fin_gmax, fin_gsum, out_o, out_lse, warp_id, lane,
                               n_v_chunks=N_V_CHUNKS, v_chunk=64, d_nope=D_NOPE, d_rope=0, n_warps=8,
                               valid_hpb=16 if tile == 0 else 8, nt_per_warp_xv=1, v_has_rope=False,
                               rope_tiles_per_warp=1)
