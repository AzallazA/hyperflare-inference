# SPDX-License-Identifier: Apache-2.0
"""TP3 fork configuration for the DeepSeek V4.1 Flash recipe: one typed, frozen object for every switch.

``TP3Config.from_environment()`` is the single place that reads the ``FLARE_DS41_*`` environment (the release
``export.sh`` sets it); everything else receives a ``TP3Config`` instance. The dataclass defaults are the bare
upstream behaviour (nothing parked, no BRRR, no vision staging), so a ``TP3Config()`` is what a test or a plain
``ignition export`` without the fork's environment gets.

Parsing keeps the historical semantics: an empty string counts as unset, integers default to 0 ("or 0"), layer sets
accept ``"20-33"`` ranges and ``"20,21"`` lists (mixed, comma separated), and flags are ``"1"`` unless noted.
"""

import hashlib
import json
import os
import re
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path

ENV_PREFIX = "FLARE_DS41_"
# b12x's own switch for programmatic dependent launch in its mHC kernels; TP3Config records it as ``mhc_pdl``.
MHC_PDL_ENV = "B12X_MHC_PDL"

# Switches whose code is not on the 2026-11-14 upstream port yet: TP3Config still parses and records them (the release
# environment keeps its meaning), but preparation refuses a non-default value until the named stage lands
# (``TP3Config.require_ported``), so an export never silently drops a saving or a kernel. Empty since S5.
UNPORTED = {}

# Routed experts per DeepSeek V4.1 MoE layer (hot_experts >= this keeps a parked layer fully resident).
ROUTED_EXPERTS = 384

# VRAM reclaim switches (2026-11-15 memory map) and their environment names. FLARE_DS41_VRAM_RECLAIM=1 is the
# umbrella: it turns every one on unless its own variable says otherwise ("1" on, any other set value off).
VRAM_RECLAIM = {
    "share_engram_token_map": "SHARE_ENGRAM_TOKEN_MAP",
    "trim_vocab_gather": "TRIM_VOCAB_GATHER",
    "skip_draft_vocab_gather": "SKIP_DRAFT_VOCAB_GATHER",
    "share_vocab_columns": "SHARE_VOCAB_COLUMNS",
}

# Optional tuning switches added after the 2026-11-15 default (151744): recorded in to_metadata only when set, so the
# record of a default export stays the reference's (the quick/combined chains require an identical record).
RECORDED_WHEN_SET = ("brrr_spin_ns", "brrr_two_shot_min", "brrr_ks_wob", "brrr_woa_small", "brrr_woa_uneven", "brrr_per_sm",
                     "prefill_graphs", "draft_vocab", "prefill_skip_suffix", "draft_head_plans", "markov_tile_rows",
                     "mla_decode_splits", "mla_h24", "index_row_split", "index_row_split_min_context",
                     "index_row_split_variant_graphs")

# On/off switches added for ds41-tp3-next (2026-11-17): recorded only when on, so a default export's record equals the
# earlier reference record.
RECORDED_WHEN_TRUE = ("mma_pad_once", "vision_arena_host", "markov_embed_host", "glue_pass2", "markov_stage_rows",
                      "rotary_once", "pin_small_qproj", "logprob_single_exchange", "no_generation_logprobs",
                      "index_row_split_check")
# Decode small-lever switches (perf/ds41-decode-small-levers, 2026-11-18): on/off, recorded only when on.
SMALL_LEVERS = ("rotary_once", "pin_small_qproj", "logprob_single_exchange", "no_generation_logprobs")
# E3c-a exact levers (perf/ds41-e3c-a, 2026-11-19; analysis/work/e3-comms-levers-20261119): on/off, recorded only when
# on, so a default export's record is unchanged. F = dcp_gather_pull, d = rotary_once_prefill.
E3C_A = ("rotary_once_prefill", "dcp_gather_pull")
RECORDED_WHEN_TRUE = RECORDED_WHEN_TRUE + E3C_A
# Steering inside BRRR (feat/ds41-steering-brrr, analysis/work/steering-sidecar-20261119/optionB): on/off, recorded
# only when on.
RECORDED_WHEN_TRUE = RECORDED_WHEN_TRUE + ("steering_plain_identity",)
# b12x's switch of the gather-pull overlay patch (tp3-patches/overlay/flashinfer-dcp-gather-pull-world3.patch)
GATHER_PULL_ENV = "B12X_PCIE_DCP_GATHER_PULL"

# tp3.pin_small_qproj: the plan triple pinned for every attention declaration of at most PIN_SMALL_ROWS rows (b12x
# offers programmatic launch / evict-first / FP32 split-K partials for MXFP8 GEMMs only up to 8 rows,
# gemm/_tuning.py ready_small_mxfp8). The values are the 11-15 cache's selections at 3 and 6 target rows (the 5-row
# declarations lost both by 0.02 / 0.01 us in their standalone races: REVIEW-FABLE.md section 1, 2026-11-18).
PIN_SMALL_ROWS = 8
PIN_SMALL_QPROJ = {
    "qkv_a": {"backend": "cutedsl", "tile_m": 32, "tile_n": 64, "tile_k": 128, "load_path": "tma", "swap_ab": False,
              "split_k_slices": 4, "large_m_unroll": True, "target_occupancy": None,
              "programmatic_serialization": True, "evict_first": False},
    "q_b": {"backend": "cutedsl", "tile_m": 16, "tile_n": 128, "tile_k": 128, "load_path": "tma", "swap_ab": False,
            "split_k_slices": 1, "large_m_unroll": False, "target_occupancy": None,
            "programmatic_serialization": True, "evict_first": True},
    "norm": {"programmatic_serialization": True},
}

# FLARE_DS41_PREFILL_SKIP_SUFFIX=1: the prefill capacities that get an encoder-only twin (the sizes that end the
# non-final chunks of long cold prompts at the 4096 chunk); a comma list names them explicitly.
SKIP_SUFFIX_DEFAULT = (3072, 3584, 4096)

# Engram table layouts at export (``engram_tables``).
ENGRAM_TABLES = ("auto", "shared", "partitioned")

# Switches whose code paths or names were removed (2026-11-08); the full construction (from_environment) rejects
# them so a stale setting fails loudly instead of being ignored.
REMOVED = {
    "FLARE_DS41_BRRR_PROLOGUE": "the BRRR attention prologue kernel measured no gain and was removed",
    "FLARE_DS41_QB_ATOMIC": "the q_b BF16 atomic split-K probe measured no gain and was removed",
    "DS41_FUSED_ATTN_PER_CHUNK": "renamed to FLARE_DS41_FUSED_ATTN_PER_CHUNK (read by TP3Config, not by the kernel)",
    # Default-off experiments not carried into the 2026-11-14 upstream port (measured level or worse; their code stays
    # in tp3-main's history).
    "FLARE_DS41_GLOBAL_COLD": "whole-model cold-expert parking (c8 -2..-7%) was not ported (2026-11-14)",
    "FLARE_DS41_JOINT_HEADROOM_MIB": "the joint-tuning compile headroom override was not ported (2026-11-14)",
    "FLARE_DS41_BRRR_SPARSE": "BRRR's one-launch sparse decode (measured level) was not ported (2026-11-14)",
    "FLARE_DS41_BRRR_SPLIT_TAIL": "BRRR item 6 (split tail) was not adopted and not ported (2026-11-14)",
    "FLARE_DS41_BRRR_FAST_ROUTE": "BRRR item 6 (fast route) was not adopted and not ported (2026-11-14)",
    "FLARE_DS41_BRRR_COUNTER_PARITY": "BRRR item 6 (counter parity) was not adopted and not ported (2026-11-14)",
    "FLARE_DS41_FUSED_QKV_NORM": "the fused qkv_a split-K MLA norm (measured level) was not ported (2026-11-14)",
}


def _atoi(value: str) -> int:
    """C ``atoi``: the leading integer of the string, 0 when there is none (the kernel's historical parser)."""
    match = re.match(r"\s*[+-]?\d+", value)
    return int(match.group()) if match else 0


def _fingerprint(parts) -> str:
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def describe_counts_dir(path: Path | None) -> str | None:
    """``<basename>#<sha256[:12] of the sorted route-count file names and sizes>``; never the full path."""
    if path is None:
        return None
    files = sorted(path.glob("expert-routes-rank0-layer*.json")) if path.is_dir() else []
    digest = _fingerprint(f"{f.name}:{f.stat().st_size}" for f in files) if files else "missing"
    return f"{path.name}#{digest}"


def describe_checkpoint(path: Path | None) -> str | None:
    """``<basename>#<sha256[:12] of the view's config.json>``; never the full path."""
    if path is None:
        return None
    config = path / "config.json"
    digest = hashlib.sha256(config.read_bytes()).hexdigest()[:12] if config.is_file() else "missing"
    return f"{path.name}#{digest}"


def describe_file(path: Path | None) -> str | None:
    """``<basename>#<sha256[:12] of the file>``; never the full path."""
    if path is None:
        return None
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12] if path.is_file() else "missing"
    return f"{path.name}#{digest}"


def parse_layers(spec: str) -> frozenset:
    """``"20-22"`` / ``"20,21"`` / ``"0-3,7"`` -> frozenset of layer indices; empty -> empty set."""
    layers = set()
    for part in filter(None, (p.strip() for p in spec.split(","))):
        a, _, b = part.partition("-")
        layers.update(range(int(a), int(b or a) + 1))
    return frozenset(layers)


def format_layers(layers) -> str:
    """Compact ``"0-3,7"`` form of a layer set (inverse of ``parse_layers``)."""
    ordered, parts = sorted(layers), []
    while ordered:
        first = last = ordered.pop(0)
        while ordered and ordered[0] == last + 1:
            last = ordered.pop(0)
        parts.append(str(first) if first == last else f"{first}-{last}")
    return ",".join(parts)


def parse_capacities(text: str, name: str = "PREFILL_GRAPHS") -> tuple[int, ...] | None:
    """``"128,256,1024"`` -> (128, 256, 1024); "" -> None. Order and range are checked by the config."""
    text = text.strip()
    if not text:
        return None
    try:
        return tuple(int(part) for part in text.split(","))
    except ValueError:
        raise ValueError(f"FLARE_DS41_{name}={text!r}: expected comma-separated token capacities") from None


def parse_draft_head_plans(text: str) -> tuple | None:
    """FLARE_DS41_DRAFT_HEAD_PLANS: "" -> None; else a JSON file (or inline JSON object) {"<rows>": {config}, ...}
    -> ((rows, canonical config json), ...) sorted by rows. The record keeps the configs, never the path."""
    text = text.strip()
    if not text:
        return None
    try:
        raw = json.loads(text) if text.startswith("{") else json.loads(Path(text).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"FLARE_DS41_DRAFT_HEAD_PLANS: {error}") from None
    if not isinstance(raw, dict) or not raw or any(not isinstance(v, dict) for v in raw.values()):
        raise ValueError("FLARE_DS41_DRAFT_HEAD_PLANS: expected {\"<rows>\": {config}, ...}")
    try:
        return tuple(sorted((int(rows), json.dumps(config, sort_keys=True)) for rows, config in raw.items()))
    except ValueError:
        raise ValueError("FLARE_DS41_DRAFT_HEAD_PLANS: row keys must be integers") from None


def parse_index_row_split(text: str) -> tuple[int, ...] | None:
    """FLARE_DS41_INDEX_ROW_SPLIT: "" / "0" -> None (off), "1" -> ``index_row_split.ROW_SPLIT_DEFAULT``, else a
    capacity list."""
    from .index_row_split import ROW_SPLIT_DEFAULT

    text = text.strip()
    if text in ("", "0"):
        return None
    return ROW_SPLIT_DEFAULT if text == "1" else parse_capacities(text, "INDEX_ROW_SPLIT")


def parse_skip_suffix(text: str) -> tuple[int, ...] | None:
    """FLARE_DS41_PREFILL_SKIP_SUFFIX: "" / "0" -> None (off), "1" -> ``SKIP_SUFFIX_DEFAULT``, else a capacity list."""
    text = text.strip()
    if text in ("", "0"):
        return None
    return SKIP_SUFFIX_DEFAULT if text == "1" else parse_capacities(text, "PREFILL_SKIP_SUFFIX")


@dataclass(frozen=True)
class TP3Config:
    # --- expert placement (weights.py, recipe.py) ---
    host_expert_layers: frozenset = field(default_factory=frozenset)
    """MoE layers whose routed-expert banks live in device-visible host memory (``"20-33"``)."""
    hot_experts: int = 0
    """Hottest experts per parked layer kept in VRAM, by the route counts in ``expert_counts_dir``; 0 = park all."""
    merged_bank: bool = False
    """Parked layers use one mixed VRAM/host bank (hot rows first). Required when ``hot_experts`` keeps experts in VRAM:
    the separate hot + cold bank layout was not carried into the 2026-11-14 port."""
    expert_counts_dir: Path | None = None
    """Directory of ``expert-routes-rank0-layer{L}.json`` route counts (vLLM route-monitor format)."""
    embed_host: bool = False
    """Token embedding read from device-visible host memory (only its tokens' rows are gathered per step)."""
    draft_vocab: Path | None = None
    """FR-Spec style draft vocabulary (FLARE_DS41_DRAFT_VOCAB=<json from tp3-tools/ds41_draft_vocab.py>): the NVFP4 DSpark
    draft head keeps only these tokens' rows (each rank its tokens in its vocabulary shard, padded to one equal multiple
    of 128 columns on every rank); the other columns of the draft logits are a large negative constant, so proposals
    never pick them (q = 0) while verification keeps the full BF16 head: outputs are exact, only acceptance on rare
    tokens can drop. Needs --draft-head-quantization nvfp4 and vocabulary-parallel sampling. None = full head (base).
    Export-time only (IDEAS 2026-11-17 item 6)."""
    markov_embed_host: bool = False
    """DSpark's Markov-head embedding (129,280 x 256 BF16, 63 MiB, replicated) read from device-visible host memory:
    the drafter's embedding lookup and the fused Markov transition bias read only the pending tokens' rows (<= one per
    request and draft step) through the mapping. Same values. False = VRAM (base). Export-time only (2026-11-15 memory
    map item 6)."""
    markov_stage_rows: bool = False
    """With ``markov_embed_host``: each draft step first copies every request's predecessor row of the parked Markov
    embedding into a [rows, 256] BF16 VRAM buffer (one small launch per step), and the fused transition bias reads that
    copy instead of the host row once per vocabulary tile (169 tiles per row on a TP3 shard; the cause of the 180256
    decode regression, ab-regression-180256/DIAG.md). Same values (CPU proof: tests/components/
    test_markov_stage_rows_interpret.py). Needs vocabulary-parallel sampling. False = tiles read the table (base).
    Export-time only (2026-11-18)."""
    mla_decode_splits: str | None = None
    """Record of ``FLARE_DS41_MLA_DECODE_SPLITS`` (canonical ``ROWS:SPLITS,...``; components/mla_decode_splits.py): DSV41
    decode split count per prepared row count for target attention. Changes output bits (BF16 split partials), so it
    is recorded when set; preparation refuses a record that disagrees with the hook installed in this process.
    None = b12x's tuned split geometry (base). Export-time only (decode lever 4a, 2026-11-18)."""
    mla_h24: str | None = None
    """Record of ``FLARE_DS41_MLA_H24`` (canonical ``ROWS:0|1,...``; components/mla_h24_decode.py): prepared row counts
    whose DSV41 BF16 target decode runs all 24 heads in one launch (one CTA per row and split gathers each KV window
    once for both head tiles) instead of b12x's 16-head grid + 8-head remainder grid. Same values as 16 + 8 at the
    same split count by construction (GPU check: tests/components/test_mla_h24_decode_cuda.py); recorded when set
    because it changes the compiled programs, and preparation refuses a record that disagrees with the installed hook.
    None = 16 + 8 grids (base). Export-time only (decode lever 4b, 2026-11-18)."""
    markov_tile_rows: int | None = None
    """Markov proposal score tiles cover this many request rows per program and read each head tile once for all of
    them (vocabulary_sampling._score_tiles_rows) instead of once per row (173 us per 5 draft steps at c16, decode trace
    2026-11-18 lever 5). Bit-identical (CPU proof: tests/components/test_markov_tile_rows_interpret.py). Needs
    vocabulary-parallel sampling. None = one program per (tile, row) (base). Export-time only (2026-11-18)."""
    embed_rows: bool = False
    """With ``embed_host``: decode, verification and draft graphs gather the parked table's rows with one vectorised
    Triton launch (all of a row's loads before its stores) and take DSpark's noise-token rows from a 10 KB device copy
    (host_embedding.py; same values). False = b12x's embedding kernel everywhere (base); prefill graphs always keep it."""
    # --- vision tower (weights.py, bootstrap.py, components/vision_staging.py) ---
    vision_host: bool = False
    """Vision tower parked in device-visible host memory; images ~3 s slower each unless staged. Implied by
    ``vision_stage`` (staging reads from the parked copy), as the pre-config recipe did."""
    vision_stage: bool = False
    """Stage the host-parked tower's weights into VRAM per block inside the vision graphs (tp3-main: image 0.11 s vs
    1.70 s; since the 2026-11-14 port the copies are forward-local and on the main stream, not overlapped).
    Requesting it parks the tower (``vision_host`` is forced on)."""
    vision_stage_copy: str = "memcpy"
    """Staging copy: ``memcpy`` (cudaMemcpy) or ``kernel`` (elementwise kernel over the mapped host range)."""
    vision_arena_host: bool = False
    """The image feature arena (``VisionState.arena``: image slots x 4,096 tokens x 5,120 BF16, 160 MiB per rank at 4
    slots) lives in device-visible host memory (``CoreStorage.host_allocate``) instead of standing VRAM. The vision
    graphs write an image's features there and the prefill graphs read them back, so image prompts move ~40 MB per
    image over PCIe each way; text-only work never touches it. Zero-filled in the package as before (disposable
    storage). False = VRAM (base). Export-time only (2026-11-15 memory map item 3)."""
    # --- collectives and planning (bootstrap.py, recipe.py, planning.py) ---
    ar_rows: int = 0
    """Rows the shared TP3 all-reduce runtime is sized for (chunked prefill reductions); 0 = a whole prefill chunk."""
    ar_part_rows: int = 0
    """Decoder attention/MLP all-reduces of more rows than this run in parts of exactly N rows plus a remainder
    (4096 at 768: 5 x 768 + 256) instead of halving to what the runtime accepts (4096 at 1024: 4 x 1024); 0 = halving
    (base). Engram reductions follow ``engram_part_rows``. Export-time only (prepared graphs)."""
    engram_part_rows: int = 0
    """The same part cut for the Engram projection all-reduce (its (rows x 5, 5120) view: a 1024-token chunk is 5,120
    rows, at 768: 6 x 768 + 512 instead of halving to 8 x 640); 0 = halving (base). Separate from ``ar_part_rows`` so
    the two can be measured apart. Export-time only (prepared graphs)."""
    mhc_pdl: bool = False
    """Record of ``B12X_MHC_PDL`` (b12x reads it once, when its mHC kernel module is imported): programmatic dependent
    launch in the mHC norm kernels. The value is part of every mHC tuning query and compile key, so turning it on
    compiles new mHC programs and re-races the multi-candidate (prefill) mHC requests. Read from ``B12X_MHC_PDL`` with
    b12x's rule (unset or "0" = off); preparation refuses a record that disagrees with the imported b12x module.
    False = off (base)."""
    tp3_graph_gib: float | None = None
    """Override of the per-rank captured-graph estimate for world size 3 (GiB); None = the 8 GiB default."""
    tp3_vocab_parallel: bool = False
    prefill_graphs: tuple[int, ...] | None = None
    """FLARE_DS41_PREFILL_GRAPHS="128,256,...": the prefill graph capacities to capture, a subset of the default ladder
    (``capacity.prefill_capacities(chunk)``: 128, 256, then steps of 512 to the chunk) that keeps 128 and the chunk;
    the server reads the catalog from the package's graphs, so a prompt chunk pads to the next kept size. Each dropped
    size saves its graph's device memory (~10-14 MiB per GPU per prefill graph in the 170303 export's capture deltas;
    the graph pool is sized by the largest graph and does not shrink). None = the full ladder (base). Export-time only."""
    """Admit vocabulary-parallel sampling at TP3 (b12x gathers enabled for world 3)."""
    prefill_skip_suffix: tuple[int, ...] | None = None
    """FLARE_DS41_PREFILL_SKIP_SUFFIX ("1" = 3072,3584,4096, or a list of prefill capacities): capture an
    encoder-only twin (graph kind ``encoder_prefill``) of each listed prefill graph. The twin runs layers 0-19 and
    layer 20's mHC plus global KV (compressor, index keys, latent pages) on every row, commits the Engram history,
    and records consumption; it has no decoder suffix (layers 20-39 on the 128-row replay), head, sampling, DSpark
    context or ``last_hidden`` write. Helios runs it for non-final chunks whose stale suffix state the next chunk
    replaces (decoder_prefill.py, scheduler ``encoder_only_chunk``). Every listed size must be on the prefill ladder
    and above 128. None = no twins (base). Export-time only (prepared graphs); serving can still turn the twins off
    with ``flare run --scheduler-encoder-prefill false``."""
    index_row_split: tuple[int, ...] | None = None
    """FLARE_DS41_INDEX_ROW_SPLIT ("1" = 3072,3584,4096, or a list of prefill capacities): in these prefill graphs (and
    their encoder-only twins) each rank runs the indexer selectors of a contiguous uneven share of the chunk's 128-row
    blocks (4096: 11/11/10) on the index layers before the candidate source (2, 8, 14), and the int32 winners are
    all-gathered over the FP32 vocabulary channel (one or two more channel rows), so every rank holds the whole
    ``selected`` (index_row_split.py). Equal to the owner rank's replicated selection bit for bit. Not byte-identical
    to the replicated indexer: the b12x selector fills the top-512 cut's tie class in arrival order, so today's ranks
    (and runs) already pick different tied members (G0, 2026-11-18); with the split every rank uses the owner's pick.
    Prompt-logprob graphs and decode/verify graphs keep the replicated indexer. Every listed size must be on the
    prefill ladder. None = replicated (base). Export-time only (prepared graphs). Experimental
    (perf/ds41-indexer-row-split, not GPU-proven yet)."""
    index_row_split_min_context: int | None = None
    """FLARE_DS41_INDEX_ROW_SPLIT_MIN_CTX=<tokens> (needs index_row_split): long-context-only split. The listed prefill
    capacities (and their encoder-only twins) keep the REPLICATED indexer in their ordinary graphs, and each gets one
    more graph, ``<name>-split-ctx<tokens>`` (same kind, capacity and uploads), with the row split. The server runs
    the split graph for a one-lane prefill chunk whose start position (the context already indexed) is at least
    <tokens>, the replicated one below (helios ``catalog::context_variant``). The split's exchange costs a near
    constant ~2.3 ms per index layer and 4096-row chunk (C2: 17-row channel gather), while the indexer work it saves
    grows with the context (G2 A/B, 2026-11-19: about +3-5 ms per 4096-row chunk constant, -0.157 ms per 1K tokens
    of context; break-even near 21-29K tokens). None = every listed graph split (G2 behaviour). Export-time only."""
    index_row_split_variant_graphs: tuple[str, ...] | None = None
    """FLARE_DS41_INDEX_ROW_SPLIT_VARIANTS=<kinds> (needs index_row_split_min_context): which graphs get the
    long-context variant, a comma list of ``prefill`` (the full prefill graph: a prompt's final chunk of 3,585-4,096
    rows) and ``encoder_prefill`` (the encoder-only twin: every non-final chunk). None = both. ``encoder_prefill``
    alone keeps nearly all of the gain (the full variant runs at most once per prompt) for one variant graph's
    capture memory instead of two. Export-time only."""
    index_row_split_check: bool = False
    """FLARE_DS41_INDEX_ROW_SPLIT_CHECK=1, debug-serve only (never export with it): the split's exchanges also carry a
    checksum per window row, and a counter op counts per layer the rows whose checksum differs after the gather
    (transport, must be 0), the assembled rows whose valid prefix is not distinct values in [0, length) (must be 0) and
    the rows whose tail is not all -1 (informational); dumped per device on request (index_row_split.check_paths).
    Needs index_row_split."""
    # --- Engram tables (engram.py Tables) ---
    engram_tables: str = "auto"
    """Engram table layout in the package: ``shared`` = every rank reads the whole tables (the layout of every TP3
    package so far; the fast-restart private-table reader was validated on it), ``partitioned`` = upstream #220's
    per-rank head partitions, ``auto`` = shared at world 3, partitioned at other multi-rank worlds (upstream)."""
    # --- graph scratch (decoder.py Scratch, engram.py; bootstrap.py sizes the pools) ---
    engram_chunk_rows: int = 0
    """Engram projection, its all-reduce and the mix run in row chunks of N inside wider graphs (prefill), so the
    Engram layer's planned graph pool and its declaration operands hold N rows of ``local``/``partial``/``projected``
    instead of a whole prefill chunk; 0 = one pass (base)."""
    share_engram_residual: bool = False
    """Declare ``engram_residual`` on ``ffn_hc.residual``'s storage (preparation memory only). In the functional
    forwards the Engram output is a value Inductor already places on dead storage, so serving graphs are the same
    either way (tp3-main's scratch alias is subsumed); False = separate declaration (base)."""
    # --- VRAM reclaim (2026-11-15 memory map items 2, 5, 8; FLARE_DS41_VRAM_RECLAIM=1 turns all four on) ---
    share_engram_token_map: bool = False
    """Every Engram hash plan reads the rank's one persistent token map (``Tables.device_token_map``, which the
    history lookups already read) instead of materializing its own 1 MiB copy (two per graph: ~152 MiB/rank).
    Same values, read-only (b12x's mutation contract); False = one copy per plan (base). Export-time only."""
    trim_vocab_gather: bool = False
    """With vocabulary-parallel sampling, size the FP32 vocabulary all-gather channel for the widest graph that still
    gathers through it (prefill heads 1 row, resume/decode heads ``max(batch_capacities)`` rows) instead of every
    verification sample (48 rows: ~41 MiB/rank). A graph needing more rows fails the export (PreparedAllGather's row
    check). No effect without vocabulary-parallel sampling. False = sized for the verification samples (base)."""
    skip_draft_vocab_gather: bool = False
    """With vocabulary-parallel sampling, do not create the BF16 draft-vocabulary all-gather channel: its only user,
    the DSpark proposal head's gather, is local-only then (``gather=False``; ~25 MiB/rank). Without
    vocabulary-parallel sampling the channel is created as before. False = always created (base)."""
    share_vocab_columns: bool = False
    """Vocabulary-parallel speculative rounds read one persistent copy of this rank's global vocabulary ids
    (``copy_columns``, 345 KB) instead of one per round (72 per rank: ~23 MiB). Same values, compared only.
    False = one copy per round (base)."""
    # --- decode glue, second pass (export-time, bit-identical; round2/glue2 notes) ---
    glue_pass2: bool = False
    """Decode/verify/draft graphs drop launches no reader needs: the zero fill of the verification window staging
    and of the draft window cache (every byte a reader addresses is written first), the draft metadata's state rows
    realised once instead of per draft layer, the first layer's mHC pre-mix one-hot from one kernel, DSpark's
    target-layer feedback concatenated once instead of copied slice by slice, and the proposal admission check
    without in-place mask updates (functional ``&``, fused). Same values. False = today's graphs (base). The
    MXFP8 MMA-scale padding fills are ``mma_pad_once`` (perf/ds41-launch-merge), not this switch."""
    # --- decode small levers (perf/ds41-decode-small-levers, 2026-11-18; export-time) ---
    rotary_once: bool = False
    """Decode, verification and DSpark draft forwards compute the FP32 RoPE (cos, sin) table once per forward and
    frequency set (``AttentionIndices``, rotary ``_angles``) and every layer's query, KV-window and index-query
    rotation reads it (``_rotate`` SHARED_ANGLES) instead of evaluating sinf/cosf per launch (slow range reduction past
    ~105K positions). Bit-identical: same expression on the same clamped positions (CPU proof
    tests/components/test_rotary_shared_angles_interpret.py). The ratio-2 compressor rotations (positions can be -2)
    and BRRR's in-kernel inverse rotation keep their own trigonometry; prefill keeps per-launch angles. False = base."""
    pin_small_qproj: bool = False
    """Attention declarations of at most ``PIN_SMALL_ROWS`` rows (decode, verification, draft) take the pinned plan
    triple ``PIN_SMALL_QPROJ`` (qkv_a split-K4 32x64 + programmatic launch, q/KV norm programmatic launch, q_b 16x128
    programmatic + evict-first) as explicit overrides instead of the tuning cache's selections (the 5-row
    declarations' selections were coin flips, 0.2%). Numerics: the 5-row rounds get the rounding the 3- and 6-row
    rounds already have. False = cache selections (base)."""
    logprob_single_exchange: bool = False
    """The verification commit's vocabulary-parallel generation-logprob reduction exchanges one statistics row per
    sample instead of two (vocabulary_logprobs._finish_single): candidates are ranked by raw logit before the
    exchange and normalised after it. Chosen/requested logprobs and every reported value are identical; ids among
    exactly tied values can differ only when >= 13 distinct raw logits of one rank collapse to one FP32 value at the
    top-20 boundary (CPU proof tests/components/test_logprob_single_exchange_interpret.py). False = two (base)."""
    no_generation_logprobs: bool = False
    """The package carries no generation-logprob storage, request flags or reductions (no logprob all-gathers, no
    logprob exchange channel): Helios answers any generation-logprob request with HTTP 400. Prompt logprobs are not
    affected. The handbook checks that need logprobs (verify_fix, consistency, engram-eval) cannot run on such a
    package. Owner decision (B2). False = generation logprobs kept (base)."""
    rotary_once_prefill: bool = False
    """``rotary_once`` for prefill graphs (and their encoder-only twins and decoder-suffix replays): every layer's
    query, KV-window and index-query rotation, and the inverse rotation of the attention output, read the forward's
    shared (cos, sin) table instead of evaluating sinf/cosf per (row, head) program (the slow range reduction above
    ~105.6K positions; seq358 M4: 3.45 ms saved per 21-layer 4,096-row chunk at >= 130K). Bit-identical (same
    expression on the same clamped positions; tests/components/test_rotary_shared_angles_interpret.py, GPU M4). The
    ratio-2 compressor and global-KV key/latent rotations keep per-launch angles (their positions can be -2).
    False = per-launch angles in prefill (base). Export-time only (E3 lever d)."""
    steering_plain_identity: bool = False
    """With a steering sidecar: tune under the checkpoint's plain preparation identity instead of upstream PR #404's
    sha256(identity || sidecar sha), so the export reuses (and races against) the sidecar-free package's tuning
    records and selects the same plans; the scale-0 output of the sidecar package can then be compared bit for bit
    with the sidecar-free one. The served config still records the sidecar sha. Needs a sidecar
    (PreparationConfig refuses it otherwise). False = upstream's sidecar identity (base)."""
    dcp_gather_pull: bool = False
    """Record of ``B12X_PCIE_DCP_GATHER_PULL`` != 0 (overlay patch flashinfer-dcp-gather-pull-world3, GLM's since
    2026-11-09): world-3 DCP column gathers (logprob/sampling/candidate/vocabulary gathers, the indexer row split's
    winner exchange) pull peers' staging (~40 GB/s) instead of pushing (the push kernel's store pattern makes it slow:
    seq358 M2 wide 17-row gather 2,326 us push vs 148 us pull); payloads under 4 KiB per rank stay push. Copies only:
    bit-identical (seq358 M2: eager check + 200-replay in-graph soak, 0 mismatches). Moves 24 b12x module fingerprints
    (everything importing _dcp_a2a_cute): the head-gather programs re-key and the persisted program discovery of every
    PCIe collective re-runs. False = push (base: export.sh forces B12X_PCIE_DCP_GATHER_PULL=0). Export-time only
    (E3 lever F)."""
    draft_head_plans: tuple | None = None
    """Prepared, not measured: explicit plan overrides for the DSpark proposal head projection by row count,
    ((rows, config-json), ...) from FLARE_DS41_DRAFT_HEAD_PLANS=<json file {"40": {...gemm.blockscaled_precision
    config...}}>; rows without an entry keep the tuning cache's selection. None = cache selections (base)."""
    # --- packed verification graph shapes for exports above 8 slots (bootstrap.PreparationConfig) ---
    packed_keep_batches: int = 0
    """Batches up to N keep exactly the packed verification shapes an export with ``--max-concurrency N`` builds (same
    graph budget), so a 16-slot package verifies 1-8 requests with today's 8-slot shapes; the wider batches get their
    own graphs (``packed_wide_graphs``) on top. 0 = upstream enumeration (one budget spread over every batch). A power
    of two; no effect when the export's request capacity is not above it."""
    packed_wide_graphs: int = 0
    """With ``packed_keep_batches``: packed graphs for the batches above it, split by the same geometric rule; 0 = only
    each wide batch's two endpoint shapes."""
    packed_wide_tokens: int = 0
    """With ``packed_keep_batches``: largest packed token capacity of the wide batches (e.g. 48: shapes only where a
    served verification budget <= 48 can select them); 0 = the export's speculative token capacity (batch x width)."""
    # --- side streams in decode/verify graphs (components/side_streams.py; export-time, bit-identical) ---
    index_side_stream: bool = False
    """Index-source layers run the indexer query chain (index_q projection, RoPE, index-weights GEMV) on a side
    stream from right after the q/KV norm, joined before the selectors; ratio-2 KV sources also run the compressor's
    gates GEMV there, joined before the compression kernel (attention.IndexSideChain). Decode/verify graphs only.
    False = serial on the main stream (base)."""
    engram_side_layers: frozenset = field(default_factory=frozenset)
    """Engram layers (``"14"``) whose readiness wait, lookup, projection and all-reduce run on a side stream during
    the previous layer's attention (started after its predecessor's tail, joined before its own tail), leaving only
    the mix in place (engram.PreparedEngram.start_side). Uses the shared all-reduce runtime: no extra VRAM.
    Decode/verify graphs only. Layer 1 is allowed but its rows are staged by the host only ~0.2 ms before layer 1
    today (hashes are published at the start of the target forward), so starting it in layer 0 can stall.
    Empty = in place (base)."""
    context_side_stream: bool = False
    """Verification rounds run DSpark's context projections (main projection, norm, context KV projection) on a
    side stream from the end of the target forward, beside verification and commit; the per-layer norms and draft
    ring writes stay after the accepted counts (components/dspark.PreparedContext). Draft acceptance only; the
    rings get the same bytes. False = after verification (base)."""
    # --- launch merging in decode/verify/draft graphs (attention.py, decoder.py, dspark.py; export-time) ---
    mma_pad_once: bool = False
    """Decode, verification and DSpark draft forwards zero the MXFP8 MMA-scale tiles of the qkv_a and q_b activations
    once per forward (one fill per tile shape) and every layer reuses them, instead of one zero fill per layer and tile
    (``mxfp8_dense_bases``: 2 x 43 Inductor fills of 20 / 5 KB per c1 round, ~0.6 us each on the critical path). Each
    layer's quantizer (b12x MXFP8 rows quantizer, mla_norm_mxfp8) still writes every real row's scales before its GEMM
    reads them; padding rows are written by nothing and stay zero. Same bytes. False = per-layer fills (base)."""
    # --- BRRR fused layer kernel (brrr.py, decoder.py) ---
    brrr_layers: frozenset = field(default_factory=frozenset)
    """Layers whose decode/verify tail (o-proj .. MLP all-reduce) runs in BRRR's fused kernel (``"0-39"``)."""
    brrr_parked_max_rows: int = 0
    """Parked layers use BRRR only in graphs of at most N rows (gate; wide verification keeps Flare's MoE); 0 = ungated."""
    brrr_ckpt: Path | None = None
    """Checkpoint view BRRR reads its staged dense copies and block scales from (required with ``brrr_layers``)."""
    brrr_native_scales: bool = False
    """Load BRRR layers' experts without CSF scale compression (the kernel reads native E8M0 scales)."""
    brrr_dense_native: bool = True
    """BRRR reads Flare's own row-major E4M3 dense values instead of staged checkpoint copies (~35-40 MB/layer).
    Environment: unset = on; any set value other than "1" (including "") = staged copies, as the base recipe did."""
    brrr_csf_check: bool = False
    """Debug: verify Flare's dense bytes and the in-kernel CSF expansion against the checkpoint at register time."""
    brrr_log: bool = False
    """Debug: print BRRR layer registration and fused-norm selection lines."""
    brrr_break: str | None = None
    """Negative control for the layer test: ``"rope"`` reverses the RoPE frequencies the tail inverts (must fail)."""
    fused_attn_per_chunk: bool = True
    """Kernel: one-shot attention exchange posts one flag per chunk (False = legacy single flag; A/B toggle).
    Environment: unset = on; a set value is read like C ``atoi`` ("" and non-numbers = off), as the kernel did."""
    brrr_spin_trap: bool = False
    """Test builds only: a BRRR protocol wait that has not finished after ~2^25 polls prints and traps instead of hanging
    the GPUs (compile-time define BRRR_SPIN_TRAP; never for timing or export)."""
    brrr_spin_ns: int | None = None
    """Kernel tuning: ns slept between polls of a BRRR wait (counter, route table, peer flag); None = the kernel default
    (200). A non-default value is its own extension build (define BRRR_SPIN_NS) and must be exported into the package.
    Timing only: the arithmetic is unchanged. Environment FLARE_DS41_BRRR_SPIN_NS (unset/empty = default)."""
    brrr_two_shot_min: int | None = None
    """Kernel tuning: rows from which the BRRR attention/FFN exchanges reduce two-shot instead of one-shot; None = the
    kernel default (17); 1 = always two-shot, 129 = never. Same sums in the same rank order either way. Own extension
    build (define BRRR_TWO_SHOT_MIN), exported into the package. Environment FLARE_DS41_BRRR_TWO_SHOT_MIN."""
    brrr_ks_wob: int | None = None
    """Kernel job layout: k32 slices of z per BRRR wo_b job; None = the kernel default (16 = 6 parts, 480 jobs at <= 16
    rows); 24 = 4 parts, 320 jobs (one round on 376 CTAs, as TP4 had). Regroups float sums only. Own extension build
    (define BRRR_KS_WOB). Environment FLARE_DS41_BRRR_KS_WOB."""
    brrr_woa_small: int | None = None
    """Kernel job layout: rows up to which BRRR wo_a jobs are 4 stages (384 jobs); None = the kernel default (16);
    0 = always 2 stages. Define BRRR_WOA_SMALL. Environment FLARE_DS41_BRRR_WOA_SMALL."""
    brrr_woa_uneven: int | None = None
    """Kernel job layout: 1 = at <= 8 rows split each wo_a row block into 7 uneven k parts (336 jobs, one round);
    None/0 = the kernel default. Define BRRR_WOA_UNEVEN. Environment FLARE_DS41_BRRR_WOA_UNEVEN."""
    brrr_per_sm: int | None = None
    """Diagnostic only: BRRR CTAs per SM (None = 2; 1 halves the weight reads in flight). Define BRRR_PER_SM.
    Environment FLARE_DS41_BRRR_PER_SM."""
    # --- debug and test tooling (config.py, decoder.py, probe.py) ---
    mini: bool = False
    """Truncated real-weight checkpoint view (make_mini.py): accept 6..40 layers, no compacted prefill feedback."""
    no_engram: bool = False
    """Debug A/B (flare#193): keep the Engram handshake but discard its contribution to the residual."""
    probe: bool = False
    """Debug-serve only: per-layer residual probes dumped to /tmp/flare_probe.pt (never export with it)."""

    def __post_init__(self):
        object.__setattr__(self, "host_expert_layers", frozenset(self.host_expert_layers))
        object.__setattr__(self, "brrr_layers", frozenset(self.brrr_layers))
        object.__setattr__(self, "engram_side_layers", frozenset(self.engram_side_layers))
        for name in ("expert_counts_dir", "brrr_ckpt"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value))
        for name in ("hot_experts", "ar_rows", "ar_part_rows", "engram_part_rows", "brrr_parked_max_rows", "engram_chunk_rows",
                     "packed_keep_batches", "packed_wide_graphs", "packed_wide_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"TP3 config: {name} must be a non-negative integer, not {value!r}")
        for name in ("merged_bank", "embed_host", "embed_rows", "vision_host", "vision_stage", "tp3_vocab_parallel",
                     "brrr_native_scales", "brrr_dense_native", "brrr_csf_check", "brrr_log", "fused_attn_per_chunk",
                     "brrr_spin_trap", "mini", "no_engram", "probe", "share_engram_residual", "mhc_pdl",
                     "index_side_stream", "context_side_stream", "glue_pass2", *VRAM_RECLAIM):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"TP3 config: {name} must be a boolean")
        if type(self.mma_pad_once) is not bool:  # own line: keeps merges with other branches' switch lists clean
            raise TypeError("TP3 config: mma_pad_once must be a boolean")
        if self.draft_vocab is not None and not isinstance(self.draft_vocab, Path):
            raise TypeError("TP3 config: draft_vocab must be a Path or None")
        for name in ("vision_arena_host", "markov_embed_host", "markov_stage_rows"):  # own line, as mma_pad_once's
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"TP3 config: {name} must be a boolean")
        for name in SMALL_LEVERS:  # own line, as mma_pad_once's
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"TP3 config: {name} must be a boolean")
        for name in E3C_A:  # own line, as mma_pad_once's
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"TP3 config: {name} must be a boolean")
        if self.logprob_single_exchange and self.no_generation_logprobs:
            raise ValueError("TP3 config: logprob_single_exchange reshapes the generation-logprob reduction that "
                             "no_generation_logprobs removes; set one of them")
        if self.draft_head_plans is not None:
            plans = self.draft_head_plans
            if (type(plans) is not tuple or not plans
                    or any(type(item) is not tuple or len(item) != 2 or type(item[0]) is not int or item[0] <= 0
                           or type(item[1]) is not str for item in plans)
                    or len({rows for rows, _ in plans}) != len(plans)):
                raise ValueError("TP3 config: draft_head_plans must be ((rows, config-json), ...) with distinct "
                                 "positive row counts")
        if self.mla_decode_splits is not None:
            from ignition.components.mla_decode_splits import format_policy, parse_policy
            policy = parse_policy(self.mla_decode_splits)
            if not policy:
                raise ValueError("TP3 config: mla_decode_splits must name at least one ROWS:SPLITS pair or be None")
            object.__setattr__(self, "mla_decode_splits", format_policy(policy))
        if self.mla_h24 is not None:
            from ignition.components.mla_h24_decode import format_policy as h24_format, parse_policy as h24_parse
            object.__setattr__(self, "mla_h24", h24_format(h24_parse(self.mla_h24)))  # raises on an empty selection
        if self.markov_tile_rows is not None and (type(self.markov_tile_rows) is not int or self.markov_tile_rows < 1):
            raise ValueError(f"TP3 config: markov_tile_rows must be a positive integer or None, not {self.markov_tile_rows!r}")
        if self.markov_stage_rows and not self.markov_embed_host:
            raise ValueError("TP3 config: markov_stage_rows stages rows of the parked Markov embedding; it needs "
                             "markov_embed_host (FLARE_DS41_MARKOV_EMBED_HOST=1)")
        if any(type(layer) is not int or layer < 0
               for layer in (*self.host_expert_layers, *self.brrr_layers, *self.engram_side_layers)):
            raise ValueError("TP3 config: layer sets must hold non-negative layer indices")
        if self.engram_tables not in ENGRAM_TABLES:
            raise ValueError(f"TP3 config: engram_tables must be one of {', '.join(ENGRAM_TABLES)}, "
                             f"not {self.engram_tables!r}")
        if self.vision_stage_copy not in ("memcpy", "kernel"):
            raise ValueError(f"TP3 config: vision_stage_copy must be 'memcpy' or 'kernel', not {self.vision_stage_copy!r}")
        if self.vision_stage and not self.vision_host:
            object.__setattr__(self, "vision_host", True)  # staging implies parking (the base parked on either switch)
        if self.vision_stage_copy == "kernel" and not self.vision_stage:
            raise ValueError("TP3 config: vision_stage_copy=kernel needs vision_stage")
        if self.hot_experts and self.host_expert_layers and self.expert_counts_dir is None:
            raise ValueError("TP3 config: hot_experts needs expert_counts_dir (route counts pick the hot set)")
        if self.hot_experts and self.host_expert_layers and not self.merged_bank:
            raise ValueError("TP3 config: hot_experts needs merged_bank (separate hot + cold banks were not ported)")
        if self.embed_rows and not self.embed_host:
            raise ValueError("TP3 config: embed_rows needs embed_host (it replaces the host-parked table's lookup)")
        if self.brrr_parked_max_rows and not self.brrr_layers:
            raise ValueError("TP3 config: brrr_parked_max_rows needs brrr_layers")
        if self.brrr_parked_max_rows and not (self.brrr_layers & self.host_expert_layers
                                              and self.merged_bank and 0 < self.hot_experts < ROUTED_EXPERTS):
            # The gate applies to layers with a merged VRAM/host bank (decoder.PreparedLayer: hot_map). With no BRRR
            # layer parked, or hot_experts 0 (whole bank in host memory, no hot_map) or >= 384 (fully resident), it
            # would gate nothing.
            raise ValueError("TP3 config: brrr_parked_max_rows gates BRRR layers parked with a merged bank; it needs "
                             "a layer in both brrr_layers and host_expert_layers, merged_bank and "
                             f"0 < hot_experts < {ROUTED_EXPERTS}")
        if self.brrr_layers and self.brrr_ckpt is None:
            raise ValueError("TP3 config: brrr_layers needs brrr_ckpt (the checkpoint view BRRR reads its dense copies from)")
        for name, low, high in (("brrr_spin_ns", 0, 1_000_000), ("brrr_two_shot_min", 1, 129), ("brrr_ks_wob", 16, 24),
                                ("brrr_woa_small", 0, 16), ("brrr_woa_uneven", 0, 1), ("brrr_per_sm", 1, 2)):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or not low <= value <= high):
                raise ValueError(f"TP3 config: {name} must be unset or an integer in [{low}, {high}], not {value!r}")
        if self.brrr_ks_wob not in (None, 16, 24) or self.brrr_woa_small not in (None, 0, 16):
            raise ValueError("TP3 config: brrr_ks_wob must be 16 or 24 and brrr_woa_small 0 or 16")
        for name in ("brrr_native_scales", "brrr_csf_check", "brrr_break", "brrr_spin_trap", "brrr_spin_ns",
                     "brrr_two_shot_min", "brrr_ks_wob", "brrr_woa_small", "brrr_woa_uneven", "brrr_per_sm"):
            if getattr(self, name) is not None and getattr(self, name) is not False and not self.brrr_layers:
                raise ValueError(f"TP3 config: {name} needs brrr_layers")
        if not self.brrr_dense_native and not self.brrr_layers:
            raise ValueError("TP3 config: brrr_dense_native=False needs brrr_layers")
        if self.brrr_break not in (None, "rope"):
            raise ValueError(f"TP3 config: brrr_break must be unset or 'rope', not {self.brrr_break!r}")
        if self.packed_keep_batches & (self.packed_keep_batches - 1):
            raise ValueError("TP3 config: packed_keep_batches must be 0 or a power of two (a batch capacity)")
        if (self.packed_wide_graphs or self.packed_wide_tokens) and not self.packed_keep_batches:
            raise ValueError("TP3 config: packed_wide_graphs / packed_wide_tokens need packed_keep_batches")
        if self.engram_chunk_rows % 128:
            raise ValueError("TP3 config: engram_chunk_rows must be a multiple of 128 (row-slice alignment of the "
                             "chunked Engram operands and the graph capacity ladder)")
        if self.prefill_graphs is not None and (
                not isinstance(self.prefill_graphs, tuple) or not self.prefill_graphs
                or any(type(c) is not int or c <= 0 or c % 64 for c in self.prefill_graphs)
                or tuple(sorted(set(self.prefill_graphs))) != self.prefill_graphs or self.prefill_graphs[0] != 128):
            raise ValueError("TP3 config: prefill_graphs must be distinct increasing multiples of 64 starting at 128 "
                             f"(got {self.prefill_graphs!r})")
        if self.prefill_skip_suffix is not None and (
                not isinstance(self.prefill_skip_suffix, tuple) or not self.prefill_skip_suffix
                or any(type(c) is not int or c <= 128 or c % 64 for c in self.prefill_skip_suffix)
                or tuple(sorted(set(self.prefill_skip_suffix))) != self.prefill_skip_suffix):
            raise ValueError("TP3 config: prefill_skip_suffix must be distinct increasing multiples of 64 above 128 "
                             f"(got {self.prefill_skip_suffix!r})")
        if self.index_row_split is not None and (
                not isinstance(self.index_row_split, tuple) or not self.index_row_split
                or any(type(c) is not int or c < 384 or c % 128 for c in self.index_row_split)
                or tuple(sorted(set(self.index_row_split))) != self.index_row_split):
            raise ValueError("TP3 config: index_row_split must be distinct increasing multiples of 128 from 384 "
                             f"(three ranks of whole 128-row selector blocks; got {self.index_row_split!r})")
        if self.index_row_split_min_context is not None and (
                not self.index_row_split or type(self.index_row_split_min_context) is not int
                or self.index_row_split_min_context <= 0):
            raise ValueError("TP3 config: index_row_split_min_context must be a positive token count and needs "
                             f"index_row_split (got {self.index_row_split_min_context!r})")
        if self.index_row_split_variant_graphs is not None and (
                self.index_row_split_min_context is None or not isinstance(self.index_row_split_variant_graphs, tuple)
                or not self.index_row_split_variant_graphs
                or any(k not in ("prefill", "encoder_prefill") for k in self.index_row_split_variant_graphs)
                or len(set(self.index_row_split_variant_graphs)) != len(self.index_row_split_variant_graphs)):
            raise ValueError("TP3 config: index_row_split_variant_graphs must be distinct names from prefill, "
                             "encoder_prefill and needs index_row_split_min_context "
                             f"(got {self.index_row_split_variant_graphs!r})")
        if self.index_row_split_check and not self.index_row_split:
            raise ValueError("TP3 config: index_row_split_check needs index_row_split")
        if self.tp3_graph_gib is not None and not self.tp3_graph_gib > 0:
            raise ValueError("TP3 config: tp3_graph_gib must be positive")

    # --- environment readers: the shared parsers, used by the full construction and by the two planner-side peeks ---

    @staticmethod
    def _get(env, name, default=""):
        value = env.get(ENV_PREFIX + name, "")
        return default if value == "" else value

    @classmethod
    def graph_gib_from_environment(cls, environ=None) -> float | None:
        """Planner-side peek at FLARE_DS41_TP3_GRAPH_GIB alone (no validation; the recipe declaration is built at
        import time for every ignition command, so this must never raise on an unrelated stale variable)."""
        value = cls._get(os.environ if environ is None else environ, "TP3_GRAPH_GIB")
        try:
            return float(value) if value else None
        except ValueError:
            return None

    @classmethod
    def vocab_parallel_from_environment(cls, environ=None) -> bool:
        """Planner-side peek at FLARE_DS41_TP3_VOCAB_PARALLEL alone (no validation; generic planner code)."""
        return cls._get(os.environ if environ is None else environ, "TP3_VOCAB_PARALLEL") not in ("", "0")

    @staticmethod
    def mhc_pdl_from_environment(environ=None) -> bool:
        """``B12X_MHC_PDL`` exactly as b12x's mHC kernel module reads it (``os.getenv(name, "0") != "0"``: unset or "0"
        is off, any other value, even "", is on). Not a ``FLARE_DS41_*`` variable: b12x owns the switch, the TP3 record
        only mirrors it."""
        return (os.environ if environ is None else environ).get(MHC_PDL_ENV, "0") != "0"

    def check_mla_decode_splits(self, installed=None):
        """Raise when the recorded split policy differs from the hook installed in this process (``installed``;
        default: read from b12x's decode module)."""
        from ignition.components import mla_decode_splits as hook
        if installed is None:
            import ignition.components.compressed_mla  # noqa: F401  (installs the hook from the environment)
            installed = hook.installed_policy()
        recorded = hook.parse_policy(self.mla_decode_splits)
        if tuple(installed) != recorded:
            raise ValueError(f"TP3 config: mla_decode_splits={self.mla_decode_splits!r} but the installed DSV41 decode "
                             f"split policy is {hook.format_policy(installed)!r}; set {hook.ENV} before the export")
        return self

    def check_mla_h24(self, installed=None):
        """Raise when the recorded 24-head decode selection differs from the hook installed in this process
        (``installed``; default: read from b12x's decode module)."""
        from ignition.components import mla_h24_decode as hook
        if installed is None:
            import ignition.components.compressed_mla  # noqa: F401  (installs the hook from the environment)
            installed = hook.installed_policy()
        recorded = hook.parse_policy(self.mla_h24)
        if tuple(installed) != recorded:
            raise ValueError(f"TP3 config: mla_h24={self.mla_h24!r} but the installed DSV41 24-head decode selection "
                             f"is {hook.format_policy(installed)!r}; set {hook.ENV} before the export")
        return self

    def check_mhc_pdl(self, compiled=None):
        """Raise when ``mhc_pdl`` disagrees with b12x's imported value (``compiled``; default: read from the b12x mHC
        kernel module, importing it). A mismatch means B12X_MHC_PDL changed after b12x was imported, or the config was
        built from another environment: the record would not describe the compiled mHC programs."""
        if compiled is None:
            from b12x.norm.mhc import _kernels
            compiled = _kernels._MHC_PDL
        if bool(compiled) != self.mhc_pdl:
            raise ValueError(f"TP3 config: mhc_pdl={self.mhc_pdl} but b12x's mHC kernels were imported with "
                             f"{MHC_PDL_ENV} {'on' if compiled else 'off'}; set {MHC_PDL_ENV} before starting the export")
        return self

    @classmethod
    def from_environment(cls, environ=None) -> "TP3Config":
        """Build and validate the whole config from ``FLARE_DS41_*`` variables (``environ`` defaults to
        ``os.environ``). Only the recipe's ``prepare`` (and the CUDA tests) call this."""
        env = os.environ if environ is None else environ

        def get(name, default=""):
            return cls._get(env, name, default)

        def flag(name):  # "1" means on; anything else (including unset and "") is off
            return get(name) == "1"

        def truthy(name):  # unset, "" and "0" are off; any other value is on
            return get(name) not in ("", "0")

        def integer(name):
            return int(get(name, "0") or 0)

        def optional(name):  # unset/empty = None (the kernel default); otherwise a strict integer
            value = get(name)
            return None if value == "" else int(value)

        for name, reason in REMOVED.items():
            if env.get(name, "") != "":
                raise ValueError(f"{name} is no longer supported: {reason}")
        umbrella = flag("VRAM_RECLAIM")

        def reclaim(name):  # per-item variable wins when set (non-empty); unset/empty follows the umbrella
            value = get(name)
            return umbrella if value == "" else value == "1"

        counts = get("EXPERT_COUNTS_DIR")
        ckpt = get("BRRR_CKPT")
        dense_native = env.get(ENV_PREFIX + "BRRR_DENSE_NATIVE")  # set-but-empty means staged copies (base semantics)
        per_chunk = env.get(ENV_PREFIX + "FUSED_ATTN_PER_CHUNK")  # set-but-empty is atoi("") = 0 (kernel semantics)
        return cls(
            host_expert_layers=parse_layers(get("HOST_EXPERT_LAYERS")),
            hot_experts=integer("HOT_EXPERTS"),
            merged_bank=flag("MERGED_BANK"),
            expert_counts_dir=None if not counts else Path(counts),
            embed_host=flag("EMBED_HOST"),
            embed_rows=flag("EMBED_ROWS"),
            vision_host=flag("VISION_HOST"),
            vision_stage=flag("VISION_STAGE"),
            vision_stage_copy=get("VISION_STAGE_COPY", "memcpy"),
            vision_arena_host=flag("VISION_ARENA_HOST"),
            markov_embed_host=flag("MARKOV_EMBED_HOST"),
            markov_stage_rows=flag("MARKOV_STAGE_ROWS"),
            markov_tile_rows=optional("MARKOV_TILE_ROWS"),
            mla_decode_splits=get("MLA_DECODE_SPLITS") or None,
            mla_h24=get("MLA_H24") or None,
            draft_vocab=None if not get("DRAFT_VOCAB") else Path(get("DRAFT_VOCAB")),
            ar_rows=integer("AR_ROWS"),
            ar_part_rows=integer("AR_PART_ROWS"),
            engram_part_rows=integer("ENGRAM_PART_ROWS"),
            mhc_pdl=cls.mhc_pdl_from_environment(env),
            tp3_graph_gib=cls.graph_gib_from_environment(env),
            tp3_vocab_parallel=cls.vocab_parallel_from_environment(env),
            prefill_graphs=parse_capacities(get("PREFILL_GRAPHS")),
            prefill_skip_suffix=parse_skip_suffix(get("PREFILL_SKIP_SUFFIX")),
            index_row_split=parse_index_row_split(get("INDEX_ROW_SPLIT")),
            index_row_split_min_context=optional("INDEX_ROW_SPLIT_MIN_CTX"),
            index_row_split_variant_graphs=tuple(k.strip() for k in get("INDEX_ROW_SPLIT_VARIANTS").split(",") if k.strip())
                                           or None,
            index_row_split_check=flag("INDEX_ROW_SPLIT_CHECK"),
            engram_tables=get("ENGRAM_TABLES", "auto"),
            engram_chunk_rows=integer("ENGRAM_CHUNK_ROWS"),
            packed_keep_batches=integer("PACKED_KEEP_BATCHES"),
            packed_wide_graphs=integer("PACKED_WIDE_GRAPHS"),
            packed_wide_tokens=integer("PACKED_WIDE_TOKENS"),
            share_engram_residual=flag("SHARE_ENGRAM_RESIDUAL"),
            glue_pass2=flag("GLUE_PASS2"),
            index_side_stream=flag("INDEX_SIDE_STREAM"),
            engram_side_layers=parse_layers(get("ENGRAM_SIDE_LAYERS")),
            context_side_stream=flag("CONTEXT_SIDE_STREAM"),
            mma_pad_once=flag("MMA_PAD_ONCE"),
            rotary_once=flag("ROTARY_ONCE"),
            pin_small_qproj=flag("PIN_SMALL_QPROJ"),
            logprob_single_exchange=flag("LOGPROB_SINGLE_EXCHANGE"),
            no_generation_logprobs=flag("NO_GENERATION_LOGPROBS"),
            rotary_once_prefill=flag("ROTARY_ONCE_PREFILL"),
            steering_plain_identity=flag("STEERING_PLAIN_IDENTITY"),
            dcp_gather_pull=cls.gather_pull_from_environment(env),
            draft_head_plans=parse_draft_head_plans(get("DRAFT_HEAD_PLANS")),
            **{field_name: reclaim(env_name) for field_name, env_name in VRAM_RECLAIM.items()},
            brrr_layers=parse_layers(get("BRRR_LAYERS")),
            brrr_parked_max_rows=integer("BRRR_PARKED_MAX_ROWS"),
            brrr_ckpt=None if not ckpt else Path(ckpt),
            brrr_native_scales=flag("BRRR_NATIVE_SCALES"),
            brrr_dense_native=dense_native is None or dense_native == "1",
            brrr_csf_check=flag("BRRR_CSF_CHECK"),
            brrr_log=flag("BRRR_LOG"),
            brrr_break=get("BRRR_BREAK") or None,
            fused_attn_per_chunk=per_chunk is None or _atoi(per_chunk) != 0,
            brrr_spin_trap=flag("BRRR_SPIN_TRAP"),
            brrr_spin_ns=optional("BRRR_SPIN_NS"),
            brrr_two_shot_min=optional("BRRR_TWO_SHOT_MIN"),
            brrr_ks_wob=optional("BRRR_KS_WOB"),
            brrr_woa_small=optional("BRRR_WOA_SMALL"),
            brrr_woa_uneven=optional("BRRR_WOA_UNEVEN"),
            brrr_per_sm=optional("BRRR_PER_SM"),
            mini=flag("MINI"),
            no_engram=truthy("NO_ENGRAM"),
            probe=truthy("PROBE"),
        )

    @staticmethod
    def gather_pull_from_environment(env) -> bool:
        """On only when set explicitly to a pulling value ("auto", "1", "on", "pull", "true"; export.sh GATHER_PULL=1
        sets "auto"). Unset counts as off: without the overlay patch b12x always pushes, and export.sh forces "0".
        Preparation checks the record against the imported b12x either way (check_b12x_switches)."""
        return env.get(GATHER_PULL_ENV, "").strip().lower() in ("auto", "1", "on", "pull", "true")

    def check_b12x_switches(self, *, world: int):
        """Refuse a dcp_gather_pull record the imported b12x cannot honour, or a patched b12x that would pull while
        the record says push (preparation calls it)."""
        if world > 1:
            from b12x.comm.pcie import _dcp_a2a_cute
            patched = hasattr(_dcp_a2a_cute, "_gather_pull_enabled")
            if self.dcp_gather_pull and not patched:
                raise ValueError("TP3 config: dcp_gather_pull needs the b12x overlay patch "
                                 "flashinfer-dcp-gather-pull-world3 (B12X_OVERLAY)")
            if patched and _dcp_a2a_cute._gather_pull_enabled(world) != self.dcp_gather_pull:
                raise ValueError("TP3 config: dcp_gather_pull differs from this process's b12x switch "
                                 f"({GATHER_PULL_ENV})")
        return self

    def unported(self) -> dict:
        """Switches set to a non-default value whose code has not been ported yet: {field: stage}."""
        defaults = {item.name: item.default_factory() if item.default is MISSING else item.default
                    for item in fields(self)}
        return {name: stage for name, stage in UNPORTED.items() if getattr(self, name) != defaults[name]}

    def require_ported(self):
        """Raise when a switch asks for code that is not on this port yet (preparation calls it at config time)."""
        pending = self.unported()
        if pending:
            raise ValueError("TP3 config: " + "; ".join(
                f"{name}={format_layers(value) if isinstance(value, frozenset) else value!r} is not ported yet "
                f"(stage {pending[name]})" for name, value in ((name, getattr(self, name)) for name in pending)))
        return self

    def check_layers(self, layers: int):
        """Raise when a layer set names a layer the model (or its mini view) does not have; preparation calls it
        with the checkpoint's layer count before loading weights."""
        for name in ("brrr_layers", "engram_side_layers"):
            beyond = sorted(layer for layer in getattr(self, name) if layer >= layers)
            if beyond:
                raise ValueError(f"TP3 config: {name} names layers {format_layers(beyond)} but the model has "
                                 f"{layers} layers (0-{layers - 1})")
        return self

    def shared_engram_tables(self, world: int) -> bool:
        """Whether a ``world``-rank export keeps whole Engram tables on every rank (see ``engram_tables``)."""
        if world == 1 or self.engram_tables == "shared":
            return True
        return self.engram_tables == "auto" and world == 3

    def to_metadata(self) -> dict:
        """JSON-ready record of every switch (RECORDED_WHEN_SET ones only when set): layer sets as ``"20-33"`` strings; the two paths as basename plus a
        short content fingerprint (never an absolute path, so the record can live in a package or a report)."""
        record = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if item.name in RECORDED_WHEN_SET and value is None:
                continue  # kernel default: absent, so a default export's record (tp3-config.json) is unchanged
            if item.name in RECORDED_WHEN_TRUE and value is False:
                continue
            if item.name == "draft_head_plans":
                if value is None:
                    continue  # base: absent, so a default export's record is unchanged
                value = {str(rows): json.loads(config) for rows, config in value}
            if isinstance(value, frozenset):
                value = format_layers(value)
            elif item.name == "expert_counts_dir":
                value = describe_counts_dir(value)
            elif item.name == "draft_vocab":
                value = describe_file(value)
            elif item.name == "brrr_ckpt":
                value = describe_checkpoint(value)
            record[item.name] = value
        return record

    def describe(self) -> str:
        """One-line summary of the switches that differ from the defaults (for export progress)."""
        record = self.to_metadata()
        changed = [f"{item.name}={record[item.name]!r}" for item in fields(self)
                   if getattr(self, item.name) != (item.default_factory() if item.default is MISSING else item.default)]
        return "TP3 config: " + (", ".join(changed) if changed else "defaults")
