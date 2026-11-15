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
import os
import re
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path

ENV_PREFIX = "FLARE_DS41_"

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
    # --- collectives and planning (bootstrap.py, recipe.py, planning.py) ---
    ar_rows: int = 0
    """Rows the shared TP3 all-reduce runtime is sized for (chunked prefill reductions); 0 = a whole prefill chunk."""
    ar_part_rows: int = 0
    """Decoder attention/MLP all-reduces of more rows than this run in parts of exactly N rows plus a remainder
    (4096 at 768: 5 x 768 + 256) instead of halving to what the runtime accepts (4096 at 1024: 4 x 1024); 0 = halving
    (base). Engram reductions keep halving. Export-time only (prepared graphs)."""
    tp3_graph_gib: float | None = None
    """Override of the per-rank captured-graph estimate for world size 3 (GiB); None = the 8 GiB default."""
    tp3_vocab_parallel: bool = False
    """Admit vocabulary-parallel sampling at TP3 (b12x gathers enabled for world 3)."""
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
        for name in ("hot_experts", "ar_rows", "ar_part_rows", "brrr_parked_max_rows", "engram_chunk_rows",
                     "packed_keep_batches", "packed_wide_graphs", "packed_wide_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"TP3 config: {name} must be a non-negative integer, not {value!r}")
        for name in ("merged_bank", "embed_host", "embed_rows", "vision_host", "vision_stage", "tp3_vocab_parallel",
                     "brrr_native_scales", "brrr_dense_native", "brrr_csf_check", "brrr_log", "fused_attn_per_chunk",
                     "brrr_spin_trap", "mini", "no_engram", "probe", "share_engram_residual",
                     "index_side_stream", "context_side_stream", *VRAM_RECLAIM):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"TP3 config: {name} must be a boolean")
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
        for name in ("brrr_native_scales", "brrr_csf_check", "brrr_break", "brrr_spin_trap"):
            if getattr(self, name) and not self.brrr_layers:
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
            ar_rows=integer("AR_ROWS"),
            ar_part_rows=integer("AR_PART_ROWS"),
            tp3_graph_gib=cls.graph_gib_from_environment(env),
            tp3_vocab_parallel=cls.vocab_parallel_from_environment(env),
            engram_tables=get("ENGRAM_TABLES", "auto"),
            engram_chunk_rows=integer("ENGRAM_CHUNK_ROWS"),
            packed_keep_batches=integer("PACKED_KEEP_BATCHES"),
            packed_wide_graphs=integer("PACKED_WIDE_GRAPHS"),
            packed_wide_tokens=integer("PACKED_WIDE_TOKENS"),
            share_engram_residual=flag("SHARE_ENGRAM_RESIDUAL"),
            index_side_stream=flag("INDEX_SIDE_STREAM"),
            engram_side_layers=parse_layers(get("ENGRAM_SIDE_LAYERS")),
            context_side_stream=flag("CONTEXT_SIDE_STREAM"),
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
            mini=flag("MINI"),
            no_engram=truthy("NO_ENGRAM"),
            probe=truthy("PROBE"),
        )

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
        """JSON-ready record of every switch: layer sets as ``"20-33"`` strings; the two paths as basename plus a
        short content fingerprint (never an absolute path, so the record can live in a package or a report)."""
        record = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, frozenset):
                value = format_layers(value)
            elif item.name == "expert_counts_dir":
                value = describe_counts_dir(value)
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
