Patches applied only to the b12x overlay (`tp3-tools/make-b12x-overlay.sh` -> `.b12x-overlay`, used by
`glm53-export-tp3.sh` via `B12X_OVERLAY`), not to the shared `.venv` (which the DS41 exports also use).

- `flashinfer-dcp-gather-pull-world3.patch`: at world 3 the PCIe DCP head gather pulls peers' staged rows instead of
  pushing into peers (payloads under 4 KiB per rank stay push). Switches: `B12X_PCIE_DCP_GATHER_PULL=auto|1|0` (auto =
  world 3 only), `B12X_PCIE_DCP_GATHER_PULL_MIN_BYTES` (4096); both are in the compile key (kernel version 15).
  Applies on top of `../flashinfer-dcp-a2a-world3.patch`. Measured 2026-11-09: head gather 923 -> 63 us at 48 rows;
  GLM a16 c1/c4/c8 +7.0/+8.2/+8.4 % (pkg gb202-tp3-202611092215 vs 082232, 3 alternating rounds); eval 43/43,
  tools 60/60, verify_fix 0, hotel c8 28/0 wrong/2 truncated.
- 2026-11-10 (branch glm53-tp2-update), same patch file: `B12X_PCIE_DCP_GATHER_PULL=1` now pulls at world 2 too (the
  generic pull plane with one peer; payloads under `_MIN_BYTES` keep the upstream TP2 push, moved unchanged into
  `_gather_push_tp2`). `auto` is unchanged (world 3 only), so TP2 exports stay on push unless `PULL=1` is exported.
  CPU check: world-2 push PTX identical to the unpatched venv, world-3 auto PTX identical to the 11-09 overlay; world-2
  pull compiles for both ranks and slot modes. Not measured on GPU yet (glm-tp2 chain stage G).
- `b12x-mla-decode-remainder-stream.patch` (DS41, 2026-11-15, branch perf/ds41-attn-one-launch): with BF16 MLA (the
  DS41 recipe default) b12x tiles 16 heads per decode CTA, so a TP3 rank's 24 heads run as a 16-head grid plus an
  8-head remainder grid that serialize (no PDL). `B12X_MLA_DECODE_REMAINDER_STREAM=1` launches the remainder grid on
  a dedicated side stream (fork/join around it, merge after the join): same kernels and arguments, disjoint output
  heads, so bit-identical. Unset/0 = serial, byte-identical to the unpatched launch order. It shapes the captured
  graph, so it takes effect at export. Build a DS41-only overlay (without the GLM pull-gather patch, which would
  change DS41's vocab gathers): `VENV=<fork>/.venv B12X_OVERLAY_PATCHES=$PWD/tp3-patches/overlay/b12x-mla-decode-remainder-stream.patch tp3-tools/make-b12x-overlay.sh <out>`,
  then export with `B12X_OVERLAY=<out> B12X_MLA_DECODE_REMAINDER_STREAM=1 tp3-tools/release/export.sh`.
  GPU checks: `tests/components/test_compressed_mla_remainder_stream_cuda.py` (bit-identity + optional timing).
- `flashinfer-pcie-twoshot-pull-world3.patch` (DS41, 2026-11-15, branch fix/ar-twoshot-pull, copied here on
  perf/ds41-exp-combined): `B12X_PCIE_TWOSHOT_PULL=1` adds a `twoshot_pull` alternative (the PCIeTwoShotBF16 runtime's
  single-launch pull transport, same thread/block sweep as `twoshot`) to every staged plain BF16 all-reduce that admits
  two-shot; unset/0 = upstream candidate set and selection keys. Host-side only (`comm/pcie/_collective_tuning.py`,
  `comm/pcie/pcie_allreduce.py`): no CuTe source-closure fingerprint moves, only the queries of those all-reduce races
  (80 in a DS41 TP3 export) change when the switch is on. Combined DS41 overlay (venv b12x + this + the remainder-stream
  patch; no GLM pull-gather patch, so DS41's 21 head-gather races keep their keys):
  `VENV=<fork>/.venv B12X_OVERLAY_PATCHES="$PWD/tp3-patches/overlay/flashinfer-pcie-twoshot-pull-world3.patch $PWD/tp3-patches/overlay/b12x-mla-decode-remainder-stream.patch" tp3-tools/make-b12x-overlay.sh <out>`.
  Note: any overlay carrying the remainder-stream patch changes `attention/_shared/mla/kernel.py`, which is in the CuTe
  source closure of 32 b12x MLA modules: every MLA program gets a new compile key and every
  `attention.compressed_sparse_mla` race (208 in a DS41 export) re-discovers and re-measures, even with the switch off.
