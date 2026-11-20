#include <type_traits>
#include <cstdlib>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include "fp8_rows.cuh"
#include "host.cuh"

#include <vector>

namespace {

constexpr int HC = 4;
constexpr int MIX = 24;
constexpr int HID = 5120;
constexpr int HCD = HC * HID;
constexpr int MAXM = 128;
constexpr int TILE = 8;
constexpr int TOPK = 6;
constexpr int ROUTES = MAXM * TOPK;
constexpr int EXPERTS = 384;
constexpr int SPAN = 16;  // TP3: routed jobs stage SPAN x KS_INTER(24) x 32 B of activations in bt; 32 x 24 x 32 = 24 KiB would not fit 2 CTAs/SM
constexpr int ENTRIES = EXPERTS + ROUTES / SPAN;
constexpr int HDIM = 512;
constexpr int ROPE = 64;
constexpr int OA = 3072;  // TP3: 3 padded o-groups x 1024 (72 heads / 3)
constexpr int OAK = 4096;
constexpr int INTER = 768;  // TP3: shared expert 2304 / 3
// TP3: b12x prepares W4A8 experts in its natural ceil-tiled layout when INTER % 128 == 0 (no 64-row tail);
// the exact N128/N64 layout exists only for tails such as TP4's 576. The loader maps between them.
constexpr bool RP_LAYOUT = (INTER % 128 == 0);  // TP3: INTER 768 has no 64-row tail, so b12x keeps its natural W4A8 layout
static_assert(!RP_LAYOUT || (INTER % 256 == 0 && HID % 256 == 0), "RP tiling needs whole 256-row tiles for both halves");
constexpr int SLICES = HID / 32;
constexpr int SH2_LD = 24;  // TP3: INTER/32 = 24 scale columns (TP4: 18 padded to 20)
constexpr int HQ_LD = 24;  // TP3: multiple of 4 >= INTER/32
constexpr float LIMIT = 10.0f;
constexpr int SINKHORN = 20;
constexpr float HC_EPS = 1e-6f;
constexpr float NORM_EPS = 1e-20f;
constexpr float ROUTE_SCALE = 1.5f;
constexpr float FIXED = 0x1p32f;

constexpr int GW = 4;
constexpr int GT = GW * 32;
constexpr int THREADS = GT + 32;
// Layout tuning switches (TP3Config brrr_per_sm / brrr_ks_wob / brrr_woa_small / brrr_woa_uneven -> -D, separately named
// builds; unset = the serving build). TP3's OA 3072 gives 384 WOA and 480 WOB jobs at <= 16 rows where TP4's OA 2048 gave
// 256 and 320: more than the 376 CTAs (2 x 188 SMs), so the last row block's WOA jobs and a quarter of the WOB jobs run
// in a second round on the layer's critical path (deep dive 2026-11-15).
#ifndef BRRR_PER_SM
#define BRRR_PER_SM 2
#endif
constexpr int PER_SM = BRRR_PER_SM;  // diagnostic only: 1 halves the weight reads in flight
constexpr int STAGES = 3;
constexpr int FRAG = 512;
constexpr int FRAGS = 16;
constexpr int SCALE_OFF = FRAGS * FRAG;
constexpr int SCALE_BYTES = 512;  // TP3 CSF: bases/codes/descriptors + up to 2x36 exception records per stage, expanded in place
constexpr int STAGE = FRAGS * FRAG + SCALE_BYTES;
constexpr int CSF_CAP = 36;  // TP3 CSF: staged exception records per 64-row half (J_R2 stages 2x for 128 rows); tail read from global
constexpr int CSF_REC = 200;  // byte offset of the staged records inside the stage scale area
static_assert(CSF_REC + 2 * CSF_CAP * 4 <= SCALE_BYTES, "staged CSF records must fit the scale area");
constexpr int OWNA_ROWS = 16;
// Tuning switches (TP3Config brrr_two_shot_min / brrr_spin_ns -> brrr.kernel_defines -> -D, a separately named
// extension build; unset = these defaults = the serving build). TWO_SHOT_MIN: rows from which the attention and FFN
// exchanges reduce two-shot (owner rank per chunk) instead of one-shot (every rank sums every chunk); 1 = always
// two-shot, MAXM + 1 = never. SPIN_NS: __nanosleep between polls of a counter, route or peer flag.
#ifndef BRRR_TWO_SHOT_MIN
#define BRRR_TWO_SHOT_MIN 17
#endif
constexpr int TWO_SHOT_MIN = BRRR_TWO_SHOT_MIN;
constexpr int BT = 16384;  // TP3: max(SPAN*KS_INTER*32 = 12288, OTOK*2*KS_WOA*64 = 16384); 2 CTAs/SM still fit
constexpr int BS = 384;  // TP3: >= max(OTOK, SPAN) * KW_INTER(6) * 4 = 16 * 6 * 4 (TP4: 32 * 5 * 4 = 640)

constexpr int CHUNKS = 40;
constexpr int CHUNK_COLS = HID / CHUNKS;
constexpr int CHUNK_VECS = CHUNK_COLS / 8;
constexpr int OTOK = 16;  // TP3: BT = OTOK*KS_INTER*32 must fit 2 CTAs/SM in sm_120 smem (24 k-slices at INTER 768)
constexpr int GROUPS = MAXM / OTOK;
constexpr int TILES = MAXM / TILE;
constexpr int MIXW = 32;
constexpr int KPARTS = 10;
constexpr int XNORM_MIN = 16;
constexpr int WOA_BLOCKS = OA / 64;
constexpr int WOB_BLOCKS = HID / 64;
constexpr int GATE_BLOCKS = EXPERTS / 64;
constexpr int SH1_BLOCKS = INTER / 32;
constexpr int SH2_BLOCKS = HID / 64;
constexpr int KS_WOA = 8;
#ifndef BRRR_KS_WOB
#define BRRR_KS_WOB 16
#endif
constexpr int KS_WOB = BRRR_KS_WOB;  // k32 slices of z per WOB job: 16 = 6 parts (480 jobs at <= 16 rows), 24 = 4 parts (320)
static_assert((OA / 32) % KS_WOB == 0 && KS_WOB % 8 == 0 && (PER_SM == 1 || PER_SM == 2), "BRRR layout switch values");
constexpr int KS_GATE = 8;
constexpr int KS_SH1 = 16;
constexpr int KP_WOB = OA / 32 / KS_WOB;
constexpr int KP_GATE = HID / 16 / KS_GATE;
constexpr int KP_SH1 = HID / 32 / KS_SH1;
constexpr int KS_INTER = INTER / 32;
constexpr int KW_INTER = HQ_LD / 4;
constexpr int INTER_STAGES = cdiv(KS_INTER, 4);
constexpr int R1_BLOCKS = INTER / 64;
constexpr int R1_PARTS = 10;
constexpr int R2_BLOCKS = HID / 128;
constexpr int KG_WOA = OAK / 128;
constexpr int KG_WOB = OA / 128;
constexpr int KG_GATE = HID * 2 / 128;
constexpr int KG_SH1 = HID / 128;
static_assert(SCALE_OFF + SCALE_BYTES <= STAGE && CHUNK_COLS == 128 && FRAGS * FRAG == 8192 && R2_BLOCKS == CHUNKS);
static_assert(OWNA_ROWS * HC * 32 * 4 <= STAGE && (MIX - OWNA_ROWS) * HC * 32 * 4 <= STAGE);
static_assert(OTOK * KS_WOA * 64 <= BT && OTOK * KS_WOB * 32 <= BT && OTOK * KS_SH1 * 32 <= BT && OTOK * KS_INTER * 32 <= BT && OTOK * KS_GATE * 32 <= BT);
static_assert(SPAN * KS_INTER * 32 <= BT && SPAN * (SLICES / R1_PARTS) * 32 <= BT, "routed jobs stage up to SPAN rows");
static_assert((OTOK > SPAN ? OTOK : SPAN) * KW_INTER * 4 <= BS && OTOK * (KS_WOB / 4) * 4 <= BS, "scale staging");

// Compile-time test switch: TP3Config.brrr_spin_trap -> brrr.kernel_defines -> -D BRRR_SPIN_TRAP=1 (a separate extension
// build). 0 = the serving kernel. (The item-6 experiments split tail / fast route / counter parity were not carried into
// the 2026-11-14 upstream port; their code stays in tp3-main's history, 7f1a2a4e.)
#ifndef BRRR_SPIN_TRAP
#define BRRR_SPIN_TRAP 0  // test builds only (TP3Config.brrr_spin_trap): a protocol wait that never ends prints and traps
#endif
constexpr int SPIN_TRAP = BRRR_SPIN_TRAP;
static_assert(SPIN_TRAP == 0 || SPIN_TRAP == 1, "BRRR switch values");

// Activation steering (upstream PR #404 semantics inside the fused tail; feat/ds41-steering-brrr, analysis/work/
// steering-sidecar-20261119/optionB/DESIGN.md). Compile-time: brrr.kernel_defines(tp3, steering=True) adds
// BRRR_STEER=1 for packages exported with a steering sidecar (its own extension build). Every steering line below is
// inside #if BRRR_STEER, so the default build (no define) preprocesses to the previous source and compiles to the same
// layer kernel. At run time a launch steers only when the layer has steering constants (set_steering), the op passes
// a per-token scale (run_steered) and some row's scale is nonzero (Smem::steer, the same on every CTA and rank); rows
// with scale 0 keep their unsteered values bit for bit, and a launch without a nonzero row runs the unsteered job graph
// (no extra wait, no extra write).
// Attention-site dot y . d: from the reduced BF16 yattn chunk right after each exchange (one fixed-point add per row
// and chunk into adot, C_YD), as upstream (y_bf16 . d); OWNA waits for all CHUNKS dots after loading its tile.
// (v2's dot before the all-reduce, published next to the attention inbox, cost +38..+50 us per launch on the GPU,
// seq375 / seq377, and was removed in v4 together with everything it added to the shared exchange code.)
// v4: the steering flag lives in shared memory (Smem::steer, read where a job branches), not in a register of Ctx,
// so the persistent loop's register allocation and the exchange code (share / reduce4 / gather) match the plain build.
#ifndef BRRR_STEER
#define BRRR_STEER 0
#endif
constexpr int STEER = BRRR_STEER;
static_assert(STEER == 0 || STEER == 1, "BRRR switch values");

// Test-only compile switch (layer test, never a serving build): BRRR_ORDERED=1 makes every FP32 accumulation of the
// kernel order-deterministic, so two launches on the same inputs give the same bits and the steering build at scale 0
// can be compared bit for bit with the plain build. The k-split partial sums (zacc / yacc / sacc / hacc) are added in part
// order (each part waits until the previous parts have added: counters that already exist); the routed outputs go to
// one row per route (yroute) and job_xf sums a token's TOPK routes in route order instead of red.add into yracc.
#ifndef BRRR_ORDERED
#define BRRR_ORDERED 0
#endif
constexpr int ORDERED = BRRR_ORDERED;
static_assert(ORDERED == 0 || ORDERED == 1, "BRRR switch values");

#if BRRR_SPIN_TRAP
constexpr long long SPIN_TRAP_POLLS = 1ll << 25;  // >= 6 s of 200 ns sleeps; every legitimate wait is microseconds

__device__ __noinline__ void spin_trap(const char* where, int index, unsigned target, unsigned seen) {
  printf("[brrr spin trap] block %d thread %d: %s index %d target %u seen %u\n", blockIdx.x, threadIdx.x, where, index, target, seen);
  __trap();
}
#endif

__host__ __device__ __forceinline__ int tiles_of(int m) { return cdiv(m, TILE); }

__host__ __device__ __forceinline__ int token_groups(int m) { return cdiv(m, OTOK); }

__host__ __device__ __forceinline__ bool split_norm(int m) { return m > XNORM_MIN; }

__host__ __device__ __forceinline__ int quads(int m) { return cdiv(m, GW); }

constexpr int WOA_SMALL = 16;  // token capacity of the 4-stage WOA job's staging (woa_group<2 * KS_WOA, WOA_SMALL>)
#ifndef BRRR_WOA_SMALL
#define BRRR_WOA_SMALL 16
#endif
constexpr int WOA_SMALL_MAX = BRRR_WOA_SMALL;  // rows up to which WOA jobs are 4 stages (16 slices); 0 = always 2 stages
static_assert(WOA_SMALL_MAX >= 0 && WOA_SMALL_MAX <= WOA_SMALL, "BRRR_WOA_SMALL");

__host__ __device__ __forceinline__ int woa_ks(int m) { return m <= WOA_SMALL_MAX ? 2 * KS_WOA : KS_WOA; }

// Uneven WOA split (BRRR_WOA_UNEVEN=1, rows <= WOA_UTOK): 7 jobs per 64-row block of 5,5,5,5,4,4,4 stages (the 32
// k128 stages of OAK 4096) = 336 jobs, so every WOA job runs in the first round of 376 CTAs. Staging 8 rows x 20 slices
// x 64 B = 10,240 B <= BT. The k split only regroups the float sums into zacc (atomic adds, unordered already).
#ifndef BRRR_WOA_UNEVEN
#define BRRR_WOA_UNEVEN 0
#endif
constexpr int WOA_UNEVEN = BRRR_WOA_UNEVEN;
constexpr int WOA_UPARTS = 7, WOA_ULONG = 4, WOA_UKS = 20, WOA_UTOK = 8;
static_assert(WOA_UNEVEN == 0 || WOA_UNEVEN == 1, "BRRR_WOA_UNEVEN");
static_assert(WOA_ULONG * 5 + (WOA_UPARTS - WOA_ULONG) * 4 == OAK / 128 && WOA_UTOK * WOA_UKS * 64 <= BT && WOA_UTOK <= OTOK && WOA_UTOK % TILE == 0);

__host__ __device__ __forceinline__ bool woa_uneven(int m) { return WOA_UNEVEN && m <= WOA_UTOK; }

__host__ __device__ __forceinline__ int woa_ustages(int b) { return b < WOA_ULONG ? 5 : 4; }

__host__ __device__ __forceinline__ int woa_ustage0(int b) { return b < WOA_ULONG ? 5 * b : 5 * WOA_ULONG + 4 * (b - WOA_ULONG); }

struct Span {
  int first, count, tiles, ts;
};

__host__ __device__ __forceinline__ Span span_of(int first, int count) {
  const int tiles = tiles_of(count);
  return {first, count, tiles, tiles == 1 ? 1 : tiles == 2 ? 2 : 4};
}

__host__ __device__ __forceinline__ Span group_of(int m, int tg) { return span_of(tg * OTOK, m - tg * OTOK < OTOK ? m - tg * OTOK : OTOK); }

__host__ __device__ __forceinline__ Span tile_span(int m, int t) { return span_of(t * TILE, m - t * TILE < TILE ? m - t * TILE : TILE); }

constexpr int OWNA_SPANS = 4;

__host__ __device__ __forceinline__ int owna_tiles(int m) { return cdiv(tiles_of(m), OWNA_SPANS); }

__host__ __device__ __forceinline__ int owna_spans(int m) { return cdiv(tiles_of(m), owna_tiles(m)); }

struct Layout {
  int m, tg, amix, wob, owna, xnorm, gate, route, sh1, mix, routed;
};

__host__ __device__ __forceinline__ Layout layout(int m) {
  Layout l{};
  l.m = m;
  l.tg = token_groups(m);
  l.amix = woa_uneven(m) ? WOA_BLOCKS * WOA_UPARTS : WOA_BLOCKS * (OAK / 32 / woa_ks(m)) * l.tg;
  l.wob = l.amix + quads(m);
  l.owna = l.wob + CHUNKS * (2 * KP_WOB * l.tg + 1);
  l.xnorm = l.owna + SLICES * owna_spans(m);
  l.gate = l.xnorm + (split_norm(m) ? KPARTS * tiles_of(m) : 0);
  l.route = l.gate + GATE_BLOCKS * KP_GATE * l.tg;
  l.sh1 = l.route + m;
  l.mix = l.sh1 + SH1_BLOCKS * KP_SH1 * l.tg;
  l.routed = l.mix + quads(m);
  return l;
}

constexpr int R1_JOBS = R1_BLOCKS * R1_PARTS;

__host__ __device__ __forceinline__ int chunk_jobs(const Layout& L, int n) { return 2 * L.tg + n + 1; }

__device__ __forceinline__ int total_jobs(const Layout& L, int n) { return L.routed + n * R1_JOBS + CHUNKS * chunk_jobs(L, n); }

enum : int { J_WOA, J_AMIX, J_WOB, J_XA, J_OWNA, J_XNORM, J_GATE, J_ROUTE, J_SH1, J_SH2, J_MIX, J_R1, J_R2, J_XF, J_END };

struct Job {
  int type, a, b, c, stages, e, f, n;
};

struct Slot {
  int chunk, item;
};

__device__ __forceinline__ int exchange_lag(int per) { return min(CHUNKS, max(1, cdiv(int(gridDim.x), per))); }

__device__ __forceinline__ Slot lagged(int r, int per) {
  const int lag = exchange_lag(per);
  if (r < lag * per) return {r / per, r % per};
  r -= lag * per;
  if (r < (CHUNKS - lag) * (per + 1)) {
    const int k = lag + r / (per + 1), i = r % (per + 1);
    return i < per ? Slot{k, i} : Slot{k - lag, per};
  }
  return {CHUNKS - lag + r - (CHUNKS - lag) * (per + 1), per};
}

__host__ __device__ __forceinline__ Job blocked(int type, int r, int parts, int tg, int stages) { return {type, r / (parts * tg), r / tg % parts, r % tg, stages}; }


__device__ __forceinline__ Job decode(const Layout& L, int n, int j) {
  if (j < L.amix) {
    if (woa_uneven(L.m)) return {J_WOA, j / WOA_UPARTS, j % WOA_UPARTS, 0, woa_ustages(j % WOA_UPARTS)};
    return blocked(J_WOA, j, OAK / 32 / woa_ks(L.m), L.tg, woa_ks(L.m) / 4);
  }
  if (j < L.wob) return {J_AMIX, j - L.amix, 0, 0, 1};
  if (j < L.owna) {
    const int per = 2 * KP_WOB * L.tg;
    const Slot sl = lagged(j - L.wob, per);
    if (sl.item == per) return {J_XA, sl.chunk, 0, 0, 1};
    const Job w = blocked(J_WOB, sl.item, KP_WOB, L.tg, KS_WOB / 4);
    return {J_WOB, 2 * sl.chunk + w.a, w.b, w.c, w.stages};
  }
  if (j < L.xnorm) return {J_OWNA, (j - L.owna) % SLICES, (j - L.owna) / SLICES, 0, 2};
  if (j < L.gate) return {J_XNORM, (j - L.xnorm) % KPARTS, (j - L.xnorm) / KPARTS, 0, 1};
  if (j < L.route) return blocked(J_GATE, j - L.gate, KP_GATE, L.tg, KS_GATE / 4);
  if (j < L.sh1) return {J_ROUTE, j - L.route, 0, 0, 1};
  if (j < L.mix) return blocked(J_SH1, j - L.sh1, KP_SH1, L.tg, KS_SH1 / 4);
  if (j < L.routed) return {J_MIX, j - L.mix, 0, 0, 1};
  int r = j - L.routed;
  if (r < n * R1_JOBS) return {J_R1, r / R1_JOBS, r / R1_PARTS % R1_BLOCKS, r % R1_PARTS, 4};
  r -= n * R1_JOBS;
  if (r < CHUNKS * chunk_jobs(L, n)) {
    const int shared = 2 * L.tg;
    const Slot sl = lagged(r, shared + n);
    if (sl.item < shared) return {J_SH2, 2 * sl.chunk + sl.item / L.tg, 0, sl.item % L.tg, INTER_STAGES};
    return sl.item < shared + n ? Job{J_R2, sl.item - shared, sl.chunk, 0, INTER_STAGES} : Job{J_XF, sl.chunk, 0, 0, 1};
  }
  return {J_END, 0, 0, 0, 1};
}

constexpr int C_CLAIM = 0;
constexpr int C_ZA = C_CLAIM + 1;
constexpr int C_Z = C_ZA + WOA_BLOCKS * GROUPS;
constexpr int C_YA = C_Z + KP_WOB * GROUPS;
constexpr int C_YC = C_YA + WOB_BLOCKS * GROUPS;
constexpr int C_YR = C_YC + CHUNKS;
constexpr int C_AMIX = C_YR + CHUNKS;
constexpr int C_A = C_AMIX + 1;
constexpr int C_XK = C_A + TILES;
constexpr int C_G = C_XK + KPARTS * GROUPS;
constexpr int C_ROUTE = C_G + GROUPS;
constexpr int C_SA = C_ROUTE + 1;
constexpr int C_HS = C_SA + SH1_BLOCKS * GROUPS;
constexpr int C_SF = C_HS + GROUPS;
constexpr int C_FIN = C_SF + CHUNKS;
constexpr int CACHED = C_FIN;
constexpr int C_PUB = C_FIN + 1;
constexpr int C_R1A = C_PUB + 4 * CHUNKS;
constexpr int C_R1D = C_R1A + ENTRIES * R1_BLOCKS;
// Cold-first routed jobs: route entries (expert slot, 16-route chunk) are kept in one table of ENTRIES, cold slots
// (slot >= Weights::hot_rows, host RAM over PCIe) filling it from the front and hot slots from the back, so job index
// a < ncold is cold. Their sum is bounded by ENTRIES (each slot adds at most ceil(count / SPAN) entries).
constexpr int C_NHOT = C_R1D + ENTRIES;
constexpr int C_NCOLD = C_NHOT + 1;
constexpr int C_ECOUNT = C_NCOLD + 1;
#if BRRR_STEER
// Steering: attention exchange jobs whose row dots are in adot (OWNA waits for CHUNKS). Past CACHED: polled.
constexpr int C_YD = C_ECOUNT + EXPERTS;
constexpr int COUNTERS = C_YD + 1;
#else
constexpr int COUNTERS = C_ECOUNT + EXPERTS;
#endif

struct Peers {
  __nv_bfloat16* pub[RANKS];
  unsigned* flags[RANKS];
};

constexpr long PUB_SLOT = long(MAXM) * HID;
constexpr long FLAG_BYTES = 8192;
static_assert(2 * 2 * RANKS * CHUNKS * sizeof(unsigned) <= FLAG_BYTES);

struct Table {
  int expert[ENTRIES];
  int chunk[ENTRIES];
  int routes[EXPERTS * MAXM];
};

struct Scratch {
  unsigned* epoch;
  unsigned* counters;
  Table* table;
  float* zacc;
  uint8_t* zq;
  uint8_t* zse;
  float* yacc;
  __nv_bfloat16* yattn;
  float* mixp;
  __nv_bfloat16* xc;
  float* xsq;
  __nv_bfloat16* xbf;
  uint8_t* xq;
  uint8_t* xse;
  long long* gacc;
  float* topk_w;
  float* sacc;
  uint8_t* hsq;
  uint8_t* hse;
  float* shacc;
  float* hacc;
  uint8_t* hq;
  uint8_t* hqe;
  float* yracc;
#if BRRR_STEER
  // Per-token steering dots in fixed point (x 2^32, like gacc: integer adds, so the sum is independent of the order
  // of the partial contributions). adot = yattn . d (replicated rows, identical on every rank), sdot = h . (W2^T d)
  // (this rank's shared-expert partial . d). Cleared by the launch's last CTA when it steered.
  long long* adot;
  long long* sdot;
#endif
#if BRRR_ORDERED
  float* yroute;  // test build: [ROUTES, HID] weighted routed output per route (replaces the red.add into yracc)
#endif
};

// TP3 CSF: b12x CsfBlockPayload (destination 2, mxfp4 1-bit codes, 3-byte group records) read in-kernel.
struct Csf {
  const uint8_t* storage;  // nullptr = plain scales in s13/s2
  const unsigned* gdesc;   // per (payload expert, group): first record index (22 bits) | count << 22
  unsigned* debug;         // optional: expanded words mirrored into the rp sfb layout (harness verification)
  unsigned* debug_desc;    // optional: per 128-row group slice, the staged descriptor word the consumer saw (debug)
  long bases, codes, code_bytes, value_off, recpos;
  int rows, column_groups, groups, sections;
  int payload_experts; long records;  // bounds for the debug guards
};

struct Weights {
  Csf csf13, csf2;
  // TP3: dense weights read in vLLM's own row-major layout (no staged copies); row strides in bytes.
  int dense_native;
  long woa_ld, wob_ld, gate_ld, sh13_ld, sh2_ld;
  unsigned* route_counts;  // TP3 dynamic expert cache: optional per-layer routed-slot counters (accumulate across steps)
  const int* hot_map;      // Flare merged bank: global expert ID -> slot in the hot-first bank (null = identity)
  int hot_rows;            // Flare merged bank: slots < hot_rows live in VRAM, the rest in pinned host RAM (EXPERTS = all hot)
  const uint8_t* woa;
  const uint8_t* woa_s;
  const uint8_t* wob;
  const uint8_t* wob_s;
  const uint8_t* gate;
  const float* gate_b;
  const float* bias_vl;
  const uint8_t* sh13;
  const uint8_t* sh13_s;
  const uint8_t* sh2;
  const uint8_t* sh2_s;
  const uint4* w13;
  const unsigned* s13;
  const uint4* w2;
  const unsigned* s2;
  long w13_stride;
  long s13_stride;
  long w2_stride;
  long s2_stride;
  const float* fn;
  const float* hc_scale;
  const float* hc_base;
  const __nv_bfloat16* norm;
#if BRRR_STEER
  // Steering constants (set_steering; null = the layer is not steered): unit direction d [HID] bf16, attention row
  // gain minus one (g - 1) [HID] bf16, u = W2_rank^T d [INTER] fp32 (this rank's dequantized shared down-projection
  // shard), alpha.
  const __nv_bfloat16* steer_dir;
  const __nv_bfloat16* steer_gain;
  const float* steer_u;
  float steer_alpha;
#endif
};

struct Params {
  const bool* pad;  // TP3 route log: vLLM's is_padding mask (persistent buffer; nullptr = count every row)
  unsigned long long* prof;  // TP3 debug: per job type [acquire wait, dependency wait, run, count] in ns (nullptr = off),
                             // then PROF_PHASES phase-mark rows (see prof_mark / prof_read)
  int test;  // TP3 debug: 1 = add CSF-style per-stage barriers on the raw path; 2 = CSF expands into sm.sx instead of in place
  int attn_per_chunk_flags;  // TP3: one-shot (m < TWO_SHOT_MIN) attention exchange posts a flag per chunk like the FFN exchange (0 = one flag for all chunks)
  Peers peers;
  Scratch s;
  Weights w;
  int rank;
  int m;
  const __nv_bfloat16* o;
  long o_stride;
  const long* positions;
  const void* input_ids;
  int ids64;
  long image_lo;
  const float* cos_sin;
  const __nv_bfloat16* resid0;
  float* attn_post;
  float* attn_comb;
  float* attn_pre;
  __nv_bfloat16* y_ffn;
  __nv_bfloat16* resid1;
  float* ffn_post;
  float* ffn_comb;
  float* ffn_pre;
#if BRRR_STEER
  const float* steer_scale;  // per-token steering scale [m] fp32 (run_steered on a steered layer), else null
#endif
};

struct Smem {
  __align__(128) uint8_t ring[STAGES][STAGE];
  unsigned long long full[STAGES];
  unsigned long long empty[STAGES];
  Job job[STAGES];
  unsigned seen[CACHED];
  __align__(16) uint8_t bt[BT];
  __align__(16) uint8_t bs[BS];
  __align__(16) unsigned sx[128];  // TP3 debug (test mode 2): out-of-place expanded scale words
  float rs[OTOK];
  int routes[SPAN];
  float cand_key[GW * TOPK];
  float cand_score[GW * TOPK];
  int cand_expert[GW * TOPK];
  int flag;
  int last;
  int ready;
#if BRRR_STEER
  int steer;  // some row of this launch has a nonzero steering scale (written once per CTA before the job loop)
#endif
};

#ifndef BRRR_SPIN_NS
#define BRRR_SPIN_NS 200
#endif
constexpr unsigned SPIN_NS = BRRR_SPIN_NS;
static_assert(TWO_SHOT_MIN >= 1 && TWO_SHOT_MIN <= MAXM + 1 && BRRR_SPIN_NS >= 0 && BRRR_SPIN_NS <= 1000000, "BRRR tuning switch values");

// TP3 debug (DS41_FUSED_PROF=1 only): per-launch phase marks after the 16 x 4 accumulators, in globaltimer ns.
// Row t < J_END (job type): [first acquire, first run start, last run end, last run start]; row PH_FIN: [first C_FIN
// arrival, last C_FIN arrival, counter reset done, 0]; row PH_KERNEL: [first CTA entry, last CTA entry, first consumer
// exit, last consumer exit]. "first" values are stored bit-inverted so one atomicMax keeps the minimum against the
// zeroed buffer; prof_read() returns them decoded (0 = never recorded). Read once per launch (prof_read resets).
constexpr int PROF_ACC = 16 * 4, PROF_PHASES = 16 * 4, PH_FIN = 14, PH_KERNEL = 15;

__device__ __forceinline__ unsigned long long globaltimer() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ void prof_first(unsigned long long* prof, int row, int col, unsigned long long t) {
  atomicMax(prof + PROF_ACC + row * 4 + col, ~t);
}

__device__ __forceinline__ void prof_last(unsigned long long* prof, int row, int col, unsigned long long t) {
  atomicMax(prof + PROF_ACC + row * 4 + col, t);
}

__device__ __forceinline__ void mbar_init(unsigned long long* bar, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(unsigned long long* bar) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(unsigned long long* bar, unsigned parity) {
  asm volatile(
      "{\n .reg .pred p;\n LAB_WAIT:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra LAB_WAIT;\n}" ::"r"(smem_u32(bar)),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void cp_async16_hint(void* dst, const void* src, unsigned long long policy) {
  asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;" ::"r"(smem_u32(dst)), "l"(src), "l"(policy) : "memory");
}

__device__ __forceinline__ unsigned long long evict_first_policy() {
  unsigned long long policy;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(policy));
  return policy;
}

__device__ __forceinline__ void cp_async8_hint(void* dst, const void* src, unsigned long long policy) {
  asm volatile("cp.async.ca.shared.global.L2::cache_hint [%0], [%1], 8, %2;" ::"r"(smem_u32(dst)), "l"(src), "l"(policy) : "memory");
}

__device__ __forceinline__ void cp_async_arrive(unsigned long long* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16(x)); }

#if BRRR_STEER
// E4M3 byte (as written by quant_byte: satfinite, never NaN) -> its exact float value.
__device__ __forceinline__ float e4m3_value(unsigned b) {
  const unsigned e = (b >> 3) & 15u, m = b & 7u;
  const float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : float(m) * 0x1p-9f;
  return (b & 0x80u) ? -v : v;
}

// One fixed-point contribution to a per-token steering dot (same 2^32 scale and rounding as the gate accumulator).
__device__ __forceinline__ void steer_dot_add(long long* dot, float part) {
  atomicAdd(reinterpret_cast<unsigned long long*>(dot), static_cast<unsigned long long>(__float2ll_rn(part * FIXED)));
}

__device__ __forceinline__ float steer_dot(const long long* dot) { return __ll2float_rn(__ldcg(dot)) / FIXED; }

// PR #404 projection of one element with the reference's operation order and roundings (eager PyTorch, FP32, no FMA
// contraction): x - ((s * alpha) * dot) * d, times (1 + s * (g - 1)) on the attention site; the caller rounds to BF16.
__device__ __forceinline__ float steer_project(float x, float t, float d) { return __fsub_rn(x, __fmul_rn(t, d)); }
#endif

__device__ __forceinline__ float lane_sums(float (&v)[32], int lane) {
#pragma unroll
  for (int w = 16; w >= 1; w >>= 1) {
    const bool hi = lane & w;
#pragma unroll
    for (int i = 0; i < w; ++i) {
      const float keep = hi ? v[i + w] : v[i], send = hi ? v[i] : v[i + w];
      v[i] = keep + __shfl_xor_sync(0xffffffffu, send, w);
    }
  }
  return v[0];
}

__device__ __forceinline__ void mma_e4m3x(int q, float (&d)[4], uint4 v, uint2 b, unsigned sa, unsigned sb) {
  switch (q) {
    case 0: mma_mx<0, 0, 0>(d, v.x, v.z, v.y, v.w, b.x, b.y, sa, sb); break;
    case 1: mma_mx<1, 0, 1>(d, v.x, v.z, v.y, v.w, b.x, b.y, sa, sb); break;
    case 2: mma_mx<2, 0, 2>(d, v.x, v.z, v.y, v.w, b.x, b.y, sa, sb); break;
    default: mma_mx<3, 0, 3>(d, v.x, v.z, v.y, v.w, b.x, b.y, sa, sb); break;
  }
}

__device__ __forceinline__ unsigned bf16x2_f16x2(unsigned v) {
  unsigned h;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h) : "f"(__uint_as_float(v & 0xffff0000u)), "f"(__uint_as_float(v << 16)));
  return h;
}

__device__ __forceinline__ unsigned fp4_lo(unsigned w) { return ((w << 4) & 0x80808080u) | ((w << 2) & 0x1c1c1c1cu); }

__device__ __forceinline__ unsigned fp4_hi(unsigned w) { return (w & 0x80808080u) | ((w >> 2) & 0x1c1c1c1cu); }

template <int BA>
__device__ __forceinline__ void mma_fp4x32(float (&t0)[4], float (&t1)[4], uint4 w, unsigned b0, unsigned b1, unsigned sa, unsigned sb) {
  mma_mx<BA, 0, BA>(t0, fp4_lo(w.x), fp4_lo(w.y), fp4_hi(w.x), fp4_hi(w.y), b0, b1, sa, sb);
  mma_mx<BA, 1, BA>(t1, fp4_lo(w.z), fp4_lo(w.w), fp4_hi(w.z), fp4_hi(w.w), b0, b1, sa, sb);
}

__device__ __forceinline__ void mma_fp4(int q, float (&t0)[4], float (&t1)[4], uint4 w, unsigned b0, unsigned b1, unsigned sa, unsigned sb) {
  switch (q) {
    case 0: mma_fp4x32<0>(t0, t1, w, b0, b1, sa, sb); break;
    case 1: mma_fp4x32<1>(t0, t1, w, b0, b1, sa, sb); break;
    case 2: mma_fp4x32<2>(t0, t1, w, b0, b1, sa, sb); break;
    default: mma_fp4x32<3>(t0, t1, w, b0, b1, sa, sb); break;
  }
}

__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ float swiglu(float gate, float up) {
  gate = fminf(gate, LIMIT);
  up = fminf(fmaxf(up, -LIMIT), LIMIT);
  return gate / (1.0f + expf(-gate)) * up;
}


struct Ctx {
  const Params& p;
  Smem& sm;
  Layout L;
  int warp;
  int lane;
  int tid;
  int it;
  unsigned epoch;
  __device__ __forceinline__ unsigned* ctr() const { return p.s.counters; }

  __device__ __forceinline__ void sync() const { asm volatile("bar.sync 1, %0;" ::"n"(GT) : "memory"); }

  __device__ void wait(int index, unsigned target) {
    if (tid == 0 && (index >= CACHED || sm.seen[index] < target)) {
      unsigned seen = ld_acquire_gpu(ctr() + index);
#if BRRR_SPIN_TRAP
      long long polls = 0;
#endif
      while (seen < target) {
        __nanosleep(SPIN_NS);
        seen = ld_acquire_gpu(ctr() + index);
#if BRRR_SPIN_TRAP
        if (++polls == SPIN_TRAP_POLLS) spin_trap("counter wait", index, target, seen);
#endif
      }
      if (index < CACHED) sm.seen[index] = seen;
    }
    sync();
  }

  __device__ void signal(int index) {
    sync();
    if (tid == 0) {
      red_release(ctr() + index, 1u);
    }
  }

  __device__ bool arrive(int index, unsigned target) {
    sync();
    if (tid == 0) {
      sm.flag = atom_add_acq_rel(ctr() + index, 1u) == target - 1;
    }
    sync();
    return sm.flag;
  }

  __device__ __forceinline__ const uint8_t* stage() const { return sm.ring[it % STAGES]; }

  __device__ __forceinline__ void full(int k) const {
    unsigned long long* bar = &sm.full[k % STAGES];
    const unsigned parity = (k / STAGES) & 1;
    mbar_wait(bar, parity);
  }

  __device__ __forceinline__ void acquire() const { full(it); }

  __device__ __forceinline__ void release() {
    __syncwarp();
    if (lane == 0) mbar_arrive(&sm.empty[it % STAGES]);
    ++it;
  }

  __device__ __forceinline__ void next() {
    release();
    acquire();
  }

  __device__ __forceinline__ const uint8_t* held(int s) const {
    full(it + s);
    return sm.ring[(it + s) % STAGES];
  }

  __device__ __forceinline__ void drop(int stages) {
    for (int s = 0; s < stages; ++s) release();
  }
};

__device__ __forceinline__ const unsigned* stage_words(const uint8_t* stage) { return reinterpret_cast<const unsigned*>(stage + SCALE_OFF); }

struct Acc8 {
  float4 lo, hi;
};

__device__ __forceinline__ Acc8 load8(const float* src, bool live) {
  const float4* a = reinterpret_cast<const float4*>(src);
  const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  return {live ? __ldcg(a) : zero, live ? __ldcg(a + 1) : zero};
}

__device__ __forceinline__ void unpack8(const Acc8& a, float (&f)[8]) {
  f[0] = a.lo.x, f[1] = a.lo.y, f[2] = a.lo.z, f[3] = a.lo.w, f[4] = a.hi.x, f[5] = a.hi.y, f[6] = a.hi.z, f[7] = a.hi.w;
}

__device__ __forceinline__ void clear8(float* dst) {
  float4* a = reinterpret_cast<float4*>(dst);
  const float4 zero = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  __stcg(a, zero);
  __stcg(a + 1, zero);
}

__device__ __forceinline__ long chunk_off(int i, int k) { return long(i / CHUNK_VECS) * HID + k * CHUNK_COLS + (i % CHUNK_VECS) * 8; }

__device__ __forceinline__ __nv_bfloat16* inbox(const Params& p, int r, int which, int stage, unsigned epoch) {
  return p.peers.pub[r] + ((long(which) * 2 + stage) * 2 + (epoch & 1)) * PUB_SLOT;
}

__device__ __forceinline__ int flag_index(int which, int stage, int src, int g) { return ((which * 2 + stage) * RANKS + src) * CHUNKS + g; }

__device__ void post(Ctx& c, int which, int stage, int g, unsigned target, unsigned epoch, unsigned mask) {
  if (target > 1) {
    if (!c.arrive(C_PUB + (which * 2 + stage) * CHUNKS + g, target)) return;
  } else {
    c.sync();
  }
  if (c.tid == 0) {
    __threadfence_system();
    for (int r = 0; r < RANKS; ++r)
      if (mask >> r & 1) st_relaxed_sys(c.p.peers.flags[r] + flag_index(which, stage, c.p.rank, g), epoch);
  }
}

__device__ void await(Ctx& c, int which, int stage, int g, unsigned epoch, unsigned mask) {
  if (c.tid < RANKS && (mask >> c.tid & 1)) {
    const unsigned* flag = c.p.peers.flags[c.p.rank] + flag_index(which, stage, c.tid, g);
#if BRRR_SPIN_TRAP
    long long polls = 0;
    while (before(ld_acquire_sys(flag), epoch)) {
      __nanosleep(SPIN_NS);
      if (++polls == SPIN_TRAP_POLLS) spin_trap("peer flag (rank = index)", c.tid, epoch, ld_acquire_sys(flag));
    }
#else
    while (before(ld_acquire_sys(flag), epoch)) __nanosleep(SPIN_NS);
#endif
  }
  c.sync();
}

constexpr int XB = 4;

__device__ void reduce4(Ctx& c, int which, int k, unsigned epoch, int nv, __nv_bfloat16* out, __nv_bfloat16* keep) {
  const Params& p = c.p;
  const __nv_bfloat16* src[RANKS];
#pragma unroll
  for (int r = 0; r < RANKS; ++r) src[r] = inbox(p, r, which, 0, epoch);
  for (int base = 0; base < nv; base += GT * XB) {
    uint4 v[XB][RANKS];
#pragma unroll
    for (int u = 0; u < XB; ++u) {
      const int i = base + c.tid + u * GT;
#pragma unroll
      for (int r = 0; r < RANKS; ++r)
        v[u][r] = i >= nv ? make_uint4(0, 0, 0, 0)
                  : r == p.rank ? __ldcg(reinterpret_cast<const uint4*>(src[r] + chunk_off(i, k)))
                                : __ldcv(reinterpret_cast<const uint4*>(src[r] + chunk_off(i, k)));
    }
#pragma unroll
    for (int u = 0; u < XB; ++u) {
      const int i = base + c.tid + u * GT;
      if (i >= nv) continue;
      float acc[8] = {};
#pragma unroll
      for (int r = 0; r < RANKS; ++r) add8(acc, v[u][r]);
      const uint4 sum = pack8(acc);
      *reinterpret_cast<uint4*>(out + chunk_off(i, k)) = sum;
      if (keep) *reinterpret_cast<uint4*>(keep + chunk_off(i, k)) = sum;
    }
  }
}

__device__ void gather(Ctx& c, const __nv_bfloat16* src, int k, int nv, __nv_bfloat16* out) {
  constexpr int GB = 2 * XB;
  for (int base = 0; base < nv; base += GT * GB) {
    uint4 v[GB];
#pragma unroll
    for (int u = 0; u < GB; ++u) {
      const int i = base + c.tid + u * GT;
      v[u] = i < nv ? __ldcv(reinterpret_cast<const uint4*>(src + chunk_off(i, k))) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int u = 0; u < GB; ++u) {
      const int i = base + c.tid + u * GT;
      if (i < nv) *reinterpret_cast<uint4*>(out + chunk_off(i, k)) = v[u];
    }
  }
}

__device__ void mix_finish(int lane, int tok, float mine, const float* scale, const float* base, float* pre, float* post, float* comb) {
  const float sq = __shfl_sync(0xffffffffu, mine, MIX);
  const float rs = rsqrtf(sq / HCD + NORM_EPS);
  const float mix = mine * rs;
  if (lane < HC) {
    pre[tok * HC + lane] = sigmoid(mix * __ldg(scale) + __ldg(base + lane)) + HC_EPS;
  } else if (lane < 2 * HC) {
    post[tok * HC + lane - HC] = 2.0f * sigmoid(mix * __ldg(scale + 1) + __ldg(base + lane));
  }
  const int idx = 2 * HC + (lane & 15);
  float cm = __shfl_sync(0xffffffffu, mix, idx) * scale[2] + base[idx];
  cm = expf(cm - quad_max(cm));
  cm = cm / quad_sum(cm) + HC_EPS;
  for (int it = 0; it < SINKHORN; ++it) {
    if (it > 0) cm /= quad_sum(cm) + HC_EPS;
    float col = cm + __shfl_xor_sync(0xffffffffu, cm, 4);
    col += __shfl_xor_sync(0xffffffffu, col, 8);
    cm /= col + HC_EPS;
  }
  if (lane < HC * HC) comb[tok * HC * HC + lane] = cm;
}

__device__ void share(Ctx& c, int which, int k, __nv_bfloat16* out) {
  const Params& p = c.p;
  const unsigned epoch = c.epoch;
  const int nv = p.m * CHUNK_VECS;
  const unsigned peers = ((1u << RANKS) - 1) & ~(1u << p.rank);
  // Attention one-shot exchange: with attn_per_chunk_flags every WOB chunk publishes its own flag (same protocol as the
  // FFN exchange, flag g = k), so a peer's OWNA slice of chunk k only waits for chunk k instead of all CHUNKS; the legacy
  // single flag (per = CHUNKS, g = 0) is the last chunk's CTA posting once for all. Flags stay within flag_index's
  // [0, 2 * 2 * RANKS * CHUNKS) words; the epoch compare in await() is unchanged.
  const int per = which == 0 && p.m < TWO_SHOT_MIN && !p.attn_per_chunk_flags ? CHUNKS : 1, g = k / per;
  if (p.m < TWO_SHOT_MIN) {
    post(c, which, 0, g, per, epoch, peers);
    await(c, which, 0, g, epoch, peers);
    reduce4(c, which, k, epoch, nv, out, nullptr);
    return;
  }
  const int owner = k % RANKS;
  post(c, which, 0, g, per, epoch, per == 1 ? 1u << owner : peers);
  if (p.rank != owner) {
    await(c, which, 1, g, epoch, 1u << owner);
    gather(c, inbox(p, owner, which, 1, epoch), k, nv, out);
    return;
  }
  await(c, which, 0, g, epoch, peers);
  reduce4(c, which, k, epoch, nv, out, inbox(p, p.rank, which, 1, epoch));
  post(c, which, 1, g, per == 1 ? 1 : per / RANKS, epoch, peers);
}

__device__ __forceinline__ void red4(float* dst, float4 v) {
  asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" ::"l"(dst), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

__device__ __forceinline__ void red4(long long* dst, float4 v) {
  asm volatile("red.global.add.u64 [%0], %1;\n\tred.global.add.u64 [%0+8], %2;\n\tred.global.add.u64 [%0+16], %3;\n\tred.global.add.u64 [%0+24], %4;" ::"l"(dst),
               "l"(__float2ll_rn(v.x * FIXED)), "l"(__float2ll_rn(v.y * FIXED)), "l"(__float2ll_rn(v.z * FIXED)), "l"(__float2ll_rn(v.w * FIXED))
               : "memory");
}

__device__ __forceinline__ float4 transpose4(const float* a, int lane) {
  const int i = (lane >> 2) & 3;
  const bool p = i & 1, q = i & 2;
  const float r0 = __shfl_xor_sync(0xffffffffu, p ? a[0] : a[1], 4), r1 = __shfl_xor_sync(0xffffffffu, p ? a[2] : a[3], 4);
  const float b0 = p ? r0 : a[0], b1 = p ? a[1] : r0, b2 = p ? r1 : a[2], b3 = p ? a[3] : r1;
  const float w0 = __shfl_xor_sync(0xffffffffu, q ? b0 : b2, 8), w1 = __shfl_xor_sync(0xffffffffu, q ? b1 : b3, 8);
  return q ? make_float4(w0, w1, b2, b3) : make_float4(b0, b1, w0, w1);
}

__device__ __forceinline__ int quad_col(int tt, int lane) { return tt * TILE + 2 * (lane & 3) + ((lane >> 2) & 1); }

__device__ __forceinline__ int quad_row(int lane) { return (lane >> 4) * 4 + ((lane >> 3) & 1) * 8; }

template <class T>
__device__ __forceinline__ void scatter(T* dst, int ld, int row0, const Span& sp, int tt, const float (&a)[4], int lane) {
  const float4 v = transpose4(a, lane);
  const int tok = quad_col(tt, lane);
  if (tok < sp.count) red4(dst + long(sp.first + tok) * ld + row0 + quad_row(lane), v);
}

__device__ __forceinline__ uint4 frag(const uint8_t* stage, int f, int lane) { return reinterpret_cast<const uint4*>(stage + f * FRAG)[lane]; }

struct Bq {
  uint2 x;
  unsigned s;
};

__device__ __forceinline__ uint8_t* bt_row(Ctx& c, int tok, int ks, int k, int bytes) { return c.sm.bt + ((tok >> 3) * ks + k) * bytes * 8 + (tok & 7) * bytes; }

__device__ __forceinline__ Bq load_q(const Ctx& c, int ks, int kw, int tt, int k) {
  return {reinterpret_cast<const uint2*>(c.sm.bt + (tt * ks + k) * 256)[c.lane], reinterpret_cast<const unsigned*>(c.sm.bs)[(tt * TILE + (c.lane >> 2)) * kw + (k >> 2)]};
}

template <class Row, class Scale>
__device__ __forceinline__ void stage_q(Ctx& c, int count, int ks, int kw, Row row, Scale scale) {
  for (int tok = c.warp; tok < count; tok += GW) {
    const uint8_t* src = row(tok);
    for (int r = c.lane; r < ks * 2; r += 32) cp_async16(bt_row(c, tok, ks, r >> 1, 32) + (r & 1) * 16, src + r * 16);
    if (c.lane < kw) cp_async4(c.sm.bs + (tok * kw + c.lane) * 4, scale(tok) + c.lane * 4);
  }
  cp_async_wait_all();
  c.sync();
}

template <class Row, class Scale>
__device__ __forceinline__ void fill_q(Ctx& c, int count, int ks, int kw, Row row, Scale scale) {
  c.sync();
  stage_q(c, count, ks, kw, row, scale);
}

struct NoPrep {
  __device__ __forceinline__ void operator()(const uint8_t*, int) const {}
};

template <int A, bool HOLD = false, class Load, class Step, class Prep = NoPrep>
__device__ __forceinline__ void gemm(Ctx& c, const Span& sp, int ksteps, float (&acc)[GW][A], Load load, Step step, Prep prep = Prep{}) {
  const int rs = GW / sp.ts, rt0 = c.warp / sp.ts, tt = c.warp % sp.ts;
  const bool live = tt < sp.tiles;
#pragma unroll
  for (int r = 0; r < GW; ++r)
#pragma unroll
    for (int e = 0; e < A; ++e) acc[r][e] = 0.0f;
  const int stages = (ksteps + 3) >> 2;
#pragma unroll 1
  for (int s = 0; s < stages; ++s) {
    if (!HOLD && s) c.next();
    const uint8_t* stage = HOLD ? c.held(s) : c.stage();
    if constexpr (!std::is_same<Prep, NoPrep>::value) {
      prep(stage, s);  // TP3 CSF: expand this stage's compressed scales in place (all GT threads), then sync
      c.sync();
    }
    if (live) {
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        if (s * 4 + q < ksteps) {
          const auto b = load(tt, s * 4 + q);
#pragma unroll
          for (int r = 0; r < GW; ++r)
            if (r < sp.ts) step(acc[r], stage, q, rt0 + r * rs, b, s);
        }
      }
    }
  }
  if (!HOLD) c.release();
}

template <int A, class Store>
__device__ __forceinline__ void stores(const Ctx& c, const Span& sp, const float (&acc)[GW][A], Store store) {
  const int rs = GW / sp.ts, rt0 = c.warp / sp.ts, tt = c.warp % sp.ts;
  if (tt >= sp.tiles) return;
#pragma unroll
  for (int r = 0; r < GW; ++r)
    if (r < sp.ts) store(rt0 + r * rs, tt, acc[r]);
}

template <class Store>
__device__ __forceinline__ void dense_q(Ctx& c, const Span& sp, int ks, int kw, Store store) {
  float acc[GW][4];
  gemm<4>(
      c, sp, ks, acc, [&](int tt, int k) { return load_q(c, ks, kw, tt, k); },
      [&](float (&a)[4], const uint8_t* stage, int q, int rt, const Bq& b, int) { mma_e4m3x(q, a, frag(stage, q * 4 + rt, c.lane), b.x, stage_words(stage)[rt >> 1], b.s); });
  stores(c, sp, acc, store);
}

#if BRRR_ORDERED
// Test build: dense_q whose stores (red.add into a k-split accumulator) wait until counter `index` reaches `part`.
template <class Store>
__device__ __forceinline__ void dense_q_ordered(Ctx& c, const Span& sp, int ks, int kw, Store store, int index, int part) {
  float acc[GW][4];
  gemm<4>(
      c, sp, ks, acc, [&](int tt, int k) { return load_q(c, ks, kw, tt, k); },
      [&](float (&a)[4], const uint8_t* stage, int q, int rt, const Bq& b, int) { mma_e4m3x(q, a, frag(stage, q * 4 + rt, c.lane), b.x, stage_words(stage)[rt >> 1], b.s); });
  c.wait(index, part);
  stores(c, sp, acc, store);
}
#endif

__device__ __forceinline__ float rot_add(float a, float b, float2 cs) { return a * cs.x + b * cs.y; }

__device__ __forceinline__ float rot_sub(float a, float b, float2 cs) { return a * cs.x - b * cs.y; }

// Flare port: RoPE angles from the 32 frequencies Flare uses (angle = float(pos) * freq[pair]) instead of a
// [positions, 64] cos/sin table, which would cost 256 MB per GPU at 1M positions.
__device__ __forceinline__ uint4 rope_freq(uint4 v, const float* freq, long pos, int d) {
  if (d < HDIM - ROPE) return v;
  unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float sn, cs;
    sincosf(float(pos) * __ldg(freq + ((d + 2 * q - (HDIM - ROPE)) >> 1)), &sn, &cs);
    const float2 r = make_float2(cs, sn);
    const float2 x = unpack_bf16(u[q]);
    u[q] = pack_bf16(rot_add(x.x, x.y, r), rot_sub(x.y, x.x, r));
  }
  return make_uint4(u[0], u[1], u[2], u[3]);
}

__device__ void group_rs(Ctx& c, const Span& sp, float* rs) {
  constexpr int V = SLICES / 4, QMIN = GT / OTOK;
  const int q = sp.count <= GW ? 32 : sp.count <= 2 * GW ? 16 : sp.count <= 4 * GW ? 8 : QMIN;
  const int i = c.tid / q, l = c.tid % q;
  float v = 0.0f;
  if (i < sp.count) {
    const float4* x = reinterpret_cast<const float4*>(c.p.s.xsq + (sp.first + i) * SLICES);
#pragma unroll
    for (int k = 0; k < cdiv(V, QMIN); ++k)
      if (l + k * q < V) {
        const float4 f = __ldcg(x + l + k * q);
        v += (f.x + f.y) + (f.z + f.w);
      }
  }
  for (int o = q / 2; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  if (l == 0 && i < sp.count) rs[i] = rsqrtf(v / HID + NORM_EPS);
  c.sync();
}

template <int PER>
__device__ __forceinline__ void select_topk(const float (&score)[PER], float (&key)[PER], const int (&expert)[PER], int lane, float& chosen_key, float& chosen_score, int& chosen_expert) {
  for (int k = 0; k < TOPK; ++k) {
    float best = -INFINITY;
    int bi = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (key[i] > best) best = key[i], bi = i;
    int be_local = 0;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (i == bi) be_local = expert[i];
    const unsigned ord = float_order(best);
    const unsigned top = __reduce_max_sync(0xffffffffu, ord);
    const int be = int(__reduce_min_sync(0xffffffffu, ord == top ? unsigned(be_local) : 0xffffffffu));
    const unsigned owner = __ffs(__ballot_sync(0xffffffffu, ord == top && be_local == be)) - 1;
    float sv = 0.0f;
#pragma unroll
    for (int i = 0; i < PER; ++i)
      if (i == bi) sv = score[i];
    sv = __shfl_sync(0xffffffffu, sv, owner);
    const float kv = __shfl_sync(0xffffffffu, best, owner);
    if (lane == int(owner))
#pragma unroll
      for (int i = 0; i < PER; ++i)
        if (i == bi) key[i] = -INFINITY;
    if (lane == k) chosen_key = kv, chosen_score = sv, chosen_expert = be;
  }
}


__device__ void job_route(Ctx& c, int tok) {
  const Params& p = c.p;
  constexpr int EW = EXPERTS / GW, PER = EW / 32;
  c.release();
  const long id = !p.input_ids ? -1 : p.ids64 ? __ldg(static_cast<const long*>(p.input_ids) + tok) : long(__ldg(static_cast<const int*>(p.input_ids) + tok));
  const float* bias = p.w.bias_vl && id >= p.image_lo && id < p.image_lo + 5 ? p.w.bias_vl : p.w.gate_b;
  float score[PER], key[PER];
  int expert[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    expert[i] = c.warp * EW + c.lane + 32 * i;
    long long* a = p.s.gacc + tok * EXPERTS + expert[i];
    const float v = __ll2float_rn(__ldcg(a)) / FIXED;
    __stcg(a, 0LL);
    score[i] = sqrtf(v > 20.0f ? v : logf(1.0f + expf(v)));
    key[i] = score[i] + __ldg(bias + expert[i]);
    if (key[i] != key[i]) key[i] = -1e30f;
  }
  float kk = -INFINITY, ks = 0.0f;
  int ke = 0;
  select_topk(score, key, expert, c.lane, kk, ks, ke);
  if (c.lane < TOPK) {
    c.sm.cand_key[c.warp * TOPK + c.lane] = kk;
    c.sm.cand_score[c.warp * TOPK + c.lane] = ks;
    c.sm.cand_expert[c.warp * TOPK + c.lane] = ke;
  }
  c.sync();
  if (c.warp == 0) {
    const bool live = c.lane < GW * TOPK;
    const float cand[1] = {live ? c.sm.cand_score[c.lane] : 0.0f};
    float ckeys[1] = {live ? c.sm.cand_key[c.lane] : -INFINITY};
    const int cexp[1] = {live ? c.sm.cand_expert[c.lane] : EXPERTS + c.lane};
    float k2 = -INFINITY, s2 = 0.0f;
    int e2 = 0;
    select_topk(cand, ckeys, cexp, c.lane, k2, s2, e2);
    float total = 0.0f;
#pragma unroll
    for (int k = 0; k < TOPK; ++k) total += __shfl_sync(0xffffffffu, s2, k);
    if (c.lane < TOPK) {
      const int r = tok * TOPK + c.lane;
      p.s.topk_w[r] = s2 * (ROUTE_SCALE / (total > 0.0f ? total : 1.0f));
      Table* tb = p.s.table;
      // Flare merged bank: routing and bias use the global ID; weights, scales and CSF records live at the bank slot.
      const int slot = p.w.hot_map ? __ldg(p.w.hot_map + e2) : e2;
      const int pos = int(atomicAdd(c.ctr() + C_ECOUNT + slot, 1u));
      if (p.w.route_counts && !(p.pad && p.pad[tok])) atomicAdd(p.w.route_counts + e2, 1u);  // skip CUDA-graph padding rows
      tb->routes[slot * MAXM + pos] = r;
      if (pos % SPAN == 0) {
        // Cold-first: entries of host-RAM slots fill the table from the front, VRAM slots from the back, so the
        // producer issues cold R1/R2 jobs (PCIe reads) before hot ones. Accumulation targets (hacc/yracc) are
        // reached through red.global.add already, so the order of experts changes only the atomic summation order.
        const bool cold = slot >= p.w.hot_rows;
        const int ent = cold ? int(atomicAdd(c.ctr() + C_NCOLD, 1u)) : ENTRIES - 1 - int(atomicAdd(c.ctr() + C_NHOT, 1u));
        tb->expert[ent] = slot;
        tb->chunk[ent] = pos / SPAN;
      }
    }
  }
  __threadfence();
  c.signal(C_ROUTE);
}

__device__ void finish_woa(Ctx& c, int rb, int tg) {
  constexpr int PER = OTOK * 8 / GT;
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  float z[PER][8];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT, tok = sp.first + i / 8;
    float v[8];
    unpack8(load8(p.s.zacc + long(tok) * OA + rb * 64 + i % 8 * 8, i / 8 < sp.count), v);
#pragma unroll
    for (int e = 0; e < 8; ++e) z[u][e] = bf(v[e]);
  }
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT, tok = sp.first + i / 8, col = rb * 64 + i % 8 * 8;
    const int e = block_exp(z[u], AMAX_FLOOR);
    if (i / 8 >= sp.count) continue;
    clear8(p.s.zacc + long(tok) * OA + col);
    *reinterpret_cast<uint2*>(p.s.zq + long(tok) * OA + col) = quant8(z[u], e);
    if (i % 4 == 0) p.s.zse[tok * (OA / 32) + col / 32] = uint8_t(e + 127);
  }
  c.signal(C_Z + rb * 2 / KS_WOB * GROUPS + tg);
}

template <int KS, int TOK>
__device__ void woa_group(Ctx& c, int rb, int kp, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  constexpr int VECS = KS * 4, PER = TOK * VECS / GT;
  static_assert(TOK * KS * 64 <= BT);
  const __nv_bfloat16* o = p.o + rb / 16 * 8 * HDIM + kp * KS * 32;
  const int r = c.tid % VECS, d = (kp * KS * 32 + r * 8) % HDIM;
  uint4 v[PER];
  long pos[PER];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int tok = (c.tid + u * GT) / VECS;
    const long t = sp.first + tok;
    v[u] = tok < sp.count ? __ldcg(reinterpret_cast<const uint4*>(o + t * p.o_stride + r * 8)) : make_uint4(0, 0, 0, 0);
    pos[u] = tok < sp.count && d >= HDIM - ROPE ? __ldg(p.positions + t) : 0;
  }
  c.sync();
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int tok = (c.tid + u * GT) / VECS;
    if (tok >= sp.count) continue;
    v[u] = rope_freq(v[u], p.cos_sin, pos[u], d);
    *reinterpret_cast<uint4*>(bt_row(c, tok, KS, r >> 2, 64) + (r & 3) * 16) = make_uint4(bf16x2_f16x2(v[u].x), bf16x2_f16x2(v[u].y), bf16x2_f16x2(v[u].z), bf16x2_f16x2(v[u].w));
  }
  c.sync();
  float acc[GW][4];
  gemm<4>(
      c, sp, KS, acc, [&](int tt, int k) { return reinterpret_cast<const uint4*>(c.sm.bt + (tt * KS + k) * 512)[c.lane]; },
      [&](float (&a)[4], const uint8_t* stage, int q, int rt, const uint4& b, int) {
        const uint4 v = frag(stage, q * 4 + rt, c.lane);
        float d[4] = {};
        mma_f16(d, e4m3x2_f16x2(v.x), e4m3x2_f16x2(v.z), e4m3x2_f16x2(v.x >> 16), e4m3x2_f16x2(v.z >> 16), b.x, b.y);
        mma_f16(d, e4m3x2_f16x2(v.y), e4m3x2_f16x2(v.w), e4m3x2_f16x2(v.y >> 16), e4m3x2_f16x2(v.w >> 16), b.z, b.w);
        const float sw = e8m0(stage[SCALE_OFF + (rt >> 1) * 4 + q]);
#pragma unroll
        for (int e = 0; e < 4; ++e) a[e] += d[e] * sw;
      });
#if BRRR_ORDERED
  c.wait(C_ZA + rb * GROUPS + tg, kp);  // test build: k parts add into zacc in part order
#endif
  stores(c, sp, acc, [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.zacc, OA, rb * 64 + rt * 16, sp, tt, a, c.lane); });
}

// BRRR_WOA_UNEVEN: one of the 7 uneven k parts of row block rb (rows <= WOA_UTOK, one token tile). Same staging and MMA
// as woa_group; the part starts at k128 stage woa_ustage0(kp) and has `stages` stages.
__device__ void woa_uneven_group(Ctx& c, int rb, int kp, int stages) {
  constexpr int KS = WOA_UKS, VECS = KS * 4, PER = WOA_UTOK * VECS / GT;
  static_assert(WOA_UTOK * VECS % GT == 0);
  const Params& p = c.p;
  const Span sp = group_of(p.m, 0);
  const int k0 = woa_ustage0(kp) * 4, ks = stages * 4;  // first k32 slice, slices in this job
  const __nv_bfloat16* o = p.o + rb / 16 * 8 * HDIM + k0 * 32;
  uint4 v[PER];
  long pos[PER];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT, tok = i / VECS, r = i % VECS, d = (k0 * 32 + r * 8) % HDIM;
    const bool live = tok < sp.count && r < ks * 4;
    const long t = sp.first + tok;
    v[u] = live ? __ldcg(reinterpret_cast<const uint4*>(o + t * p.o_stride + r * 8)) : make_uint4(0, 0, 0, 0);
    pos[u] = live && d >= HDIM - ROPE ? __ldg(p.positions + t) : 0;
  }
  c.sync();
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT, tok = i / VECS, r = i % VECS, d = (k0 * 32 + r * 8) % HDIM;
    if (tok >= sp.count || r >= ks * 4) continue;
    v[u] = rope_freq(v[u], p.cos_sin, pos[u], d);
    *reinterpret_cast<uint4*>(bt_row(c, tok, KS, r >> 2, 64) + (r & 3) * 16) = make_uint4(bf16x2_f16x2(v[u].x), bf16x2_f16x2(v[u].y), bf16x2_f16x2(v[u].z), bf16x2_f16x2(v[u].w));
  }
  c.sync();
  float acc[GW][4];
  gemm<4>(
      c, sp, ks, acc, [&](int tt, int k) { return reinterpret_cast<const uint4*>(c.sm.bt + (tt * KS + k) * 512)[c.lane]; },
      [&](float (&a)[4], const uint8_t* stage, int q, int rt, const uint4& b, int) {
        const uint4 v = frag(stage, q * 4 + rt, c.lane);
        float d[4] = {};
        mma_f16(d, e4m3x2_f16x2(v.x), e4m3x2_f16x2(v.z), e4m3x2_f16x2(v.x >> 16), e4m3x2_f16x2(v.z >> 16), b.x, b.y);
        mma_f16(d, e4m3x2_f16x2(v.y), e4m3x2_f16x2(v.w), e4m3x2_f16x2(v.y >> 16), e4m3x2_f16x2(v.w >> 16), b.z, b.w);
        const float sw = e8m0(stage[SCALE_OFF + (rt >> 1) * 4 + q]);
#pragma unroll
        for (int e = 0; e < 4; ++e) a[e] += d[e] * sw;
      });
#if BRRR_ORDERED
  c.wait(C_ZA + rb * GROUPS, kp);  // test build: k parts add into zacc in part order (one token group)
#endif
  stores(c, sp, acc, [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.zacc, OA, rb * 64 + rt * 16, sp, tt, a, c.lane); });
}

__device__ void job_woa(Ctx& c, int rb, int kp, int tg, int stages) {
  if (woa_uneven(c.p.m)) {
    woa_uneven_group(c, rb, kp, stages);
    if (c.arrive(C_ZA + rb * GROUPS + tg, WOA_UPARTS)) finish_woa(c, rb, tg);
    return;
  }
  if (stages * 4 == KS_WOA) woa_group<KS_WOA, OTOK>(c, rb, kp, tg);
  else woa_group<2 * KS_WOA, WOA_SMALL>(c, rb, kp, tg);
  if (!c.arrive(C_ZA + rb * GROUPS + tg, OAK / 32 / (stages * 4))) return;
  finish_woa(c, rb, tg);
}

__device__ void finish_wob(Ctx& c, int rb, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  __nv_bfloat16* mine = inbox(p, p.rank, 0, 0, c.epoch);
  constexpr int PER = OTOK * 8 / GT;
  Acc8 v[PER];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT;
    v[u] = load8(p.s.yacc + long(sp.first + i / 8) * HID + rb * 64 + (i % 8) * 8, i < sp.count * 8);
  }
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.tid + u * GT, tok = sp.first + i / 8, col = rb * 64 + (i % 8) * 8;
    if (i >= sp.count * 8) break;
    clear8(p.s.yacc + long(tok) * HID + col);
    float y[8];
    unpack8(v[u], y);
    *reinterpret_cast<uint4*>(mine + long(tok) * HID + col) = pack8(y);
  }
  c.signal(C_YC + rb / 2);
}

__device__ void job_wob(Ctx& c, int rb, int kp, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  fill_q(
      c, sp.count, KS_WOB, KS_WOB / 4, [&](int tok) { return p.s.zq + long(sp.first + tok) * OA + kp * KS_WOB * 32; },
      [&](int tok) { return p.s.zse + (sp.first + tok) * (OA / 32) + kp * KS_WOB; });
#if BRRR_ORDERED
  dense_q_ordered(c, sp, KS_WOB, KS_WOB / 4, [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.yacc, HID, rb * 64 + rt * 16, sp, tt, a, c.lane); },
                  C_YA + rb * GROUPS + tg, kp);
#else
  dense_q(c, sp, KS_WOB, KS_WOB / 4, [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.yacc, HID, rb * 64 + rt * 16, sp, tt, a, c.lane); });
#endif
  if (c.arrive(C_YA + rb * GROUPS + tg, KP_WOB)) finish_wob(c, rb, tg);
}


#if BRRR_STEER
// Post mode: this chunk's part of every row's dot, from the reduced BF16 values this CTA just wrote (L2), one uint4
// (8 columns) per thread: the 16 threads of a row (a half warp) sum their parts, one fixed-point add per row into adot.
__device__ void steer_xa_dot(Ctx& c, int k) {
  const Params& p = c.p;
  constexpr int RV = CHUNK_COLS / 8;  // 16 vectors per row
  static_assert(RV == 16 && GT % 32 == 0);
  const int cv = c.tid % RV;
  const uint4 dv = __ldg(reinterpret_cast<const uint4*>(p.w.steer_dir + k * CHUNK_COLS + cv * 8));
  const unsigned dw[4] = {dv.x, dv.y, dv.z, dv.w};
  for (int base = 0; base < p.m * RV; base += GT) {
    const int i = base + c.tid, tok = i / RV;
    float part = 0.0f;
    if (tok < p.m) {
      const uint4 yv = __ldcg(reinterpret_cast<const uint4*>(p.s.yattn + long(tok) * HID + k * CHUNK_COLS + cv * 8));
      const unsigned yw[4] = {yv.x, yv.y, yv.z, yv.w};
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float2 y = unpack_bf16(yw[e]), d = unpack_bf16(dw[e]);
        part += y.x * d.x + y.y * d.y;
      }
    }
    part += __shfl_xor_sync(0xffffffffu, part, 8);
    part += __shfl_xor_sync(0xffffffffu, part, 4);
    part += __shfl_xor_sync(0xffffffffu, part, 2);
    part += __shfl_xor_sync(0xffffffffu, part, 1);
    if (cv == 0 && tok < p.m) steer_dot_add(p.s.adot + tok, part);
  }
  c.signal(C_YD);
}
#endif

__device__ void job_xa(Ctx& c, int k) {
  c.release();
#if BRRR_STEER
  share(c, 0, k, c.p.s.yattn);
#else
  share(c, 0, k, c.p.s.yattn);
#endif
  c.signal(C_YR + k);
#if BRRR_STEER
  if (c.sm.steer) steer_xa_dot(c, k);
#endif
}

__device__ void job_amix(Ctx& c, int) {
  // The attention mHC mix stays in Flare (the opt-in BRRR prologue that produced it was removed, 2026-11-08);
  // the job only keeps its place in the counter protocol.
  c.release();
  c.signal(C_AMIX);
}

__device__ void owna_tile(Ctx& c, int s, int t8, const float* fn0, const float* fn1) {
  constexpr int PER = TILE / GW;
  const Params& p = c.p;
  const Span sp = tile_span(p.m, t8);
  const int col = s * 32 + c.lane;
  float ya[PER], rk[PER][HC];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.warp + u * GW, tok = sp.first + (i < sp.count ? i : 0);
    ya[u] = __bfloat162float(__ldcg(p.s.yattn + tok * HID + col));
#pragma unroll
    for (int k = 0; k < HC; ++k) rk[u][k] = __bfloat162float(__ldg(p.resid0 + (long(tok) * HC + k) * HID + col));
  }
#if BRRR_STEER
  // Steering, attention site (PR #404 steer_attention on the replicated post-all-reduce row): a row with scale 0 keeps
  // its loaded value untouched. Wait (after this tile's loads were issued) until every chunk added its part of adot.
  if (c.sm.steer) {
    c.wait(C_YD, CHUNKS);
    const float d = __bfloat162float(__ldg(p.w.steer_dir + col)), g = __bfloat162float(__ldg(p.w.steer_gain + col));
#pragma unroll
    for (int u = 0; u < PER; ++u) {
      const int i = c.warp + u * GW, tok = sp.first + i;
      if (i >= sp.count) break;
      const float s = __ldcg(p.steer_scale + tok);
      if (s != 0.0f) {
        const float t = __fmul_rn(__fmul_rn(s, p.w.steer_alpha), steer_dot(p.s.adot + tok));
        ya[u] = bf(__fmul_rn(steer_project(ya[u], t, d), __fadd_rn(1.0f, __fmul_rn(s, g))));
      }
    }
  }
#endif
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int i = c.warp + u * GW, tok = sp.first + i;
    if (i >= sp.count) break;
    float r1[HC];
    float x = 0.0f, sq = 0.0f;
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      float v = __ldg(p.attn_post + tok * HC + j) * ya[u];
#pragma unroll
      for (int k = 0; k < HC; ++k) v += __ldg(p.attn_comb + tok * HC * HC + k * HC + j) * rk[u][k];
      r1[j] = bf(v);
      p.resid1[(long(tok) * HC + j) * HID + col] = __float2bfloat16(r1[j]);
      sq += r1[j] * r1[j];
      x += __ldg(p.attn_pre + tok * HC + j) * r1[j];
    }
    static_assert(MIX + 2 <= MIXW);
    float v[32] = {};
#pragma unroll
    for (int r = 0; r < MIX; ++r)
#pragma unroll
      for (int j = 0; j < HC; ++j) v[r] += (r < OWNA_ROWS ? fn0[(r * HC + j) * 32 + c.lane] : fn1[((r - OWNA_ROWS) * HC + j) * 32 + c.lane]) * r1[j];
    x = bf(x);
    p.s.xc[tok * HID + col] = __float2bfloat16(x);
    v[MIX] = sq;
    v[MIX + 1] = x * x;
    const float total = lane_sums(v, c.lane);
    p.s.mixp[(long(tok) * SLICES + s) * MIXW + c.lane] = total;
    if (c.lane == MIX + 1) p.s.xsq[tok * SLICES + s] = total;
  }
  c.signal(C_A + t8);
}

__device__ void job_owna(Ctx& c, int s, int span) {
  const float* fn0 = reinterpret_cast<const float*>(c.stage());
  const float* fn1 = reinterpret_cast<const float*>(c.held(1));
  const int per = owna_tiles(c.p.m), last = min(tiles_of(c.p.m), (span + 1) * per);
  for (int t8 = span * per; t8 < last; ++t8) owna_tile(c, s, t8, fn0, fn1);
  c.drop(2);
}

__device__ void job_xnorm(Ctx& c, int kp, int t8) {
  constexpr int PER = TILE * 64 / GT;
  const Params& p = c.p;
  const Span sp = tile_span(p.m, t8);
  c.release();
  const int k0 = kp * 512 + c.tid % 64 * 8;
  uint4 xv[PER];
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int row = (c.tid + u * GT) / 64;
    xv[u] = row < sp.count ? __ldcg(reinterpret_cast<const uint4*>(p.s.xc + (sp.first + row) * HID + k0)) : make_uint4(0, 0, 0, 0);
  }
  float nw[8] = {};
  add8(nw, __ldg(reinterpret_cast<const uint4*>(p.w.norm + k0)));
  float* rs = c.sm.rs;
  group_rs(c, sp, rs);
#pragma unroll
  for (int u = 0; u < PER; ++u) {
    const int row = (c.tid + u * GT) / 64, tok = sp.first + row;
    float xc[8] = {}, x8[8];
    add8(xc, xv[u]);
    const float r = row < sp.count ? rs[row] : 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) x8[e] = bf(nw[e] * (xc[e] * r));
    const int ex = block_exp(x8, AMAX_FLOOR);
    if (row < sp.count) {
      *reinterpret_cast<uint2*>(p.s.xq + tok * HID + k0) = quant8(x8, ex);
      *reinterpret_cast<uint4*>(p.s.xbf + tok * HID + k0) = pack8(x8);
      if ((c.tid & 3) == 0) p.s.xse[tok * SLICES + k0 / 32] = uint8_t(ex + 127);
    }
  }
  c.signal(C_XK + kp * GROUPS + t8 * TILE / OTOK);
}

__device__ void gate_group(Ctx& c, int rb, int kp, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  const int k0 = kp * KS_GATE * 16;
  const bool split = split_norm(p.m);
  if (split) {
    c.wait(C_XK + k0 / 512 * GROUPS + tg, sp.tiles);
    stage_q(
        c, sp.count, KS_GATE, 0, [&](int tok) { return reinterpret_cast<const uint8_t*>(p.s.xbf + long(sp.first + tok) * HID + k0); },
        [](int) -> const uint8_t* { return nullptr; });
  } else {
    group_rs(c, sp, c.sm.rs);
    for (int i = c.tid; i < sp.count * (KS_GATE / 2); i += GT) {
      const int tok = i / (KS_GATE / 2), j = i % (KS_GATE / 2), k = k0 + j * 32;
      const long t = sp.first + tok;
      const float r = c.sm.rs[tok];
      float v[32];
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        float xf[8] = {}, nf[8] = {};
        add8(xf, __ldcg(reinterpret_cast<const uint4*>(p.s.xc + t * HID + k + u * 8)));
        add8(nf, __ldg(reinterpret_cast<const uint4*>(p.w.norm + k + u * 8)));
#pragma unroll
        for (int e = 0; e < 8; ++e) v[u * 8 + e] = bf(nf[e] * (xf[e] * r));
      }
      uint4 packed[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) packed[u] = pack8(reinterpret_cast<const float(&)[8]>(v[u * 8]));
      uint4* d0 = reinterpret_cast<uint4*>(bt_row(c, tok, KS_GATE, 2 * j, 32));
      uint4* d1 = reinterpret_cast<uint4*>(bt_row(c, tok, KS_GATE, 2 * j + 1, 32));
      d0[0] = packed[0];
      d0[1] = packed[1];
      d1[0] = packed[2];
      d1[1] = packed[3];
      if (rb == 0) {
        float amax = 0.0f;
#pragma unroll
        for (int e = 0; e < 32; ++e) amax = fmaxf(amax, fabsf(v[e]));
        const int ex = quant_exp(amax, AMAX_FLOOR);
        uint2* q = reinterpret_cast<uint2*>(p.s.xq + t * HID + k);
#pragma unroll
        for (int u = 0; u < 4; ++u) q[u] = quant8(reinterpret_cast<const float(&)[8]>(v[u * 8]), ex);
        p.s.xse[t * SLICES + k / 32] = uint8_t(ex + 127);
      }
    }
    c.sync();
  }
  float acc[GW][4];
  gemm<4, true>(
      c, sp, KS_GATE, acc, [&](int tt, int k) { return reinterpret_cast<const uint2*>(c.sm.bt + (tt * KS_GATE + k) * 256)[c.lane]; },
      [&](float (&a)[4], const uint8_t* stage, int q, int rt, const uint2& b, int) {
        const uint4 v = frag(stage, q * 4 + rt, c.lane);
        mma_bf16(a, v.x, v.z, v.y, v.w, b.x, b.y);
      });
  stores(c, sp, acc, [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.gacc, EXPERTS, rb * 64 + rt * 16, sp, tt, a, c.lane); });
  if (!split && rb == 0) c.signal(C_XK + k0 / 512 * GROUPS);
  c.signal(C_G + tg);
}

__device__ void job_gate(Ctx& c, int rb, int kp, int tg) {
  gate_group(c, rb, kp, tg);
  c.drop(KS_GATE / 4);
}

__device__ void finish_sh1(Ctx& c, int b, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  constexpr int PER = OTOK / GW;
  const int col = b * 32 + c.lane;
  float g[PER], u[PER];
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int t = c.warp + i * GW;
    const float* h = p.s.sacc + long(sp.first + t) * 2 * INTER + col;
    g[i] = t < sp.count ? __ldcg(h) : 0.0f;
    u[i] = t < sp.count ? __ldcg(h + INTER) : 0.0f;
  }
#pragma unroll
  for (int i = 0; i < PER; ++i) {
    const int t = c.warp + i * GW, tok = sp.first + t;
    if (t >= sp.count) break;
    float* h = p.s.sacc + long(tok) * 2 * INTER + col;
    __stcg(h, 0.0f);
    __stcg(h + INTER, 0.0f);
    const float a = bf(swiglu(bf(g[i]), bf(u[i])));
    const int e = quant_exp(warp_max(fabsf(a)), AMAX_FLOOR);
#if BRRR_STEER
    const uint8_t q = quant_byte(a, e);
    p.s.hsq[tok * INTER + col] = q;
    if (c.lane == 0) p.s.hse[tok * HQ_LD + b] = uint8_t(e + 127);
    // Steering, shared site: this rank's shared partial sh = W2_rank h (h = the E4M3 activation J_SH2 multiplies), so
    // sh . d = h . (W2_rank^T d) = h . u. The 32 columns of this block go into sdot now; J_XF reads it after C_SF,
    // which every J_SH2 of the row's group signals after C_HS (all of these blocks), so no extra wait.
    if (c.sm.steer) {
      const float part = warp_sum(e4m3_value(q) * pow2(e) * __ldg(p.w.steer_u + col));
      if (c.lane == 0) steer_dot_add(p.s.sdot + tok, part);
    }
#else
    p.s.hsq[tok * INTER + col] = quant_byte(a, e);
    if (c.lane == 0) p.s.hse[tok * HQ_LD + b] = uint8_t(e + 127);
#endif
  }
  c.signal(C_HS + tg);
}

__device__ void job_sh1(Ctx& c, int b, int kp, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  fill_q(
      c, sp.count, KS_SH1, KS_SH1 / 4, [&](int tok) { return p.s.xq + long(sp.first + tok) * HID + kp * KS_SH1 * 32; },
      [&](int tok) { return p.s.xse + (sp.first + tok) * SLICES + kp * KS_SH1; });
#if BRRR_ORDERED
  dense_q_ordered(c, sp, KS_SH1, KS_SH1 / 4,
                  [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.sacc, 2 * INTER, (rt < 2 ? b * 32 : INTER + b * 32 - 32) + rt * 16, sp, tt, a, c.lane); },
                  C_SA + b * GROUPS + tg, kp);
#else
  dense_q(c, sp, KS_SH1, KS_SH1 / 4,
          [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.sacc, 2 * INTER, (rt < 2 ? b * 32 : INTER + b * 32 - 32) + rt * 16, sp, tt, a, c.lane); });
#endif
  if (c.arrive(C_SA + b * GROUPS + tg, KP_SH1)) finish_sh1(c, b, tg);
}

__device__ void job_sh2(Ctx& c, int rb, int tg) {
  const Params& p = c.p;
  const Span sp = group_of(p.m, tg);
  fill_q(
      c, sp.count, KS_INTER, KW_INTER, [&](int tok) { return p.s.hsq + long(sp.first + tok) * INTER; }, [&](int tok) { return p.s.hse + (sp.first + tok) * HQ_LD; });
  dense_q(c, sp, KS_INTER, KW_INTER, [&](int rt, int tt, const float (&a)[4]) { scatter(p.s.shacc, HID, rb * 64 + rt * 16, sp, tt, a, c.lane); });
  c.signal(C_SF + rb / 2);
}

__device__ __forceinline__ float mix_reduce(const float* mixp, int tok, int lane) {
  constexpr int ROWS = SLICES / 4, BATCH = 10;
  const float4* src = reinterpret_cast<const float4*>(mixp + long(tok) * SLICES * MIXW) + (lane >> 3) * (MIXW / 4) + (lane & 7);
  float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
  for (int b = 0; b < ROWS; b += BATCH) {
    float4 v[BATCH];
#pragma unroll
    for (int i = 0; i < BATCH; ++i) v[i] = __ldcg(src + (b + i) * MIXW);
#pragma unroll
    for (int i = 0; i < BATCH; ++i) acc.x += v[i].x, acc.y += v[i].y, acc.z += v[i].z, acc.w += v[i].w;
  }
#pragma unroll
  for (int o = 8; o < 32; o <<= 1) {
    acc.x += __shfl_xor_sync(0xffffffffu, acc.x, o);
    acc.y += __shfl_xor_sync(0xffffffffu, acc.y, o);
    acc.z += __shfl_xor_sync(0xffffffffu, acc.z, o);
    acc.w += __shfl_xor_sync(0xffffffffu, acc.w, o);
  }
  const int from = (lane >> 2) & 7, part = lane & 3;
  const float a0 = __shfl_sync(0xffffffffu, acc.x, from), a1 = __shfl_sync(0xffffffffu, acc.y, from);
  const float a2 = __shfl_sync(0xffffffffu, acc.z, from), a3 = __shfl_sync(0xffffffffu, acc.w, from);
  return part == 0 ? a0 : part == 1 ? a1 : part == 2 ? a2 : a3;
}

__device__ void job_mix(Ctx& c, int q) {
  const Params& p = c.p;
  c.release();
  const int tok = q * GW + c.warp;
  if (tok >= p.m) return;
  mix_finish(c.lane, tok, mix_reduce(p.s.mixp, tok, c.lane), p.w.hc_scale, p.w.hc_base, p.ffn_pre, p.ffn_post, p.ffn_comb);
}

__device__ __forceinline__ float (&lo4(float (&a)[8]))[4] { return reinterpret_cast<float(&)[4]>(a[0]); }

__device__ __forceinline__ float (&hi4(float (&a)[8]))[4] { return reinterpret_cast<float(&)[4]>(a[4]); }

__device__ __forceinline__ Span entry(Ctx& c, const Job& job) {
  const int* routes = c.p.s.table->routes + job.f;
  c.sync();
  if (c.tid < job.n) c.sm.routes[c.tid] = __ldcg(routes + c.tid);
  return span_of(0, job.n);
}

// TP3 CSF: expand one stage's compressed scales (see issue_csf layout) into words[0..128): J_R1 two 64-row halves
// (word h*64 + r), J_R2 one 128-row group. r0 = first row of the staged slice within its 128-row payload group.
__device__ __forceinline__ void expand_csf(const Ctx& c, const Csf& cs, const uint8_t* stage, int halves, int r0, unsigned desc, long dbg0, long dbg1) {
  const uint8_t* area = stage + SCALE_OFF;
  unsigned* words = c.p.test == 2 ? c.sm.sx : const_cast<unsigned*>(stage_words(stage));  // in place over the compressed bytes (two phases)
  const int nrows = halves == 2 ? 64 : 128, h = c.tid / nrows, r = c.tid % nrows;
  const unsigned nib = (area[128 + h * 32 + r / 2] >> ((r & 1) * 4)) & 15u;
  unsigned spread = nib;
  spread = (spread | (spread << 14)) & 0x00030003u;
  spread = (spread | (spread << 7)) & 0x01010101u;
  unsigned word = unsigned(area[h * 64 + r]) * 0x01010101u + spread;
  const unsigned first = desc & 0x3fffffu, count = desc >> 22;  // 22-bit record index, 10-bit count (this thread's half)
  if (long(first) + count > cs.records) { printf("[csf guard] expand blk %d tid %d desc %#x first %u count %u records %ld\n", blockIdx.x, c.tid, desc, first, count, cs.records); __trap(); }
  const unsigned short* pos_p = reinterpret_cast<const unsigned short*>(cs.storage + cs.recpos) + first;
  const uint8_t* val_p = cs.storage + cs.value_off + first;
  for (unsigned i = 0; i < count; ++i) {  // exception records straight from the payload (L1/L2 broadcast; usually count == 0)
    const unsigned pos = __ldg(pos_p + i), val = __ldg(val_p + i);
    if (int(pos >> 2) - r0 == r) word = (word & ~(0xffu << ((pos & 3) * 8))) | (val << ((pos & 3) * 8));
  }
  c.sync();  // everyone has read its compressed inputs before the expanded words overwrite them
  words[h * nrows + r] = word;
  if (cs.debug) cs.debug[(h ? dbg1 : dbg0) + r] = word;
  if (cs.debug_desc && r == 0) cs.debug_desc[(h ? dbg1 : dbg0) / 128] = desc;
}

template <class Store>
__device__ __forceinline__ void routed(Ctx& c, const Span& en, int ks, int kw, Store store, const Csf* csf = nullptr, int halves = 1, int r0 = 0,
                                       long dbg0 = 0, long dbg1 = 0, int se0 = 0, int se1 = 0, int sgbase = 0) {
  const int g = c.lane >> 2, t = c.lane & 3;
  float acc[GW][8];
  unsigned descs[6] = {};  // CSF: this thread's half's record descriptors for every stage of the job, prefetched up front
  if (csf) {
    const int se = (halves == 2 && c.tid >= 64) ? se1 : se0;
    if (se < 0 || se >= csf->payload_experts || sgbase < 0 || sgbase + (ks + 3) / 4 > csf->groups) {
      printf("[csf guard] consumer blk %d tid %d type %s e=%d se=%d sgbase=%d ks=%d halves=%d n=%d\n", blockIdx.x, c.tid, halves == 2 ? "R1" : "R2", se0 / csf->sections, se, sgbase, ks, halves, en.count);
      __trap();
    }
    const unsigned* gd = csf->gdesc + long(se) * csf->groups + sgbase;
#pragma unroll
    for (int s = 0; s < 6; ++s)
      if (s * 4 < ks) descs[s] = __ldg(gd + s);
  }
  auto step = [&](float (&a)[8], const uint8_t* stage, int q, int rt, const Bq& b, int s) {
    const unsigned* words = (csf && c.p.test == 2) ? c.sm.sx : stage_words(stage);
    const int idx = rt * 32 + t * 8 + g;
    const unsigned sa = (s * 4 + 4 <= ks ? words[idx] : unsigned(reinterpret_cast<const unsigned short*>(words)[idx])) + 0x06060606u;
    mma_fp4(q, lo4(a), hi4(a), frag(stage, q * 4 + rt, c.lane), __byte_perm(b.x.x, b.x.y, 0x6420), __byte_perm(b.x.x, b.x.y, 0x7531), sa, b.s);
  };
  auto load = [&](int tt, int k) { return load_q(c, ks, kw, tt, k); };
  // descs[] stays in registers: the stage index s is a runtime loop variable in gemm(), and indexing descs[s] directly
  // turned the array into a local-memory stack slot (one LDL per routed stage). Select it with constant indices instead.
  auto desc_at = [&](int s) {
    unsigned d = descs[0];
#pragma unroll
    for (int i = 1; i < 6; ++i)
      if (s == i) d = descs[i];
    return d;
  };
  if (csf) gemm<8>(c, en, ks, acc, load, step, [&](const uint8_t* stage, int s) { expand_csf(c, *csf, stage, halves, r0, desc_at(s), dbg0 + s * 256, dbg1 + s * 256); });
  else if (c.p.test == 1) gemm<8>(c, en, ks, acc, load, step, [&](const uint8_t* stage, int) { volatile unsigned sink = stage_words(stage)[c.tid]; (void)sink; c.sync(); });
  else gemm<8>(c, en, ks, acc, load, step);
  stores(c, en, acc, [&](int rt, int tt, const float (&a)[8]) {
    const float4 lo = transpose4(a, c.lane), hi = transpose4(a + 4, c.lane);
    const int i = quad_col(tt, c.lane), row = rt * 32 + quad_row(c.lane);
    if (i < en.count) {
      store(c.sm.routes[i], row, lo);
      store(c.sm.routes[i], row + 16, hi);
    }
  });
}

__device__ void finish_r1(Ctx& c, int ei, int bb, const Span& en) {
  const Params& p = c.p;
  for (int item = c.warp; item < en.count * 2; item += GW) {
    const int route = c.sm.routes[item / 2], col = bb * 64 + (item & 1) * 32 + c.lane;
    float* h = p.s.hacc + long(route) * 2 * INTER + col;
    const float g = __ldcg(h), u = __ldcg(h + INTER);
    __stcg(h, 0.0f);
    __stcg(h + INTER, 0.0f);
    const float a = swiglu(g, u);
    const int e = quant_exp(warp_max(fabsf(a)), AMAX_FLOOR);
    p.s.hq[long(route) * INTER + col] = quant_byte(a, e);
    if (c.lane == 0) p.s.hqe[route * HQ_LD + col / 32] = uint8_t(e + 127);
  }
  c.signal(C_R1D + ei);
}

__device__ void job_r1(Ctx& c, const Job& job) {
  constexpr int KS = SLICES / R1_PARTS;
  const Params& p = c.p;
  const int ei = job.a, bb = job.b, kp = job.c;
  const Span en = entry(c, job);
  fill_q(
      c, en.count, KS, KS / 4, [&](int i) { return p.s.xq + long(c.sm.routes[i] / TOPK) * HID + kp * KS * 32; },
      [&](int i) { return p.s.xse + (c.sm.routes[i] / TOPK) * SLICES + kp * KS; });
#if BRRR_ORDERED
  c.wait(C_R1A + ei * R1_BLOCKS + bb, kp);  // test build: k parts add into hacc in part order
#endif
  routed(c, en, KS, KS / 4, [&](int route, int r, float4 v) { red4(p.s.hacc + long(route) * 2 * INTER + (r < 64 ? 0 : INTER - 64) + bb * 64 + r, v); },
         p.w.csf13.storage ? &p.w.csf13 : nullptr, 2, (bb % 2) * 64,
         ((long(job.e) * (2 * INTER / 256) + 1 * (INTER / 256) + (bb / 2) / 2) * (HID / 128) + kp * 4) * 256 + ((bb / 2) % 2) * 128 + (bb % 2) * 64,
         ((long(job.e) * (2 * INTER / 256) + 0 * (INTER / 256) + (bb / 2) / 2) * (HID / 128) + kp * 4) * 256 + ((bb / 2) % 2) * 128 + (bb % 2) * 64,
         job.e * p.w.csf13.sections + 1, job.e * p.w.csf13.sections + 0, (bb / 2) * p.w.csf13.column_groups + kp * 4);  // h=0 -> section 1 (half = 1 - h)
  if (c.arrive(C_R1A + ei * R1_BLOCKS + bb, R1_PARTS)) finish_r1(c, ei, bb, en);
}

__device__ void job_r2(Ctx& c, const Job& job) {
  const Params& p = c.p;
  const int rb = job.b;
  const Span en = entry(c, job);
  fill_q(
      c, en.count, KS_INTER, KW_INTER, [&](int i) { return p.s.hq + long(c.sm.routes[i]) * INTER; }, [&](int i) { return p.s.hqe + c.sm.routes[i] * HQ_LD; });
  routed(c, en, KS_INTER, KW_INTER, [&](int route, int r, float4 v) {
    const float w = __ldcg(p.s.topk_w + route);
#if BRRR_ORDERED
    __stcg(reinterpret_cast<float4*>(p.s.yroute + long(route) * HID + rb * 128 + r), make_float4(w * v.x, w * v.y, w * v.z, w * v.w));
#else
    red4(p.s.yracc + long(route / TOPK) * HID + rb * 128 + r, make_float4(w * v.x, w * v.y, w * v.z, w * v.w));
#endif
  }, p.w.csf2.storage ? &p.w.csf2 : nullptr, 1, 0, ((long(job.e) * (HID / 256) + rb / 2) * (INTER / 128)) * 256 + (rb % 2) * 128, 0,
         job.e, job.e, rb * p.w.csf2.column_groups);
  c.signal(C_SF + rb);
}

#if BRRR_ORDERED
// Test build: a token's routed output for 8 columns at chunk offset `off`, its TOPK route rows summed in route order.
__device__ __forceinline__ Acc8 routed_rows(const Params& p, long off, bool live) {
  const long tok = off / HID, col = off % HID;
  float sum[8] = {};
  for (int k = 0; k < TOPK && live; ++k) {
    float v[8];
    unpack8(load8(p.s.yroute + (tok * TOPK + k) * HID + col, true), v);
#pragma unroll
    for (int e = 0; e < 8; ++e) sum[e] += v[e];
  }
  return {make_float4(sum[0], sum[1], sum[2], sum[3]), make_float4(sum[4], sum[5], sum[6], sum[7])};
}
#endif

#if BRRR_STEER
// STEERED (steering builds, a launch with a nonzero scale only): the shared-site projection in the loop; the false
// instantiation is the unsteered loop. Only the loop is instantiated twice: the exchange (share) after it is one copy
// (v3 instantiated share twice, which duplicated reduce4 / gather and added spills to them).
template <bool STEERED>
__device__ void xf_mix(Ctx& c, int k, __nv_bfloat16* mine) {
  const Params& p = c.p;
  constexpr int XU = 2;
  for (int base = 0; base < p.m * CHUNK_VECS; base += GT * XU) {
    Acc8 rv[XU], sv[XU];
#pragma unroll
    for (int u = 0; u < XU; ++u) {
      const int i = base + c.tid + u * GT;
      const bool live = i < p.m * CHUNK_VECS;
      const long off = chunk_off(live ? i : 0, k);
#if BRRR_ORDERED
      rv[u] = routed_rows(p, off, live);
#else
      rv[u] = load8(p.s.yracc + off, live);
#endif
      sv[u] = load8(p.s.shacc + off, live);
    }
#pragma unroll
    for (int u = 0; u < XU; ++u) {
      const int i = base + c.tid + u * GT;
      if (i >= p.m * CHUNK_VECS) break;
      const long off = chunk_off(i, k);
      clear8(p.s.yracc + off);
      clear8(p.s.shacc + off);
      float routed[8], shared[8], y[8];
      unpack8(rv[u], routed);
      unpack8(sv[u], shared);
#pragma unroll
      for (int e = 0; e < 8; ++e) shared[e] = bf(shared[e]);
#if BRRR_STEER
      // Steering, shared site (PR #404 steer_shared_expert on the rank-local BF16 partial before the MLP all-reduce).
      if constexpr (STEERED) {
        const int tok = i / CHUNK_VECS;
        const float s = __ldcg(p.steer_scale + tok);
        if (s != 0.0f) {
          const float t = __fmul_rn(__fmul_rn(s, p.w.steer_alpha), steer_dot(p.s.sdot + tok));
          const uint4 dv = __ldg(reinterpret_cast<const uint4*>(p.w.steer_dir + k * CHUNK_COLS + (i % CHUNK_VECS) * 8));
          const unsigned dw[4] = {dv.x, dv.y, dv.z, dv.w};
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const float2 d = unpack_bf16(dw[e / 2]);
            shared[e] = bf(steer_project(shared[e], t, e % 2 ? d.y : d.x));
          }
        }
      }
#endif
#pragma unroll
      for (int e = 0; e < 8; ++e) y[e] = bf(routed[e]) + shared[e];
      *reinterpret_cast<uint4*>(mine + off) = pack8(y);
    }
  }
}

__device__ void job_xf(Ctx& c, int k) {
  const Params& p = c.p;
  c.release();
  __nv_bfloat16* mine = inbox(p, p.rank, 1, 0, c.epoch);
#if BRRR_STEER
  if (c.sm.steer) xf_mix<true>(c, k, mine);
  else
#endif
    xf_mix<false>(c, k, mine);
  share(c, 1, k, p.y_ffn);
}
#else
__device__ void job_xf(Ctx& c, int k) {
  const Params& p = c.p;
  c.release();
  __nv_bfloat16* mine = inbox(p, p.rank, 1, 0, c.epoch);
  constexpr int XU = 2;
  for (int base = 0; base < p.m * CHUNK_VECS; base += GT * XU) {
    Acc8 rv[XU], sv[XU];
#pragma unroll
    for (int u = 0; u < XU; ++u) {
      const int i = base + c.tid + u * GT;
      const bool live = i < p.m * CHUNK_VECS;
      const long off = chunk_off(live ? i : 0, k);
#if BRRR_ORDERED
      rv[u] = routed_rows(p, off, live);
#else
      rv[u] = load8(p.s.yracc + off, live);
#endif
      sv[u] = load8(p.s.shacc + off, live);
    }
#pragma unroll
    for (int u = 0; u < XU; ++u) {
      const int i = base + c.tid + u * GT;
      if (i >= p.m * CHUNK_VECS) break;
      const long off = chunk_off(i, k);
      clear8(p.s.yracc + off);
      clear8(p.s.shacc + off);
      float routed[8], shared[8], y[8];
      unpack8(rv[u], routed);
      unpack8(sv[u], shared);
#pragma unroll
      for (int e = 0; e < 8; ++e) shared[e] = bf(shared[e]);
#pragma unroll
      for (int e = 0; e < 8; ++e) y[e] = bf(routed[e]) + shared[e];
      *reinterpret_cast<uint4*>(mine + off) = pack8(y);
    }
  }
  share(c, 1, k, p.y_ffn);
}
#endif

__device__ void deps(Ctx& c, const Job& job) {
  const Params& p = c.p;
  switch (job.type) {
    case J_WOB: return c.wait(C_Z + job.b * GROUPS + job.c, KS_WOB / 2);
    case J_XA: return c.wait(C_YC + job.a, 2 * c.L.tg);
    case J_OWNA: return c.wait(C_YR + job.a / (CHUNK_COLS / 32), 1);
    case J_XNORM: return c.wait(C_A + job.b, SLICES);
    case J_GATE:
      if (!split_norm(p.m))
        for (int t = 0; t < tiles_of(p.m); ++t) c.wait(C_A + t, SLICES);
      return;
    case J_ROUTE: return c.wait(C_G + job.a / OTOK, GATE_BLOCKS * KP_GATE);
    case J_SH1: {
      const int k0 = job.b * KS_SH1 * 32;
      const unsigned target = split_norm(p.m) ? group_of(p.m, job.c).tiles : 512 / (KS_GATE * 16);
      for (int part = k0 / 512; part < (k0 + KS_SH1 * 32 + 511) / 512; ++part) c.wait(C_XK + part * GROUPS + job.c, target);
      return;
    }
    case J_SH2: return c.wait(C_HS + job.c, SH1_BLOCKS);
    case J_MIX: return c.wait(C_A + job.a * GW / TILE, SLICES);
    case J_R2: return c.wait(C_R1D + job.a, R1_BLOCKS);
    case J_XF: return c.wait(C_SF + job.a, __ldcg(c.ctr() + C_NHOT) + __ldcg(c.ctr() + C_NCOLD) + 2 * c.L.tg);
    default: return;
  }
}

__device__ void run_job(Ctx& c, const Job& job) {
  switch (job.type) {
    case J_WOA: return job_woa(c, job.a, job.b, job.c, job.stages);
    case J_AMIX: return job_amix(c, job.a);
    case J_WOB: return job_wob(c, job.a, job.b, job.c);
    case J_XA: return job_xa(c, job.a);
    case J_OWNA: return job_owna(c, job.a, job.b);
    case J_XNORM: return job_xnorm(c, job.a, job.b);
    case J_GATE: return job_gate(c, job.a, job.b, job.c);
    case J_ROUTE: return job_route(c, job.a);
    case J_SH1: return job_sh1(c, job.a, job.b, job.c);
    case J_SH2: return job_sh2(c, job.a, job.c);
    case J_MIX: return job_mix(c, job.a);
    case J_R1: return job_r1(c, job);
    case J_R2: return job_r2(c, job);
    default: return job_xf(c, job.a);
  }
}

// Total route entries (hot + cold) once every J_ROUTE job has signalled; ncold = how many fill the table from the front.
__device__ int route_table(const Params& p, const unsigned* ctrs, int lane, int& ncold) {
  int n = 0, cold = 0;
  if (lane == 0) {
#if BRRR_SPIN_TRAP
    for (long long polls = 0; ld_acquire_gpu(ctrs + C_ROUTE) < unsigned(p.m); __nanosleep(SPIN_NS))
      if (++polls == SPIN_TRAP_POLLS) spin_trap("route table", C_ROUTE, unsigned(p.m), ld_acquire_gpu(ctrs + C_ROUTE));
#else
    while (ld_acquire_gpu(ctrs + C_ROUTE) < unsigned(p.m)) __nanosleep(SPIN_NS);
#endif
    cold = int(__ldcg(ctrs + C_NCOLD));
    n = cold + int(__ldcg(ctrs + C_NHOT));
  }
  ncold = __shfl_sync(0xffffffffu, cold, 0);
  return __shfl_sync(0xffffffffu, n, 0);
}

// Job entry index a in [0, n) -> table row: cold entries [0, ncold) from the front, hot ones from the back (see job_route).
__device__ __forceinline__ int entry_row(int a, int ncold) { return a < ncold ? a : ENTRIES - 1 - (a - ncold); }

struct Issue {
  uint8_t* stage;
  int lane;
  unsigned long long policy;

  __device__ __forceinline__ void block(const uint8_t* src) const {
#pragma unroll
    for (int f = 0; f < FRAGS; ++f) cp_async16_hint(stage + f * FRAG + lane * 16, src + f * FRAG + lane * 16, policy);
  }

  __device__ __forceinline__ void frag(int f, const uint4* src) const { cp_async16_hint(stage + f * FRAG + lane * 16, src + lane, policy); }

  // Same smem image as block() on a stages()-repacked matrix, gathered from the row-major original: frag f = (k32 group
  // e = f / 4, 16-row group b = f % 4); lane (d = lane / 4, q = lane % 4) holds rows +c*8+d (c = 0, 1), bytes k0+e*32+q*8 .. +8.
  // SH13 = gate/up interleaved in 32-row blocks (logical row r -> native (r % 64) / 32 * INTER + r / 64 * 32 + r % 32).
  template <bool SH13 = false>
  __device__ __forceinline__ void native(const uint8_t* w, long ld, int row0, long k0) const {
    const int d = lane >> 2, q = lane & 3;
#pragma unroll
    for (int f = 0; f < FRAGS; ++f) {
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        int row = row0 + (f & 3) * 16 + c * 8 + d;
        if constexpr (SH13) row = (row % 64) / 32 * INTER + row / 64 * 32 + row % 32;
        cp_async8_hint(stage + f * FRAG + lane * 16 + c * 8, w + long(row) * ld + k0 + (f >> 2) * 32 + q * 8, policy);
      }
    }
  }

  __device__ __forceinline__ void scales(const uint8_t* src, long ld) const {
    if (lane < 2) cp_async4(stage + SCALE_OFF + lane * 4, src + lane * ld);
  }

  __device__ __forceinline__ void words(int offset, const unsigned* src, int bytes) const {
    for (int o = lane * 16; o < bytes; o += 32 * 16) cp_async16(stage + SCALE_OFF + offset + o, reinterpret_cast<const uint8_t*>(src) + o);
  }
};

// TP3 CSF stage scale area (SCALE_BYTES = 512 B): [0,128) row bases (half h at h*64; J_R2: 128 rows), [128,192) 1-bit codes
// (half h at 128 + h*32; 4 bits per row = 4 k32 columns), [192,200) group descriptors (no longer staged here: the consumer prefetches gdesc; format first | count << 22, 10-bit count, half h at 192 + h*4),
// [CSF_REC,512) exception records as (position | value << 16) words (half h at CSF_REC + h*CSF_CAP*4; J_R2: up to 2*CSF_CAP).

// Design C: the producer only cp.asyncs the row bases and 1-bit codes; the consumer fetches the (rare) exception records
// straight from the payload using descriptors it prefetched at job start (no plain smem stores in the producer).
__device__ __forceinline__ void issue_csf(const Csf& cs, uint8_t* area, int lane, int h, int se, int rg, int cg, int r0, int nrows) {
  if (se < 0 || se >= cs.payload_experts || rg < 0 || rg >= cs.rows / 128 || cg < 0 || cg >= cs.column_groups) {
    if (lane == 0) printf("[csf guard] producer blk %d se=%d rg=%d cg=%d r0=%d nrows=%d\n", blockIdx.x, se, rg, cg, r0, nrows);
    __trap();
  }
  const uint8_t* bases = cs.storage + cs.bases + long(se) * cs.rows + rg * 128 + r0;
  const uint8_t* codes = cs.storage + cs.codes + long(se) * cs.code_bytes + long(rg) * (128 * cs.column_groups * 4 / 8) + cg * 64 + r0 / 2;
  if (lane < nrows / 16) cp_async16(area + h * 64 + lane * 16, bases + lane * 16);
  else if (lane < nrows / 16 + nrows / 32) cp_async16(area + 128 + h * 32 + (lane - nrows / 16) * 16, codes + (lane - nrows / 16) * 16);
}

__device__ void issue(const Params& p, const Job& job, int s, const Issue& is) {
  const Weights& w = p.w;
  switch (job.type) {
    case J_WOA: {
      const int kg = (woa_uneven(p.m) ? woa_ustage0(job.b) : job.b * job.stages) + s;
      if (w.dense_native) is.native(w.woa, w.woa_ld, job.a * 64, long(kg) * 128);
      else is.block(w.woa + (long(job.a) * KG_WOA + kg) * FRAGS * FRAG);
      is.scales(w.woa_s + long(job.a * 2) * (OAK / 32) + kg * 4, OAK / 32);
      return;
    }
    case J_WOB: {
      const int kg = job.b * (KS_WOB / 4) + s;
      if (w.dense_native) is.native(w.wob, w.wob_ld, job.a * 64, long(kg) * 128);
      else is.block(w.wob + (long(job.a) * KG_WOB + kg) * FRAGS * FRAG);
      is.scales(w.wob_s + long(job.a * 2) * (OA / 32) + kg * 4, OA / 32);
      return;
    }
    case J_GATE: {
      const int kg = job.b * (KS_GATE / 4) + s;
      if (w.dense_native) is.native(w.gate, w.gate_ld, job.a * 64, long(kg) * 128);
      else is.block(w.gate + (long(job.a) * KG_GATE + kg) * FRAGS * FRAG);
      return;
    }
    case J_SH1: {
      const int kg = job.b * (KS_SH1 / 4) + s;
      if (w.dense_native) is.native<true>(w.sh13, w.sh13_ld, job.a * 64, long(kg) * 128);
      else is.block(w.sh13 + (long(job.a) * KG_SH1 + kg) * FRAGS * FRAG);
      is.scales(w.sh13_s + long(job.a * 2) * SLICES + kg * 4, SLICES);
      return;
    }
    case J_SH2: {
      const int ks = s;
      if (w.dense_native) is.native(w.sh2, w.sh2_ld, job.a * 64, long(ks) * 128);
      else is.block(w.sh2 + (long(job.a) * INTER_STAGES + ks) * FRAGS * FRAG);
      is.scales(w.sh2_s + long(job.a * 2) * SH2_LD + ks * 4, SH2_LD);
      return;
    }
    case J_OWNA:
      for (int i = is.lane; i < (s ? MIX - OWNA_ROWS : OWNA_ROWS) * HC * 8; i += 32) {
        const int piece = s * OWNA_ROWS * HC + i / 8, part = i % 8, r = piece / HC, hc = piece % HC;
        cp_async16(is.stage + (i / 8) * 128 + part * 16, w.fn + long(r) * HCD + hc * HID + job.a * 32 + part * 4);
      }
      return;
    case J_R1: {
      // TP3: only b12x's natural W4A8 layout is reachable (RP_LAYOUT is a constexpr of INTER); the exact N128/N64
      // layout arm for 64-row tails (TP4's INTER 576) was removed as dead code.
      static_assert(RP_LAYOUT, "the fused tail reads b12x's natural W4A8 tile layout");
      const int e = job.e, nt = job.b / 2, n0 = (job.b % 2) * 2;
      // b12x's natural W4A8 layout (N256 x K128 tiles of 4096 words; within a tile [k32][32-row block][inner 128 words],
      // the same inner order as the exact N128 layout). Scales: per (256-row tile, 128-K tile) 256 row-ordered words
      // (4 k32 scales per row).
      const int kt = job.c * 4 + s;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int half = 1 - h, nt256 = half * (INTER / 256) + nt / 2, n8c0 = (nt % 2) * 4 + n0;
        const uint4* tile = w.w13 + e * w.w13_stride + (long(nt256) * (HID / 128) + kt) * 1024;
        for (int q = 0; q < 4; ++q) {
          is.frag(q * 4 + h * 2, tile + (q * 8 + n8c0) * 32);
          is.frag(q * 4 + h * 2 + 1, tile + (q * 8 + n8c0 + 1) * 32);
        }
        // scales: this k-tile's 128-row half (nt % 2) of the 256-row tile, rows n0*32 .. n0*32+63 (one word per row).
        if (w.csf13.storage) issue_csf(w.csf13, is.stage + SCALE_OFF, is.lane, h, e * w.csf13.sections + half, nt, kt, n0 * 32, 64);
        else is.words(h * 256, w.s13 + e * w.s13_stride + (long(nt256) * (HID / 128) + kt) * 256 + (nt % 2) * 128 + n0 * 32, 256);
      }
      return;
    }
    case J_R2: {
      const int ks = s;
      const int e = job.e, kq = ks == INTER / 128 ? (INTER % 128) / 32 : 4;
      const int nt = job.b, kt = ks, nt256 = nt / 2;
      const uint4* tile = w.w2 + e * w.w2_stride + (long(nt256) * (INTER / 128) + kt) * 1024;
      for (int q = 0; q < kq; ++q)
#pragma unroll
        for (int rt = 0; rt < 4; ++rt) is.frag(q * 4 + rt, tile + (q * 8 + (nt % 2) * 4 + rt) * 32);
      // scales: k-tile kt, the 128 rows of half (nt % 2) of the 256-row tile (one word per row = 4 k32 scales).
      if (w.csf2.storage) {
        issue_csf(w.csf2, is.stage + SCALE_OFF, is.lane, 0, e, nt, kt, 0, 128);
      } else {
        is.words(0, w.s2 + e * w.s2_stride + (long(nt256) * (INTER / 128) + kt) * 256 + (nt % 2) * 128, kq * 128);
      }
      return;
    }
    default: return;
  }
}

__device__ void producer(const Params& p, Smem& sm) {
  const int lane = threadIdx.x & 31;
  const Layout L = layout(p.m);
  const unsigned long long policy = evict_first_policy();
  const volatile int* ready = &sm.ready;
  int n = -1, ncold = 0, it = 0, j = int(blockIdx.x);
  unsigned* const ctrs = p.s.counters;
  for (int seq = 0;; ++seq) {
    if (n < 0 && j >= L.routed) n = route_table(p, ctrs, lane, ncold);
    Job job = n < 0 || j < total_jobs(L, n) ? decode(L, n, j) : Job{J_END, 0, 0, 0, 1};
    if (job.type == J_R1 || job.type == J_R2) {
      // job.a stays the dense entry id (R1 arrive/done counters); the table row is cold-first permuted.
      const int row = entry_row(job.a, ncold);
      const int e = __ldcg(p.s.table->expert + row), ch = __ldcg(p.s.table->chunk + row);
      job.e = e;
      job.f = e * MAXM + ch * SPAN;
      job.n = min(SPAN, int(__ldcg(ctrs + C_ECOUNT + e)) - ch * SPAN);
    }
    bool claimed = false;
    unsigned next = 0;
    for (int s = 0; s < job.stages; ++s, ++it) {
      const int slot = it % STAGES;
      mbar_wait(&sm.empty[slot], ((it / STAGES) & 1) ^ 1);
      issue(p, job, s, Issue{sm.ring[slot], lane, policy});
      if (lane == 0 && s == 0) sm.job[slot] = job;
      cp_async_arrive(&sm.full[slot]);
      if (lane == 0) {
        mbar_arrive(&sm.full[slot]);
        if (!claimed && seq > 0 && job.type != J_END && *ready >= seq) {
          next = atomicAdd(ctrs + C_CLAIM, 1u);
          claimed = true;
        }
      }
    }
    if (job.type == J_END) return;
    if (seq == 0) pdl_wait();
    if (lane == 0 && !claimed) {
      while (*ready < seq) __nanosleep(32);
      next = atomicAdd(ctrs + C_CLAIM, 1u);
    }
    j = int(gridDim.x) + int(__shfl_sync(0xffffffffu, next, 0));
  }
}

__global__ void __launch_bounds__(THREADS, PER_SM) layer_kernel(Params p) {
  extern __shared__ __align__(128) uint8_t raw[];
  Smem& sm = *reinterpret_cast<Smem*>(raw);
  for (int i = threadIdx.x; i < CACHED; i += THREADS) sm.seen[i] = 0;
  if (threadIdx.x == 0) {
    sm.ready = -1;
#if BRRR_STEER
    sm.steer = 0;
#endif
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(&sm.full[s], 33);
      mbar_init(&sm.empty[s], GW);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x >= GT) {
    producer(p, sm);
  } else {
    const int tid = threadIdx.x;
    Ctx c{p, sm, layout(p.m), tid >> 5, tid & 31, tid, 0, 0u};
    if (p.prof && tid == 0) {
      const unsigned long long t_entry = globaltimer();
      prof_first(p.prof, PH_KERNEL, 0, t_entry);
      prof_last(p.prof, PH_KERNEL, 1, t_entry);
    }
    pdl_wait();
#if BRRR_STEER
    // Steering flag: any nonzero row scale. The scale loads are issued before the epoch load so their latencies
    // overlap; every consumer warp of every CTA (and every rank: the scale is a replicated graph value) decides alike.
    float scale[MAXM / 32] = {};
    if (p.steer_scale) {
#pragma unroll
      for (int j = 0; j < MAXM / 32; ++j)
        if (c.lane + 32 * j < p.m) scale[j] = __ldcg(p.steer_scale + c.lane + 32 * j);
    }
#endif
    c.epoch = ld_volatile(p.s.epoch) + 1;
#if BRRR_STEER
    bool nonzero = false;
#pragma unroll
    for (int j = 0; j < MAXM / 32; ++j) nonzero |= scale[j] != 0.0f;
    nonzero = __any_sync(0xffffffffu, nonzero);
    if (tid == 0) sm.steer = nonzero;
    c.sync();  // every consumer warp reads sm.steer only after this
#endif
    unsigned long long t_begin = 0, t0 = 0, t1 = 0, t2 = 0;
    if (p.prof && tid == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_begin));
    for (int seq = 0;; ++seq) {
      if (p.prof && tid == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
      c.acquire();
      const Job job = sm.job[c.it % STAGES];
      if (job.type == J_END) break;
      if (p.prof && tid == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
      deps(c, job);
      if (p.prof && tid == 0) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t2));
      const bool exchange = job.type == J_XA || job.type == J_XF;
      if (tid == 0 && !exchange) *reinterpret_cast<volatile int*>(&sm.ready) = seq;
      run_job(c, job);
      if (tid == 0 && exchange) *reinterpret_cast<volatile int*>(&sm.ready) = seq;
      if (p.prof && tid == 0) {
        unsigned long long t3;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t3));
        unsigned long long* q = p.prof + job.type * 4;
        atomicAdd(q + 0, t1 - t0); atomicAdd(q + 1, t2 - t1); atomicAdd(q + 2, t3 - t2); atomicAdd(q + 3, 1ull);
        prof_first(p.prof, job.type, 0, t1);
        prof_first(p.prof, job.type, 1, t2);
        prof_last(p.prof, job.type, 2, t3);
        prof_last(p.prof, job.type, 3, t2);
      }
    }
    if (p.prof && tid == 0) {
      unsigned long long t_end;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t_end));
      atomicAdd(p.prof + 15 * 4 + 0, t_end - t_begin); atomicAdd(p.prof + 15 * 4 + 3, 1ull);
      prof_first(p.prof, PH_KERNEL, 2, t_end);
      prof_last(p.prof, PH_KERNEL, 3, t_end);
    }
  }
  __syncthreads();
  // No pdl_launch_dependents() here: an early trigger at this point measured ~2% slower at c4/c8 (2026-11-08 bisect,
  // 3 alternating rounds). No kernel in this file triggers dependents early.
  if (threadIdx.x == 0) {
    __threadfence();
    sm.last = atomicAdd(p.s.counters + C_FIN, 1u) == gridDim.x - 1;
    if (p.prof) {
      const unsigned long long t_fin = globaltimer();
      prof_first(p.prof, PH_FIN, 0, t_fin);
      prof_last(p.prof, PH_FIN, 1, t_fin);
    }
  }
  __syncthreads();
  if (!sm.last) return;
  __threadfence();
  for (int i = threadIdx.x; i < COUNTERS; i += THREADS) p.s.counters[i] = 0;
#if BRRR_STEER
  // Every CTA has finished (C_FIN), so no job reads sdot / adot any more; the next launch starts from zero.
  if (sm.steer)
    for (int i = threadIdx.x; i < p.m; i += THREADS) p.s.sdot[i] = 0, p.s.adot[i] = 0;
#endif
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence();
    atomicAdd(p.s.epoch, 1u);
    __threadfence();
    if (p.prof) prof_last(p.prof, PH_FIN, 2, globaltimer());
  }
}

struct Layer {
  Weights w;
  long image_lo;
  std::vector<torch::Tensor> keep;
};

struct Engine {
  Peers peers;
  Scratch s;
  int rank;
  int ctas;
  int attn_per_chunk_flags;  // TP3: one-shot attention exchange posts a flag per chunk (TP3Config.fused_attn_per_chunk)
  std::vector<Layer> layers;
};

std::vector<Engine> engines;

template <class T>
long row_words(const torch::Tensor& t) {
  return t.stride(0) * t.element_size() / long(sizeof(T));
}

}

// TP3 CSF self-check: expand a whole payload with the same addressing the fused loader/consumer use, into rp sfb words.
__global__ void csf_expand_all_kernel(Csf cs, int E, unsigned* out) {
  const int se = blockIdx.y, sg = blockIdx.x, r = threadIdx.x;  // 128 threads = 128 rows of one (row group, column group)
  const int CG = cs.column_groups, rg = sg / CG, cg = sg % CG;
  const uint8_t* bases = cs.storage + cs.bases + long(se) * cs.rows + rg * 128;
  const uint8_t* codes = cs.storage + cs.codes + long(se) * cs.code_bytes + long(rg) * (128 * CG * 4 / 8) + cg * 64;
  const unsigned nib = (codes[r / 2] >> ((r & 1) * 4)) & 15u;
  unsigned spread = nib;
  spread = (spread | (spread << 14)) & 0x00030003u;
  spread = (spread | (spread << 7)) & 0x01010101u;
  unsigned word = unsigned(bases[r]) * 0x01010101u + spread;
  const unsigned desc = cs.gdesc[long(se) * cs.groups + sg];
  const unsigned first = desc & 0x3fffffu, count = desc >> 22;
  const unsigned short* pos_p = reinterpret_cast<const unsigned short*>(cs.storage + cs.recpos) + first;
  const uint8_t* val_p = cs.storage + cs.value_off + first;
  for (unsigned i = 0; i < count; ++i) {
    const unsigned pos = pos_p[i], val = val_p[i];
    if (int(pos >> 2) == r) word = (word & ~(0xffu << ((pos & 3) * 8))) | (val << ((pos & 3) * 8));
  }
  const int e = se / cs.sections, sec = se % cs.sections, rgs = cs.rows / 128;
  const int rg_out = sec * rgs + rg, tiles_per_e = (cs.sections * rgs / 2) * CG;
  const long tile = long(e) * tiles_per_e + long(rg_out / 2) * CG + cg;
  out[tile * 256 + (rg_out % 2) * 128 + r] = word;
}

void csf_expand_all(torch::Tensor storage, torch::Tensor gdesc, std::vector<int64_t> meta, torch::Tensor out) {
  Csf cs{};
  TORCH_CHECK(meta.size() == 9 && storage.scalar_type() == at::kByte && gdesc.scalar_type() == at::kInt && out.scalar_type() == at::kInt);
  cs.storage = storage.data_ptr<uint8_t>();
  cs.gdesc = reinterpret_cast<const unsigned*>(gdesc.data_ptr<int>());
  cs.bases = meta[0], cs.codes = meta[1], cs.code_bytes = meta[2], cs.value_off = meta[3], cs.recpos = meta[4];
  cs.rows = int(meta[5]), cs.column_groups = int(meta[6]), cs.groups = int(meta[7]), cs.sections = int(meta[8]);
  const int Es = int(gdesc.numel() / cs.groups), E = Es / cs.sections;
  TORCH_CHECK(out.numel() == long(E) * cs.sections * cs.rows * cs.column_groups, "csf_expand_all: out size");
  const c10::cuda::CUDAGuard guard(storage.device());
  csf_expand_all_kernel<<<dim3(cs.groups, Es), 128, 0, c10::cuda::getCurrentCUDAStream()>>>(cs, E, reinterpret_cast<unsigned*>(out.data_ptr<int>()));
  TORCH_CHECK(cudaGetLastError() == cudaSuccess, "csf_expand_all launch failed");
}

static unsigned long long* g_prof = nullptr;
static const bool* g_pad = nullptr;
void set_pad(torch::Tensor t) { TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBool); g_pad = t.data_ptr<bool>(); }
// Rows 0..15: accumulators [acquire wait, dependency wait, run, count] per job type (row 15: consumer loop total);
// rows 16..31: phase marks (PROF_PHASES, absolute globaltimer ns, "first" columns decoded, 0 = not recorded).
torch::Tensor prof_read() {
  auto out = torch::zeros({32, 4}, torch::kInt64);
  if (!g_prof) return out;
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  TORCH_CHECK(cudaMemcpy(out.data_ptr<int64_t>(), g_prof, (PROF_ACC + PROF_PHASES) * 8, cudaMemcpyDeviceToHost) == cudaSuccess);
  TORCH_CHECK(cudaMemset(g_prof, 0, (PROF_ACC + PROF_PHASES) * 8) == cudaSuccess);
  auto* v = reinterpret_cast<unsigned long long*>(out.data_ptr<int64_t>()) + PROF_ACC;
  auto decode = [&](int row, int col) { unsigned long long& x = v[row * 4 + col]; x = x ? ~x : 0; };
  for (int t = 0; t < J_END; ++t) { decode(t, 0); decode(t, 1); }
  decode(PH_FIN, 0);
  decode(PH_KERNEL, 0);
  decode(PH_KERNEL, 2);
  return out;
}

int64_t flag_bytes() { return FLAG_BYTES; }

// TP3 test hook (host only): the last launch's routing state of an engine. topk_w [ROUTES] fp32 route weights (route
// r = token * TOPK + k) and routes [EXPERTS, MAXM] (bank slot s, position -> route r, -1 = unused when cleared before
// the launch with clear_routes). Lets the layer test compare two builds' (slot, weight) per route bit for bit.
std::vector<torch::Tensor> route_state(int64_t handle) {
  const auto& e = engines.at(handle);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  auto w = torch::empty({ROUTES}, torch::kFloat32), r = torch::empty({EXPERTS, MAXM}, torch::kInt32);
  TORCH_CHECK(cudaMemcpy(w.data_ptr<float>(), e.s.topk_w, ROUTES * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
  TORCH_CHECK(cudaMemcpy(r.data_ptr<int>(), e.s.table->routes, sizeof(Table::routes), cudaMemcpyDeviceToHost) == cudaSuccess);
  return {w, r};
}

// TP3 test hook (host only): the last launch's reduced attention output yattn [m, HID] BF16 (replicated; steering never
// writes it: OWNA steers its registers), so the layer test can apply upstream's steer_attention to the kernel's own y.
torch::Tensor last_yattn(int64_t handle, int64_t m) {
  const auto& e = engines.at(handle);
  TORCH_CHECK(m >= 1 && m <= MAXM, "last_yattn: rows");
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  auto out = torch::empty({m, HID}, torch::kBFloat16);
  TORCH_CHECK(cudaMemcpy(out.data_ptr(), e.s.yattn, size_t(m) * HID * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost) == cudaSuccess);
  return out;
}

void clear_routes(int64_t handle) {
  const auto& e = engines.at(handle);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  TORCH_CHECK(cudaMemset(e.s.table->routes, 0xff, sizeof(Table::routes)) == cudaSuccess);
  TORCH_CHECK(cudaMemset(e.s.topk_w, 0, ROUTES * sizeof(float)) == cudaSuccess);
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
}


int64_t inbox_bytes() { return 2L * 2 * 2 * PUB_SLOT * sizeof(__nv_bfloat16); }

int64_t create(std::vector<int64_t> buffers, int64_t rank, int64_t epoch_addr, int64_t attn_per_chunk_flags) {
  TORCH_CHECK(buffers.size() == RANKS);
  Engine e{};
  e.attn_per_chunk_flags = int(attn_per_chunk_flags);
  for (int r = 0; r < RANKS; ++r) {
    e.peers.flags[r] = reinterpret_cast<unsigned*>(buffers[r]);
    e.peers.pub[r] = reinterpret_cast<__nv_bfloat16*>(buffers[r] + FLAG_BYTES);
  }
  e.rank = int(rank);
  // Flare port: the epoch may live in the rank's own (exported, reset-to-zero) flag page, so flags and epoch restart together.
  e.s.epoch = epoch_addr ? reinterpret_cast<unsigned*>(epoch_addr) : zeroed<unsigned>(1);
  e.s.counters = zeroed<unsigned>(COUNTERS);
  e.s.table = zeroed<Table>(1);
  e.s.zacc = zeroed<float>(long(MAXM) * OA);
  e.s.zq = zeroed<uint8_t>(long(MAXM) * OA);
  e.s.zse = zeroed<uint8_t>(long(MAXM) * (OA / 32));
  e.s.yacc = zeroed<float>(long(MAXM) * HID);
  e.s.yattn = zeroed<__nv_bfloat16>(long(MAXM) * HID);
  e.s.mixp = zeroed<float>(long(SLICES) * MAXM * MIXW);
  e.s.xc = zeroed<__nv_bfloat16>(long(MAXM) * HID);
  e.s.xsq = zeroed<float>(long(SLICES) * MAXM);
  e.s.xbf = zeroed<__nv_bfloat16>(long(MAXM) * HID);
  e.s.xq = zeroed<uint8_t>(long(MAXM) * HID);
  e.s.xse = zeroed<uint8_t>(long(MAXM) * SLICES);
  e.s.gacc = zeroed<long long>(long(MAXM) * EXPERTS);
  e.s.topk_w = zeroed<float>(ROUTES);
  e.s.sacc = zeroed<float>(long(MAXM) * 2 * INTER);
  e.s.hsq = zeroed<uint8_t>(long(MAXM) * INTER);
  e.s.hse = zeroed<uint8_t>(long(MAXM) * HQ_LD);
  e.s.shacc = zeroed<float>(long(MAXM) * HID);
  e.s.hacc = zeroed<float>(long(ROUTES) * 2 * INTER);
  e.s.hq = zeroed<uint8_t>(long(ROUTES) * INTER);
  e.s.hqe = zeroed<uint8_t>(long(ROUTES) * HQ_LD);
  e.s.yracc = zeroed<float>(long(MAXM) * HID);
#if BRRR_ORDERED
  e.s.yroute = zeroed<float>(long(ROUTES) * HID);
#endif
#if BRRR_STEER
  e.s.adot = zeroed<long long>(MAXM);
  e.s.sdot = zeroed<long long>(MAXM);
#endif
  int sms = 0, per = 0;
  TORCH_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, c10::cuda::current_device()) == cudaSuccess);
  opt_in_smem<layer_kernel>(sizeof(Smem));
  TORCH_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, layer_kernel, THREADS, sizeof(Smem)) == cudaSuccess);
  TORCH_CHECK(per >= PER_SM, "layer kernel fits ", per, " CTAs per SM, needs ", PER_SM);
  e.ctas = sms * PER_SM;
  TORCH_CHECK(cudaDeviceSynchronize() == cudaSuccess);
  engines.push_back(e);
  return long(engines.size()) - 1;
}

int64_t add_layer(int64_t handle, torch::Tensor woa, torch::Tensor woa_s, torch::Tensor wob, torch::Tensor wob_s, torch::Tensor gate, torch::Tensor gate_b,
                  std::optional<torch::Tensor> bias_vl, torch::Tensor sh13, torch::Tensor sh13_s, torch::Tensor sh2, torch::Tensor sh2_s, torch::Tensor w13,
                  torch::Tensor s13, torch::Tensor w2, torch::Tensor s2, torch::Tensor fn, torch::Tensor hc_scale, torch::Tensor hc_base, torch::Tensor norm,
                  int64_t image_lo, std::optional<torch::Tensor> csf13_storage, std::optional<torch::Tensor> csf13_gdesc, std::vector<int64_t> csf13_meta,
                  std::optional<torch::Tensor> csf2_storage, std::optional<torch::Tensor> csf2_gdesc, std::vector<int64_t> csf2_meta,
                  std::optional<torch::Tensor> csf13_debug, std::optional<torch::Tensor> csf2_debug,
                  std::optional<torch::Tensor> csf13_debug_desc = std::nullopt, std::optional<torch::Tensor> csf2_debug_desc = std::nullopt,
                  bool dense_native = false, std::optional<torch::Tensor> route_counts = std::nullopt,
                  std::optional<torch::Tensor> hot_map = std::nullopt, int64_t hot_rows = EXPERTS) {
  TORCH_CHECK(woa.numel() == long(OA) * OAK && sh2_s.size(-1) == SH2_LD);
  auto ld_bytes = [&](const torch::Tensor& t, long rows) {
    TORCH_CHECK(t.dim() == 2 && t.size(0) == rows && t.stride(1) == 1 && (t.stride(0) * t.element_size()) % 16 == 0 && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
                "native dense weight must be row-major with 16-byte aligned rows");
    return long(t.stride(0) * t.element_size());
  };
  for (const auto& t : {woa_s, wob_s, sh13_s, sh2_s}) TORCH_CHECK(t.scalar_type() == at::kByte, "dense scales must be e8m0 bytes");
  Layer l{};
  l.w.dense_native = dense_native;
  l.w.route_counts = nullptr;
  if (route_counts.has_value()) {
    TORCH_CHECK(route_counts->is_cuda() && route_counts->scalar_type() == at::kInt && route_counts->numel() >= EXPERTS);
    l.w.route_counts = reinterpret_cast<unsigned*>(route_counts->data_ptr<int>());
  }
  if (route_counts.has_value()) l.keep.push_back(*route_counts);
  l.w.hot_map = nullptr;
  if (hot_map.has_value()) {
    TORCH_CHECK(hot_map->is_cuda() && hot_map->scalar_type() == at::kInt && hot_map->numel() == EXPERTS && hot_map->is_contiguous());
    l.w.hot_map = hot_map->data_ptr<int>();
    l.keep.push_back(*hot_map);
  }
  // Cold-first job ordering: slots >= hot_rows are host-RAM experts (merged bank is hot-first). EXPERTS = all resident.
  TORCH_CHECK(hot_rows >= 0 && hot_rows <= EXPERTS, "hot_rows must be in [0, ", EXPERTS, "]");
  l.w.hot_rows = int(hot_rows);
  if (dense_native) {
    l.w.woa_ld = ld_bytes(woa, OA);
    l.w.wob_ld = ld_bytes(wob, HID);
    l.w.gate_ld = ld_bytes(gate, EXPERTS);
    l.w.sh13_ld = ld_bytes(sh13, 2 * INTER);
    l.w.sh2_ld = ld_bytes(sh2, HID);
  }
  l.w.woa = ptr<uint8_t>(woa);
  l.w.woa_s = ptr<uint8_t>(woa_s);
  l.w.wob = ptr<uint8_t>(wob);
  l.w.wob_s = ptr<uint8_t>(wob_s);
  l.w.gate = ptr<uint8_t>(gate);
  l.w.gate_b = ptr<float>(gate_b);
  l.w.bias_vl = bias_vl ? ptr<float>(*bias_vl) : nullptr;
  l.w.sh13 = ptr<uint8_t>(sh13);
  l.w.sh13_s = ptr<uint8_t>(sh13_s);
  l.w.sh2 = ptr<uint8_t>(sh2);
  l.w.sh2_s = ptr<uint8_t>(sh2_s);
  l.w.w13 = ptr<uint4>(w13);
  l.w.s13 = ptr<unsigned>(s13);
  l.w.w2 = ptr<uint4>(w2);
  l.w.s2 = ptr<unsigned>(s2);
  l.w.w13_stride = row_words<uint4>(w13);
  l.w.s13_stride = row_words<unsigned>(s13);
  l.w.w2_stride = row_words<uint4>(w2);
  l.w.s2_stride = row_words<unsigned>(s2);
  // TP3 CSF: optional compact scale payloads (b12x CsfBlockPayload storage + per-group record descriptors).
  auto set_csf = [&](Csf& cs, const std::optional<torch::Tensor>& st, const std::optional<torch::Tensor>& gd, const std::vector<int64_t>& meta,
                     const std::optional<torch::Tensor>& dbg) {
    cs = Csf{};
    if (!st.has_value()) return;
    if (dbg.has_value()) {
      TORCH_CHECK(dbg->scalar_type() == at::kInt && dbg->is_cuda());
      cs.debug = reinterpret_cast<unsigned*>(dbg->data_ptr<int>());
      l.keep.push_back(*dbg);
    }
    TORCH_CHECK(gd.has_value() && meta.size() == 9 && st->scalar_type() == at::kByte && gd->scalar_type() == at::kInt && st->is_cuda() && gd->is_cuda());
    cs.storage = st->data_ptr<uint8_t>();
    cs.gdesc = reinterpret_cast<const unsigned*>(gd->data_ptr<int>());
    cs.bases = meta[0], cs.codes = meta[1], cs.code_bytes = meta[2], cs.value_off = meta[3], cs.recpos = meta[4];
    cs.rows = int(meta[5]), cs.column_groups = int(meta[6]), cs.groups = int(meta[7]), cs.sections = int(meta[8]);
    TORCH_CHECK(gd->numel() >= long(cs.groups) * (cs.sections * EXPERTS), "csf gdesc too small");
    cs.payload_experts = int(gd->numel() / cs.groups);
    cs.records = (cs.value_off - cs.recpos) / 2;
    l.keep.push_back(*st);
    l.keep.push_back(*gd);
  };
  set_csf(l.w.csf13, csf13_storage, csf13_gdesc, csf13_meta, csf13_debug);
  set_csf(l.w.csf2, csf2_storage, csf2_gdesc, csf2_meta, csf2_debug);
  auto set_dd = [&](Csf& cs, const std::optional<torch::Tensor>& dd) {
    cs.debug_desc = nullptr;
    if (dd.has_value()) { cs.debug_desc = reinterpret_cast<unsigned*>(dd->data_ptr<int>()); l.keep.push_back(*dd); }
  };
  set_dd(l.w.csf13, csf13_debug_desc);
  set_dd(l.w.csf2, csf2_debug_desc);
  l.w.fn = ptr<float>(fn);
  l.w.hc_scale = ptr<float>(hc_scale);
  l.w.hc_base = ptr<float>(hc_base);
  l.w.norm = ptr<__nv_bfloat16>(norm);
  l.image_lo = image_lo;
  // TP3 CSF: append, never reassign: set_csf/set_dd above already pushed the CSF payload, descriptor and debug tensors.
  for (const auto& t : {woa, woa_s, wob, wob_s, gate, gate_b, sh13, sh13_s, sh2, sh2_s, w13, s13, w2, s2, fn, hc_scale, hc_base, norm}) l.keep.push_back(t);
  if (bias_vl) l.keep.push_back(*bias_vl);
  auto& layers = engines.at(handle).layers;
  layers.push_back(std::move(l));
  return long(layers.size()) - 1;
}

#if BRRR_STEER
// Steering constants of one registered layer (brrr.Engine.register_steering): the kernel keeps raw pointers, the
// layer keeps the tensors (persistent-arena copies on the Flare side).
void set_steering(int64_t handle, int64_t layer, torch::Tensor direction, torch::Tensor gain_delta, torch::Tensor u, double alpha) {
  auto& l = engines.at(handle).layers.at(layer);
  for (const auto& t : {direction, gain_delta})
    TORCH_CHECK(t.is_cuda() && t.is_contiguous() && t.scalar_type() == at::kBFloat16 && t.numel() == HID, "steering direction/gain: contiguous bf16 [", HID, "]");
  TORCH_CHECK(u.is_cuda() && u.is_contiguous() && u.scalar_type() == at::kFloat && u.numel() == INTER, "steering u = W2^T d: contiguous fp32 [", INTER, "]");
  TORCH_CHECK(std::isfinite(alpha) && alpha > 0.0, "steering alpha must be finite and positive");
  for (const auto& t : {direction, gain_delta, u}) TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, "steering constants must be 16-byte aligned");
  l.w.steer_dir = ptr<__nv_bfloat16>(direction);
  l.w.steer_gain = ptr<__nv_bfloat16>(gain_delta);
  l.w.steer_u = ptr<float>(u);
  l.w.steer_alpha = float(alpha);
  for (const auto& t : {direction, gain_delta, u}) l.keep.push_back(t);
}
#endif

static void launch(int64_t handle, int64_t layer, torch::Tensor o, torch::Tensor positions, std::optional<torch::Tensor> input_ids, torch::Tensor cos_sin, torch::Tensor resid0,
                   torch::Tensor attn_post, torch::Tensor attn_comb, torch::Tensor attn_pre, torch::Tensor y_ffn, torch::Tensor resid1, torch::Tensor ffn_post, torch::Tensor ffn_comb, torch::Tensor ffn_pre,
                   const torch::Tensor* steer_scale) {
  const auto& e = engines.at(handle);
  const auto& l = e.layers.at(layer);
  const int m = int(resid0.size(0));
  TORCH_CHECK(m >= 1 && m <= MAXM && o.stride(-1) == 1 && o.size(-1) == HDIM && o.stride(1) == HDIM && positions.scalar_type() == at::kLong);
  TORCH_CHECK(o.scalar_type() == at::kBFloat16, "BRRR tail expects a bf16 attention output");
  TORCH_CHECK(cos_sin.scalar_type() == at::kFloat && cos_sin.numel() == ROPE / 2, "BRRR tail expects ", ROPE / 2, " fp32 RoPE frequencies");
  const bool ids = l.w.bias_vl && input_ids && input_ids->numel() && input_ids->is_contiguous();
  TORCH_CHECK(!ids || input_ids->scalar_type() == at::kLong || input_ids->scalar_type() == at::kInt);
  // Scratch contract (Flare MHCScratch / decoder Scratch layout, indexed by the kernel as tok * HC, tok * HC * HC,
  // (tok * HC + k) * HID and tok * HID with unit inner strides): residual [m, HC, HID] bf16, post [m, HC] f32,
  // comb [m, HC, HC] f32, pre/attn mix [m, HC] f32, y [m, HID] bf16; capacities may exceed m, the leading rows are used.
  auto dense = [&](const torch::Tensor& t, const char* what, at::ScalarType dtype, std::initializer_list<long> inner) {
    TORCH_CHECK(t.is_cuda() && t.device() == resid0.device(), "BRRR tail: ", what, " must live on the layer's device");
    TORCH_CHECK(t.scalar_type() == dtype, "BRRR tail: ", what, " dtype ", t.scalar_type(), ", expected ", dtype);
    TORCH_CHECK(t.dim() == long(inner.size()) + 1 && t.size(0) >= m, "BRRR tail: ", what, " shape ", t.sizes(), " needs [>=", m, ", ...]");
    long expect = 1;
    int d = t.dim() - 1;
    for (auto it = std::rbegin(inner); it != std::rend(inner); ++it, --d) {
      TORCH_CHECK(t.size(d) == *it && t.stride(d) == expect, "BRRR tail: ", what, " shape ", t.sizes(), " strides ", t.strides(),
                  " must be dense with inner dims ", std::vector<long>(inner));
      expect *= *it;
    }
    TORCH_CHECK(t.stride(0) == expect, "BRRR tail: ", what, " row stride ", t.stride(0), " must be ", expect);
  };
  dense(resid0, "resid0", at::kBFloat16, {HC, HID});
  dense(resid1, "resid1", at::kBFloat16, {HC, HID});
  dense(y_ffn, "y", at::kBFloat16, {HID});
  dense(attn_post, "attn_post", at::kFloat, {HC});
  dense(ffn_post, "post1", at::kFloat, {HC});
  dense(attn_comb, "attn_comb", at::kFloat, {HC, HC});
  dense(ffn_comb, "comb1", at::kFloat, {HC, HC});
  dense(attn_pre, "attn_pre", at::kFloat, {HC});
  dense(ffn_pre, "pre1", at::kFloat, {HC});
  TORCH_CHECK(positions.is_contiguous() && positions.numel() >= m, "BRRR tail: positions must hold ", m, " contiguous int64 rows");
  TORCH_CHECK(o.size(0) >= m, "BRRR tail: attention output has ", o.size(0), " rows, needs ", m);
  const c10::cuda::CUDAGuard guard(resid0.device());
  Params p{};
  static const bool prof_on = getenv("DS41_FUSED_PROF") && atoi(getenv("DS41_FUSED_PROF"));
  if (prof_on && !g_prof) {
    TORCH_CHECK(cudaMalloc(&g_prof, (PROF_ACC + PROF_PHASES) * 8) == cudaSuccess);
    TORCH_CHECK(cudaMemset(g_prof, 0, (PROF_ACC + PROF_PHASES) * 8) == cudaSuccess);
  }
  p.prof = prof_on ? g_prof : nullptr;
  p.pad = g_pad;
  static const int test_mode = getenv("DS41_FUSED_TEST") ? atoi(getenv("DS41_FUSED_TEST")) : 0;
  p.test = test_mode;
  // 0 restores the legacy single attention flag for m < TWO_SHOT_MIN (A/B toggle, TP3Config.fused_attn_per_chunk via create()).
  p.attn_per_chunk_flags = e.attn_per_chunk_flags;
  // The producer's seq-0 job (blockIdx.x) must lie before the routed section: it is decoded before pdl_wait, and the
  // route table may only be read after it.
  TORCH_CHECK(layout(m).routed >= e.ctas, "BRRR tail: ", e.ctas, " CTAs exceed the ", layout(m).routed, " pre-route jobs at m=", m);
  p.peers = e.peers;
  p.s = e.s;
  p.w = l.w;
  p.rank = e.rank;
  p.m = m;
  p.o = data_as<const __nv_bfloat16>(o);
  p.o_stride = o.stride(0);
  p.positions = ptr<long>(positions);
  p.input_ids = ids ? ptr<void>(*input_ids) : nullptr;
  p.ids64 = ids && input_ids->scalar_type() == at::kLong;
  p.image_lo = l.image_lo;
  p.cos_sin = ptr<float>(cos_sin);
  p.resid0 = ptr<__nv_bfloat16>(resid0);
  p.attn_post = mut<float>(attn_post);
  p.attn_comb = mut<float>(attn_comb);
  p.attn_pre = mut<float>(attn_pre);
  p.y_ffn = mut<__nv_bfloat16>(y_ffn);
  p.resid1 = mut<__nv_bfloat16>(resid1);
  p.ffn_post = mut<float>(ffn_post);
  p.ffn_comb = mut<float>(ffn_comb);
  p.ffn_pre = mut<float>(ffn_pre);
#if BRRR_STEER
  p.steer_scale = nullptr;
  if (steer_scale) {
    TORCH_CHECK(l.w.steer_dir, "BRRR tail: run_steered on a layer without steering constants (set_steering)");
    TORCH_CHECK(steer_scale->is_cuda() && steer_scale->device() == resid0.device() && steer_scale->scalar_type() == at::kFloat &&
                    steer_scale->is_contiguous() && steer_scale->numel() >= m,
                "BRRR tail: steering scale must be a contiguous fp32 tensor of >= ", m, " rows on the layer's device");
    p.steer_scale = ptr<float>(*steer_scale);
  }
#else
  TORCH_CHECK(!steer_scale, "BRRR tail: this build has no steering (BRRR_STEER)");
#endif
  launch_pdl("layer kernel", layer_kernel, dim3(e.ctas), dim3(THREADS), sizeof(Smem), p);
}

void run(int64_t handle, int64_t layer, torch::Tensor o, torch::Tensor positions, std::optional<torch::Tensor> input_ids, torch::Tensor cos_sin, torch::Tensor resid0,
         torch::Tensor attn_post, torch::Tensor attn_comb, torch::Tensor attn_pre, torch::Tensor y_ffn, torch::Tensor resid1, torch::Tensor ffn_post, torch::Tensor ffn_comb, torch::Tensor ffn_pre) {
  launch(handle, layer, o, positions, input_ids, cos_sin, resid0, attn_post, attn_comb, attn_pre, y_ffn, resid1, ffn_post, ffn_comb, ffn_pre, nullptr);
}

#if BRRR_STEER
void run_steered(int64_t handle, int64_t layer, torch::Tensor o, torch::Tensor positions, std::optional<torch::Tensor> input_ids, torch::Tensor cos_sin, torch::Tensor resid0,
                 torch::Tensor attn_post, torch::Tensor attn_comb, torch::Tensor attn_pre, torch::Tensor steer_scale, torch::Tensor y_ffn, torch::Tensor resid1, torch::Tensor ffn_post,
                 torch::Tensor ffn_comb, torch::Tensor ffn_pre) {
  launch(handle, layer, o, positions, input_ids, cos_sin, resid0, attn_post, attn_comb, attn_pre, y_ffn, resid1, ffn_post, ffn_comb, ffn_pre, &steer_scale);
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.attr("RANKS") = RANKS;
  m.attr("MAXM") = MAXM;
  m.attr("HID") = HID;
  m.attr("HC") = HC;
  m.attr("EXPERTS") = EXPERTS;
  m.attr("INTER") = INTER;
  m.attr("OA") = OA;
  m.attr("OAK") = OAK;
  m.attr("SH2_LD") = SH2_LD;
  m.attr("RP_LAYOUT") = int(RP_LAYOUT);
  // Constants contract: Engine.register asserts these against Flare's model config and mHC options.
  m.attr("MIX") = MIX;
  m.attr("TOPK") = TOPK;
  m.attr("ROPE") = ROPE;
  m.attr("LIMIT") = double(LIMIT);
  m.attr("ROUTE_SCALE") = double(ROUTE_SCALE);
  m.attr("SINKHORN") = SINKHORN;
  m.attr("HC_EPS") = double(HC_EPS);
  m.attr("NORM_EPS") = double(NORM_EPS);
  m.attr("TWO_SHOT_MIN") = TWO_SHOT_MIN;
  m.attr("SPIN_NS") = int(SPIN_NS);
  m.attr("PER_SM") = PER_SM;
  m.attr("KS_WOB") = KS_WOB;
  m.attr("WOA_SMALL") = WOA_SMALL_MAX;
  m.attr("WOA_UNEVEN") = WOA_UNEVEN;
  // Compile-time switch of this build (brrr.kernel_defines); Engine checks it against its TP3Config.
  m.attr("SPIN_TRAP") = SPIN_TRAP;
  m.attr("STEER") = STEER;
  m.attr("ORDERED") = ORDERED;
  m.def("route_state", &route_state);
  m.def("clear_routes", &clear_routes);
  m.def("last_yattn", &last_yattn);
  m.def("flag_bytes", &flag_bytes);
  m.def("prof_read", &prof_read);
  m.def("set_pad", &set_pad);
  m.def("csf_expand_all", &csf_expand_all);
  m.def("inbox_bytes", &inbox_bytes);
  m.def("alloc_ipc", &alloc_ipc);
  m.def("open_ipc", &open_ipc);
  m.def("create", &create, py::arg("buffers"), py::arg("rank"), py::arg("epoch_addr") = 0, py::arg("attn_per_chunk_flags") = 1);
  m.def("add_layer", &add_layer, py::arg("handle"), py::arg("woa"), py::arg("woa_s"), py::arg("wob"), py::arg("wob_s"), py::arg("gate"), py::arg("gate_b"),
        py::arg("bias_vl"), py::arg("sh13"), py::arg("sh13_s"), py::arg("sh2"), py::arg("sh2_s"), py::arg("w13"), py::arg("s13"), py::arg("w2"), py::arg("s2"),
        py::arg("fn"), py::arg("hc_scale"), py::arg("hc_base"), py::arg("norm"), py::arg("image_lo"), py::arg("csf13_storage") = py::none(),
        py::arg("csf13_gdesc") = py::none(), py::arg("csf13_meta") = std::vector<int64_t>{}, py::arg("csf2_storage") = py::none(),
        py::arg("csf2_gdesc") = py::none(), py::arg("csf2_meta") = std::vector<int64_t>{}, py::arg("csf13_debug") = py::none(),
        py::arg("csf2_debug") = py::none(), py::arg("csf13_debug_desc") = py::none(), py::arg("csf2_debug_desc") = py::none(),
        py::arg("dense_native") = false, py::arg("route_counts") = py::none(), py::arg("hot_map") = py::none(),
        py::arg("hot_rows") = int64_t(EXPERTS));
  m.def("run", &run);
#if BRRR_STEER
  m.def("set_steering", &set_steering);
  m.def("run_steered", &run_steered);
#endif
}
