/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host- or graph-produced n-gram requests and asynchronous compact-row delivery.

mod cache;
#[cfg(unix)]
mod prepared;
mod read_ahead;
mod reader;
mod row_ids;
use crate::{
    Error, Result,
    cuda::{Event, PinnedHost, Stream},
    execution::{ExecutionManifest, GraphKind, Upload},
};
#[cfg(unix)]
pub use prepared::{PlaneSource, PreparedTables, SourceSegment, TableSource};
pub(crate) use reader::Plane;
pub(crate) use read_ahead::read_ahead_requested;
pub(crate) use row_ids::RowHash;
use read_ahead::Advisory;
#[cfg(feature = "server")]
use std::fmt::Write;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle, Thread},
    time::{Duration, Instant},
};

#[derive(Default)]
pub(crate) struct Stats {
    gathers: AtomicU64,
    gather_ns: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    evictions: AtomicU64,
    resident_rows: AtomicU64,
    capacity_rows: AtomicU64,
    cache_bytes: AtomicU64,
    resident_scale_bytes: AtomicU64,
    disk_reads: AtomicU64,
    disk_bytes: AtomicU64,
    disk_plan_ns: AtomicU64,
    disk_submit_ns: AtomicU64,
    disk_wait_ns: AtomicU64,
    disk_complete_ns: AtomicU64,
    /// TP3 fork (perf/engram-readahead): set when any table of this set reads ahead; the
    /// counters below are rendered only then, so /metrics is unchanged with read-ahead off.
    read_ahead: AtomicBool,
    read_ahead_queued: AtomicU64,
    read_ahead_loaded: AtomicU64,
    read_ahead_dropped: AtomicU64,
    read_ahead_errors: AtomicU64,
    read_ahead_skipped: AtomicU64,
}

impl Stats {
    #[cfg(feature = "server")]
    pub(crate) fn render(&self, text: &mut String) {
        for (name, kind, help, counter) in [
            (
                "gathers_total",
                "counter",
                "Embedding row gathers, excluding the wait for GPU-produced IDs.",
                &self.gathers,
            ),
            (
                "gather_nanoseconds_total",
                "counter",
                "Wall time inside embedding row gathers.",
                &self.gather_ns,
            ),
            (
                "row_cache_hits_total",
                "counter",
                "Cached row hits; duplicate IDs within one gather count once.",
                &self.hits,
            ),
            (
                "row_cache_misses_total",
                "counter",
                "Cached row misses; duplicate IDs within one gather count once.",
                &self.misses,
            ),
            (
                "row_cache_evictions_total",
                "counter",
                "Resident rows replaced by other row IDs.",
                &self.evictions,
            ),
            (
                "row_cache_resident_rows",
                "gauge",
                "Rows held in the embedding row cache.",
                &self.resident_rows,
            ),
            (
                "row_cache_capacity_rows",
                "gauge",
                "Maximum rows held in the embedding row cache.",
                &self.capacity_rows,
            ),
            (
                "row_cache_bytes",
                "gauge",
                "Allocated embedding row payload, usage counters, index and bounded miss workspace bytes.",
                &self.cache_bytes,
            ),
            (
                "resident_scale_bytes",
                "gauge",
                "Resident scale-plane allocation bytes for disk-backed embedding tables, separate from the row cache.",
                &self.resident_scale_bytes,
            ),
            (
                "disk_reads_total",
                "counter",
                "Direct-I/O embedding read requests queued.",
                &self.disk_reads,
            ),
            (
                "disk_bytes_total",
                "counter",
                "Bytes requested by direct-I/O embedding reads.",
                &self.disk_bytes,
            ),
            (
                "disk_plan_nanoseconds_total",
                "counter",
                "Wall time building, sorting and coalescing disk row reads.",
                &self.disk_plan_ns,
            ),
            (
                "disk_submit_nanoseconds_total",
                "counter",
                "Cumulative worker wall time preparing and issuing disk read batches; Linux kernel submission is included in wait time.",
                &self.disk_submit_ns,
            ),
            (
                "disk_wait_nanoseconds_total",
                "counter",
                "Cumulative worker wall time in disk completion waits, including Linux io_uring submission.",
                &self.disk_wait_ns,
            ),
            (
                "disk_complete_nanoseconds_total",
                "counter",
                "Cumulative worker wall time processing disk completions and copying rows, excluding completion waits.",
                &self.disk_complete_ns,
            ),
        ] {
            let _ = writeln!(
                text,
                "# HELP flare_engram_{name} {help}\n# TYPE flare_engram_{name} {kind}\nflare_engram_{name} {}",
                counter.load(Ordering::Relaxed)
            );
        }
        if !self.read_ahead.load(Ordering::Relaxed) {
            return;
        }
        for (name, help, counter) in [
            ("read_ahead_rows_total", "Predicted prefill rows handed to disk-table readers ahead of their graph.", &self.read_ahead_queued),
            ("read_ahead_rows_loaded_total", "Rows read ahead from disk into the row cache (resident rows are pinned, not counted).", &self.read_ahead_loaded),
            ("read_ahead_rows_dropped_total", "Predicted rows not read ahead because the advisory queue or a batch was full.", &self.read_ahead_dropped),
            ("read_ahead_windows_skipped_total", "Read-ahead windows dropped unread because their gather was already served or their request left.", &self.read_ahead_skipped),
            ("read_ahead_errors_total", "Read-ahead batches that failed; a failure fails the table as a failed gather does.", &self.read_ahead_errors),
        ] {
            let _ = writeln!(
                text,
                "# HELP flare_engram_{name} {help}\n# TYPE flare_engram_{name} counter\nflare_engram_{name} {}",
                counter.load(Ordering::Relaxed)
            );
        }
    }
}

pub use reader::default_gather_threads;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Storage {
    #[default]
    Disk,
    Ram,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Table {
    pub(crate) planes: [Plane; 2],
    pub(crate) heads: usize,
    /// True when each rank stages only its own local head subset (e.g. 8 of 24 heads on TP3),
    /// so every rank publishes distinct row IDs and receives its own staging frames even when
    /// reading from a shared unpartitioned host table.
    pub(crate) per_rank: bool,
    pub(crate) weights: String,
    pub(crate) scales: String,
    pub(crate) hashes: String,
    pub(crate) event: String,
    pub(crate) request: Option<String>,
    /// TP3 fork (perf/engram-readahead): the declared row hash when the host can predict this
    /// table's rows (unpartitioned `XOR_MODULO`); None never reads ahead.
    pub(crate) hash: Option<Arc<RowHash>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Index {
    pub(crate) tables: Vec<Table>,
}

impl Table {
    fn staging_payload_bytes(&self, max_tokens: usize) -> Result<usize> {
        self.planes.iter().try_fold(0usize, |total, plane| total.checked_add(plane.row_bytes))
            .and_then(|bytes| bytes.checked_mul(self.heads))
            .and_then(|bytes| bytes.checked_mul(max_tokens))
            .ok_or_else(|| "embedding staging exceeds host range".into())
    }
}

pub(crate) fn staging_bytes(index: &Index, max_tokens: usize, ranks: usize) -> Result<u64> {
    index.tables.iter().try_fold(0u64, |total, table| {
        let lanes = if table.per_rank { ranks.max(1) } else { 1 } as u64;
        let payload = table.staging_payload_bytes(max_tokens)? as u64;
        payload.checked_add(4)
            .and_then(|bytes| bytes.checked_mul(STAGING_FRAMES as u64))
            .and_then(|bytes| bytes.checked_mul(lanes))
            .and_then(|bytes| total.checked_add(bytes))
            .ok_or_else(|| "embedding staging exceeds the memory budget".into())
    })
}

struct Staging {
    data: PinnedHost,
    ready: PinnedHost,
    error: Mutex<Option<Error>>,
}
impl Staging {
    fn signal(&self, generation: u32) {
        unsafe { &*(self.ready.address() as *const AtomicU32) }
            .store(generation, Ordering::Release);
    }
}
enum Requests {
    Host(Vec<i64>),
    Device {
        address: usize,
        flag: usize,
        sequence: u64,
        rows: usize,
    },
}
impl Requests {
    /// Whether `read` would return without waiting for the graph to publish its ids.
    fn ready(&self) -> bool {
        match self {
            Self::Host(_) => true,
            Self::Device { flag, sequence, .. } => {
                unsafe { &*(*flag as *const AtomicU64) }.load(Ordering::Acquire) == *sequence
            }
        }
    }
    fn rows(&self) -> usize {
        match self {
            Self::Host(ids) => ids.len(),
            Self::Device { rows, .. } => *rows,
        }
    }
    fn read(self, cancelled: &AtomicBool) -> Result<Vec<i64>> {
        match self {
            Self::Host(ids) => Ok(ids),
            Self::Device {
                address,
                flag,
                sequence,
                rows,
            } => {
                let ready = unsafe { &*(flag as *const AtomicU64) };
                // The graph publishes these ids part-way through the step that
                // consumes them, so this wait spins only while they may still be in
                // flight and gives the core up once they are not. Optional
                // FLARE_ENGRAM_SPIN_US (default 0) extends the initial spin window.
                let spin_budget_us = engram_spin_us();
                let mut spins = 0u32;
                let mut deadline: Option<Instant> = None;
                while ready.load(Ordering::Acquire) != sequence {
                    if cancelled.load(Ordering::Acquire) {
                        return Err("n-gram request cancelled".into());
                    }
                    if spin_budget_us > 0 {
                        if spins & 0x3ff == 0 {
                            let limit = *deadline.get_or_insert_with(|| {
                                Instant::now() + Duration::from_micros(spin_budget_us)
                            });
                            if Instant::now() >= limit {
                                yield_core((spins >> 10).saturating_sub(1));
                                spins = spins.saturating_add(1);
                                continue;
                            }
                        }
                        std::hint::spin_loop();
                    } else if spins < SPIN_ROUNDS {
                        std::hint::spin_loop();
                    } else {
                        yield_core(spins - SPIN_ROUNDS);
                    }
                    spins = spins.saturating_add(1);
                }
                Ok(unsafe { std::slice::from_raw_parts(address as *const i64, rows) }.to_vec())
            }
        }
    }
}
fn engram_spin_us() -> u64 {
    static SPIN_US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SPIN_US.get_or_init(|| {
        std::env::var("FLARE_ENGRAM_SPIN_US")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(0)
    })
}
const STAGING_FRAMES: usize = 2;
/// Rounds a waiting gather spins before it starts napping, enough to catch ids
/// the graph published before the wait began.
const SPIN_ROUNDS: u32 = 1 << 12;
/// Nap growth per round after the spin; the tenth round reaches `NAP_MAX`.
const NAP_GROWTH: Duration = Duration::from_micros(25);
/// Longest nap between rounds, under one decode step so that a nap delays the
/// published ids by no more than its own length.
const NAP_MAX: Duration = Duration::from_micros(250);

/// Gives up the core for one round of a wait that has spent its spin, napping
/// `NAP_GROWTH` longer each round up to `NAP_MAX`.
///
/// Windows keeps the spin: its timer granularity is longer than `NAP_MAX`, so a
/// nap there could delay the ids by more than the decode step this wait stays
/// under.
#[cfg(not(windows))]
fn yield_core(rounds: u32) {
    thread::sleep(NAP_GROWTH.saturating_mul(rounds.saturating_add(1)).min(NAP_MAX));
}

#[cfg(windows)]
fn yield_core(_: u32) {
    std::hint::spin_loop();
}

struct Job {
    lane: usize,
    frame: usize,
    generation: u32,
    /// Submission sequence (TP3 fork, perf/engram-readahead: read-ahead drops windows whose
    /// gather was served).
    sequence: u64,
    requests: Requests,
}
#[derive(Clone, Copy, Default)]
struct State {
    sequence: u64,
    generation: u32,
    rows: usize,
    submitted: usize,
    completed: usize,
}
impl State {
    fn prepare(&mut self, rank: usize, sequence: u64, rows: usize, ranks: usize) -> Result<bool> {
        if sequence == 0 || rank >= ranks {
            return Err("invalid Engram submission identity".into());
        }
        let fresh = self.sequence != sequence;
        if fresh {
            if self.sequence > sequence
                || (self.sequence != 0 && self.completed != (1 << ranks) - 1)
            {
                return Err("Engram staging reused before all ranks completed".into());
            }
            *self = Self {
                sequence,
                generation: self.generation.wrapping_add(1),
                rows,
                submitted: 0,
                completed: 0,
            };
        }
        if self.rows != rows {
            return Err("ranks submitted different Engram row counts".into());
        }
        if self.submitted & (1 << rank) != 0 {
            return Err("rank submitted Engram generation twice".into());
        }
        self.submitted |= 1 << rank;
        Ok(fresh)
    }

    fn finish(&mut self, rank: usize, sequence: u64) -> Result<()> {
        if self.sequence != sequence || self.submitted & (1 << rank) == 0 {
            return Err("Engram completion does not match its submission".into());
        }
        if self.completed & (1 << rank) != 0 {
            return Err("rank completed Engram generation twice".into());
        }
        self.completed |= 1 << rank;
        Ok(())
    }
}

struct Layer {
    per_rank: bool,
    staging: Vec<Arc<[Staging; STAGING_FRAMES]>>,
    senders: Vec<Option<mpsc::SyncSender<Job>>>,
    workers: Vec<JoinHandle<()>>,
    states: Vec<Mutex<[State; STAGING_FRAMES]>>,
    cancelled: Arc<AtomicBool>,
    /// TP3 fork (perf/engram-readahead): Some only for a table that reads ahead; its reader
    /// thread then parks instead of blocking on `sender`, so producers unpark `thread`.
    advisory: Option<(Arc<Advisory>, Thread)>,
}
impl Layer {
    fn wake(&self) {
        if let Some((_, thread)) = &self.advisory {
            thread.unpark();
        }
    }
    fn lane_routing(&self, rank: usize, ranks: usize) -> (usize, usize, usize) {
        if self.per_rank {
            (rank, 0, 1)
        } else {
            (0, rank, ranks)
        }
    }
}
impl Drop for Layer {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        for sender in &mut self.senders {
            sender.take();
        }
        // A read-ahead thread parks rather than blocking on the closed channel.
        self.wake();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Serve one real gather into its staging frame and release the frame (the reader loop body of
/// tp3-main, shared unchanged by both loops below).
fn serve(
    reader: &mut reader::Reader,
    planes: &[Plane; 2],
    buffers: &[Staging; STAGING_FRAMES],
    row_bytes: usize,
    stop: &AtomicBool,
    job: Job,
) {
    // Each frame stays owned until every rank retires its consuming graph.
    // The reader alone writes its data before publishing the ready flag.
    let buffers = &buffers[job.frame];
    let output = unsafe {
        std::slice::from_raw_parts_mut(
            buffers.data.address() as *mut u8,
            job.requests.rows() * row_bytes,
        )
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if stop.load(Ordering::Acquire) {
            return Err("n-gram request cancelled".into());
        }
        let ids = job.requests.read(stop)?;
        reader.gather(planes, &ids, output)
    }))
    .unwrap_or_else(|_| Err("Engram reader panicked".into()));
    if let Err(error) = result {
        output.fill(0);
        *buffers.error.lock().unwrap() = Some(error);
        stop.store(true, Ordering::Release);
    }
    // Release failed reads too, so graph shutdown can drain before reporting the error.
    buffers.signal(job.generation);
}

pub(crate) struct Shared {
    pub index: Arc<Index>,
    layers: Vec<Layer>,
    ranks: usize,
    max_tokens: usize,
    stats: Arc<Stats>,
    /// TP3 fork (perf/engram-readahead): any layer reads ahead.
    #[cfg_attr(not(feature = "server"), allow(dead_code))]
    read_ahead: bool,
}
impl Shared {
    #[cfg(test)]
    pub(crate) fn layout_fixture(index: Index, ranks: usize) -> Arc<Self> {
        Arc::new(Self {
            index: Arc::new(index),
            layers: Vec::new(),
            ranks,
            max_tokens: 4096,
            stats: Arc::new(Stats::default()),
            read_ahead: false,
        })
    }

    pub fn new(
        index: Index,
        files: &[Arc<std::fs::File>],
        ranks: usize,
        max_tokens: usize,
        context: usize,
        device: i32,
        storage: &[Storage],
        cache_bytes: u64,
        workers: usize,
        gather_threads: usize,
        resident_scales: bool,
        pin_cpu_base: usize,
    ) -> Result<Arc<Self>> {
        if storage.len() != index.tables.len() {
            return Err("embedding table storage list does not match the table count".into());
        }
        if cache_bytes != 0 && storage.iter().all(|s| *s == Storage::Ram) {
            return Err("embedding row cache requires disk table storage".into());
        }
        let index = Arc::new(index);
        let stats = Arc::new(Stats::default());
        // Disk tables of this set share the row-cache budget in proportion to their bytes;
        // RAM-resident tables take none of it.
        let disk_bytes: u128 = index
            .tables
            .iter()
            .zip(storage)
            .filter(|(_, storage)| **storage == Storage::Disk)
            .flat_map(|(table, _)| &table.planes)
            .map(|plane| plane.nbytes as u128)
            .sum::<u128>()
            .max(1);
        let readers = index
            .tables
            .iter()
            .zip(storage)
            .map(|(table, storage)| {
                let ram = *storage == Storage::Ram;
                let bytes: u128 = table.planes.iter().map(|plane| plane.nbytes as u128).sum();
                let budget = if ram {
                    0
                } else {
                    usize::try_from(cache_bytes as u128 * bytes / disk_bytes)
                        .map_err(|_| "embedding row cache exceeds host address range")?
                };
                let rows = max_tokens
                    .checked_mul(table.heads)
                    .ok_or("embedding row capacity exceeds host range")?;
                reader::Reader::from_files(
                    files,
                    &table.planes,
                    ram,
                    budget,
                    rows,
                    stats.clone(),
                    workers,
                    resident_scales && !ram,
                )
                .map(|reader| reader.with_gather_threads(gather_threads))
            })
            .collect::<Result<Vec<_>>>()?;
        Self::with_readers(
            index,
            readers,
            ranks,
            max_tokens,
            context,
            device,
            stats,
            pin_cpu_base,
        )
    }

    fn with_readers(
        index: Arc<Index>,
        readers: Vec<reader::Reader>,
        ranks: usize,
        max_tokens: usize,
        context: usize,
        device: i32,
        stats: Arc<Stats>,
        pin_cpu_base: usize,
    ) -> Result<Arc<Self>> {
        if ranks == 0 || ranks > 16 || max_tokens == 0 || readers.len() != index.tables.len() {
            return Err("invalid embedding delivery geometry".into());
        }
        let stream = Stream::open(context, device)?;
        let mut layers = Vec::new();
        let mut read_ahead = false;
        let mut next_pin_cpu = pin_cpu_base;
        for (table, mut reader) in index.tables.iter().zip(readers) {
            let per_rank = table.per_rank && ranks > 1;
            let lanes = if per_rank { ranks } else { 1 };
            let staging_bytes = table.staging_payload_bytes(max_tokens)?;
            let mut staging = Vec::with_capacity(lanes);
            for _ in 0..lanes {
                let mut frames = Vec::with_capacity(STAGING_FRAMES);
                for _ in 0..STAGING_FRAMES {
                    frames.push(Staging {
                        data: stream.mapped_host_buffer(staging_bytes)?,
                        ready: stream.mapped_host_buffer(4)?,
                        error: Mutex::new(None),
                    });
                }
                let frames: Arc<[Staging; STAGING_FRAMES]> = Arc::new(
                    frames
                        .try_into()
                        .map_err(|_| "invalid Engram staging frame count")?,
                );
                staging.push(frames);
            }
            let planes = table.planes;
            let row_bytes = planes.iter().map(|p| p.row_bytes).sum::<usize>();
            let cancelled = Arc::new(AtomicBool::new(false));
            let mut senders = Vec::with_capacity(lanes);
            let mut workers = Vec::new();
            let advisory = if per_rank && reader.ram_planes().is_some() {
                // Compact per-rank staging on a shared RAM table: keep one backing resident
                // table in host RAM and give each rank its own parallel reader thread and
                // compact staging frames.
                let ram_planes = reader.ram_planes().unwrap().clone();
                let gather_threads = reader.gather_threads();
                drop(reader);
                for (lane, lane_staging) in staging.iter().enumerate() {
                    let (sender, receiver) = mpsc::sync_channel::<Job>(STAGING_FRAMES);
                    let buffers = lane_staging.clone();
                    let stop = cancelled.clone();
                    let mut lane_reader = reader::Reader::resident(ram_planes.clone(), stats.clone())
                        .with_gather_threads(gather_threads);
                    let pin_cpu = next_pin_cpu;
                    next_pin_cpu += 1;
                    let worker = thread::Builder::new()
                        .name(format!("flare-{}-r{lane}", table.event))
                        .spawn(move || {
                            crate::os::pin_current_thread(pin_cpu);
                            while let Ok(job) = receiver.recv() {
                                serve(&mut lane_reader, &planes, &buffers, row_bytes, &stop, job);
                            }
                        })
                        .map_err(|e| format!("start Engram reader: {e}"))?;
                    senders.push(Some(sender));
                    workers.push(worker);
                }
                None
            } else {
                let reader_stats = stats.clone();
                let (sender, receiver) = mpsc::sync_channel::<Job>(STAGING_FRAMES * lanes);
                let all_buffers = staging.clone();
                let stop = cancelled.clone();
                // TP3 fork (perf/engram-readahead): only a disk table with a row cache and a
                // predictable hash, with FLARE_ENGRAM_READ_AHEAD=1, gets advisory state. Every other
                // table (all RAM-resident ones) runs the tp3-main loop below and allocates nothing.
                let advisory = read_ahead::advisory_for(&reader, table.hash.as_ref());
                let worker_advisory = advisory.clone();
                let hash = advisory.as_ref().and(table.hash.clone());
                let pin_cpu = next_pin_cpu;
                next_pin_cpu += 1;
                // One worker serves every rank whose rows this table holds; a rank-partitioned
                // table becomes a separate `Shared` with its own workers.
                let worker = thread::Builder::new()
                    .name(format!("flare-{}", table.event))
                    .spawn(move || {
                        crate::os::pin_current_thread(pin_cpu);
                        let (Some(advisory), Some(hash)) = (worker_advisory, hash) else {
                            while let Ok(job) = receiver.recv() {
                                let lane = job.lane;
                                serve(&mut reader, &planes, &all_buffers[lane], row_bytes, &stop, job);
                            }
                            return;
                        };
                        read_ahead::run_reader(
                            receiver, &advisory, &hash, &mut reader, &planes, &all_buffers, row_bytes,
                            &stop, &reader_stats,
                        );
                    })
                    .map_err(|e| format!("start Engram reader: {e}"))?;
                let advisory = advisory.map(|advisory| (advisory, worker.thread().clone()));
                for _ in 0..lanes {
                    senders.push(Some(sender.clone()));
                }
                workers.push(worker);
                advisory
            };
            read_ahead |= advisory.is_some();
            let states = (0..lanes)
                .map(|_| Mutex::new(std::array::from_fn(|_| State::default())))
                .collect();
            layers.push(Layer {
                per_rank,
                staging,
                cancelled,
                senders,
                workers,
                states,
                advisory,
            });
        }
        stats.read_ahead.store(read_ahead, Ordering::Relaxed);
        Ok(Arc::new(Self {
            index,
            layers,
            ranks,
            max_tokens,
            stats,
            read_ahead,
        }))
    }

    #[cfg(feature = "server")]
    pub(crate) fn stats(&self) -> Arc<Stats> {
        self.stats.clone()
    }

    pub(crate) fn cache_bytes(&self) -> u64 {
        self.stats.cache_bytes.load(Ordering::Relaxed)
    }

    fn validate_start(&self, table: usize, rank: usize, sequence: u64, rows: usize) -> Result<()> {
        if rank >= self.ranks {
            return Err("invalid Engram submission identity".into());
        }
        if rows > self.max_tokens * self.index.tables[table].heads {
            return Err("Engram input exceeds staged capacity".into());
        }
        let layer = &self.layers[table];
        let (lane, state_rank, state_ranks) = layer.lane_routing(rank, self.ranks);
        let states = layer.states[lane].lock().unwrap();
        for staging in layer.staging[lane].iter() {
            if let Some(error) = staging.error.lock().unwrap().as_ref() {
                return Err(error.clone());
            }
        }
        let mut state = states[sequence as usize % STAGING_FRAMES];
        state.prepare(state_rank, sequence, rows, state_ranks)?;
        Ok(())
    }

    fn start(
        &self,
        table: usize,
        rank: usize,
        sequence: u64,
        requests: Requests,
    ) -> Result<(usize, u32)> {
        if rank >= self.ranks {
            return Err("invalid Engram submission identity".into());
        }
        let rows = requests.rows();
        if rows > self.max_tokens * self.index.tables[table].heads {
            return Err("Engram input exceeds staged capacity".into());
        }
        let layer = &self.layers[table];
        let (lane, state_rank, state_ranks) = layer.lane_routing(rank, self.ranks);
        let mut states = layer.states[lane].lock().unwrap();
        for staging in layer.staging[lane].iter() {
            if let Some(error) = staging.error.lock().unwrap().as_ref() {
                return Err(error.clone());
            }
        }
        let frame = sequence as usize % STAGING_FRAMES;
        let mut state = states[frame];
        let fresh = state.prepare(state_rank, sequence, rows, state_ranks)?;
        if fresh {
            layer.senders[lane]
                .as_ref()
                .unwrap()
                .send(Job {
                    lane,
                    frame,
                    generation: state.generation,
                    sequence,
                    requests,
                })
                .map_err(|_| "Engram reader disconnected")?;
            // No-op unless this table reads ahead (its thread parks instead of blocking in recv).
            layer.wake();
        }
        states[frame] = state;
        Ok((rows, state.generation))
    }
    fn finish(&self, rank: usize, sequence: u64) -> Result<()> {
        if rank >= self.ranks {
            return Err("invalid Engram submission identity".into());
        }
        let mut error = None;
        for layer in &self.layers {
            let (lane, state_rank, _) = layer.lane_routing(rank, self.ranks);
            let mut states = layer.states[lane].lock().unwrap();
            let frame = sequence as usize % STAGING_FRAMES;
            states[frame].finish(state_rank, sequence)?;
            error = error.or_else(|| layer.staging[lane][frame].error.lock().unwrap().clone());
        }
        error.map_or(Ok(()), Err)
    }
}

#[derive(Clone)]
pub struct Binding {
    pub(crate) shared: Arc<Shared>,
    pub(crate) rank: usize,
    pub(crate) events: Vec<usize>,
}
impl std::fmt::Debug for Binding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngramBinding")
            .field("rank", &self.rank)
            .finish()
    }
}
struct Transfer {
    stream: Stream,
    flags: [usize; STAGING_FRAMES],
    weights: usize,
    scales: usize,
    event: usize,
    request: Option<(usize, usize)>,
}
pub(crate) struct Transfers {
    binding: Binding,
    layers: Vec<Transfer>,
}
impl Transfers {
    pub fn new(binding: Binding, manifest: &ExecutionManifest) -> Result<Self> {
        if let Some(buffer) = manifest
            .buffers
            .iter()
            .find(|buffer| buffer.name == "readiness_timestamps")
        {
            if buffer.shape[1] / 3 != binding.shared.index.tables.len() {
                return Err(
                    "readiness interval count differs from the embedding table catalog".into(),
                );
            }
        }
        let mut layers = Vec::new();
        for (i, table) in binding.shared.index.tables.iter().enumerate() {
            let stream = Stream::open(manifest.context, manifest.device)?;
            let lane = if binding.shared.layers[i].per_rank {
                binding.rank
            } else {
                0
            };
            let mut flags = [0; STAGING_FRAMES];
            for (flag, staging) in flags
                .iter_mut()
                .zip(binding.shared.layers[i].staging[lane].iter())
            {
                *flag = stream.mapped_address(&staging.ready)?;
            }
            let request = table
                .request
                .as_ref()
                .map(|name| {
                    let buffer = manifest.buffer(name)?;
                    let address = stream.host_address(buffer.address, buffer.nbytes)?;
                    Ok::<_, Error>((address, address + buffer.nbytes - 8))
                })
                .transpose()?;
            layers.push(Transfer {
                stream,
                flags,
                weights: manifest.buffer(&table.weights)?.address,
                scales: manifest.buffer(&table.scales)?.address,
                event: binding.events[i],
                request,
            });
        }
        Ok(Self { binding, layers })
    }
    pub fn validate(&self, sequence: u64, tokens: usize, uploads: &[Upload<'_>]) -> Result<()> {
        for (i, (transfer, table)) in self
            .layers
            .iter()
            .zip(&self.binding.shared.index.tables)
            .enumerate()
        {
            let rows = if transfer.request.is_some() {
                tokens
                    .checked_mul(table.heads)
                    .ok_or("Engram row count overflow")?
            } else {
                let upload = uploads
                    .iter()
                    .find(|u| u.buffer == table.hashes)
                    .ok_or("step lacks n-gram hash IDs")?;
                if upload.offset != 0 || upload.bytes.len() % (table.heads * 8) != 0 {
                    return Err("invalid n-gram hash upload".into());
                }
                upload.bytes.len() / 8
            };
            self.binding
                .shared
                .validate_start(i, self.binding.rank, sequence, rows)?;
        }
        Ok(())
    }

    pub fn enqueue(
        &self,
        sequence: u64,
        tokens: usize,
        uploads: &[Upload<'_>],
        inputs_ready: &Event,
    ) -> Result<()> {
        self.validate(sequence, tokens, uploads)?;
        for (i, (transfer, table)) in self
            .layers
            .iter()
            .zip(&self.binding.shared.index.tables)
            .enumerate()
        {
            let heads = table.heads;
            let requests = if let Some((address, flag)) = transfer.request {
                Requests::Device {
                    address,
                    flag,
                    sequence,
                    rows: tokens * heads,
                }
            } else {
                let upload = uploads
                    .iter()
                    .find(|u| u.buffer == table.hashes)
                    .ok_or("step lacks n-gram hash IDs")?;
                if upload.offset != 0 || upload.bytes.len() % (heads * 8) != 0 {
                    return Err("invalid n-gram hash upload".into());
                }
                Requests::Host(
                    upload
                        .bytes
                        .chunks_exact(8)
                        .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                        .collect(),
                )
            };
            let (rows, generation) =
                self.binding
                    .shared
                    .start(i, self.binding.rank, sequence, requests)?;
            let frame = sequence as usize % STAGING_FRAMES;
            let lane = if self.binding.shared.layers[i].per_rank {
                self.binding.rank
            } else {
                0
            };
            let staging = &self.binding.shared.layers[i].staging[lane][frame];
            // The model stream records inputs_ready after the preceding graph, so DMA
            // cannot overwrite tables still consumed by that graph. Each graph waits for
            // its host gather before the following graph can overwrite the mailbox.
            transfer.stream.wait(inputs_ready)?;
            unsafe {
                transfer
                    .stream
                    .wait_value(transfer.flags[frame], generation)?;
                transfer.stream.upload(
                    transfer.weights,
                    &staging.data,
                    0,
                    rows * table.planes[0].row_bytes,
                )?;
                transfer.stream.upload(
                    transfer.scales,
                    &staging.data,
                    rows * table.planes[0].row_bytes,
                    rows * table.planes[1].row_bytes,
                )?;
                // The consumer is enqueued before a following submission re-records this event.
                transfer.stream.record_borrowed(transfer.event)?;
            }
        }
        Ok(())
    }
    pub fn finish(&self, sequence: u64) -> Result<()> {
        self.binding.shared.finish(self.binding.rank, sequence)
    }
    pub fn cancel(&self) {
        for layer in &self.binding.shared.layers {
            layer.cancelled.store(true, Ordering::Release);
        }
    }
    pub fn drain(&self) -> Result<()> {
        let mut error = None;
        for transfer in &self.layers {
            error = error.or(transfer.stream.synchronize().err());
        }
        error.map_or(Ok(()), Err)
    }
}

pub(crate) fn consumes(kind: GraphKind) -> bool {
    matches!(
        kind,
        GraphKind::Prefill
            | GraphKind::TerminalPrefill
            | GraphKind::ScoredPrefill
            | GraphKind::Decode
            | GraphKind::Speculative
            | GraphKind::Mixed
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn staging_budget_covers_two_payloads_and_ready_flags_per_table() {
        let table = Table {
            planes: [Plane { file: 0, offset: 0, nbytes: 1024, row_bytes: 64 },
                Plane { file: 0, offset: 1024, nbytes: 128, row_bytes: 8 }],
            heads: 4, per_rank: false, weights: "weights".into(), scales: "scales".into(),
            hashes: "hashes".into(), event: "ready".into(), request: None, hash: None,
        };
        let index = Index { tables: vec![table.clone(), table] };
        assert_eq!(staging_bytes(&index, 128, 3).unwrap(), 2 * 2 * (128 * 4 * (64 + 8) + 4));
        assert!(staging_bytes(&index, usize::MAX, 1).is_err());

        // Per-rank compact staging multiplies staging frames by the rank count in the group.
        let compact = Table {
            planes: [Plane { file: 0, offset: 0, nbytes: 1024, row_bytes: 256 },
                Plane { file: 0, offset: 1024, nbytes: 32, row_bytes: 8 }],
            heads: 8, per_rank: true, weights: "weights".into(), scales: "scales".into(),
            hashes: "hashes".into(), event: "ready".into(), request: Some("requests".into()), hash: None,
        };
        let legacy = Table { heads: 24, per_rank: false, ..compact.clone() };
        let compact_bytes = staging_bytes(&Index { tables: vec![compact] }, 16384, 3).unwrap();
        let legacy_bytes = staging_bytes(&Index { tables: vec![legacy] }, 16384, 3).unwrap();
        // Same total payload bytes across 3 ranks (3 * 8 == 24), plus 2 extra ready flags/frame.
        assert_eq!(compact_bytes - 3 * 2 * 4, legacy_bytes - 2 * 4);
    }

    #[test]
    fn two_frames_require_all_rank_retirement_before_reuse() {
        let mut states: [State; STAGING_FRAMES] = std::array::from_fn(|_| State::default());
        for sequence in 1..=2 {
            let state = &mut states[sequence as usize % STAGING_FRAMES];
            assert!(state.prepare(0, sequence, 24, 2).unwrap());
            assert!(!state.prepare(1, sequence, 24, 2).unwrap());
        }
        let first = &mut states[1];
        first.finish(0, 1).unwrap();
        assert!(first.prepare(0, 3, 48, 2).is_err());
        assert_eq!(first.sequence, 1);
        first.finish(1, 1).unwrap();
        assert!(first.prepare(0, 3, 48, 2).unwrap());
        assert_eq!(first.generation, 2);
        assert_eq!(first.rows, 48);
        assert_eq!(states[0].sequence, 2);
        assert_eq!(states[0].completed, 0);
    }

    #[test]
    fn frame_rejects_stale_duplicate_and_mismatched_rank_work() {
        let mut state = State::default();
        assert!(state.prepare(0, 0, 24, 2).is_err());
        assert!(state.prepare(2, 1, 24, 2).is_err());
        state.prepare(0, 1, 24, 2).unwrap();
        assert!(state.prepare(0, 1, 24, 2).is_err());
        assert!(state.prepare(1, 1, 48, 2).is_err());
        assert!(state.finish(1, 1).is_err());
        assert!(state.finish(0, 3).is_err());
        state.prepare(1, 1, 24, 2).unwrap();
        state.finish(0, 1).unwrap();
        assert!(state.finish(0, 1).is_err());
        state.finish(1, 1).unwrap();
        state.prepare(0, 3, 24, 2).unwrap();
        assert!(state.prepare(1, 1, 24, 2).is_err());
    }

    /// A device row set whose publication flag and cancellation the test owns.
    fn device_request(flag: &AtomicU64, ids: &[i64]) -> Requests {
        Requests::Device {
            address: ids.as_ptr() as usize,
            flag: flag as *const AtomicU64 as usize,
            sequence: 7,
            rows: ids.len(),
        }
    }

    /// Waits out the spin budget on any host, so a flag stored after it is stored
    /// while the reader is napping.
    const AFTER_SPIN: Duration = Duration::from_millis(5);

    #[test]
    fn device_request_wait_returns_ids_already_published() {
        let ids = [17i64, 31, 47];
        let flag = AtomicU64::new(7);
        assert_eq!(
            device_request(&flag, &ids).read(&AtomicBool::new(false)).unwrap(),
            ids
        );
    }

    #[test]
    fn device_request_wait_returns_ids_published_after_the_spin() {
        let ids = [17i64, 31, 47];
        let flag = AtomicU64::new(0);
        thread::scope(|scope| {
            let worker = scope.spawn(|| device_request(&flag, &ids).read(&AtomicBool::new(false)));
            thread::sleep(AFTER_SPIN);
            flag.store(7, Ordering::Release);
            assert_eq!(worker.join().unwrap().unwrap(), ids);
        });
    }

    #[test]
    fn device_request_wait_observes_cancellation_before_publication() {
        let ids = [17i64, 31, 47];
        let flag = AtomicU64::new(0);
        assert!(device_request(&flag, &ids)
            .read(&AtomicBool::new(true))
            .is_err());
    }

    #[test]
    fn device_request_wait_observes_cancellation_after_the_spin() {
        let ids = [17i64, 31, 47];
        let flag = AtomicU64::new(0);
        let cancelled = AtomicBool::new(false);
        thread::scope(|scope| {
            let worker = scope.spawn(|| device_request(&flag, &ids).read(&cancelled));
            thread::sleep(AFTER_SPIN);
            cancelled.store(true, Ordering::Release);
            assert!(worker.join().unwrap().is_err());
        });
    }

    #[test]
    fn shared_ram_compact_tp3_gathers_interleaved_heads_per_rank_without_duplicating_table() {
        // Option B contract: 1x shared 24-head RAM table in host memory, while TP3 ranks
        // 0, 1, 2 each gather only their own 8 interleaved heads:
        //   rank 0 -> [0, 3, 6, 9, 12, 15, 18, 21]
        //   rank 1 -> [1, 4, 7, 10, 13, 16, 19, 22]
        //   rank 2 -> [2, 5, 8, 11, 14, 17, 20, 23]
        let world = 3usize;
        let canonical_heads = 24usize;
        let compact_heads = canonical_heads / world;
        let rows_per_head = 4usize;
        let total_rows = canonical_heads * rows_per_head;
        let weight_bytes = 256usize;
        let scale_bytes = 8usize;
        let row_bytes = weight_bytes + scale_bytes;

        let path = std::env::temp_dir().join(format!(
            "helios-engram-compact-tp3-shared-{}.bin",
            std::process::id()
        ));
        let mut blob = vec![0u8; total_rows * row_bytes];
        for row in 0..total_rows {
            let head = (row / rows_per_head) as u8;
            let local = (row % rows_per_head) as u8;
            let w_off = row * weight_bytes;
            for col in 0..weight_bytes {
                blob[w_off + col] = head.wrapping_mul(11).wrapping_add(local.wrapping_mul(3)).wrapping_add(col as u8);
            }
            let s_off = total_rows * weight_bytes + row * scale_bytes;
            for col in 0..scale_bytes {
                blob[s_off + col] = 100u8.wrapping_add(head).wrapping_add((col as u8) * 5).wrapping_add(local);
            }
        }
        std::fs::write(&path, &blob).unwrap();

        let planes = [
            Plane {
                file: 0,
                offset: 0,
                nbytes: total_rows * weight_bytes,
                row_bytes: weight_bytes,
            },
            Plane {
                file: 0,
                offset: (total_rows * weight_bytes) as u64,
                nbytes: total_rows * scale_bytes,
                row_bytes: scale_bytes,
            },
        ];
        let stats = Arc::new(Stats::default());
        let master = reader::Reader::open(&path, &planes, true, 0, 16, stats.clone()).unwrap();
        let ram_planes = master.ram_planes().expect("RAM table must expose shared Resident Arcs").clone();
        // Verify all 3 ranks share the exact same Resident allocations (1x host RAM, not 3x).
        let rank_readers: Vec<_> = (0..world)
            .map(|_| reader::Reader::resident(ram_planes.clone(), stats.clone()).with_gather_threads(1))
            .collect();
        drop(master);
        // 1 copy in `ram_planes` + 3 clones in `rank_readers` == 4 strong refs to the SAME Resident.
        assert_eq!(Arc::strong_count(&ram_planes[0]), 1 + world);
        assert_eq!(Arc::strong_count(&ram_planes[1]), 1 + world);

        // Two tokens: token 0 picks offset 1 within each head; token 1 picks offset 3.
        let tokens = 2usize;
        let expected_heads: [[usize; 8]; 3] = [
            [0, 3, 6, 9, 12, 15, 18, 21],
            [1, 4, 7, 10, 13, 16, 19, 22],
            [2, 5, 8, 11, 14, 17, 20, 23],
        ];
        let mut states: Vec<[State; STAGING_FRAMES]> = (0..world)
            .map(|_| std::array::from_fn(|_| State::default()))
            .collect();
        for (rank, mut reader) in rank_readers.into_iter().enumerate() {
            // Each per-rank lane has independent State (ranks = 1), so every rank submits fresh=true.
            assert!(states[rank][1].prepare(0, 1, tokens * compact_heads, 1).unwrap());
            let mut ids = Vec::with_capacity(tokens * compact_heads);
            for (t, offset_in_head) in [1usize, 3usize].into_iter().enumerate() {
                let _ = t;
                for &head in &expected_heads[rank] {
                    ids.push((head * rows_per_head + offset_in_head) as i64);
                }
            }
            let mut out = vec![0u8; ids.len() * row_bytes];
            reader.gather(&planes, &ids, &mut out).unwrap();
            let scale_base = ids.len() * weight_bytes;
            for (slot, &row_id) in ids.iter().enumerate() {
                let row = row_id as usize;
                assert_eq!(
                    &out[slot * weight_bytes..(slot + 1) * weight_bytes],
                    &blob[row * weight_bytes..(row + 1) * weight_bytes],
                    "rank {rank} slot {slot} weight mismatch for row {row}"
                );
                assert_eq!(
                    &out[scale_base + slot * scale_bytes..scale_base + (slot + 1) * scale_bytes],
                    &blob[total_rows * weight_bytes + row * scale_bytes
                        ..total_rows * weight_bytes + (row + 1) * scale_bytes],
                    "rank {rank} slot {slot} scale mismatch for row {row}"
                );
            }
            states[rank][1].finish(0, 1).unwrap();
        }
        std::fs::remove_file(path).unwrap();
    }

    #[cfg(feature = "server")]
    #[test]
    fn metrics_are_unchanged_without_read_ahead() {
        let stats = Stats::default();
        let mut off = String::new();
        stats.render(&mut off);
        assert!(!off.contains("read_ahead"));
        stats.read_ahead.store(true, Ordering::Relaxed);
        let mut on = String::new();
        stats.render(&mut on);
        assert!(on.starts_with(&off) && on.contains("flare_engram_read_ahead_rows_total 0"));
    }

}
