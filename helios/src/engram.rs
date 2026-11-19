/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host- or graph-produced n-gram requests and asynchronous compact-row delivery.

mod cache;
#[cfg(unix)]
mod prepared;
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
pub(crate) use row_ids::RowHash;
#[cfg(feature = "server")]
use std::fmt::Write;
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        mpsc,
    },
    thread::{self, JoinHandle, Thread},
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

/// TP3 fork (perf/engram-readahead): `FLARE_ENGRAM_READ_AHEAD=1` lets disk-backed tables with a
/// row cache and a predictable hash read prefill rows ahead of their graph. Default off. Tables in
/// RAM (both DS41 tables in the TP3 default) never read ahead whatever the switch says.
pub(crate) fn read_ahead_requested() -> bool {
    static REQUESTED: OnceLock<bool> = OnceLock::new();
    *REQUESTED.get_or_init(|| {
        parse_read_ahead(std::env::var("FLARE_ENGRAM_READ_AHEAD").ok().as_deref())
    })
}

fn parse_read_ahead(value: Option<&str>) -> bool {
    matches!(value.map(str::trim), Some("1" | "on" | "true" | "yes"))
}

/// Rows in one advisory batch: a published gather waits at most one batch.
#[cfg_attr(not(feature = "server"), allow(dead_code))]
const READ_AHEAD_BATCH_ROWS: usize = 4096;
/// Advisory batches one table may hold queued (about a million rows).
const READ_AHEAD_SLOTS: usize = 256;

/// One advisory batch: prompt tokens whose rows a coming prefill step will request. `tokens`
/// holds the query positions from `first` on, preceded by up to the hash's history (fewer only at
/// the start of the prompt, where the hash reads pad). Excluded positions (images) are stored as
/// `u32::MAX`, which no token map holds, so they stop n-grams as in the graph. The reader thread
/// hashes the window itself, which keeps that work off the serving worker's submit path.
struct Window {
    tokens: Vec<u32>,
    first: usize,
}

/// Fill `window` with the tokens of query positions `[position, next)` and up to `history`
/// tokens before them, excluded positions as `u32::MAX`; returns where the queries start in it.
#[cfg_attr(not(feature = "server"), allow(dead_code))]
fn fill_window(
    history: usize,
    tokens: &[u32],
    position: usize,
    next: usize,
    excluded: &impl Fn(usize) -> bool,
    window: &mut Vec<u32>,
) -> usize {
    let base = position.saturating_sub(history);
    window.clear();
    window.extend((base..next).map(|at| if excluded(at) { u32::MAX } else { tokens[at] }));
    position - base
}

/// Advisory read-ahead state of one disk-backed table, shared by its producer (the serving
/// worker) and its reader thread. Exists only for tables that read ahead.
#[derive(Default)]
struct Advisory {
    queue: Mutex<VecDeque<Window>>,
    /// Recycled window buffers, so steady-state read-ahead allocates nothing.
    pool: Mutex<Vec<Vec<u32>>>,
}

impl Advisory {
    fn pop(&self) -> Option<Window> {
        self.queue.lock().unwrap().pop_front()
    }

    fn recycle(&self, mut tokens: Vec<u32>) {
        tokens.clear();
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < READ_AHEAD_SLOTS {
            pool.push(tokens);
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

pub(crate) fn staging_bytes(index: &Index, max_tokens: usize) -> Result<u64> {
    index.tables.iter().try_fold(0u64, |total, table| {
        let payload = table.staging_payload_bytes(max_tokens)? as u64;
        payload.checked_add(4).and_then(|bytes| bytes.checked_mul(STAGING_FRAMES as u64))
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
                while ready.load(Ordering::Acquire) != sequence {
                    if cancelled.load(Ordering::Acquire) {
                        return Err("n-gram request cancelled".into());
                    }
                    std::hint::spin_loop();
                }
                Ok(unsafe { std::slice::from_raw_parts(address as *const i64, rows) }.to_vec())
            }
        }
    }
}
const STAGING_FRAMES: usize = 2;

struct Job {
    frame: usize,
    generation: u32,
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
    staging: Arc<[Staging; STAGING_FRAMES]>,
    sender: Option<mpsc::SyncSender<Job>>,
    worker: Option<JoinHandle<()>>,
    state: Mutex<[State; STAGING_FRAMES]>,
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
}
impl Drop for Layer {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Release);
        self.sender.take();
        // A read-ahead thread parks rather than blocking on the closed channel.
        self.wake();
        if let Some(worker) = self.worker.take() {
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

/// What a read-ahead reader thread does next.
enum Work {
    /// A real gather whose ids are readable (or cancelled): always outranks warming.
    Gather(Job),
    /// One advisory batch.
    Warm(Window),
    /// A held gather still waits for its graph and nothing is queued: spin, as `read` does.
    Spin,
    /// Nothing held or queued: park until a producer unparks.
    Park,
    /// The submission channel is closed and drained.
    Stop,
}

/// TP3 fork (perf/engram-readahead, the c2606aae reader fix): a submission is taken into `held`
/// as soon as it arrives but gathered only once its ids are readable. Device ids exist only
/// after the step's graph starts, so a thread blocked on them would idle through the replay that
/// read-ahead is meant to overlap; while the held gather waits, advisory batches run one at a
/// time, so the gather starts at most one batch after its ids appear.
fn next_work(
    receiver: &mpsc::Receiver<Job>,
    advisory: &Advisory,
    held: &mut Option<Job>,
    cancelled: &AtomicBool,
) -> Work {
    if held.is_none() {
        match receiver.try_recv() {
            Ok(job) => *held = Some(job),
            Err(mpsc::TryRecvError::Disconnected) => return Work::Stop,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }
    if let Some(job) = held.as_ref() {
        if job.requests.ready() || cancelled.load(Ordering::Acquire) {
            return Work::Gather(held.take().unwrap());
        }
        return advisory.pop().map_or(Work::Spin, Work::Warm);
    }
    if cancelled.load(Ordering::Acquire) {
        // A cancelled or failed table warms nothing more; it only releases submissions.
        advisory.queue.lock().unwrap().clear();
        return Work::Park;
    }
    advisory.pop().map_or(Work::Park, Work::Warm)
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
        Self::with_readers(index, readers, ranks, max_tokens, context, device, stats)
    }

    fn with_readers(
        index: Arc<Index>,
        readers: Vec<reader::Reader>,
        ranks: usize,
        max_tokens: usize,
        context: usize,
        device: i32,
        stats: Arc<Stats>,
    ) -> Result<Arc<Self>> {
        if ranks == 0 || ranks > 16 || max_tokens == 0 || readers.len() != index.tables.len() {
            return Err("invalid embedding delivery geometry".into());
        }
        let stream = Stream::open(context, device)?;
        let mut layers = Vec::new();
        let mut read_ahead = false;
        for (table, mut reader) in index.tables.iter().zip(readers) {
            let reader_stats = stats.clone();
            let staging_bytes = table.staging_payload_bytes(max_tokens)?;
            let mut staging = Vec::with_capacity(STAGING_FRAMES);
            for _ in 0..STAGING_FRAMES {
                staging.push(Staging {
                    data: stream.mapped_host_buffer(staging_bytes)?,
                    ready: stream.mapped_host_buffer(4)?,
                    error: Mutex::new(None),
                });
            }
            let staging: Arc<[Staging; STAGING_FRAMES]> = Arc::new(
                staging
                    .try_into()
                    .map_err(|_| "invalid Engram staging frame count")?,
            );
            let (sender, receiver) = mpsc::sync_channel::<Job>(STAGING_FRAMES);
            let buffers = staging.clone();
            let planes = table.planes;
            let row_bytes = planes.iter().map(|p| p.row_bytes).sum::<usize>();
            let cancelled = Arc::new(AtomicBool::new(false));
            let stop = cancelled.clone();
            // TP3 fork (perf/engram-readahead): only a disk table with a row cache and a
            // predictable hash, with FLARE_ENGRAM_READ_AHEAD=1, gets advisory state. Every other
            // table (all RAM-resident ones) runs the tp3-main loop below and allocates nothing.
            let advisory = (read_ahead_requested() && reader.caches_rows() && table.hash.is_some())
                .then(|| Arc::new(Advisory::default()));
            let worker_advisory = advisory.clone();
            let hash = advisory.as_ref().and(table.hash.clone());
            let worker = thread::Builder::new()
                .name(format!("flare-{}", table.event))
                .spawn(move || {
                    let Some(advisory) = worker_advisory else {
                        while let Ok(job) = receiver.recv() {
                            serve(&mut reader, &planes, &buffers, row_bytes, &stop, job);
                        }
                        return;
                    };
                    let mut held = None;
                    let hash = hash.expect("read-ahead tables carry a row hash");
                    let (mut history, mut rows) = (Vec::new(), Vec::new());
                    loop {
                        match next_work(&receiver, &advisory, &mut held, &stop) {
                            Work::Gather(job) => {
                                serve(&mut reader, &planes, &buffers, row_bytes, &stop, job)
                            }
                            Work::Warm(batch) => {
                                rows.clear();
                                hash.predict(&batch.tokens, batch.first, batch.tokens.len(), |_| false, &mut history, &mut rows);
                                reader_stats.read_ahead_queued.fetch_add(rows.len() as u64, Ordering::Relaxed);
                                let warmed = std::panic::catch_unwind(
                                    std::panic::AssertUnwindSafe(|| reader.prefetch(&planes, &rows)),
                                );
                                let failure: Option<Error> = match warmed {
                                    Ok(Ok(_)) => None,
                                    Ok(Err(error)) => Some(error),
                                    Err(_) => Some("Engram read-ahead panicked".into()),
                                };
                                if let Some(error) = failure {
                                    // A failed disk read may leave the reader's ring or staging in a
                                    // state the next gather must not trust, and the real gather of
                                    // these rows would meet the same disk condition: fail the table
                                    // as a failed gather does (c2606aae instead swallows the error).
                                    reader_stats.read_ahead_errors.fetch_add(1, Ordering::Relaxed);
                                    for frame in buffers.iter() {
                                        frame.error.lock().unwrap().get_or_insert_with(|| error.clone());
                                    }
                                    stop.store(true, Ordering::Release);
                                }
                                advisory.recycle(batch.tokens);
                            }
                            Work::Spin => std::hint::spin_loop(),
                            Work::Park => thread::park(),
                            Work::Stop => break,
                        }
                    }
                })
                .map_err(|e| format!("start Engram reader: {e}"))?;
            let advisory = advisory.map(|advisory| (advisory, worker.thread().clone()));
            read_ahead |= advisory.is_some();
            layers.push(Layer {
                staging,
                cancelled,
                sender: Some(sender),
                worker: Some(worker),
                state: Mutex::new(std::array::from_fn(|_| State::default())),
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
        if rows > self.max_tokens * self.index.tables[table].heads {
            return Err("Engram input exceeds staged capacity".into());
        }
        let layer = &self.layers[table];
        let states = layer.state.lock().unwrap();
        for staging in layer.staging.iter() {
            if let Some(error) = staging.error.lock().unwrap().as_ref() {
                return Err(error.clone());
            }
        }
        let mut state = states[sequence as usize % STAGING_FRAMES];
        state.prepare(rank, sequence, rows, self.ranks)?;
        Ok(())
    }

    fn start(
        &self,
        table: usize,
        rank: usize,
        sequence: u64,
        requests: Requests,
    ) -> Result<(usize, u32)> {
        let rows = requests.rows();
        if rows > self.max_tokens * self.index.tables[table].heads {
            return Err("Engram input exceeds staged capacity".into());
        }
        let layer = &self.layers[table];
        let mut states = layer.state.lock().unwrap();
        for staging in layer.staging.iter() {
            if let Some(error) = staging.error.lock().unwrap().as_ref() {
                return Err(error.clone());
            }
        }
        let frame = sequence as usize % STAGING_FRAMES;
        let mut state = states[frame];
        let fresh = state.prepare(rank, sequence, rows, self.ranks)?;
        if fresh {
            layer
                .sender
                .as_ref()
                .unwrap()
                .send(Job {
                    frame,
                    generation: state.generation,
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
        let mut error = None;
        for layer in &self.layers {
            let mut states = layer.state.lock().unwrap();
            let frame = sequence as usize % STAGING_FRAMES;
            states[frame].finish(rank, sequence)?;
            error = error.or_else(|| layer.staging[frame].error.lock().unwrap().clone());
        }
        error.map_or(Ok(()), Err)
    }
}

/// TP3 fork (perf/engram-readahead): the serving worker's read-ahead entry points.
#[cfg_attr(not(feature = "server"), allow(dead_code))]
impl Shared {
    /// TP3 fork (perf/engram-readahead): whether any table of this set reads ahead.
    pub(crate) fn reads_ahead(&self) -> bool {
        self.read_ahead
    }

    /// Whether `table` reads ahead.
    pub(crate) fn table_reads_ahead(&self, table: usize) -> bool {
        self.layers.get(table).is_some_and(|layer| layer.advisory.is_some())
    }

    /// Query positions one advisory batch of `table` covers.
    pub(crate) fn read_ahead_batch_tokens(&self, table: usize) -> usize {
        (READ_AHEAD_BATCH_ROWS / self.index.tables[table].heads.max(1)).max(1)
    }

    /// Query positions `table` may hold read ahead of the step being launched: a quarter of its
    /// share of the row cache, so pinned rows never crowd out rows in steady use, and no more
    /// than its advisory queue holds.
    pub(crate) fn read_ahead_tokens(&self, table: usize) -> usize {
        let Some(entry) = self.index.tables.get(table) else {
            return 0;
        };
        let disk_tables = self.layers.iter().filter(|layer| layer.advisory.is_some()).count().max(1);
        let capacity = self.stats.capacity_rows.load(Ordering::Relaxed) as usize / disk_tables;
        (capacity / 4).min(READ_AHEAD_SLOTS * READ_AHEAD_BATCH_ROWS) / entry.heads.max(1)
    }

    /// Hand the prompt positions `[first, end)` of `tokens` to `table`'s reader thread, which
    /// predicts their rows and reads them ahead, in batches of about `READ_AHEAD_BATCH_ROWS`
    /// rows. `excluded` reports positions that never enter an n-gram (images). Advisory: never
    /// blocks or fails. Returns the position reached: `end`, or less when the queue is full or
    /// the table does not read ahead, so the caller retries the rest in a later step.
    pub(crate) fn read_ahead(
        &self,
        table: usize,
        tokens: &[u32],
        first: usize,
        end: usize,
        excluded: impl Fn(usize) -> bool,
    ) -> usize {
        let Some(layer) = self.layers.get(table) else {
            return first;
        };
        let (Some((advisory, _)), Some(hash)) = (&layer.advisory, &self.index.tables[table].hash) else {
            return first;
        };
        if layer.cancelled.load(Ordering::Acquire) {
            return first;
        }
        let end = end.min(tokens.len());
        let step = self.read_ahead_batch_tokens(table);
        let mut position = first;
        while position < end {
            let next = (position + step).min(end);
            let mut window = advisory.pool.lock().unwrap().pop().unwrap_or_default();
            let start = fill_window(hash.history(), tokens, position, next, &excluded, &mut window);
            {
                let mut queue = advisory.queue.lock().unwrap();
                if queue.len() >= READ_AHEAD_SLOTS {
                    drop(queue);
                    advisory.recycle(window);
                    let heads = self.index.tables[table].heads;
                    self.stats.read_ahead_dropped.fetch_add(((end - position) * heads) as u64, Ordering::Relaxed);
                    return position;
                }
                queue.push_back(Window { tokens: window, first: start });
            }
            // Unpark after enqueueing: a park that races this still finds the token.
            layer.wake();
            position = next;
        }
        position
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
            let mut flags = [0; STAGING_FRAMES];
            for (flag, staging) in flags
                .iter_mut()
                .zip(binding.shared.layers[i].staging.iter())
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
            let staging = &self.binding.shared.layers[i].staging[frame];
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
            | GraphKind::ScoredPrefill
            | GraphKind::EncoderPrefill
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
            heads: 4, weights: "weights".into(), scales: "scales".into(),
            hashes: "hashes".into(), event: "ready".into(), request: None, hash: None,
        };
        let index = Index { tables: vec![table.clone(), table] };
        assert_eq!(staging_bytes(&index, 128).unwrap(), 2 * 2 * (128 * 4 * (64 + 8) + 4));
        assert!(staging_bytes(&index, usize::MAX).is_err());
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

    #[test]
    fn device_request_wait_observes_publication_and_cancellation() {
        let ids = [17i64, 31, 47];
        let flag = AtomicU64::new(0);
        let cancelled = AtomicBool::new(false);
        let request = || Requests::Device {
            address: ids.as_ptr() as usize,
            flag: &flag as *const AtomicU64 as usize,
            sequence: 7,
            rows: ids.len(),
        };
        thread::scope(|scope| {
            let worker = scope.spawn(|| request().read(&cancelled));
            flag.store(7, Ordering::Release);
            assert_eq!(worker.join().unwrap().unwrap(), ids);
        });
        flag.store(0, Ordering::Release);
        cancelled.store(true, Ordering::Release);
        assert!(request().read(&cancelled).is_err());
    }

    // TP3 fork (perf/engram-readahead).
    #[test]
    fn read_ahead_switch_is_off_unless_asked() {
        for value in [None, Some(""), Some("0"), Some("off"), Some("2"), Some("TRUE")] {
            assert!(!parse_read_ahead(value), "{value:?}");
        }
        for value in ["1", " 1 ", "on", "true", "yes"] {
            assert!(parse_read_ahead(Some(value)), "{value}");
        }
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

    #[test]
    fn windows_handed_to_the_reader_predict_the_whole_prompt() {
        let hash = RowHash {
            orders: (2..=4).flat_map(|order| [order; 8]).collect(),
            multipliers: vec![3, 5, 7, 11],
            primes: (0..24).map(|head| 101 + 2 * head).collect(),
            offsets: (0..24).map(|head| head * 160).collect(),
            rows: 24 * 160,
            token_map: (0..50u32).map(|token| (token * 17) % 41).collect::<Vec<_>>().into(),
            pad_token_id: 2,
        };
        let tokens: Vec<u32> = (0..300).map(|i| (i * 31 + i / 7) % 60).collect();
        let excluded = |position: usize| (100..104).contains(&position) || position == 7;
        let (mut history, mut whole) = (Vec::new(), Vec::new());
        hash.predict(&tokens, 0, tokens.len(), excluded, &mut history, &mut whole);
        assert!(whole.len() > 24 * 250);
        for step in [1, 3, 170, 300] {
            let (mut window, mut rows) = (Vec::new(), Vec::new());
            for position in (0..tokens.len()).step_by(step) {
                let next = (position + step).min(tokens.len());
                let first = fill_window(hash.history(), &tokens, position, next, &excluded, &mut window);
                hash.predict(&window, first, window.len(), |_| false, &mut history, &mut rows);
            }
            assert_eq!(rows, whole, "step {step}");
        }
    }

    fn device_job(flag: &AtomicU64, sequence: u64) -> Job {
        Job {
            frame: 0,
            generation: 1,
            requests: Requests::Device {
                address: 0,
                flag: flag as *const AtomicU64 as usize,
                sequence,
                rows: 0,
            },
        }
    }

    fn window(first: usize) -> Window {
        Window { tokens: vec![1, 2, 3], first }
    }

    #[test]
    fn reader_warms_while_a_held_gather_waits_and_serves_it_once_published() {
        let (sender, receiver) = mpsc::sync_channel::<Job>(2);
        let advisory = Advisory::default();
        let cancelled = AtomicBool::new(false);
        let flag = AtomicU64::new(0);
        let mut held = None;
        // Idle with nothing queued: park.
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Park));
        // Warming runs while idle.
        advisory.queue.lock().unwrap().push_back(window(0));
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Warm(_)));
        // A submission whose graph has not published yet is held; warming runs meanwhile.
        sender.send(device_job(&flag, 5)).unwrap();
        advisory.queue.lock().unwrap().push_back(window(1));
        advisory.queue.lock().unwrap().push_back(window(2));
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Warm(w) if w.first == 1));
        assert!(held.is_some());
        // Published ids outrank queued warming.
        flag.store(5, Ordering::Release);
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Gather(_)));
        assert!(held.is_none());
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Warm(w) if w.first == 2));
        // Held and unpublished with nothing queued: spin on the flag as `read` does.
        sender.send(device_job(&flag, 6)).unwrap();
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Spin));
        // Cancellation releases the held gather so it reports the cancellation.
        cancelled.store(true, Ordering::Release);
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Gather(_)));
        // A cancelled table drops queued warming and parks until a submission or shutdown.
        advisory.queue.lock().unwrap().push_back(window(9));
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Park));
        assert!(advisory.queue.lock().unwrap().is_empty());
        // Host-produced ids are always readable.
        cancelled.store(false, Ordering::Release);
        sender.send(Job { frame: 1, generation: 2, requests: Requests::Host(vec![1]) }).unwrap();
        advisory.queue.lock().unwrap().push_back(window(3));
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Gather(_)));
        // A closed submission channel stops the thread once nothing is held; queued warming is
        // dropped with it (shutdown).
        drop(sender);
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Stop));
    }

    #[test]
    fn a_parked_reader_thread_wakes_for_work_and_for_shutdown() {
        let (sender, receiver) = mpsc::sync_channel::<Job>(2);
        let advisory = Arc::new(Advisory::default());
        let cancelled = Arc::new(AtomicBool::new(false));
        let (done, served) = mpsc::channel();
        let thread_advisory = advisory.clone();
        let thread_cancelled = cancelled.clone();
        let worker = thread::spawn(move || {
            let mut held = None;
            loop {
                match next_work(&receiver, &thread_advisory, &mut held, &thread_cancelled) {
                    Work::Gather(_) => done.send("gather").unwrap(),
                    Work::Warm(_) => done.send("warm").unwrap(),
                    Work::Spin => std::hint::spin_loop(),
                    Work::Park => thread::park(),
                    Work::Stop => break,
                }
            }
        });
        let unpark = || worker.thread().unpark();
        let wait = || served.recv_timeout(std::time::Duration::from_secs(10)).unwrap();
        advisory.queue.lock().unwrap().push_back(window(0));
        unpark();
        assert_eq!(wait(), "warm");
        sender.send(Job { frame: 0, generation: 1, requests: Requests::Host(vec![]) }).unwrap();
        unpark();
        assert_eq!(wait(), "gather");
        drop(sender);
        unpark();
        worker.join().unwrap();
    }

}
