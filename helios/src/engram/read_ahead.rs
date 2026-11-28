/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! TP3 fork (perf/engram-readahead): opt-in read-ahead of disk-backed Engram rows for prefill,
//! after c2606aae (randomvariable topic/engramcache, NVIDIA/hyperflare-inference #267).
//!
//! The serving worker hands the prompt positions of prefill lanes to the reader thread of every
//! disk table with a row cache before each step is launched. The reader thread hashes them
//! ([`RowHash`]) and warms the row cache while earlier steps replay, pinning the rows until their
//! gather uses them. The captured graph stays the authority: it publishes the ids its lookup
//! consumes and the gather serves exactly those, so a wrong prediction only costs disk reads.
//!
//! Inert unless `FLARE_ENGRAM_READ_AHEAD=1` and a table is on disk with a row cache: every other
//! table keeps the tp3-main reader loop and has no state here.

use super::{
    Error, Job, Plane, RowHash, STAGING_FRAMES, Shared, Staging, Stats, reader, serve,
};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        mpsc,
    },
    thread,
};

/// `FLARE_ENGRAM_READ_AHEAD=1` lets disk-backed tables with a row cache and a predictable hash
/// read prefill rows ahead of their graph. Default off. Tables in RAM (both DS41 tables in the TP3
/// default) never read ahead whatever the switch says.
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
const BATCH_ROWS: usize = 4096;
/// Advisory batches one table may hold queued (about a million rows).
const SLOTS: usize = 256;
/// Launches remembered per request slot; at most two frames are in flight.
const LAUNCHES: usize = 8;

/// One advisory batch: prompt tokens whose rows a coming prefill step will request. `tokens`
/// holds the query positions from `first` on, preceded by up to the hash's history (fewer only at
/// the start of the prompt, where the hash reads pad). Excluded positions (images) are stored as
/// `u32::MAX`, which no token map holds, so they stop n-grams as in the graph. The reader thread
/// hashes the window itself, which keeps that work off the serving worker's submit path.
pub(super) struct Window {
    tokens: Vec<u32>,
    first: usize,
    /// Request slot and generation the positions belong to.
    slot: usize,
    generation: u64,
    /// Prompt position after the window's last query.
    end: usize,
}

/// Fill `window` with the tokens of query positions `[position, next)` and up to `history`
/// tokens before them, excluded positions as `u32::MAX`; returns where the queries start in it.
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

/// What the reader knows of one request slot: the request occupying it, the prompt positions
/// launched by recent steps (by submission sequence), and the positions whose gather was served.
#[derive(Default)]
struct Slot {
    generation: u64,
    launches: VecDeque<(u64, usize)>,
    served_through: usize,
}

/// Advisory read-ahead state of one disk-backed table, shared by its producer (the serving
/// worker) and its reader thread. Exists only for tables that read ahead.
#[derive(Default)]
pub(super) struct Advisory {
    queue: Mutex<VecDeque<Window>>,
    /// Windows queued, read before taking `queue` so a spinning reader never locks an empty queue.
    queued: AtomicUsize,
    /// Recycled window buffers, so steady-state read-ahead allocates nothing.
    pool: Mutex<Vec<Vec<u32>>>,
    slots: Mutex<Vec<Slot>>,
    /// Submission sequence of the last gather the reader served.
    served: AtomicU64,
}

impl Advisory {
    fn pop(&self) -> Option<Window> {
        if self.queued.load(Ordering::Acquire) == 0 {
            return None;
        }
        let window = self.queue.lock().unwrap().pop_front();
        if window.is_some() {
            self.queued.fetch_sub(1, Ordering::AcqRel);
        }
        window
    }

    fn clear(&self) {
        if self.queued.load(Ordering::Acquire) != 0 {
            let mut queue = self.queue.lock().unwrap();
            self.queued.fetch_sub(queue.len(), Ordering::AcqRel);
            queue.clear();
        }
    }

    fn recycle(&self, mut tokens: Vec<u32>) {
        tokens.clear();
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < SLOTS {
            pool.push(tokens);
        }
    }

    /// Record that the step with submission `sequence` launches the prompt of the request in
    /// `slot` (identified by `generation`) through position `through`.
    fn launch(&self, slot: usize, generation: u64, sequence: u64, through: usize) {
        let mut slots = self.slots.lock().unwrap();
        if slots.len() <= slot {
            slots.resize_with(slot + 1, Slot::default);
        }
        let state = &mut slots[slot];
        if state.generation != generation {
            *state = Slot { generation, ..Slot::default() };
        }
        if state.launches.len() == LAUNCHES {
            state.launches.pop_front();
        }
        state.launches.push_back((sequence, through));
    }

    /// The request in `slot` left: its queued windows are dropped as they come up.
    fn release(&self, slot: usize) {
        if let Some(state) = self.slots.lock().unwrap().get_mut(slot) {
            *state = Slot::default();
        }
    }

    fn served(&self, sequence: u64) {
        self.served.fetch_max(sequence, Ordering::AcqRel);
    }

    /// Whether `window` can no longer help: its request left the slot, or the gather that
    /// consumes all of its positions was already served. Such a window is dropped without a pin
    /// or a read, so rows are pinned only ahead of their use and never re-read after it.
    fn stale(&self, window: &Window) -> bool {
        let served = self.served.load(Ordering::Acquire);
        let mut slots = self.slots.lock().unwrap();
        let Some(state) = slots.get_mut(window.slot) else {
            return true;
        };
        if state.generation != window.generation {
            return true;
        }
        while let Some(&(sequence, through)) = state.launches.front() {
            if sequence > served {
                break;
            }
            state.served_through = state.served_through.max(through);
            state.launches.pop_front();
        }
        window.end <= state.served_through
    }
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

/// The c2606aae reader fix: a submission is taken into `held` as soon as it arrives but gathered
/// only once its ids are readable. Device ids exist only after the step's graph starts, so a
/// thread blocked on them would idle through the replay that read-ahead is meant to overlap;
/// while the held gather waits, advisory batches run one at a time, so the gather starts at most
/// one batch after its ids appear.
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
        advisory.clear();
        return Work::Park;
    }
    advisory.pop().map_or(Work::Park, Work::Warm)
}

/// Reader-thread scratch for warming.
#[derive(Default)]
struct Scratch {
    history: Vec<i64>,
    rows: Vec<i64>,
}

/// Warm the row cache with one window's rows unless it is stale. Returns whether it read.
fn warm(
    advisory: &Advisory,
    hash: &RowHash,
    reader: &mut reader::Reader,
    planes: &[Plane; 2],
    window: &Window,
    scratch: &mut Scratch,
    stats: &Stats,
) -> Result<bool, Error> {
    if advisory.stale(window) {
        stats.read_ahead_skipped.fetch_add(1, Ordering::Relaxed);
        return Ok(false);
    }
    let Scratch { history, rows } = scratch;
    let warmed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        rows.clear();
        hash.predict(&window.tokens, window.first, window.tokens.len(), |_| false, history, rows);
        stats.read_ahead_queued.fetch_add(rows.len() as u64, Ordering::Relaxed);
        reader.prefetch(planes, rows)
    }));
    match warmed {
        Ok(Ok(_)) => Ok(true),
        Ok(Err(error)) => Err(error),
        Err(_) => Err("Engram read-ahead panicked".into()),
    }
}

/// The reader thread of a table that reads ahead (the tp3-main loop serves every other table).
#[allow(clippy::too_many_arguments)]
pub(super) fn run_reader(
    receiver: mpsc::Receiver<Job>,
    advisory: &Advisory,
    hash: &RowHash,
    reader: &mut reader::Reader,
    planes: &[Plane; 2],
    buffers: &[Arc<[Staging; STAGING_FRAMES]>],
    row_bytes: usize,
    stop: &AtomicBool,
    stats: &Stats,
) {
    let mut held = None;
    let mut scratch = Scratch::default();
    loop {
        match next_work(&receiver, advisory, &mut held, stop) {
            Work::Gather(job) => {
                let sequence = job.sequence;
                let lane = job.lane;
                serve(reader, planes, &buffers[lane], row_bytes, stop, job);
                advisory.served(sequence);
            }
            Work::Warm(window) => {
                if let Err(error) = warm(advisory, hash, reader, planes, &window, &mut scratch, stats) {
                    // A failed disk read may leave the reader's ring or staging in a state the
                    // next gather must not trust, and the real gather of these rows would meet
                    // the same disk condition: fail the table as a failed gather does (c2606aae
                    // instead swallows the error). The cause is logged here because a held gather
                    // then reports only its cancellation.
                    stats.read_ahead_errors.fetch_add(1, Ordering::Relaxed);
                    eprintln!("Engram read-ahead failed; failing the table: {error}");
                    for lane_buffers in buffers {
                        for frame in lane_buffers.iter() {
                            frame.error.lock().unwrap().get_or_insert_with(|| error.clone());
                        }
                    }
                    stop.store(true, Ordering::Release);
                }
                advisory.recycle(window.tokens);
            }
            Work::Spin => std::hint::spin_loop(),
            Work::Park => thread::park(),
            Work::Stop => break,
        }
    }
}

/// Whether a table gets advisory state: switch on, disk with a row cache, predictable hash.
pub(super) fn advisory_for(reader: &reader::Reader, hash: Option<&Arc<RowHash>>) -> Option<Arc<Advisory>> {
    (read_ahead_requested() && reader.caches_rows() && hash.is_some()).then(|| Arc::new(Advisory::default()))
}

/// The serving worker's read-ahead entry points.
#[cfg_attr(not(feature = "server"), allow(dead_code))]
impl Shared {
    /// Whether any table of this set reads ahead.
    pub(crate) fn reads_ahead(&self) -> bool {
        self.read_ahead
    }

    /// Whether `table` reads ahead.
    pub(crate) fn table_reads_ahead(&self, table: usize) -> bool {
        self.layers.get(table).is_some_and(|layer| layer.advisory.is_some())
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
        let predicted_heads = entry.hash.as_ref().map_or(entry.heads, |hash| hash.heads()).max(1);
        (capacity / 4).min(SLOTS * BATCH_ROWS) / predicted_heads
    }

    /// Record, for every table that reads ahead, that step `sequence` launches the prompt of
    /// the request in `slot` through `through`.
    pub(crate) fn read_ahead_launch(&self, slot: usize, generation: u64, sequence: u64, through: usize) {
        for (advisory, _) in self.layers.iter().filter_map(|layer| layer.advisory.as_ref()) {
            advisory.launch(slot, generation, sequence, through);
        }
    }

    /// The request in `slot` left; its queued windows are dropped.
    pub(crate) fn read_ahead_release(&self, slot: usize) {
        for (advisory, _) in self.layers.iter().filter_map(|layer| layer.advisory.as_ref()) {
            advisory.release(slot);
        }
    }

    /// Hand the prompt positions `[first, end)` of `tokens`, of the request in `slot`, to
    /// `table`'s reader thread, which predicts their rows and reads them ahead, in windows of
    /// about `BATCH_ROWS` rows. `excluded` reports positions that never enter an n-gram (images).
    /// Advisory: never blocks or fails. Returns the position reached: `end`, or less when the
    /// queue is full or the table does not read ahead, so the caller retries the rest later.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn read_ahead(
        &self,
        table: usize,
        slot: usize,
        generation: u64,
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
        let heads = hash.heads().max(1);
        let step = (BATCH_ROWS / heads).max(1);
        let mut position = first;
        {
            // One lock for the whole hand-over; the reader takes it once per window.
            let mut queue = advisory.queue.lock().unwrap();
            while position < end {
                if queue.len() >= SLOTS {
                    self.stats.read_ahead_dropped.fetch_add(((end - position) * heads) as u64, Ordering::Relaxed);
                    break;
                }
                let next = (position + step).min(end);
                let mut buffer = advisory.pool.lock().unwrap().pop().unwrap_or_default();
                let start = fill_window(hash.history(), tokens, position, next, &excluded, &mut buffer);
                queue.push_back(Window { tokens: buffer, first: start, slot, generation, end: next });
                advisory.queued.fetch_add(1, Ordering::AcqRel);
                position = next;
            }
        }
        if position > first {
            // Unpark after enqueueing: a park that races this still finds the token.
            layer.wake();
        }
        position
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engram::{Requests, reader::Reader};

    #[test]
    fn read_ahead_switch_is_off_unless_asked() {
        for value in [None, Some(""), Some("0"), Some("off"), Some("2"), Some("TRUE")] {
            assert!(!parse_read_ahead(value), "{value:?}");
        }
        for value in ["1", " 1 ", "on", "true", "yes"] {
            assert!(parse_read_ahead(Some(value)), "{value}");
        }
    }

    fn hash() -> RowHash {
        RowHash {
            orders: (2..=4).flat_map(|order| [order; 8]).collect(),
            multipliers: vec![3, 5, 7, 11],
            primes: (0..24).map(|head| 101 + 2 * head).collect(),
            offsets: (0..24).map(|head| head * 160).collect(),
            rows: 24 * 160,
            token_map: (0..50u32).map(|token| (token * 17) % 41).collect::<Vec<_>>().into(),
            pad_token_id: 2,
        }
    }

    #[test]
    fn windows_handed_to_the_reader_predict_the_whole_prompt() {
        let hash = hash();
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

    fn window(slot: usize, generation: u64, first: usize, end: usize, tokens: &[u32]) -> Window {
        let mut buffer = Vec::new();
        let start = fill_window(3, tokens, first, end, &|_| false, &mut buffer);
        Window { tokens: buffer, first: start, slot, generation, end }
    }

    fn device_job(flag: &AtomicU64, sequence: u64) -> Job {
        Job {
            lane: 0,
            frame: 0,
            generation: 1,
            sequence,
            requests: Requests::Device {
                address: 0,
                flag: flag as *const AtomicU64 as usize,
                sequence,
                rows: 0,
                advisory: None,
            },
        }
    }

    fn push(advisory: &Advisory, window: Window) {
        advisory.queue.lock().unwrap().push_back(window);
        advisory.queued.fetch_add(1, Ordering::AcqRel);
    }

    #[test]
    fn reader_warms_while_a_held_gather_waits_and_serves_it_once_published() {
        let (sender, receiver) = mpsc::sync_channel::<Job>(2);
        let advisory = Advisory::default();
        let cancelled = AtomicBool::new(false);
        let flag = AtomicU64::new(0);
        let mut held = None;
        let tokens = [1u32; 8];
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Park));
        push(&advisory, window(0, 1, 0, 1, &tokens));
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Warm(_)));
        sender.send(device_job(&flag, 5)).unwrap();
        push(&advisory, window(0, 1, 1, 2, &tokens));
        push(&advisory, window(0, 1, 2, 3, &tokens));
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Warm(w) if w.end == 2));
        assert!(held.is_some());
        flag.store(5, Ordering::Release);
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Gather(_)));
        assert!(held.is_none());
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Warm(w) if w.end == 3));
        assert_eq!(advisory.queued.load(Ordering::Acquire), 0);
        sender.send(device_job(&flag, 6)).unwrap();
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Spin));
        cancelled.store(true, Ordering::Release);
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Gather(_)));
        push(&advisory, window(0, 1, 3, 4, &tokens));
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Park));
        assert_eq!(advisory.queued.load(Ordering::Acquire), 0);
        assert!(advisory.queue.lock().unwrap().is_empty());
        cancelled.store(false, Ordering::Release);
        sender
            .send(Job { lane: 0, frame: 1, generation: 2, sequence: 7, requests: Requests::Host(vec![1]) })
            .unwrap();
        push(&advisory, window(0, 1, 4, 5, &tokens));
        assert!(matches!(next_work(&receiver, &advisory, &mut held, &cancelled), Work::Gather(_)));
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
        push(&advisory, window(0, 1, 0, 1, &[1, 2]));
        unpark();
        assert_eq!(wait(), "warm");
        sender
            .send(Job { lane: 0, frame: 0, generation: 1, sequence: 1, requests: Requests::Host(vec![]) })
            .unwrap();
        unpark();
        assert_eq!(wait(), "gather");
        drop(sender);
        unpark();
        worker.join().unwrap();
    }

    #[test]
    fn windows_are_stale_once_their_gather_is_served_or_their_request_leaves() {
        let advisory = Advisory::default();
        let tokens = [1u32; 64];
        // Unknown slot or request: stale.
        assert!(advisory.stale(&window(3, 1, 0, 8, &tokens)));
        // Step 5 launches slot 0 through 16; step 6 through 32.
        advisory.launch(0, 1, 5, 16);
        advisory.launch(0, 1, 6, 32);
        let first = window(0, 1, 8, 16, &tokens);
        let second = window(0, 1, 16, 32, &tokens);
        let ahead = window(0, 1, 32, 48, &tokens);
        assert!(!advisory.stale(&first) && !advisory.stale(&second));
        advisory.served(5);
        assert!(advisory.stale(&first), "served positions must not be pinned or re-read");
        assert!(!advisory.stale(&second));
        advisory.served(6);
        assert!(advisory.stale(&second));
        assert!(!advisory.stale(&ahead), "positions not launched yet stay useful");
        // A window straddling the served boundary keeps its unserved part.
        assert!(!advisory.stale(&window(0, 1, 24, 40, &tokens)));
        // Another request in the slot, or a release, makes old windows stale.
        advisory.launch(0, 2, 7, 4);
        assert!(advisory.stale(&ahead));
        assert!(!advisory.stale(&window(0, 2, 4, 8, &tokens)));
        advisory.release(0);
        assert!(advisory.stale(&window(0, 2, 4, 8, &tokens)));
    }

    #[test]
    fn a_window_processed_after_its_gather_pins_nothing_and_reads_nothing() {
        let hash = hash();
        let rows = hash.rows as usize;
        let path = std::env::temp_dir().join(format!("helios-engram-ra-{}.bin", std::process::id()));
        let second = rows * 80;
        let bytes: Vec<u8> = (0..second + rows * 10).map(|i| (i * 7 + i / 131) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane { file: 0, offset: 0, nbytes: rows * 80, row_bytes: 80 },
            Plane { file: 0, offset: second as u64, nbytes: rows * 10, row_bytes: 10 },
        ];
        let stats = Arc::new(Stats::default());
        let mut reader = Reader::open(&path, &planes, false, 4 << 20, 4096, stats.clone()).unwrap();
        let advisory = Advisory::default();
        let tokens: Vec<u32> = (0..400).map(|i| (i * 13 + 5) % 50).collect();
        let mut scratch = Scratch::default();
        advisory.launch(0, 9, 11, 200);
        // Before its gather is served the window is read ahead.
        let useful = window(0, 9, 0, 100, &tokens);
        assert!(warm(&advisory, &hash, &mut reader, &planes, &useful, &mut scratch, &stats).unwrap());
        let reads = stats.disk_reads.load(Ordering::Relaxed);
        let loaded = stats.read_ahead_loaded.load(Ordering::Relaxed);
        assert!(reads > 0 && loaded > 0);
        // After the gather of step 11 is served, a window of those positions is dropped
        // without a read, a pin or an insert.
        advisory.served(11);
        let late = window(0, 9, 100, 200, &tokens);
        assert!(!warm(&advisory, &hash, &mut reader, &planes, &late, &mut scratch, &stats).unwrap());
        assert_eq!(stats.disk_reads.load(Ordering::Relaxed), reads);
        assert_eq!(stats.read_ahead_loaded.load(Ordering::Relaxed), loaded);
        assert_eq!(stats.read_ahead_skipped.load(Ordering::Relaxed), 1);
        // Positions beyond the served launch still warm.
        let ahead = window(0, 9, 200, 300, &tokens);
        assert!(warm(&advisory, &hash, &mut reader, &planes, &ahead, &mut scratch, &stats).unwrap());
        assert!(stats.disk_reads.load(Ordering::Relaxed) > reads);
        std::fs::remove_file(path).unwrap();
    }
}
