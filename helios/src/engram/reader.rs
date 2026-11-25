/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bounded, direct-I/O row gathering into the same buffers used by resident tables.

use super::{Stats, cache::RowCache};
use crate::Result;
#[cfg(unix)]
use io_uring::{IoUring, opcode, types};
use std::{
    fs::File,
    sync::{Arc, atomic::Ordering},
    time::Instant,
};

#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};

#[cfg(test)]
use std::path::Path;

#[cfg(windows)]
mod windows;

const BLOCK: usize = 4096;
const CHUNK: usize = 64 << 10;
const DEPTH: usize = 128;
const LOAD_CHUNK: usize = 4 << 20;
const LOAD_READERS: usize = 4;
const PARALLEL_GATHER_MIN_ROWS: usize = 2048;
const MAX_GATHER_THREADS: usize = 16;
const PREFETCH_DISTANCE: usize = 8;
const MAX_LOAD_READERS: usize = 64;

/// Threads that read one RAM-resident plane at startup: `FLARE_ENGRAM_LOAD_READERS` (1..=64), default 4.
/// The reads are disjoint 4 MiB ranges, so the count only changes startup speed.
fn load_readers() -> usize {
    static READERS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *READERS.get_or_init(|| parse_load_readers(std::env::var("FLARE_ENGRAM_LOAD_READERS").ok().as_deref()))
}

fn parse_load_readers(value: Option<&str>) -> usize {
    match value.map(str::trim) {
        None | Some("") => LOAD_READERS,
        Some(text) => match text.parse::<usize>() {
            Ok(count) if (1..=MAX_LOAD_READERS).contains(&count) => count,
            _ => {
                eprintln!(
                    "FLARE_ENGRAM_LOAD_READERS={text:?} is not 1..={MAX_LOAD_READERS}; using {LOAD_READERS}"
                );
                LOAD_READERS
            }
        },
    }
}

/// `FLARE_ENGRAM_PRIVATE_TABLES=1`: RAM-resident planes that `Reader::from_files` loads live in
/// anonymous, process-private memory (read-only after loading) instead of a sealed memfd.
/// Anonymous memory makes the load disk-bound: DS41 "Prepare embedding tables" 25.7 s -> 11.4 s on
/// 4 KiB pages (`MADV_NOHUGEPAGE`, see `ResidentMapping::anonymous`). Release is not faster without
/// huge pages (8.4 s), so pair it with `FLARE_FAST_EXIT`.
/// Private planes have no descriptor to share, so the prepared multi-process tables
/// (`PreparedTables`, `Resident::load_segments`) keep the memfd. Default off (unchanged behaviour).
#[cfg(unix)]
fn private_tables() -> bool {
    static PRIVATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PRIVATE.get_or_init(|| {
        matches!(
            std::env::var("FLARE_ENGRAM_PRIVATE_TABLES").ok().as_deref().map(str::trim),
            Some("1" | "true" | "on" | "yes")
        )
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plane {
    pub file: usize,
    pub offset: u64,
    pub nbytes: usize,
    pub row_bytes: usize,
}

impl Plane {
    pub(crate) fn resident_bytes(self) -> Result<usize> {
        ((self.offset % BLOCK as u64) as usize)
            .checked_add(self.nbytes)
            .and_then(|n| n.checked_add(BLOCK - 1))
            .map(|n| (n / BLOCK * BLOCK).max(BLOCK))
            .ok_or_else(|| "resident embedding allocation exceeds host range".into())
    }
}

#[derive(Clone, Copy)]
struct Fragment {
    file: usize,
    source: u64,
    destination: usize,
    nbytes: usize,
}

struct Read {
    file: usize,
    offset: u64,
    nbytes: usize,
    required: usize,
    fragments: std::ops::Range<usize>,
}

struct Aligned(usize, usize);
impl Aligned {
    fn new(nbytes: usize) -> Result<Self> {
        Ok(Self(
            crate::os::allocate_aligned(nbytes).map_err(|e| format!("Engram allocation: {e}"))?,
            nbytes,
        ))
    }
    fn at(&self, slot: usize) -> *mut u8 {
        (self.0 + slot * CHUNK) as *mut u8
    }
}

#[cfg(test)]
pub(super) fn open_file(path: &Path) -> Result<File> {
    crate::os::direct_file(path, false).map_err(|e| format!("open Engram core: {e}").into())
}

pub(crate) struct Resident {
    #[cfg(unix)]
    storage: ResidentMapping,
    #[cfg(windows)]
    storage: Aligned,
    /// The sealed memfd behind a shareable plane; `None` for a process-private plane.
    #[cfg(unix)]
    file: Option<File>,
    skip: usize,
    nbytes: usize,
}

#[cfg(unix)]
struct ResidentMapping {
    address: usize,
    nbytes: usize,
}

#[cfg(unix)]
impl ResidentMapping {
    fn new(file: &File, nbytes: usize, writable: bool) -> Result<Self> {
        let protection = libc::PROT_READ | if writable { libc::PROT_WRITE } else { 0 };
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                nbytes,
                protection,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(format!(
                "map resident embedding table: {}",
                std::io::Error::last_os_error()
            )
            .into());
        }
        let mapping = Self {
            address: address as usize,
            nbytes,
        };
        // The sealed read-only remap has its own page tables, even after loading
        // populated the backing RAM. Prepare them before row delivery starts.
        if !writable && unsafe { libc::madvise(address, nbytes, libc::MADV_POPULATE_READ) } != 0 {
            let error = std::io::Error::last_os_error();
            return Err(format!("populate resident embedding page tables: {error}").into());
        }
        Ok(mapping)
    }
}

#[cfg(unix)]
impl ResidentMapping {
    /// Process-private anonymous memory of `nbytes` on 4 KiB pages. `MADV_NOHUGEPAGE`: with THP
    /// "always" a fragmented host backed ~60% of the DS41 tables with huge pages and the rest with
    /// 4 KiB pages, and decode gathers paid for it (11-14 A/B, 32K c4: 640 vs 749 tok/s, dTLB load
    /// misses x2.6). Without huge pages c4 matches the memfd tables and the load stays as fast
    /// (11.4 s vs 25.7 s); the slower release is skipped by `FLARE_FAST_EXIT`.
    fn anonymous(nbytes: usize) -> Result<Self> {
        let nbytes = nbytes.next_multiple_of(crate::os::page_size());
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                nbytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(format!(
                "map private embedding table: {}",
                std::io::Error::last_os_error()
            )
            .into());
        }
        let mapping = Self { address: address as usize, nbytes };
        #[cfg(target_os = "linux")]
        if unsafe { libc::madvise(address, nbytes, libc::MADV_NOHUGEPAGE) } != 0 {
            let error = std::io::Error::last_os_error();
            if !matches!(error.raw_os_error(), Some(libc::EINVAL | libc::EPERM)) {
                return Err(format!(
                    "disable huge pages for private embedding table: {error}"
                )
                .into());
            }
        }
        Ok(mapping)
    }

    fn make_read_only(&self) -> Result<()> {
        if unsafe { libc::mprotect(self.address as *mut libc::c_void, self.nbytes, libc::PROT_READ) } != 0 {
            return Err(format!(
                "protect private embedding table: {}",
                std::io::Error::last_os_error()
            )
            .into());
        }
        Ok(())
    }
}

#[cfg(unix)]
impl Drop for ResidentMapping {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.address as *mut libc::c_void, self.nbytes) };
    }
}

#[cfg(unix)]
const RESIDENT_SEALS: i32 =
    libc::F_SEAL_WRITE | libc::F_SEAL_GROW | libc::F_SEAL_SHRINK | libc::F_SEAL_SEAL;

#[cfg(unix)]
fn seal_resident(file: &File) -> Result<()> {
    // The loading threads have joined and their writable mapping is gone.
    // Kernel-held page references can still make F_SEAL_WRITE temporarily busy.
    // Publish only after every required seal succeeds; persistent busy state fails.
    for attempt in 0..4 {
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, RESIDENT_SEALS) } == 0 {
            return Ok(());
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EBUSY) || attempt == 3 {
            return Err(format!("seal resident embedding table: {error}").into());
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    unreachable!()
}

impl Resident {
    pub(super) fn load(file: &File, plane: Plane) -> Result<Self> {
        Self::initialize(plane, |address, extent, required| {
            Self::read_plane(file, plane, address, extent, required, load_readers())
        })
    }

    /// Load `plane` into process-private anonymous memory (see `private_tables`).
    #[cfg(unix)]
    pub(super) fn load_private(file: &File, plane: Plane) -> Result<Self> {
        let skip = (plane.offset % BLOCK as u64) as usize;
        let required = skip
            .checked_add(plane.nbytes)
            .ok_or("resident Engram plane exceeds host range")?;
        let extent = plane.resident_bytes()?;
        let storage = ResidentMapping::anonymous(extent)?;
        Self::read_plane(file, plane, storage.address, extent, required, load_readers())?;
        // Readers joined; nothing writes the plane after this point.
        storage.make_read_only()?;
        Ok(Self {
            storage,
            file: None,
            skip,
            nbytes: plane.nbytes,
        })
    }

    /// Fill `extent` bytes at `address` from the block-aligned start of `plane` with `readers`
    /// threads reading disjoint `LOAD_CHUNK` ranges. The first `required` bytes must be present.
    fn read_plane(
        file: &File,
        plane: Plane,
        address: usize,
        extent: usize,
        required: usize,
        readers: usize,
    ) -> Result<()> {
        {
            let skip = (plane.offset % BLOCK as u64) as usize;
            let offset = plane.offset - skip as u64;
            let readers = readers.clamp(1, MAX_LOAD_READERS);
            std::thread::scope(|scope| -> Result<()> {
                let mut workers = Vec::new();
                for reader in 0..readers.min(extent.div_ceil(LOAD_CHUNK)) {
                    workers.push(
                        std::thread::Builder::new()
                            .name(format!("flare-table-loader-{reader}"))
                            .spawn_scoped(scope, move || -> Result<()> {
                                for begin in (reader * LOAD_CHUNK..extent)
                                    .step_by(readers * LOAD_CHUNK)
                                {
                                    let nbytes = LOAD_CHUNK.min(extent - begin);
                                    let source = offset
                                        .checked_add(begin as u64)
                                        .ok_or("resident Engram file offset exceeds range")?;
                                    // SAFETY: Each reader initializes a disjoint aligned allocation range.
                                    let count = unsafe { crate::os::read_at_raw(file, (address + begin) as *mut u8, nbytes, source) }
                                        .map_err(|e| format!("resident Engram read at {source}: {e}"))?;
                                    let needed = nbytes.min(required - begin);
                                    if count < needed {
                                        return Err(format!(
                                            "resident Engram read at {source} returned {count}, requires {needed} bytes"
                                        ).into());
                                    }
                                }
                                Ok(())
                            })
                            .map_err(|e| format!("start resident Engram loader: {e}"))?,
                    );
                }
                for worker in workers {
                    worker
                        .join()
                        .map_err(|_| "resident Engram loader panicked")??;
                }
                Ok(())
            })
        }
    }

    #[cfg(unix)]
    pub(super) fn load_segments(plane: Plane, segments: &[(File, u64, usize)]) -> Result<Self> {
        if plane.offset != 0 {
            return Err("assembled embedding planes must start at offset zero".into());
        }
        Self::initialize(plane, |address, _, _| {
            let mut destination = 0usize;
            for (file, offset, nbytes) in segments {
                let end = destination
                    .checked_add(*nbytes)
                    .filter(|end| *end <= plane.nbytes)
                    .ok_or("embedding source segments exceed their plane")?;
                let mut copied = 0;
                while copied < *nbytes {
                    let count = (*nbytes - copied).min(LOAD_CHUNK);
                    // SAFETY: Source segments initialize consecutive, disjoint ranges
                    // inside the resident allocation before it becomes immutable.
                    let read = unsafe {
                        crate::os::read_at_raw(
                            file,
                            (address + destination + copied) as *mut u8,
                            count,
                            offset + copied as u64,
                        )
                    }
                    .map_err(|error| format!("read embedding source segment: {error}"))?;
                    if read == 0 {
                        return Err(
                            "embedding source segment ended before its declared size".into()
                        );
                    }
                    copied += read;
                }
                destination = end;
            }
            if destination != plane.nbytes {
                return Err("embedding source segments do not fill their plane".into());
            }
            Ok(())
        })
    }

    fn initialize(
        plane: Plane,
        fill: impl FnOnce(usize, usize, usize) -> Result<()>,
    ) -> Result<Self> {
        let skip = (plane.offset % BLOCK as u64) as usize;
        let required = skip
            .checked_add(plane.nbytes)
            .ok_or("resident Engram plane exceeds host range")?;
        let extent = plane.resident_bytes()?;
        #[cfg(unix)]
        let backing = {
            let descriptor = unsafe {
                libc::memfd_create(
                    c"flare-embedding-table".as_ptr(),
                    libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
                )
            };
            if descriptor < 0 {
                return Err(format!(
                    "create resident embedding table: {}",
                    std::io::Error::last_os_error()
                )
                .into());
            }
            // The descriptor is newly created and owned exclusively until published.
            let backing = unsafe { File::from_raw_fd(descriptor) };
            backing
                .set_len(extent as u64)
                .map_err(|e| format!("size resident embedding table: {e}"))?;
            backing
        };
        #[cfg(unix)]
        let storage = ResidentMapping::new(&backing, extent, true)?;
        #[cfg(unix)]
        let address = storage.address;
        #[cfg(windows)]
        let storage = Aligned::new(extent)?;
        #[cfg(windows)]
        let address = storage.0;
        fill(address, extent, required)?;
        #[cfg(unix)]
        {
            drop(storage);
            seal_resident(&backing)?;
            Self::from_shared(backing, plane)
        }
        #[cfg(windows)]
        {
            Ok(Self {
                storage,
                skip,
                nbytes: plane.nbytes,
            })
        }
    }

    #[cfg(unix)]
    pub(super) fn from_shared(file: File, plane: Plane) -> Result<Self> {
        let seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
        if seals < 0 || seals & RESIDENT_SEALS != RESIDENT_SEALS {
            return Err("resident embedding table must have immutable size and contents".into());
        }
        let skip = (plane.offset % BLOCK as u64) as usize;
        let extent = plane.resident_bytes()?;
        if file
            .metadata()
            .map_err(|e| format!("inspect resident embedding table: {e}"))?
            .len()
            != extent as u64
        {
            return Err("shared resident embedding table extent differs from its plane".into());
        }
        Ok(Self {
            storage: ResidentMapping::new(&file, extent, false)?,
            file: Some(file),
            skip,
            nbytes: plane.nbytes,
        })
    }

    /// The shareable memfd of this plane; `None` for a process-private plane.
    #[cfg(unix)]
    pub(super) fn descriptor(&self) -> Option<i32> {
        self.file.as_ref().map(AsRawFd::as_raw_fd)
    }

    fn bytes(&self) -> &[u8] {
        #[cfg(unix)]
        let address = self.storage.address;
        #[cfg(windows)]
        let address = self.storage.0;
        // SAFETY: Every logical byte was initialized by successful reads, and
        // scoped readers joined before this immutable view becomes available.
        unsafe { std::slice::from_raw_parts((address + self.skip) as *const u8, self.nbytes) }
    }
}
impl Drop for Aligned {
    fn drop(&mut self) {
        unsafe { crate::os::free_aligned(self.0, self.1) };
    }
}

#[cfg(unix)]
struct UnixLane {
    // Drop the ring before releasing the registered staging allocation.
    ring: IoUring,
    staging: Aligned,
    depth: usize,
}

#[cfg(unix)]
#[derive(Clone, Copy)]
struct UnixOutput {
    address: usize,
    len: usize,
}

#[cfg(unix)]
impl UnixOutput {
    // SAFETY: The planner assigns each destination byte to exactly one fragment,
    // and each read (with its fragments) belongs to one worker lane. The scoped
    // join finishes every worker before the caller accesses or releases output.
    unsafe fn copy(self, fragment: &Fragment, source: *const u8) {
        assert!(fragment.destination <= self.len);
        assert!(fragment.nbytes <= self.len - fragment.destination);
        unsafe {
            std::ptr::copy_nonoverlapping(
                source,
                (self.address as *mut u8).add(fragment.destination),
                fragment.nbytes,
            );
        }
    }
}

#[cfg(unix)]
impl UnixLane {
    fn new(depth: usize) -> Result<Self> {
        let staging = Aligned::new(depth * CHUNK)?;
        let ring = IoUring::new(depth as u32).map_err(|e| format!("Engram io_uring: {e}"))?;
        let buffers: Vec<_> = (0..depth)
            .map(|i| libc::iovec {
                iov_base: staging.at(i).cast(),
                iov_len: CHUNK,
            })
            .collect();
        unsafe { ring.submitter().register_buffers(&buffers) }
            .map_err(|e| format!("register Engram staging: {e}"))?;
        Ok(Self {
            ring,
            staging,
            depth,
        })
    }

    fn gather(
        &mut self,
        files: &[Arc<File>],
        reads: &[Read],
        fragments: &[Fragment],
        output: UnixOutput,
        stats: &Stats,
    ) -> Result<()> {
        let mut timing = DiskTimings {
            stats,
            plan: 0,
            submit: 0,
            wait: 0,
            complete: 0,
        };
        for batch in reads.chunks(self.depth) {
            let begin = Instant::now();
            for (slot, read) in batch.iter().enumerate() {
                let entry = opcode::ReadFixed::new(
                    types::Fd(files[read.file].as_raw_fd()),
                    self.staging.at(slot),
                    read.nbytes as u32,
                    slot as u16,
                )
                .offset(read.offset)
                .build()
                .user_data(slot as u64);
                unsafe { self.ring.submission().push(&entry) }
                    .map_err(|_| "Engram submission queue full")?;
            }
            stats
                .disk_reads
                .fetch_add(batch.len() as u64, Ordering::Relaxed);
            stats.disk_bytes.fetch_add(
                batch.iter().map(|read| read.nbytes as u64).sum(),
                Ordering::Relaxed,
            );
            timing.submit += begin.elapsed().as_nanos() as u64;
            let mut completed = 0;
            let mut failure = None;
            while completed < batch.len() {
                let begin = Instant::now();
                let result = self.ring.submit_and_wait(1);
                timing.wait += begin.elapsed().as_nanos() as u64;
                match result {
                    Ok(_) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(format!("submit Engram reads: {error}").into()),
                }
                let begin = Instant::now();
                for entry in self.ring.completion() {
                    completed += 1;
                    let slot = entry.user_data() as usize;
                    let read = &batch[slot];
                    let count = entry.result();
                    if count < 0 || (count as usize) < read.required {
                        failure = Some(format!(
                            "Engram read at {} returned {count}, requires {} bytes",
                            read.offset, read.required
                        ));
                        continue;
                    }
                    for fragment in &fragments[read.fragments.clone()] {
                        // SAFETY: This lane exclusively owns each fragment; the
                        // staging slot stays untouched until completion finishes.
                        unsafe {
                            output.copy(
                                fragment,
                                self.staging
                                    .at(slot)
                                    .add((fragment.source - read.offset) as usize),
                            );
                        }
                    }
                }
                timing.complete += begin.elapsed().as_nanos() as u64;
            }
            if let Some(error) = failure {
                return Err(error.into());
            }
        }
        Ok(())
    }
}

pub(crate) struct DiskReader {
    // Destroying each lane drops its io_uring before its registered staging buffer.
    #[cfg(unix)]
    lanes: Vec<UnixLane>,
    #[cfg(windows)]
    windows: windows::Readers,
    #[cfg(unix)]
    files: Vec<Arc<File>>,
    fragments: Vec<Fragment>,
    reads: Vec<Read>,
    resident_scales: Option<Resident>,
    stats: Arc<Stats>,
}

// Accumulate locally so timing does not add atomic operations to each completion.
// Drop also records completed phases when a gather exits with an I/O error.
struct DiskTimings<'a> {
    stats: &'a Stats,
    plan: u64,
    submit: u64,
    wait: u64,
    complete: u64,
}

impl Drop for DiskTimings<'_> {
    fn drop(&mut self) {
        for (counter, value) in [
            (&self.stats.disk_plan_ns, self.plan),
            (&self.stats.disk_submit_ns, self.submit),
            (&self.stats.disk_wait_ns, self.wait),
            (&self.stats.disk_complete_ns, self.complete),
        ] {
            counter.fetch_add(value, Ordering::Relaxed);
        }
    }
}

impl DiskReader {
    fn from_files(files: &[Arc<File>], stats: Arc<Stats>, workers: usize) -> Result<Self> {
        if !matches!(workers, 1 | 2 | 4 | 8) {
            return Err("embedding table workers must be 1, 2, 4, or 8".into());
        }
        #[cfg(unix)]
        let lanes = {
            let depth = DEPTH / workers;
            let mut lanes = Vec::with_capacity(workers);
            for _ in 0..workers {
                lanes.push(UnixLane::new(depth)?);
            }
            lanes
        };
        #[cfg(windows)]
        let windows = windows::Readers::new(files, workers)?;
        Ok(Self {
            #[cfg(windows)]
            windows,
            #[cfg(unix)]
            lanes,
            #[cfg(unix)]
            files: files.to_vec(),
            fragments: Vec::new(),
            reads: Vec::new(),
            resident_scales: None,
            stats,
        })
    }

    pub fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        let mut timing = DiskTimings {
            stats: &self.stats,
            plan: 0,
            submit: 0,
            wait: 0,
            complete: 0,
        };
        let begin = Instant::now();
        self.fragments.clear();
        let mut base = 0;
        for (ordinal, plane) in planes.iter().enumerate() {
            if ordinal == 1 {
                if let Some(scales) = &self.resident_scales {
                    // Scale rows are a few bytes each; the whole plane copies faster than a thread starts.
                    gather_resident(scales, plane.row_bytes, ids, &mut output[base..], 1);
                    base += ids.len() * plane.row_bytes;
                    continue;
                }
            }
            for (row, id) in ids.iter().copied().enumerate() {
                let destination = base + row * plane.row_bytes;
                if id < 0 {
                    output[destination..destination + plane.row_bytes].fill(0);
                    continue;
                }
                let source = plane.offset + id as u64 * plane.row_bytes as u64;
                let mut copied = 0;
                while copied < plane.row_bytes {
                    let address = source + copied as u64;
                    let nbytes = (plane.row_bytes - copied).min(BLOCK - address as usize % BLOCK);
                    self.fragments.push(Fragment {
                        file: plane.file,
                        source: address,
                        destination: destination + copied,
                        nbytes,
                    });
                    copied += nbytes;
                }
            }
            base += ids.len() * plane.row_bytes;
        }
        self.fragments.sort_unstable_by_key(|f| (f.file, f.source));
        self.reads.clear();
        for (i, fragment) in self.fragments.iter().enumerate() {
            let block = fragment.source / BLOCK as u64 * BLOCK as u64;
            if let Some(read) = self.reads.last_mut().filter(|r| {
                r.file == fragment.file
                    && block <= r.offset + r.nbytes as u64
                    && block + BLOCK as u64 <= r.offset + CHUNK as u64
            }) {
                read.nbytes = (block - read.offset) as usize + BLOCK;
                read.required = read
                    .required
                    .max((fragment.source - read.offset) as usize + fragment.nbytes);
                read.fragments.end = i + 1;
            } else {
                self.reads.push(Read {
                    file: fragment.file,
                    offset: block,
                    nbytes: BLOCK,
                    required: (fragment.source - block) as usize + fragment.nbytes,
                    fragments: i..i + 1,
                });
            }
        }
        timing.plan = begin.elapsed().as_nanos() as u64;
        #[cfg(unix)]
        {
            let output = UnixOutput {
                address: output.as_mut_ptr() as usize,
                len: output.len(),
            };
            if self.lanes.len() == 1 || self.reads.len() <= DEPTH {
                self.lanes[0].gather(&self.files, &self.reads, &self.fragments, output, &self.stats)?;
            } else {
                let chunk = self.reads.len().div_ceil(self.lanes.len());
                let files = &self.files;
                let fragments = &self.fragments;
                let stats = &self.stats;
                std::thread::scope(|scope| {
                    let (local, helpers) = self.lanes.split_first_mut().unwrap();
                    let mut jobs = self.reads.chunks(chunk);
                    let first = jobs.next().unwrap();
                    let mut joins = Vec::with_capacity(helpers.len());
                    for (index, (lane, reads)) in helpers.iter_mut().zip(jobs).enumerate() {
                        joins.push(
                            std::thread::Builder::new()
                                .name(format!("flare-table-io-{}", index + 1))
                                .spawn_scoped(scope, move || {
                                    crate::os::unpin_current_thread();
                                    lane.gather(files, reads, fragments, output, stats)
                                })
                                .map_err(|e| format!("start embedding read worker: {e}"))?,
                        );
                    }
                    let mut result = local.gather(files, first, fragments, output, stats);
                    for worker in joins {
                        let next = worker
                            .join()
                            .unwrap_or_else(|_| Err("embedding read worker panicked".into()));
                        if result.is_ok() {
                            result = next;
                        }
                    }
                    result
                })?;
            }
        }
        #[cfg(windows)]
        self.windows
            .gather(&self.reads, &self.fragments, output, &self.stats)?;

        Ok(())
    }
}

#[inline]
fn prefetch_read(ptr: *const u8) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        std::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) ptr,
            options(nostack, readonly, preserves_flags)
        );
    }
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(ptr.cast::<i8>(), std::arch::x86_64::_MM_HINT_T0);
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    let _ = ptr;
}

fn gather_thread_count(work_items: usize) -> usize {
    if work_items < PARALLEL_GATHER_MIN_ROWS {
        return 1;
    }
    std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .clamp(1, MAX_GATHER_THREADS)
        .min(work_items.div_ceil(PARALLEL_GATHER_MIN_ROWS / 2))
}

struct CachedReader {
    disk: DiskReader,
    cache: RowCache,
    requests: Vec<(i64, usize)>,
    miss_ids: Vec<i64>,
    miss_groups: Vec<(usize, usize)>,
    hit_groups: Vec<(usize, usize, usize)>,
    miss_output: Vec<u8>,
    max_rows: usize,
}

fn copy_cached_hits_slice(
    hits: &[(usize, usize, usize)],
    requests: &[(i64, usize)],
    output_base: usize,
    scale_base: usize,
    weights: usize,
    scales: usize,
) {
    for (index, &(source, begin, end)) in hits.iter().enumerate() {
        if let Some(&(next_source, _, _)) = hits.get(index + PREFETCH_DISTANCE) {
            prefetch_read(next_source as *const u8);
        }
        // SAFETY: `source` addresses one immutable cached row of `weights + scales`
        // bytes while no thread mutates `RowCache`.
        let row = unsafe { std::slice::from_raw_parts(source as *const u8, weights + scales) };
        for &(_, target) in &requests[begin..end] {
            // SAFETY: Request groups partition `requests` by unique row ID, so each
            // `target` output row is written by at most one thread.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    row.as_ptr(),
                    (output_base + target * weights) as *mut u8,
                    weights,
                );
                std::ptr::copy_nonoverlapping(
                    row.as_ptr().add(weights),
                    (output_base + scale_base + target * scales) as *mut u8,
                    scales,
                );
            }
        }
    }
}

fn copy_cached_hits(
    hits: &[(usize, usize, usize)],
    requests: &[(i64, usize)],
    output: &mut [u8],
    weights: usize,
    scales: usize,
    total_rows: usize,
) {
    let scale_base = total_rows * weights;
    let output_base = output.as_mut_ptr() as usize;
    let threads = gather_thread_count(hits.len());
    if threads <= 1 {
        copy_cached_hits_slice(hits, requests, output_base, scale_base, weights, scales);
        return;
    }
    let chunk = hits.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let mut parts = hits.chunks(chunk);
        let first = parts.next().unwrap_or(&[]);
        for part in parts {
            scope.spawn(move || {
                crate::os::unpin_current_thread();
                copy_cached_hits_slice(part, requests, output_base, scale_base, weights, scales);
            });
        }
        copy_cached_hits_slice(first, requests, output_base, scale_base, weights, scales);
    });
}

impl CachedReader {
    fn new(disk: DiskReader, planes: &[Plane; 2], budget: usize, max_rows: usize) -> Result<Self> {
        if max_rows == 0 {
            return Err("embedding row cache requires a nonzero gather capacity".into());
        }
        let row_bytes = planes.iter().map(|plane| plane.row_bytes).sum::<usize>();
        let workspace = max_rows
            .checked_mul(
                size_of::<(i64, usize)>()
                    + size_of::<i64>()
                    + size_of::<(usize, usize)>()
                    + size_of::<(usize, usize, usize)>()
                    + row_bytes,
            )
            .and_then(|n| n.checked_add(size_of::<Self>()))
            .ok_or("embedding row cache workspace size overflow")?;
        let payload_budget = budget
            .checked_sub(workspace)
            .ok_or("embedding row cache budget cannot hold its bounded miss workspace")?;
        fn allocated<T: Copy>(len: usize, value: T) -> Result<Vec<T>> {
            let mut data = Vec::new();
            data.try_reserve_exact(len)
                .map_err(|e| format!("embedding row cache workspace: {e}"))?;
            data.resize(len, value);
            // Touch the entire workspace before KV sizing, including vectors
            // whose length is cleared after allocation.
            for index in (0..len).step_by((4096 / size_of::<T>()).max(1)) {
                unsafe { data.as_mut_ptr().add(index).write_volatile(data[index]) };
            }
            Ok(data)
        }
        let mut requests = allocated(max_rows, (0i64, 0usize))?;
        let mut miss_ids = allocated(max_rows, 0i64)?;
        let mut miss_groups = allocated(max_rows, (0usize, 0usize))?;
        let mut hit_groups = allocated(max_rows, (0usize, 0usize, 0usize))?;
        requests.clear();
        miss_ids.clear();
        miss_groups.clear();
        hit_groups.clear();
        let miss_output = allocated(max_rows * row_bytes, 0u8)?;
        let cache = RowCache::new(
            payload_budget,
            planes[0].nbytes / planes[0].row_bytes,
            row_bytes,
        )?;
        let bytes = cache.bytes() + workspace;
        disk.stats
            .capacity_rows
            .fetch_add(cache.capacity() as u64, Ordering::Relaxed);
        disk.stats
            .cache_bytes
            .fetch_add(bytes as u64, Ordering::Relaxed);
        Ok(Self {
            disk,
            cache,
            requests,
            miss_ids,
            miss_groups,
            hit_groups,
            miss_output,
            max_rows,
        })
    }

    fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        if ids.len() > self.max_rows {
            return Err("embedding gather exceeds the row cache workspace capacity".into());
        }
        let [weights, scales] = planes.map(|plane| plane.row_bytes);
        let scale_base = ids.len() * weights;
        self.requests.clear();
        self.miss_ids.clear();
        self.miss_groups.clear();
        self.hit_groups.clear();
        for (row, &id) in ids.iter().enumerate() {
            if id < 0 {
                output[row * weights..(row + 1) * weights].fill(0);
                output[scale_base + row * scales..scale_base + (row + 1) * scales].fill(0);
            } else {
                self.requests.push((id, row));
            }
        }
        self.requests.sort_unstable_by_key(|&(id, _)| id);
        let mut begin = 0;
        let mut hits = 0;
        while begin < self.requests.len() {
            let id = self.requests[begin].0;
            let end = begin + self.requests[begin..].partition_point(|&(next, _)| next == id);
            if let Some(payload) = self.cache.get(id) {
                hits += 1;
                self.hit_groups
                    .push((payload.as_ptr() as usize, begin, end));
            } else {
                self.miss_ids.push(id);
                self.miss_groups.push((begin, end));
            }
            begin = end;
        }
        copy_cached_hits(
            &self.hit_groups,
            &self.requests,
            output,
            weights,
            scales,
            ids.len(),
        );
        self.disk.stats.hits.fetch_add(hits, Ordering::Relaxed);
        self.disk
            .stats
            .misses
            .fetch_add(self.miss_ids.len() as u64, Ordering::Relaxed);
        if self.miss_ids.is_empty() {
            return Ok(());
        }
        let count = self.miss_ids.len();
        self.disk.gather(
            planes,
            &self.miss_ids,
            &mut self.miss_output[..count * (weights + scales)],
        )?;
        for (index, (&id, &(begin, end))) in self.miss_ids.iter().zip(&self.miss_groups).enumerate()
        {
            let w = &self.miss_output[index * weights..(index + 1) * weights];
            let s = &self.miss_output
                [count * weights + index * scales..count * weights + (index + 1) * scales];
            for &(_, row) in &self.requests[begin..end] {
                output[row * weights..(row + 1) * weights].copy_from_slice(w);
                output[scale_base + row * scales..scale_base + (row + 1) * scales]
                    .copy_from_slice(s);
            }
            // Only complete successful disk gathers enter the immutable cache.
            let inserted = self.cache.insert(id, w, s);
            self.disk
                .stats
                .resident_rows
                .fetch_add(u64::from(inserted.added), Ordering::Relaxed);
            self.disk
                .stats
                .evictions
                .fetch_add(u64::from(inserted.evicted), Ordering::Relaxed);
        }
        Ok(())
    }
}

impl CachedReader {
    /// TP3 fork (perf/engram-readahead, after c2606aae): warm the row cache with predicted ids,
    /// without a staging frame. Advisory: ids that are negative, outside the table, already
    /// resident (those are pinned for their coming use instead) or repeated are skipped, and a
    /// failed read is returned for the caller to count; the real gather still reads the rows.
    /// Disk read counters include these reads; gather counters, hits and misses do not.
    fn prefetch(&mut self, planes: &[Plane; 2], ids: &[i64]) -> Result<usize> {
        let [weights, scales] = planes.map(|plane| plane.row_bytes);
        let rows = planes[0].nbytes / planes[0].row_bytes;
        self.miss_ids.clear();
        self.miss_ids.extend(ids.iter().copied().filter(|&id| id >= 0 && (id as usize) < rows));
        self.miss_ids.sort_unstable();
        self.miss_ids.dedup();
        self.miss_ids.retain(|&id| !self.cache.pin(id));
        let stats = &self.disk.stats;
        if self.miss_ids.len() > self.max_rows {
            stats.read_ahead_dropped.fetch_add((self.miss_ids.len() - self.max_rows) as u64, Ordering::Relaxed);
            self.miss_ids.truncate(self.max_rows);
        }
        let count = self.miss_ids.len();
        if count == 0 {
            return Ok(0);
        }
        self.disk.gather(planes, &self.miss_ids, &mut self.miss_output[..count * (weights + scales)])?;
        let mut admitted = 0usize;
        for index in 0..count {
            let id = self.miss_ids[index];
            let w = &self.miss_output[index * weights..(index + 1) * weights];
            let s = &self.miss_output[count * weights + index * scales..count * weights + (index + 1) * scales];
            let inserted = self.cache.insert_ahead(id, w, s);
            admitted += usize::from(inserted.admitted);
            self.disk.stats.resident_rows.fetch_add(u64::from(inserted.added), Ordering::Relaxed);
            self.disk.stats.evictions.fetch_add(u64::from(inserted.evicted), Ordering::Relaxed);
        }
        self.disk.stats.read_ahead_loaded.fetch_add(admitted as u64, Ordering::Relaxed);
        Ok(admitted)
    }
}

enum Data {
    Disk(DiskReader),
    Ram([Arc<Resident>; 2]),
    Cached(Box<CachedReader>),
}

/// Bytes one gather thread copies at minimum; smaller planes (the 64 KiB scale plane, decode-sized
/// weight gathers) stay on the calling thread rather than paying thread start-up per call.
const GATHER_BYTES_PER_THREAD: usize = 128 << 10;

/// Default cap on CPU threads one RAM-resident gather fans out across.
/// A RAM-resident plane for one process's readers: process-private when `private` (Unix only).
fn load_resident(file: &File, plane: Plane, private: bool) -> Result<Resident> {
    #[cfg(unix)]
    if private {
        return Resident::load_private(file, plane);
    }
    let _ = private;
    Resident::load(file, plane)
}

pub fn default_gather_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get()).min(MAX_GATHER_THREADS)
}

/// Threads a gather of `bytes` may use, capped by the configured `max_threads`.
fn gather_threads(bytes: usize, max_threads: usize) -> usize {
    (bytes / GATHER_BYTES_PER_THREAD).clamp(1, max_threads.max(1))
}

struct GatherJob {
    data: *const u8,
    data_len: usize,
    ids: *const i64,
    ids_len: usize,
    output: *mut u8,
    row_bytes: usize,
    threads: usize,
    generation: u64,
}
// SAFETY: Pointers inside GatherJob reference caller-owned slices whose lifetime strictly outlives
// the dispatch barrier (`remaining == 0`) before `GatherPool::run` returns.
unsafe impl Send for GatherJob {}
unsafe impl Sync for GatherJob {}

struct GatherShared {
    job: std::sync::Mutex<GatherJob>,
    wake: std::sync::Condvar,
    remaining: std::sync::atomic::AtomicUsize,
    panicked: std::sync::atomic::AtomicBool,
}

struct GatherPool {
    dispatch_lock: std::sync::Mutex<()>,
    shared: Arc<GatherShared>,
    workers: usize,
}

impl GatherPool {
    fn global() -> &'static Self {
        static POOL: std::sync::OnceLock<GatherPool> = std::sync::OnceLock::new();
        POOL.get_or_init(|| {
            let workers = default_gather_threads().saturating_sub(1);
            let shared = Arc::new(GatherShared {
                job: std::sync::Mutex::new(GatherJob {
                    data: std::ptr::null(),
                    data_len: 0,
                    ids: std::ptr::null(),
                    ids_len: 0,
                    output: std::ptr::null_mut(),
                    row_bytes: 0,
                    threads: 0,
                    generation: 0,
                }),
                wake: std::sync::Condvar::new(),
                remaining: std::sync::atomic::AtomicUsize::new(0),
                panicked: std::sync::atomic::AtomicBool::new(false),
            });
            let mut spawned = 0usize;
            for _ in 0..workers {
                let worker_id = spawned;
                let s = shared.clone();
                if std::thread::Builder::new()
                    .name(format!("flare-gather-{}", worker_id + 1))
                    .spawn(move || {
                        crate::os::unpin_current_thread();
                        let mut seen = 0u64;
                        loop {
                            let (data, data_len, ids, ids_len, output, row_bytes, threads) = {
                                let mut guard = s.job.lock().unwrap_or_else(|e| e.into_inner());
                                while guard.generation == seen {
                                    guard = s.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
                                }
                                seen = guard.generation;
                                (
                                    guard.data,
                                    guard.data_len,
                                    guard.ids,
                                    guard.ids_len,
                                    guard.output,
                                    guard.row_bytes,
                                    guard.threads,
                                )
                            };
                            let chunk_idx = worker_id + 1;
                            if chunk_idx < threads {
                                let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    let rows_per = ids_len.div_ceil(threads);
                                    let start = chunk_idx * rows_per;
                                    if start < ids_len {
                                        let end = (start + rows_per).min(ids_len);
                                        unsafe {
                                            let data_slice = std::slice::from_raw_parts(data, data_len);
                                            let ids_slice = std::slice::from_raw_parts(ids.add(start), end - start);
                                            let out_slice = std::slice::from_raw_parts_mut(
                                                output.add(start * row_bytes),
                                                (end - start) * row_bytes,
                                            );
                                            copy_rows(data_slice, row_bytes, ids_slice, out_slice);
                                        }
                                    }
                                }));
                                if res.is_err() {
                                    s.panicked.store(true, Ordering::Release);
                                }
                                s.remaining.fetch_sub(1, Ordering::Release);
                            }
                        }
                    })
                    .is_ok()
                {
                    spawned += 1;
                }
            }
            GatherPool {
                dispatch_lock: std::sync::Mutex::new(()),
                shared,
                workers: spawned,
            }
        })
    }

    fn run(&self, data: &[u8], row_bytes: usize, ids: &[i64], output: &mut [u8], threads: usize) {
        let threads = threads.min(self.workers + 1);
        if threads <= 1 {
            copy_rows(data, row_bytes, ids, output);
            return;
        }
        let _dispatch = self.dispatch_lock.lock().unwrap_or_else(|e| e.into_inner());
        let rows_per = ids.len().div_ceil(threads);
        let active_chunks = ids.len().div_ceil(rows_per);
        if active_chunks <= 1 {
            copy_rows(data, row_bytes, ids, output);
            return;
        }
        let worker_chunks = active_chunks - 1;
        self.shared.panicked.store(false, Ordering::Release);
        self.shared.remaining.store(worker_chunks, Ordering::Release);
        {
            let mut job = self.shared.job.lock().unwrap_or_else(|e| e.into_inner());
            job.data = data.as_ptr();
            job.data_len = data.len();
            job.ids = ids.as_ptr();
            job.ids_len = ids.len();
            job.output = output.as_mut_ptr();
            job.row_bytes = row_bytes;
            job.threads = active_chunks;
            job.generation = job.generation.wrapping_add(1);
            self.shared.wake.notify_all();
        }
        // Calling thread executes chunk 0 immediately while persistent workers execute chunks 1..active_chunks.
        let end0 = rows_per.min(ids.len());
        let local_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            copy_rows(data, row_bytes, &ids[..end0], &mut output[..end0 * row_bytes]);
        }));
        while self.shared.remaining.load(Ordering::Acquire) != 0 {
            std::hint::spin_loop();
        }
        if let Err(payload) = local_res {
            std::panic::resume_unwind(payload);
        }
        if self.shared.panicked.load(Ordering::Acquire) {
            panic!("engram gather worker panicked");
        }
    }
}

fn gather_resident(data: &Resident, row_bytes: usize, ids: &[i64], output: &mut [u8], max_threads: usize) {
    let data = data.bytes();
    let output = &mut output[..ids.len() * row_bytes];
    // Prefill gathers thousands of random rows; split large gathers across persistent CPU workers.
    let threads = gather_threads(output.len(), max_threads);
    if threads > 1 {
        GatherPool::global().run(data, row_bytes, ids, output, threads);
        return;
    }
    copy_rows(data, row_bytes, ids, output);
}

fn copy_rows(data: &[u8], row_bytes: usize, ids: &[i64], output: &mut [u8]) {
    let base = data.as_ptr();
    for (row, &id) in ids.iter().enumerate() {
        if let Some(&next_id) = ids.get(row + PREFETCH_DISTANCE) {
            if next_id >= 0 {
                // SAFETY: Every nonnegative ID in `ids` was validated against the plane extent.
                prefetch_read(unsafe { base.add(next_id as usize * row_bytes) });
            }
        }
        let target = &mut output[row * row_bytes..(row + 1) * row_bytes];
        if id < 0 {
            target.fill(0);
        } else {
            let start = id as usize * row_bytes;
            target.copy_from_slice(&data[start..start + row_bytes]);
        }
    }
}

#[cfg(test)]
fn gather_resident_slice(data: &[u8], row_bytes: usize, ids: &[i64], output: &mut [u8]) {
    copy_rows(data, row_bytes, ids, output);
}
pub(crate) struct Reader {
    data: Data,
    stats: Arc<Stats>,
    /// Upper bound on threads one RAM-resident gather fans out across.
    gather_threads: usize,
}
impl Reader {
    #[cfg(test)]
    pub(super) fn open(
        path: &Path,
        planes: &[Plane; 2],
        resident: bool,
        cache_bytes: usize,
        max_rows: usize,
        stats: Arc<Stats>,
    ) -> Result<Self> {
        // Cap 4 so resident gathers of a few MiB exercise the chunked parallel path.
        Self::from_files(
            &[Arc::new(open_file(path)?)],
            planes,
            resident,
            cache_bytes,
            max_rows,
            stats,
            1,
            false,
        )
        .map(|reader| reader.with_gather_threads(4))
    }
    pub fn from_files(
        files: &[Arc<File>],
        planes: &[Plane; 2],
        resident: bool,
        cache_bytes: usize,
        max_rows: usize,
        stats: Arc<Stats>,
        workers: usize,
        resident_scales: bool,
    ) -> Result<Self> {
        if resident && cache_bytes != 0 {
            return Err("embedding row cache requires disk table storage".into());
        }
        if planes.iter().any(|plane| plane.file >= files.len()) {
            return Err("embedding plane references an unknown file".into());
        }
        let data = if resident {
            #[cfg(unix)]
            let private = private_tables();
            #[cfg(not(unix))]
            let private = false;
            Data::Ram([
                Arc::new(load_resident(&files[planes[0].file], planes[0], private)?),
                Arc::new(load_resident(&files[planes[1].file], planes[1], private)?),
            ])
        } else {
            let mut disk = DiskReader::from_files(files, stats.clone(), workers)?;
            if resident_scales {
                disk.resident_scales = Some(Resident::load(&files[planes[1].file], planes[1])?);
                stats.resident_scale_bytes.fetch_add(planes[1].resident_bytes()? as u64, Ordering::Relaxed);
            }
            if cache_bytes != 0 {
                Data::Cached(Box::new(CachedReader::new(
                    disk,
                    planes,
                    cache_bytes,
                    max_rows,
                )?))
            } else {
                Data::Disk(disk)
            }
        };
        Ok(Self { data, stats, gather_threads: 1 })
    }
    /// Allow RAM-resident gathers to fan out across up to `threads` CPU threads.
    pub fn with_gather_threads(mut self, threads: usize) -> Self {
        self.gather_threads = threads.max(1);
        self
    }
    /// A reader over already-loaded resident planes (prepared tables); prefill-sized gathers
    /// fan out with the same default cap the package loader configures.
    #[cfg(unix)]
    pub(super) fn resident(data: [Arc<Resident>; 2], stats: Arc<Stats>) -> Self {
        Self {
            data: Data::Ram(data),
            stats,
            gather_threads: default_gather_threads(),
        }
    }
    /// Whether rows read ahead can be kept: a disk table with a row cache.
    pub fn caches_rows(&self) -> bool {
        matches!(self.data, Data::Cached(_))
    }
    /// Warm the row cache with predicted ids (see [`CachedReader::prefetch`]); no work for
    /// RAM-resident or uncached tables.
    pub fn prefetch(&mut self, planes: &[Plane; 2], ids: &[i64]) -> Result<usize> {
        match &mut self.data {
            Data::Cached(reader) => reader.prefetch(planes, ids),
            Data::Disk(_) | Data::Ram(_) => Ok(0),
        }
    }
    pub fn gather(&mut self, planes: &[Plane; 2], ids: &[i64], output: &mut [u8]) -> Result<()> {
        if output.len() != ids.len() * planes.iter().map(|p| p.row_bytes).sum::<usize>() {
            return Err("Engram staging size differs from row count".into());
        }
        for plane in planes {
            if ids
                .iter()
                .any(|id| *id < -1 || (*id >= 0 && *id as usize >= plane.nbytes / plane.row_bytes))
            {
                return Err("Engram row exceeds its table".into());
            }
        }
        let begin = Instant::now();
        let result = match &mut self.data {
            Data::Disk(reader) => reader.gather(planes, ids, output),
            Data::Cached(reader) => reader.gather(planes, ids, output),
            Data::Ram(data) => {
                let mut base = 0;
                for (plane, data) in planes.iter().zip(data) {
                    gather_resident(data, plane.row_bytes, ids, &mut output[base..], self.gather_threads);
                    base += ids.len() * plane.row_bytes;
                }
                Ok(())
            }
        };
        self.stats.gathers.fetch_add(1, Ordering::Relaxed);
        self.stats.gather_ns.fetch_add(
            begin.elapsed().as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::OpenOptions;

    #[test]
    fn gather_fan_out_follows_bytes_and_the_configured_cap() {
        // The 64 KiB scale plane and decode-sized weight gathers stay on the calling thread.
        assert_eq!(gather_threads(64 << 10, 16), 1);
        assert_eq!(gather_threads(8 * 256, 16), 1);
        // A 2 MiB prefill gather fans out, bounded by the configured thread count.
        assert_eq!(gather_threads(2 << 20, 16), 16);
        assert_eq!(gather_threads(2 << 20, 4), 4);
        assert_eq!(gather_threads(1 << 20, 16), 8);
        // A zero cap never divides by zero or disables the copy.
        assert_eq!(gather_threads(2 << 20, 0), 1);
    }

    #[test]
    fn resident_scales_preserve_bytes_and_cache_hits_without_reading_the_scale_file() {
        for (weight_row_bytes, scale_row_bytes) in [(256, 8), (80, 10), (128, 8), (128, 16)] {
            check_resident_scale_rows(weight_row_bytes, scale_row_bytes);
        }
    }

    fn check_resident_scale_rows(weight_row_bytes: usize, scale_row_bytes: usize) {
        let root = std::env::temp_dir().join(format!("helios-engram-scales-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let rows = 4097;
        let planes = [
            Plane { file: 0, offset: 4093, nbytes: rows * weight_row_bytes, row_bytes: weight_row_bytes },
            Plane { file: 1, offset: 4093, nbytes: rows * scale_row_bytes, row_bytes: scale_row_bytes },
        ];
        let row_bytes = weight_row_bytes + scale_row_bytes;
        let data: Vec<Vec<u8>> = planes.iter().map(|plane| {
            (0..plane.offset as usize + plane.nbytes).map(|i| (i * 19 + i / 257) as u8).collect()
        }).collect();
        for cache in [0, 65536] {
            let mut files = Vec::new();
            for (i, bytes) in data.iter().enumerate() {
                let path = root.join(i.to_string());
                std::fs::write(&path, bytes).unwrap();
                files.push(Arc::new(open_file(&path).unwrap()));
            }
            let stats = Arc::new(Stats::default());
            let mut reader = Reader::from_files(&files, &planes, false, cache, 8, stats.clone(), 1, true).unwrap();
            assert_eq!(stats.resident_scale_bytes.load(Ordering::Relaxed), planes[1].resident_bytes().unwrap() as u64);
            OpenOptions::new().write(true).open(root.join("1")).unwrap().set_len(0).unwrap();
            for ids in [vec![0, 4096, -1, 0], vec![4096, 0, -1, 0], vec![17, -1, 19]] {
                let mut expected = Vec::new();
                for (plane, bytes) in planes.iter().zip(&data) {
                    for &id in &ids {
                        if id < 0 {
                            expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                        } else {
                            let start = plane.offset as usize + id as usize * plane.row_bytes;
                            expected.extend_from_slice(&bytes[start..start + plane.row_bytes]);
                        }
                    }
                }
                let mut output = vec![0xa5; expected.len()];
                reader.gather(&planes, &ids, &mut output).unwrap();
                assert_eq!(output, expected);
            }
            assert!(stats.disk_reads.load(Ordering::Relaxed) > 0);
            if cache > 0 {
                assert_eq!(stats.hits.load(Ordering::Relaxed), 2);
                assert_eq!(stats.misses.load(Ordering::Relaxed), 4);
                let reads = stats.disk_reads.load(Ordering::Relaxed);
                reader.gather(&planes, &[17, 19], &mut vec![0; row_bytes * 2]).unwrap();
                assert_eq!(stats.disk_reads.load(Ordering::Relaxed), reads);
            }
            assert!(reader.gather(&planes, &[rows as i64], &mut vec![0; row_bytes]).is_err());
            assert!(reader.gather(&planes, &[-2], &mut vec![0; row_bytes]).is_err());
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(target_os = "linux")]
    fn unsealed_resident_file() -> File {
        let descriptor = unsafe {
            libc::memfd_create(
                c"helios-resident-seal-test".as_ptr(),
                libc::MFD_CLOEXEC | libc::MFD_ALLOW_SEALING,
            )
        };
        assert!(descriptor >= 0);
        let file = unsafe { File::from_raw_fd(descriptor) };
        file.set_len(BLOCK as u64).unwrap();
        file
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn resident_sealing_waits_for_unmapped_pages_held_by_io() {
        use std::io::Read;
        use std::sync::mpsc::{self, RecvTimeoutError};
        use std::time::Duration;

        let file = unsealed_resident_file();
        let mapping = ResidentMapping::new(&file, BLOCK, true).unwrap();
        unsafe { std::ptr::write_bytes(mapping.address as *mut u8, 0x39, BLOCK) };
        let mut descriptors = [-1; 2];
        assert_eq!(
            unsafe { libc::pipe2(descriptors.as_mut_ptr(), libc::O_CLOEXEC) },
            0
        );
        let mut reader = unsafe { File::from_raw_fd(descriptors[0]) };
        let writer = unsafe { File::from_raw_fd(descriptors[1]) };
        let vector = libc::iovec {
            iov_base: mapping.address as *mut libc::c_void,
            iov_len: BLOCK,
        };
        assert_eq!(
            unsafe { libc::vmsplice(writer.as_raw_fd(), &vector, 1, 0) },
            BLOCK as isize
        );
        drop(mapping);
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_ADD_SEALS, RESIDENT_SEALS) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EBUSY)
        );

        std::thread::scope(|scope| {
            let (send, receive) = mpsc::channel();
            let file = &file;
            scope.spawn(move || send.send(seal_resident(file)).unwrap());
            // The I/O reference remains live beyond the kernel's initial pin wait.
            assert!(matches!(
                receive.recv_timeout(Duration::from_millis(250)),
                Err(RecvTimeoutError::Timeout)
            ));
            let mut bytes = [0; BLOCK];
            reader.read_exact(&mut bytes).unwrap();
            assert_eq!(bytes, [0x39; BLOCK]);
            receive.recv_timeout(Duration::from_secs(2)).unwrap().unwrap();
        });
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) } & RESIDENT_SEALS,
            RESIDENT_SEALS
        );
        assert!(file.set_len(1).is_err());
        let value = 0u8;
        assert_eq!(
            unsafe { libc::pwrite(file.as_raw_fd(), (&value as *const u8).cast(), 1, 0) },
            -1
        );
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::EPERM)
        );
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn resident_sealing_rejects_a_retained_writable_mapping() {
        let file = unsealed_resident_file();
        let initial_seals = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) };
        let mapping = ResidentMapping::new(&file, BLOCK, true).unwrap();
        assert!(seal_resident(&file).is_err());
        assert_eq!(
            unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GET_SEALS) },
            initial_seals
        );
        drop(mapping);
        seal_resident(&file).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn shared_resident_rows_are_readable_without_first_access_page_faults() {
        let path = std::env::temp_dir().join(format!(
            "helios-engram-populated-{}.bin",
            std::process::id()
        ));
        let rows = 65536;
        let second = rows * 256;
        let bytes: Vec<_> = (0..second + rows * 8)
            .map(|i| (i * 19 + i / 7) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: 0,
                nbytes: second,
                row_bytes: 256,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        let file = open_file(&path).unwrap();
        let original = planes.map(|plane| Arc::new(Resident::load(&file, plane).unwrap()));
        let shared = std::array::from_fn(|i| {
            Arc::new(
                Resident::from_shared(original[i].file.as_ref().unwrap().try_clone().unwrap(), planes[i]).unwrap(),
            )
        });
        std::fs::remove_file(path).unwrap();
        let ids: Vec<_> = (0..128).map(|i| ((i * 509 + 17) % rows) as i64).collect();
        let mut output = vec![0; ids.len() * 264];
        let mut expected = Vec::with_capacity(output.len());
        for plane in &planes {
            for &id in &ids {
                let first = plane.offset as usize + id as usize * plane.row_bytes;
                expected.extend_from_slice(&bytes[first..first + plane.row_bytes]);
            }
        }
        for data in [original, shared] {
            let mut reader = Reader::resident(data, Arc::default());
            // Prime the copy code and output storage without touching the selected table pages.
            reader.gather(&planes, &[0], &mut output[..264]).unwrap();
            output.fill(0xA5);
            let mut before = unsafe { std::mem::zeroed::<libc::rusage>() };
            let mut after = unsafe { std::mem::zeroed::<libc::rusage>() };
            assert_eq!(
                unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut before) },
                0
            );
            reader.gather(&planes, &ids, &mut output).unwrap();
            assert_eq!(
                unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut after) },
                0
            );
            assert_eq!(
                after.ru_minflt, before.ru_minflt,
                "row gathering faulted resident pages"
            );
            assert_eq!(after.ru_majflt, before.ru_majflt);
            assert_eq!(output, expected);
        }
    }

    #[test]
    #[cfg(unix)]
    fn shared_resident_plane_outlives_source_and_loader_and_rejects_mutation() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-shared-{}.bin", std::process::id()));
        let bytes: Vec<_> = (0..12345).map(|i| (i * 19 + i / 7) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let plane = Plane {
            file: 0,
            offset: 97,
            nbytes: 8192,
            row_bytes: 128,
        };
        let original = Resident::load(&open_file(&path).unwrap(), plane).unwrap();
        let peer = Resident::from_shared(original.file.as_ref().unwrap().try_clone().unwrap(), plane).unwrap();
        use std::os::unix::fs::MetadataExt;
        assert_eq!(
            original.file.as_ref().unwrap().metadata().unwrap().ino(),
            peer.file.as_ref().unwrap().metadata().unwrap().ino()
        );
        assert!(original.file.as_ref().unwrap().set_len(1).is_err());
        let value = 255u8;
        assert_eq!(
            unsafe { libc::pwrite(original.descriptor().unwrap(), (&value as *const u8).cast(), 1, 97) },
            -1
        );
        let wrong = Plane {
            nbytes: plane.nbytes + BLOCK,
            ..plane
        };
        assert!(Resident::from_shared(original.file.as_ref().unwrap().try_clone().unwrap(), wrong).is_err());
        std::fs::remove_file(path).unwrap();
        drop(original);
        assert_eq!(peer.bytes(), &bytes[97..97 + 8192]);
    }

    #[test]
    fn resident_load_covers_parallel_chunks_and_rejects_truncated_planes() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-bulk-{}.bin", std::process::id()));
        let rows = LOAD_CHUNK * 5 / 80 + 1;
        let first = 123;
        let second = first + rows * 80 + 37;
        let bytes: Vec<_> = (0..second + rows * 10)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 80,
                row_bytes: 80,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 10,
                row_bytes: 10,
            },
        ];
        let file = open_file(&path).unwrap();
        for plane in planes {
            let data = Resident::load(&file, plane).unwrap();
            let begin = plane.offset as usize;
            assert_eq!(data.bytes(), &bytes[begin..begin + plane.nbytes]);
        }
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((first + LOAD_CHUNK + 17) as u64)
            .unwrap();
        assert!(
            Resident::load(&file, planes[0])
                .err()
                .unwrap()
                .to_string()
                .contains("requires")
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn private_planes_match_shared_planes_and_are_read_only() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-private-{}.bin", std::process::id()));
        let rows = LOAD_CHUNK * 3 / 96 + 5;
        let first = 4096 + 77;
        let second = first + rows * 96 + 13;
        let bytes: Vec<_> = (0..second + rows * 12)
            .map(|i| (i * 31 + i / 101) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane { file: 0, offset: first as u64, nbytes: rows * 96, row_bytes: 96 },
            Plane { file: 0, offset: second as u64, nbytes: rows * 12, row_bytes: 12 },
        ];
        let file = open_file(&path).unwrap();
        let shared = planes.map(|plane| Arc::new(load_resident(&file, plane, false).unwrap()));
        let private = planes.map(|plane| Arc::new(load_resident(&file, plane, true).unwrap()));
        for (plane, (shared, private)) in planes.iter().zip(shared.iter().zip(&private)) {
            let begin = plane.offset as usize;
            assert_eq!(private.bytes(), &bytes[begin..begin + plane.nbytes]);
            assert_eq!(private.bytes(), shared.bytes());
            assert!(shared.descriptor().is_some());
            assert!(private.descriptor().is_none());
            // Read-only private mapping without huge pages (VmFlags `nh`).
            let start = private.storage.address;
            let smaps = std::fs::read_to_string("/proc/self/smaps").unwrap();
            let mut lines = smaps.lines().skip_while(|line| {
                let range = line.split_whitespace().next().unwrap_or("");
                let Some((low, high)) = range.split_once('-') else { return true };
                match (usize::from_str_radix(low, 16), usize::from_str_radix(high, 16)) {
                    (Ok(low), Ok(high)) => !(low <= start && start < high),
                    _ => true,
                }
            });
            let header = lines.next().expect("private plane mapping");
            assert_eq!(header.split_whitespace().nth(1).unwrap(), "r--p", "{header}");
            let flags = lines.find(|line| line.starts_with("VmFlags:")).unwrap();
            assert!(flags.split_whitespace().any(|flag| flag == "nh"), "{flags}");
        }
        // Gathers over private planes return the same rows as over shared planes.
        let ids: Vec<_> = (0..300).map(|i| ((i * 7919 + 3) % rows) as i64).chain([-1]).collect();
        let row = 96 + 12;
        let mut expected = vec![0; ids.len() * row];
        let mut actual = vec![0xA5; ids.len() * row];
        Reader::resident(shared, Arc::default()).gather(&planes, &ids, &mut expected).unwrap();
        Reader::resident(private, Arc::default()).gather(&planes, &ids, &mut actual).unwrap();
        assert_eq!(actual, expected);
        // A truncated source fails the private load as it fails the shared one.
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len((first + LOAD_CHUNK + 17) as u64)
            .unwrap();
        assert!(
            load_resident(&file, planes[0], true)
                .err()
                .unwrap()
                .to_string()
                .contains("requires")
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn load_reader_count_parses_and_falls_back() {
        assert_eq!(parse_load_readers(None), LOAD_READERS);
        assert_eq!(parse_load_readers(Some("")), LOAD_READERS);
        assert_eq!(parse_load_readers(Some(" 16 ")), 16);
        assert_eq!(parse_load_readers(Some("64")), 64);
        for bad in ["0", "65", "-1", "eight"] {
            assert_eq!(parse_load_readers(Some(bad)), LOAD_READERS, "{bad}");
        }
    }

    #[test]
    fn plane_reads_are_identical_for_any_reader_count() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-readers-{}.bin", std::process::id()));
        let bytes: Vec<_> = (0..LOAD_CHUNK * 9 + 4321).map(|i| (i * 7 + i / 4099) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let plane = Plane { file: 0, offset: 513, nbytes: LOAD_CHUNK * 9, row_bytes: 64 };
        let file = open_file(&path).unwrap();
        let extent = plane.resident_bytes().unwrap();
        let required = 513 + plane.nbytes;
        for readers in [1, 3, 4, 16, 64] {
            let buffer = Aligned::new(extent).unwrap();
            Resident::read_plane(&file, plane, buffer.0, extent, required, readers).unwrap();
            let view = unsafe { std::slice::from_raw_parts((buffer.0 + 513) as *const u8, plane.nbytes) };
            assert_eq!(view, &bytes[513..513 + plane.nbytes], "{readers} readers");
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn warm_rows_avoid_disk_and_failed_misses_never_enter_the_cache() {
        let path =
            std::env::temp_dir().join(format!("helios-engram-cache-{}.bin", std::process::id()));
        let rows = 128;
        let first = 4093;
        let second = first + rows * 80 + 7;
        let bytes: Vec<u8> = (0..second + rows * 10)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 80,
                row_bytes: 80,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 10,
                row_bytes: 10,
            },
        ];
        let stats = Arc::new(Stats::default());
        let mut reader = Reader::open(&path, &planes, false, 65536, 32, stats.clone()).unwrap();
        assert!(stats.cache_bytes.load(Ordering::Relaxed) <= 65536);
        fn check(reader: &mut Reader, planes: &[Plane; 2], bytes: &[u8], ids: &[i64]) {
            let mut expected = Vec::new();
            for plane in planes {
                for &id in ids {
                    if id < 0 {
                        expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                    } else {
                        let start = plane.offset as usize + id as usize * plane.row_bytes;
                        expected.extend_from_slice(&bytes[start..start + plane.row_bytes]);
                    }
                }
            }
            let mut output = vec![0xa5; ids.len() * 90];
            reader.gather(planes, ids, &mut output).unwrap();
            assert_eq!(output, expected);
        }
        check(&mut reader, &planes, &bytes, &[2, 9, 2, -1, 17]);
        assert_eq!(stats.misses.load(Ordering::Relaxed), 3);
        assert_eq!(stats.resident_rows.load(Ordering::Relaxed), 3);
        let reads = stats.disk_reads.load(Ordering::Relaxed);
        assert!(reads > 0);
        check(&mut reader, &planes, &bytes, &[17, 9, -1, 2, 17]);
        assert_eq!(stats.hits.load(Ordering::Relaxed), 3);
        assert_eq!(stats.disk_reads.load(Ordering::Relaxed), reads);
        assert!(reader.gather(&planes, &[128], &mut [0; 90]).is_err());
        assert!(reader.gather(&planes, &[-2], &mut [0; 90]).is_err());
        assert!(reader.gather(&planes, &[2; 33], &mut [0; 33 * 90]).is_err());
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(0)
            .unwrap();
        check(&mut reader, &planes, &bytes, &[9, 2]);
        assert!(reader.gather(&planes, &[20], &mut [0; 90]).is_err());
        assert_eq!(stats.resident_rows.load(Ordering::Relaxed), 3);
        std::fs::write(&path, &bytes).unwrap();
        check(&mut reader, &planes, &bytes, &[20]);
        assert_eq!(stats.resident_rows.load(Ordering::Relaxed), 4);
        std::fs::remove_file(path).unwrap();
    }

    /// TP3 fork (perf/engram-readahead).
    #[test]
    fn read_ahead_warms_the_cache_so_the_gather_reads_no_disk_and_other_readers_do_nothing() {
        let path = std::env::temp_dir().join(format!("helios-engram-ahead-{}.bin", std::process::id()));
        let rows = 256;
        let second = rows * 80 + 3;
        let bytes: Vec<u8> = (0..second + rows * 10).map(|i| (i * 7 + i / 131) as u8).collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane { file: 0, offset: 0, nbytes: rows * 80, row_bytes: 80 },
            Plane { file: 0, offset: second as u64, nbytes: rows * 10, row_bytes: 10 },
        ];
        let expected = |ids: &[i64]| {
            let mut out = Vec::new();
            for plane in &planes {
                for &id in ids {
                    let start = plane.offset as usize + id as usize * plane.row_bytes;
                    out.extend_from_slice(&bytes[start..start + plane.row_bytes]);
                }
            }
            out
        };
        // RAM-resident and uncached disk readers cannot keep rows: no work, no disk reads.
        for resident in [true, false] {
            let stats = Arc::new(Stats::default());
            let mut other = Reader::open(&path, &planes, resident, 0, 64, stats.clone()).unwrap();
            assert!(!other.caches_rows());
            assert_eq!(other.prefetch(&planes, &[1, 2, 3]).unwrap(), 0);
            assert_eq!(stats.disk_reads.load(Ordering::Relaxed), 0);
        }
        let stats = Arc::new(Stats::default());
        let mut reader = Reader::open(&path, &planes, false, 1 << 20, 64, stats.clone()).unwrap();
        assert!(reader.caches_rows());
        // Duplicates, negatives and rows outside the table are skipped without an error.
        let loaded = reader.prefetch(&planes, &[40, 7, 40, -1, 9999, 200, 7]).unwrap();
        assert_eq!(loaded, 3);
        assert_eq!(stats.read_ahead_loaded.load(Ordering::Relaxed), 3);
        let reads = stats.disk_reads.load(Ordering::Relaxed);
        assert!(reads > 0);
        // Read-ahead is not a use: no hits or misses counted, no gather recorded.
        assert_eq!(stats.hits.load(Ordering::Relaxed) + stats.misses.load(Ordering::Relaxed), 0);
        assert_eq!(stats.gathers.load(Ordering::Relaxed), 0);
        let ids = [200, 7, 40, 7];
        let mut output = vec![0; ids.len() * 90];
        reader.gather(&planes, &ids, &mut output).unwrap();
        assert_eq!(output, expected(&ids));
        assert_eq!(stats.misses.load(Ordering::Relaxed), 0);
        assert_eq!(stats.hits.load(Ordering::Relaxed), 3);
        assert_eq!(stats.disk_reads.load(Ordering::Relaxed), reads, "the gather went to disk");
        // Resident rows are pinned rather than read again.
        assert_eq!(reader.prefetch(&planes, &[7, 40]).unwrap(), 0);
        assert_eq!(stats.disk_reads.load(Ordering::Relaxed), reads);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn disk_and_ram_gather_identical_rows_across_unaligned_planes() {
        let path = std::env::temp_dir().join(format!("helios-engram-{}.bin", std::process::id()));
        let rows = 4097;
        let first = 4093;
        let second = first + rows * 256 + 7;
        let bytes: Vec<_> = (0..second + rows * 8)
            .map(|i| (i * 13 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 256,
                row_bytes: 256,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        let ids: Vec<_> = (0..8192)
            .map(|i| {
                if i % 19 == 0 {
                    -1
                } else {
                    (i * 101 % rows) as i64
                }
            })
            .collect();
        let mut expected = Vec::new();
        for plane in &planes {
            for id in &ids {
                if *id == -1 {
                    expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                } else {
                    let offset = plane.offset as usize + *id as usize * plane.row_bytes;
                    expected.extend_from_slice(&bytes[offset..offset + plane.row_bytes]);
                }
            }
        }
        for (resident, cache_bytes) in [(false, 0), (true, 0), (false, 3 << 20)] {
            let mut reader = Reader::open(
                &path,
                &planes,
                resident,
                cache_bytes,
                ids.len(),
                Arc::default(),
            )
            .unwrap();
            for _ in 0..3 {
                let mut actual = vec![0xA5; expected.len()];
                reader.gather(&planes, &ids, &mut actual).unwrap();
                assert_eq!(actual, expected);
            }
            assert!(
                reader
                    .gather(&planes, &[rows as i64], &mut [0; 264])
                    .is_err()
            );
        }
        let mut reader = Reader::open(&path, &planes, false, 0, ids.len(), Arc::default()).unwrap();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(4096)
            .unwrap();
        assert!(reader.gather(&planes, &[100], &mut [0; 264]).is_err());
        std::fs::remove_file(&path).unwrap();
    }
    #[test]
    fn equal_offsets_in_separate_files_never_coalesce() {
        let root = std::env::temp_dir().join(format!("helios-engram-files-{}", std::process::id()));
        std::fs::create_dir_all(&root).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: 4093,
                nbytes: 80 * 128,
                row_bytes: 80,
            },
            Plane {
                file: 1,
                offset: 4093,
                nbytes: 10 * 128,
                row_bytes: 10,
            },
        ];
        let mut files = Vec::new();
        for (i, plane) in planes.iter().enumerate() {
            let path = root.join(i.to_string());
            std::fs::write(
                &path,
                vec![i as u8 + 1; plane.offset as usize + plane.nbytes],
            )
            .unwrap();
            files.push(Arc::new(open_file(&path).unwrap()));
        }
        for (resident, cache) in [(false, 0), (false, 65536), (true, 0)] {
            let mut reader =
                Reader::from_files(&files, &planes, resident, cache, 16, Arc::default(), 1, false).unwrap();
            for _ in 0..3 {
                let mut output = vec![0xa5; 4 * 90];
                reader
                    .gather(&planes, &[17, 2, -1, 17], &mut output)
                    .unwrap();
                assert_eq!(&output[..80], &[1; 80]);
                assert_eq!(&output[160..240], &[0; 80]);
                assert_eq!(&output[320..340], &[2; 20]);
                assert_eq!(&output[340..350], &[0; 10]);
                assert_eq!(&output[350..360], &[2; 10]);
            }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn parallel_resident_gather_matches_sequential_with_padding_rows() {
        let path = std::env::temp_dir().join(format!(
            "helios-engram-parallel-ram-{}.bin",
            std::process::id()
        ));
        let rows = 8192;
        let first = 123;
        let second = first + rows * 256 + 17;
        let bytes: Vec<u8> = (0..second + rows * 8)
            .map(|i| (i * 31 + i / 127) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 256,
                row_bytes: 256,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        let ids: Vec<i64> = (0..PARALLEL_GATHER_MIN_ROWS * 3 + 77)
            .map(|i| {
                if i % 23 == 0 {
                    -1
                } else {
                    ((i * 197 + 13) % rows) as i64
                }
            })
            .collect();
        let mut expected = vec![0u8; ids.len() * 264];
        let file = open_file(&path).unwrap();
        let resident = planes.map(|plane| Arc::new(Resident::load(&file, plane).unwrap()));
        gather_resident_slice(
            resident[0].bytes(),
            256,
            &ids,
            &mut expected[..ids.len() * 256],
        );
        gather_resident_slice(
            resident[1].bytes(),
            8,
            &ids,
            &mut expected[ids.len() * 256..],
        );
        let mut reader = Reader::open(&path, &planes, true, 0, ids.len(), Arc::default()).unwrap();
        let mut actual = vec![0xA5u8; expected.len()];
        reader.gather(&planes, &ids, &mut actual).unwrap();
        assert_eq!(actual, expected);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn parallel_cached_hit_gather_matches_disk_across_padding_and_duplicates() {
        let path = std::env::temp_dir().join(format!(
            "helios-engram-parallel-cache-{}.bin",
            std::process::id()
        ));
        let rows = 4096;
        let first = 4093;
        let second = first + rows * 256 + 11;
        let bytes: Vec<u8> = (0..second + rows * 8)
            .map(|i| (i * 17 + i / 131) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 256,
                row_bytes: 256,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        // Exceed PARALLEL_GATHER_MIN_ROWS unique cached hits with duplicates and -1 padding.
        let ids: Vec<i64> = (0..PARALLEL_GATHER_MIN_ROWS * 2 + 113)
            .map(|i| {
                if i % 29 == 0 {
                    -1
                } else {
                    ((i * 37) % 3072) as i64
                }
            })
            .collect();
        let mut expected = Vec::with_capacity(ids.len() * 264);
        for plane in &planes {
            for &id in &ids {
                if id < 0 {
                    expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                } else {
                    let start = plane.offset as usize + id as usize * plane.row_bytes;
                    expected.extend_from_slice(&bytes[start..start + plane.row_bytes]);
                }
            }
        }
        let stats = Arc::new(Stats::default());
        let mut reader =
            Reader::open(&path, &planes, false, 64 << 20, ids.len(), stats.clone()).unwrap();
        // Warm cache on first pass so the follow-up gather exercises parallel copy_cached_hits.
        let mut warm = vec![0u8; expected.len()];
        reader.gather(&planes, &ids, &mut warm).unwrap();
        assert_eq!(warm, expected);
        let hits_before = stats.hits.load(Ordering::Relaxed);
        let mut actual = vec![0x5Au8; expected.len()];
        reader.gather(&planes, &ids, &mut actual).unwrap();
        assert_eq!(actual, expected);
        assert!(
            stats.hits.load(Ordering::Relaxed) - hits_before >= PARALLEL_GATHER_MIN_ROWS as u64
        );
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    #[cfg(unix)]
    fn parallel_unix_disk_workers_preserve_rows_and_propagate_truncation() {
        let path = std::env::temp_dir().join(format!(
            "helios-engram-unix-workers-{}.bin",
            std::process::id()
        ));
        // Space rows more than CHUNK (64 KiB) apart so reads cannot coalesce below DEPTH (128).
        let rows = 48000;
        let first = 4093;
        let second = first + rows * 256 + 7;
        let bytes: Vec<u8> = (0..second + rows * 8)
            .map(|i| (i * 19 + i / 257) as u8)
            .collect();
        std::fs::write(&path, &bytes).unwrap();
        let planes = [
            Plane {
                file: 0,
                offset: first as u64,
                nbytes: rows * 256,
                row_bytes: 256,
            },
            Plane {
                file: 0,
                offset: second as u64,
                nbytes: rows * 8,
                row_bytes: 8,
            },
        ];
        // Stride 300 rows = 76.8 KiB (> 64 KiB CHUNK), producing 150+ distinct disk reads (> DEPTH).
        let ids: Vec<i64> = (0..160)
            .flat_map(|i| {
                let row = (i * 300) as i64;
                if i % 13 == 0 {
                    vec![-1, row, row]
                } else {
                    vec![row]
                }
            })
            .collect();
        let mut expected = Vec::with_capacity(ids.len() * 264);
        for plane in &planes {
            for &id in &ids {
                if id < 0 {
                    expected.extend(std::iter::repeat_n(0, plane.row_bytes));
                } else {
                    let start = plane.offset as usize + id as usize * plane.row_bytes;
                    expected.extend_from_slice(&bytes[start..start + plane.row_bytes]);
                }
            }
        }
        let files = [Arc::new(open_file(&path).unwrap())];
        for workers in [1, 2, 4, 8] {
            let stats = Arc::new(Stats::default());
            let mut reader = Reader::from_files(
                &files,
                &planes,
                false,
                0,
                ids.len(),
                stats.clone(),
                workers,
                false,
            )
            .unwrap();
            let mut actual = vec![0xA5u8; expected.len()];
            reader.gather(&planes, &ids, &mut actual).unwrap();
            assert_eq!(actual, expected, "worker count {workers} corrupted rows");
            assert!(stats.disk_reads.load(Ordering::Relaxed) > DEPTH as u64);
        }
        assert!(Reader::from_files(&files, &planes, false, 0, ids.len(), Arc::default(), 3, false).is_err());
        let mut reader =
            Reader::from_files(&files, &planes, false, 0, ids.len(), Arc::default(), 4, false)
                .unwrap();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(4096)
            .unwrap();
        assert!(reader.gather(&planes, &ids, &mut vec![0u8; expected.len()]).is_err());
        std::fs::remove_file(&path).unwrap();
    }
}
