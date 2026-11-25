/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Optional override of the CUDA per-thread stack limit a core records.
//!
//! An export records the exporting context's limits (`cuCtxGetLimit`), and a load replays them
//! with `cuCtxSetLimit`. The recorded `CU_LIMIT_STACK_SIZE` is the driver's default of 1,024
//! bytes. The driver backs the per-thread stack with device memory for every thread the device
//! can hold resident at once, so a limit far above what any kernel uses holds memory the KV
//! cache could use. The override replaces only the stack entry, before any module loads. The
//! driver raises the limit by itself when a launched kernel's static stack requirement exceeds
//! it, so the load refuses an override below the largest requirement of any kernel the core
//! registers (read from each module's `EIATTR_MIN_STACK_SIZE` / `EIATTR_FRAME_SIZE` records and
//! `.nv.local.<kernel>` sections), and any override when a registered kernel's requirement
//! cannot be bounded (no stack record, or a cycle in the module call graph).

use std::collections::{HashMap, HashSet};

/// `CUlimit` id of `CU_LIMIT_STACK_SIZE`.
pub(super) const STACK_LIMIT_ID: usize = 0;

/// Environment variable read when `LoadOptions::cuda_stack_limit` is absent.
pub const STACK_LIMIT_ENV: &str = "FLARE_CUDA_STACK_LIMIT";

/// Largest override accepted (1 MiB per thread); larger values are almost certainly a typo.
const MAX_STACK_LIMIT: usize = 1 << 20;

/// Parse an override: a positive number of bytes per thread, at most 1 MiB.
pub fn parse_stack_limit(value: &str) -> Result<usize, String> {
    let bytes: usize = value
        .trim()
        .parse()
        .map_err(|_| format!("CUDA stack limit must be a number of bytes per thread, got {value:?}"))?;
    if bytes == 0 || bytes > MAX_STACK_LIMIT {
        return Err(format!(
            "CUDA stack limit must be between 1 and {MAX_STACK_LIMIT} bytes per thread, got {bytes}"
        ));
    }
    Ok(bytes)
}

/// The override in effect: the option when given, else a nonempty environment value.
pub(super) fn resolve_stack_limit(
    option: Option<usize>,
    environment: Option<&str>,
) -> crate::Result<Option<usize>> {
    if let Some(bytes) = option {
        return Ok(Some(parse_stack_limit(&bytes.to_string())?));
    }
    match environment.map(str::trim) {
        None | Some("") => Ok(None),
        Some(value) => parse_stack_limit(value)
            .map(Some)
            .map_err(|error| format!("{STACK_LIMIT_ENV}: {error}").into()),
    }
}

/// The context limits with the stack entry replaced (or added) by `bytes`.
pub(super) fn with_stack_limit(limits: &[[usize; 2]], bytes: usize) -> Vec<[usize; 2]> {
    let mut limits: Vec<_> = limits
        .iter()
        .copied()
        .filter(|[id, _]| *id != STACK_LIMIT_ID)
        .collect();
    // The stack goes first, as in recorded cores, so it is set before the other limits.
    limits.insert(0, [STACK_LIMIT_ID, bytes]);
    limits
}

/// The recorded stack limit, if the core records one.
pub(super) fn recorded_stack_limit(limits: &[[usize; 2]]) -> Option<usize> {
    limits
        .iter()
        .find(|[id, _]| *id == STACK_LIMIT_ID)
        .map(|[_, value]| *value)
}

/// Per-thread stack bytes of each kernel in one cubin, keyed by symbol; `None` marks a kernel
/// whose requirement cannot be bounded statically (call-graph cycle in its module).
pub(super) type StackNeeds = HashMap<String, Option<u64>>;

const SHT_SYMTAB: u32 = 2;
const SHT_CUDA_INFO: u32 = 0x7000_0000;
const SHT_CUDA_CALLGRAPH: u32 = 0x7000_0001;
const EIFMT_SVAL: u8 = 4;
const EIATTR_FRAME_SIZE: u8 = 0x11;
const EIATTR_MIN_STACK_SIZE: u8 = 0x12;

struct Section<'a> {
    name: &'a [u8],
    kind: u32,
    link: u32,
    data: &'a [u8],
    size: u64,
}

fn read<const N: usize>(image: &[u8], offset: usize) -> Result<[u8; N], String> {
    offset
        .checked_add(N)
        .and_then(|end| image.get(offset..end))
        .map(|bytes| bytes.try_into().unwrap())
        .ok_or_else(|| "cubin is truncated".to_owned())
}
fn u16_at(image: &[u8], offset: usize) -> Result<u16, String> {
    read::<2>(image, offset).map(u16::from_le_bytes)
}
fn u32_at(image: &[u8], offset: usize) -> Result<u32, String> {
    read::<4>(image, offset).map(u32::from_le_bytes)
}
fn u64_at(image: &[u8], offset: usize) -> Result<u64, String> {
    read::<8>(image, offset).map(u64::from_le_bytes)
}
fn c_string(table: &[u8], offset: usize) -> Result<&[u8], String> {
    let tail = table.get(offset..).ok_or("cubin string offset out of range")?;
    let end = tail.iter().position(|&b| b == 0).ok_or("cubin string is unterminated")?;
    Ok(&tail[..end])
}

fn sections(image: &[u8]) -> Result<Vec<Section<'_>>, String> {
    if !image.starts_with(b"\x7fELF") || image.get(4) != Some(&2) || image.get(5) != Some(&1) {
        return Err("module image is not a little-endian ELF64 cubin".into());
    }
    let table = u64_at(image, 0x28)? as usize;
    let entry = u16_at(image, 0x3a)? as usize;
    let count = u16_at(image, 0x3c)? as usize;
    let names = u16_at(image, 0x3e)? as usize;
    if entry < 64 || names >= count {
        return Err("cubin section table is malformed".into());
    }
    let mut raw = Vec::with_capacity(count);
    for index in 0..count {
        let header = table
            .checked_add(index.checked_mul(entry).ok_or("cubin section table overflows")?)
            .ok_or("cubin section table overflows")?;
        let (name, kind) = (u32_at(image, header)?, u32_at(image, header + 4)?);
        let (offset, size) = (u64_at(image, header + 24)?, u64_at(image, header + 32)?);
        let link = u32_at(image, header + 40)?;
        // NOBITS sections (type 8) occupy no file bytes; only their size matters.
        let data = if kind == 8 {
            &image[0..0]
        } else {
            let start = usize::try_from(offset).map_err(|_| "cubin section offset overflows")?;
            let end = usize::try_from(offset.checked_add(size).ok_or("cubin section overflows")?)
                .map_err(|_| "cubin section overflows")?;
            image.get(start..end).ok_or("cubin section lies outside the image")?
        };
        raw.push((name, kind, link, data, size));
    }
    let strings = raw[names].3;
    raw.into_iter()
        .map(|(name, kind, link, data, size)| {
            Ok(Section { name: c_string(strings, name as usize)?, kind, link, data, size })
        })
        .collect()
}

/// Read the per-thread stack requirement of every function symbol in a cubin.
pub(super) fn kernel_stack_needs(image: &[u8]) -> Result<StackNeeds, String> {
    let sections = sections(image)?;
    let symtab = sections
        .iter()
        .find(|s| s.kind == SHT_SYMTAB)
        .ok_or("cubin has no symbol table")?;
    let strtab = sections
        .get(symtab.link as usize)
        .ok_or("cubin symbol table has no string table")?
        .data;
    // Largest stack record per symbol index (min stack covers callees; frame is the own frame).
    let mut stack: HashMap<u32, u64> = HashMap::new();
    for info in sections.iter().filter(|s| s.kind == SHT_CUDA_INFO && s.name == b".nv.info") {
        let data = info.data;
        let mut at = 0;
        while at < data.len() {
            let (format, attribute) = (data[at], *data.get(at + 1).ok_or("cubin .nv.info is truncated")?);
            let value = u16_at(data, at + 2)? as usize;
            at += 4;
            if format == EIFMT_SVAL {
                let payload = data.get(at..at + value).ok_or("cubin .nv.info is truncated")?;
                if matches!(attribute, EIATTR_MIN_STACK_SIZE | EIATTR_FRAME_SIZE) {
                    if payload.len() != 8 {
                        return Err("cubin stack record has an unexpected size".into());
                    }
                    let symbol = u32_at(payload, 0)?;
                    let bytes = u64::from(u32_at(payload, 4)?);
                    let entry = stack.entry(symbol).or_default();
                    *entry = (*entry).max(bytes);
                }
                at += value;
            } else if !(1..=3).contains(&format) {
                return Err(format!("cubin .nv.info record format {format} is unknown"));
            }
        }
    }
    // Recursion makes a function's stack unbounded; the records then understate it.
    let mut edges: HashMap<u32, Vec<u32>> = HashMap::new();
    for graph in sections.iter().filter(|s| s.kind == SHT_CUDA_CALLGRAPH) {
        for pair in graph.data.chunks_exact(8) {
            let (caller, callee) = (u32_at(pair, 0)?, u32_at(pair, 4)?);
            // Zero callers and negative callees mark entries without a call edge.
            if caller != 0 && (callee as i32) > 0 {
                edges.entry(caller).or_default().push(callee);
            }
        }
    }
    let recursive = has_cycle(&edges);
    let mut local: HashMap<&[u8], u64> = HashMap::new();
    for section in &sections {
        if let Some(name) = section.name.strip_prefix(b".nv.local.") {
            *local.entry(name).or_default() += section.size;
        }
    }
    let mut needs = StackNeeds::new();
    for (&symbol, &bytes) in &stack {
        let name = symbol_name(symtab.data, strtab, symbol)?;
        let need = (!recursive).then(|| bytes + local.get(name).copied().unwrap_or(0));
        needs.insert(String::from_utf8_lossy(name).into_owned(), need);
    }
    Ok(needs)
}

fn symbol_name<'a>(symtab: &[u8], strtab: &'a [u8], index: u32) -> Result<&'a [u8], String> {
    let entry = (index as usize).checked_mul(24).ok_or("cubin symbol index overflows")?;
    c_string(strtab, u32_at(symtab, entry)? as usize)
}

fn has_cycle(edges: &HashMap<u32, Vec<u32>>) -> bool {
    // Iterative three-colour depth-first search.
    let (mut done, mut open) = (HashSet::new(), HashSet::new());
    for &start in edges.keys() {
        if done.contains(&start) {
            continue;
        }
        let mut path = vec![(start, 0usize)];
        open.insert(start);
        while let Some((node, next)) = path.last_mut() {
            let node = *node;
            match edges.get(&node).and_then(|callees| callees.get(*next)) {
                Some(&callee) => {
                    *next += 1;
                    if open.contains(&callee) {
                        return true;
                    }
                    if !done.contains(&callee) {
                        open.insert(callee);
                        path.push((callee, 0));
                    }
                }
                None => {
                    open.remove(&node);
                    done.insert(node);
                    path.pop();
                }
            }
        }
    }
    false
}

/// The largest stack requirement among `registered` kernels (`(symbol, needs of its module)`),
/// or an error naming the first kernel the requirement cannot be read for.
pub(super) fn registered_stack_need<'a>(
    registered: impl IntoIterator<Item = (&'a str, Result<&'a StackNeeds, &'a String>)>,
) -> Result<(u64, String), String> {
    let mut largest = (0u64, String::new());
    for (symbol, needs) in registered {
        let needs = needs.map_err(|error| format!("kernel {symbol}: {error}"))?;
        match needs.get(symbol) {
            Some(Some(bytes)) if largest.1.is_empty() || *bytes > largest.0 => {
                largest = (*bytes, symbol.to_owned());
            }
            Some(Some(_)) => {}
            Some(None) => return Err(format!("kernel {symbol} is recursive; its stack is unbounded")),
            None => return Err(format!("kernel {symbol} has no stack record in its module")),
        }
    }
    Ok(largest)
}

/// Refuse a stack override below the largest requirement of the registered kernels.
pub(super) fn check_stack_limit(
    limit: usize,
    rank: usize,
    need: &Result<(u64, String), String>,
) -> crate::Result<()> {
    match need {
        Ok((bytes, _)) if *bytes <= limit as u64 => Ok(()),
        Ok((bytes, symbol)) => Err(format!(
            "CUDA stack limit {limit} B is below rank {rank}'s largest kernel stack {bytes} B \
             ({symbol}); use at least {bytes} or drop the override"
        )
        .into()),
        Err(error) => Err(format!(
            "CUDA stack limit {limit} B refused: rank {rank}'s kernel stack cannot be bounded ({error})"
        )
        .into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal ELF64 cubin: null, .shstrtab, .strtab, .symtab, .nv.info, optional
    /// .nv.callgraph, and one NOBITS .nv.local.<name> per entry of `local`.
    fn cubin(
        symbols: &[&str],
        records: &[(u8, u32, u32)],
        calls: &[(u32, i32)],
        local: &[(&str, u64)],
    ) -> Vec<u8> {
        let mut shstr = vec![0u8];
        let mut name = |text: &str| {
            let at = shstr.len() as u32;
            shstr.extend_from_slice(text.as_bytes());
            shstr.push(0);
            at
        };
        let mut sections: Vec<(u32, u32, u32, Vec<u8>, u64)> = vec![(0, 0, 0, vec![], 0)];
        let shstrtab = name(".shstrtab");
        let strtab_name = name(".strtab");
        let symtab_name = name(".symtab");
        let info_name = name(".nv.info");
        let graph_name = name(".nv.callgraph");
        let locals: Vec<_> = local.iter().map(|(n, size)| (name(&format!(".nv.local.{n}")), *size)).collect();
        let mut strtab = vec![0u8];
        let mut symtab = vec![0u8; 24];
        for symbol in symbols {
            let at = strtab.len() as u32;
            strtab.extend_from_slice(symbol.as_bytes());
            strtab.push(0);
            let mut entry = vec![0u8; 24];
            entry[0..4].copy_from_slice(&at.to_le_bytes());
            entry[4] = 0x12;
            symtab.extend(entry);
        }
        let mut info = Vec::new();
        for &(attribute, symbol, bytes) in records {
            info.extend([EIFMT_SVAL, attribute, 8, 0]);
            info.extend(symbol.to_le_bytes());
            info.extend(bytes.to_le_bytes());
        }
        info.extend([3, 0x1b, 168, 0]); // an HVAL record (register count) to skip
        let mut graph = Vec::new();
        for &(caller, callee) in calls {
            graph.extend(caller.to_le_bytes());
            graph.extend(callee.to_le_bytes());
        }
        sections.push((shstrtab, 3, 0, shstr.clone(), 0));
        sections.push((strtab_name, 3, 0, strtab, 0));
        sections.push((symtab_name, SHT_SYMTAB, 2, symtab, 0));
        sections.push((info_name, SHT_CUDA_INFO, 3, info, 0));
        sections.push((graph_name, SHT_CUDA_CALLGRAPH, 3, graph, 0));
        for (n, size) in locals {
            sections.push((n, 8, 0, vec![], size));
        }
        // The section-name table must hold every name, including ones added after it was cloned.
        sections[1].3 = shstr;
        let mut image = vec![0u8; 64];
        image[0..4].copy_from_slice(b"\x7fELF");
        image[4] = 2;
        image[5] = 1;
        let mut offsets = Vec::new();
        for section in &sections {
            offsets.push(image.len() as u64);
            image.extend(&section.3);
        }
        let table = image.len() as u64;
        image[0x28..0x30].copy_from_slice(&table.to_le_bytes());
        image[0x3a..0x3c].copy_from_slice(&64u16.to_le_bytes());
        image[0x3c..0x3e].copy_from_slice(&(sections.len() as u16).to_le_bytes());
        image[0x3e..0x40].copy_from_slice(&1u16.to_le_bytes());
        for ((name, kind, link, data, nobits), offset) in sections.iter().zip(offsets) {
            let mut header = vec![0u8; 64];
            header[0..4].copy_from_slice(&name.to_le_bytes());
            header[4..8].copy_from_slice(&kind.to_le_bytes());
            header[24..32].copy_from_slice(&offset.to_le_bytes());
            let size = if *kind == 8 { *nobits } else { data.len() as u64 };
            header[32..40].copy_from_slice(&size.to_le_bytes());
            header[40..44].copy_from_slice(&link.to_le_bytes());
            image.extend(header);
        }
        image
    }

    /// Reads every `*.cubin` in `FLARE_STACK_TEST_CUBINS` (cubins split from a package's kernel
    /// payload) and prints the kernel count and the largest stack, for comparison with
    /// `cuobjdump --dump-resource-usage`.
    #[test]
    #[ignore = "needs FLARE_STACK_TEST_CUBINS"]
    fn real_cubins_parse() {
        let directory = std::env::var_os("FLARE_STACK_TEST_CUBINS").expect("FLARE_STACK_TEST_CUBINS");
        let (mut kernels, mut largest, mut unbounded) = (0usize, (0u64, String::new()), 0usize);
        for entry in std::fs::read_dir(directory).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "cubin") {
                let needs = kernel_stack_needs(&std::fs::read(&path).unwrap())
                    .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                for (symbol, need) in needs {
                    kernels += 1;
                    match need {
                        Some(bytes) if bytes > largest.0 => largest = (bytes, symbol),
                        Some(_) => {}
                        None => unbounded += 1,
                    }
                }
            }
        }
        println!("functions with stack records {kernels}, unbounded {unbounded}, largest {} B {}", largest.0, largest.1);
    }

    #[test]
    fn parses_bytes_and_rejects_zero_huge_and_text() {
        assert_eq!(parse_stack_limit("256"), Ok(256));
        assert_eq!(parse_stack_limit(" 1024 "), Ok(1024));
        assert_eq!(parse_stack_limit("1048576"), Ok(1 << 20));
        for bad in ["0", "1048577", "-256", "256B", "0x100", "", "2.5"] {
            assert!(parse_stack_limit(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn option_wins_over_environment_and_empty_environment_means_none() {
        assert_eq!(resolve_stack_limit(None, None).unwrap(), None);
        assert_eq!(resolve_stack_limit(None, Some("")).unwrap(), None);
        assert_eq!(resolve_stack_limit(None, Some(" ")).unwrap(), None);
        assert_eq!(resolve_stack_limit(None, Some("256")).unwrap(), Some(256));
        assert_eq!(resolve_stack_limit(Some(512), Some("256")).unwrap(), Some(512));
        assert_eq!(resolve_stack_limit(Some(512), Some("junk")).unwrap(), Some(512));
        let error = resolve_stack_limit(None, Some("junk")).unwrap_err().to_string();
        assert!(error.contains(STACK_LIMIT_ENV), "{error}");
        assert!(resolve_stack_limit(Some(0), None).is_err());
    }

    #[test]
    fn override_replaces_only_the_stack_entry() {
        let recorded = [[0, 1024], [1, 9_240_576], [2, 8 << 20], [6, 25_165_824]];
        assert_eq!(recorded_stack_limit(&recorded), Some(1024));
        let limits = with_stack_limit(&recorded, 256);
        assert_eq!(limits, [[0, 256], [1, 9_240_576], [2, 8 << 20], [6, 25_165_824]]);
        assert_eq!(with_stack_limit(&[[1, 7]], 256), [[0, 256], [1, 7]]);
        assert_eq!(recorded_stack_limit(&[[1, 7]]), None);
    }

    #[test]
    fn reads_min_stack_frame_and_local_per_kernel() {
        // Symbols 1..=3: a kernel with a 192 B min stack, one with frame 40 / min 32 plus 16 B
        // of local memory, and a callee helper with only a frame record.
        let image = cubin(
            &["layer_kernel", "small_kernel", "helper"],
            &[
                (EIATTR_FRAME_SIZE, 1, 192),
                (EIATTR_MIN_STACK_SIZE, 1, 192),
                (EIATTR_FRAME_SIZE, 2, 40),
                (EIATTR_MIN_STACK_SIZE, 2, 32),
                (EIATTR_FRAME_SIZE, 3, 192),
            ],
            &[(0, -1), (1, 3), (0, -2)],
            &[("small_kernel", 16)],
        );
        let needs = kernel_stack_needs(&image).unwrap();
        assert_eq!(needs["layer_kernel"], Some(192));
        assert_eq!(needs["small_kernel"], Some(56));
        assert_eq!(needs["helper"], Some(192));
    }

    #[test]
    fn recursion_marks_every_kernel_of_the_module_unbounded() {
        let image = cubin(
            &["a", "b"],
            &[(EIATTR_MIN_STACK_SIZE, 1, 16), (EIATTR_MIN_STACK_SIZE, 2, 16)],
            &[(1, 2), (2, 1)],
            &[],
        );
        let needs = kernel_stack_needs(&image).unwrap();
        assert_eq!(needs["a"], None);
        assert_eq!(needs["b"], None);
        let self_call = cubin(&["a"], &[(EIATTR_MIN_STACK_SIZE, 1, 16)], &[(1, 1)], &[]);
        assert_eq!(kernel_stack_needs(&self_call).unwrap()["a"], None);
        let chain = cubin(&["a", "b", "c"], &[(EIATTR_MIN_STACK_SIZE, 1, 16)], &[(1, 2), (2, 3), (1, 3)], &[]);
        assert_eq!(kernel_stack_needs(&chain).unwrap()["a"], Some(16));
    }

    #[test]
    fn malformed_images_are_errors() {
        assert!(kernel_stack_needs(b"not an elf").is_err());
        let mut image = cubin(&["a"], &[(EIATTR_MIN_STACK_SIZE, 1, 16)], &[], &[]);
        image.truncate(image.len() - 40); // cuts the last section header's offset and size
        assert!(kernel_stack_needs(&image).is_err());
    }

    #[test]
    fn check_refuses_below_the_largest_registered_kernel_and_unbounded_kernels() {
        let big = cubin(&["layer_kernel", "unused"], &[(EIATTR_MIN_STACK_SIZE, 1, 192), (EIATTR_MIN_STACK_SIZE, 2, 832)], &[], &[]);
        let small = cubin(&["reduce"], &[(EIATTR_MIN_STACK_SIZE, 1, 56)], &[], &[]);
        let (big, small) = (kernel_stack_needs(&big), kernel_stack_needs(&small));
        let registered = [("layer_kernel", big.as_ref()), ("reduce", small.as_ref())];
        // The unregistered 832 B kernel of the same module does not count.
        let need = registered_stack_need(registered);
        assert_eq!(need, Ok((192, "layer_kernel".into())));
        check_stack_limit(256, 0, &need).unwrap();
        check_stack_limit(192, 0, &need).unwrap();
        let error = check_stack_limit(191, 2, &need).unwrap_err().to_string();
        assert!(error.contains("rank 2") && error.contains("192 B") && error.contains("layer_kernel"), "{error}");
        let missing = registered_stack_need([("absent", small.as_ref())]);
        assert!(check_stack_limit(1024, 0, &missing).is_err());
        let failed: Result<StackNeeds, String> = Err("module image is not a little-endian ELF64 cubin".into());
        let unreadable = registered_stack_need([("reduce", failed.as_ref())]);
        assert!(check_stack_limit(1024, 1, &unreadable).unwrap_err().to_string().contains("cannot be bounded"));
        let recursive = kernel_stack_needs(&cubin(&["a"], &[(EIATTR_MIN_STACK_SIZE, 1, 16)], &[(1, 1)], &[]));
        let unbounded = registered_stack_need([("a", recursive.as_ref())]);
        assert!(check_stack_limit(1 << 20, 0, &unbounded).unwrap_err().to_string().contains("recursive"));
        assert_eq!(registered_stack_need(std::iter::empty()), Ok((0, String::new())));
    }
}
