/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host operating-system operations for native serving.

pub(crate) use flare_eos::*;

#[cfg(unix)]
pub(crate) fn cuda_library() -> Result<libloading::Library, libloading::Error> {
    unsafe { libloading::Library::new("libcuda.so.1") }
}

#[cfg(windows)]
pub(crate) fn cuda_library() -> Result<libloading::Library, libloading::Error> {
    unsafe {
        libloading::os::windows::Library::load_with_flags(
            "nvcuda.dll",
            windows_sys::Win32::System::LibraryLoader::LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
        .map(Into::into)
    }
}

#[cfg(target_os = "linux")]
fn initial_affinity() -> Option<libc::cpu_set_t> {
    static INITIAL: std::sync::OnceLock<Option<libc::cpu_set_t>> = std::sync::OnceLock::new();
    *INITIAL.get_or_init(|| unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) == 0 {
            Some(set)
        } else {
            None
        }
    })
}

#[cfg(target_os = "linux")]
fn parse_cpu_list(text: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in text.trim().split(',').filter(|part| !part.is_empty()) {
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) =
                (start.trim().parse::<usize>(), end.trim().parse::<usize>())
            {
                if start <= end {
                    cpus.extend(start..=end);
                }
            }
        } else if let Ok(cpu) = part.trim().parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus
}

#[cfg(target_os = "linux")]
fn select_physical_cores<F>(allowed: &[usize], mut read_siblings: F) -> Vec<usize>
where
    F: FnMut(usize) -> Option<String>,
{
    let mut seen_cores = std::collections::HashSet::new();
    let mut non_zero = Vec::new();
    let mut zero_core = None;

    for &cpu in allowed {
        let siblings = read_siblings(cpu)
            .map(|text| parse_cpu_list(&text))
            .filter(|list| !list.is_empty());
        let core_key = siblings
            .as_ref()
            .and_then(|list| list.iter().copied().min())
            .unwrap_or(cpu);
        if !seen_cores.insert(core_key) {
            continue;
        }
        let has_cpu0 = cpu == 0 || siblings.as_ref().is_some_and(|list| list.contains(&0));
        if has_cpu0 {
            zero_core = Some(cpu);
        } else {
            non_zero.push(cpu);
        }
    }

    if let Some(cpu) = zero_core {
        non_zero.push(cpu);
    }
    non_zero
}

#[cfg(target_os = "linux")]
fn physical_cores() -> &'static [usize] {
    static CORES: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    CORES.get_or_init(|| {
        let Some(set) = initial_affinity() else {
            return Vec::new();
        };
        let mut allowed = Vec::new();
        for cpu in 0..libc::CPU_SETSIZE as usize {
            if unsafe { libc::CPU_ISSET(cpu, &set) } {
                allowed.push(cpu);
            }
        }
        select_physical_cores(&allowed, |cpu| {
            std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            ))
            .ok()
        })
    })
}

#[cfg(target_os = "linux")]
pub(crate) fn pin_current_thread(slot_index: usize) {
    let cores = physical_cores();
    if std::env::var_os("FLARE_PIN_CORES").is_some_and(|v| v == "0") {
        return;
    }
    if cores.len() < 4 || slot_index >= cores.len() {
        return;
    }
    let cpu = cores[slot_index];
    if cpu >= libc::CPU_SETSIZE as usize {
        return;
    }
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn pin_current_thread(_slot_index: usize) {}

#[cfg(target_os = "linux")]
pub(crate) fn unpin_current_thread() {
    if let Some(set) = initial_affinity() {
        unsafe {
            libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn unpin_current_thread() {}

#[cfg(unix)]
pub(crate) fn terminal_size() -> (usize, usize) {
    let mut size: libc::winsize = unsafe { std::mem::zeroed() };
    unsafe {
        libc::ioctl(libc::STDERR_FILENO, libc::TIOCGWINSZ, &mut size);
    }
    (
        if size.ws_col == 0 {
            80
        } else {
            size.ws_col as usize
        },
        if size.ws_row == 0 {
            24
        } else {
            size.ws_row as usize
        },
    )
}

#[cfg(windows)]
pub(crate) fn terminal_size() -> (usize, usize) {
    use windows_sys::Win32::System::Console::{
        CONSOLE_SCREEN_BUFFER_INFO, GetConsoleScreenBufferInfo, GetStdHandle, STD_ERROR_HANDLE,
    };
    let mut info: CONSOLE_SCREEN_BUFFER_INFO = unsafe { std::mem::zeroed() };
    if unsafe { GetConsoleScreenBufferInfo(GetStdHandle(STD_ERROR_HANDLE), &mut info) } == 0 {
        return (80, 24);
    }
    (
        (info.srWindow.Right - info.srWindow.Left + 1) as usize,
        (info.srWindow.Bottom - info.srWindow.Top + 1) as usize,
    )
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn parse_cpu_list_handles_ranges_and_commas() {
        assert_eq!(parse_cpu_list("0,24\n"), vec![0, 24]);
        assert_eq!(parse_cpu_list("0-3,8-9"), vec![0, 1, 2, 3, 8, 9]);
        assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
    }

    #[test]
    fn select_physical_cores_deduplicates_smt_and_deprioritizes_cpu0() {
        let allowed = vec![0, 1, 2, 3, 4, 5, 6, 7];
        // Siblings: (0,4), (1,5), (2,6), (3,7)
        let cores = select_physical_cores(&allowed, |cpu| Some(format!("{},{}", cpu % 4, cpu % 4 + 4)));
        assert_eq!(cores, vec![1, 2, 3, 0]);

        // Siblings: (0,1), (2,3), (4,5), (6,7)
        let cores = select_physical_cores(&allowed, |cpu| {
            let base = (cpu / 2) * 2;
            Some(format!("{base}-{}", base + 1))
        });
        assert_eq!(cores, vec![2, 4, 6, 0]);

        // Restricted to SMT siblings of core 0..3 without CPU 0 itself; core 0 sibling (4) is still deprioritized.
        let cores = select_physical_cores(&[4, 5, 6, 7], |cpu| Some(format!("{},{}", cpu - 4, cpu)));
        assert_eq!(cores, vec![5, 6, 7, 4]);

        // Sysfs unavailable: falls back to allowed CPUs with CPU 0 deprioritized.
        let cores = select_physical_cores(&[0, 2, 4, 6], |_| None);
        assert_eq!(cores, vec![2, 4, 6, 0]);
    }
}
