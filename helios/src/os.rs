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
fn online_cpus() -> usize {
    let n = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
    if n > 0 { n as usize } else { 1 }
}

#[cfg(target_os = "linux")]
pub(crate) fn pin_current_thread(cpu: usize) {
    if std::env::var_os("FLARE_PIN_CORES").is_some_and(|v| v == "0") {
        return;
    }
    let ncpus = online_cpus();
    if ncpus < 8 || cpu >= ncpus || cpu >= libc::CPU_SETSIZE as usize {
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
pub(crate) fn pin_current_thread(_cpu: usize) {}

#[cfg(target_os = "linux")]
pub(crate) fn unpin_current_thread() {
    let ncpus = online_cpus().min(libc::CPU_SETSIZE as usize);
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for cpu in 0..ncpus {
            libc::CPU_SET(cpu, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
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
