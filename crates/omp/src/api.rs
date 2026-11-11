//! The `omp_*` user-level API.

use std::sync::atomic::Ordering;
use std::time::Instant;

use once_cell::sync::Lazy;

use crate::icv::{self, ICV};
use crate::team::current;

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_num_threads() -> i32 {
    current().team.nproc
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_thread_num() -> i32 {
    current().tid
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_max_threads() -> i32 {
    icv::nthreads()
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_set_num_threads(n: i32) {
    if n > 0 {
        ICV.nthreads.store(n, Ordering::Relaxed);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_num_procs() -> i32 {
    ICV.num_procs
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_in_parallel() -> i32 {
    i32::from(current().active_level > 0)
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_set_dynamic(flag: i32) {
    ICV.dynamic.store(flag != 0, Ordering::Relaxed);
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_dynamic() -> i32 {
    i32::from(ICV.dynamic.load(Ordering::Relaxed))
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_cancellation() -> i32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_set_schedule(kind: i32, chunk: i32) {
    let base = kind & !i32::MIN; // strip omp_sched_monotonic
    if (icv::SCHED_STATIC..=icv::SCHED_AUTO).contains(&base) {
        ICV.run_sched_kind.store(base, Ordering::Relaxed);
        // libomp: chunk < 1 means "default"; auto ignores the chunk
        let c = if base == icv::SCHED_AUTO || chunk < 1 { 0 } else { chunk };
        ICV.run_sched_chunk.store(c, Ordering::Relaxed);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn omp_get_schedule(kind: *mut i32, chunk: *mut i32) {
    let k = ICV.run_sched_kind.load(Ordering::Relaxed);
    let mut c = ICV.run_sched_chunk.load(Ordering::Relaxed);
    if c == 0 && k != icv::SCHED_STATIC && k != icv::SCHED_AUTO {
        c = 1; // libomp reports the default dynamic/guided chunk
    }
    unsafe {
        if !kind.is_null() {
            *kind = k;
        }
        if !chunk.is_null() {
            *chunk = c;
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_thread_limit() -> i32 {
    icv::THREAD_LIMIT
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_set_max_active_levels(n: i32) {
    if n >= 0 {
        ICV.max_active_levels.store(n, Ordering::Relaxed);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_max_active_levels() -> i32 {
    icv::max_active_levels()
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_level() -> i32 {
    current().level
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_active_level() -> i32 {
    current().active_level
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_ancestor_thread_num(level: i32) -> i32 {
    current().ancestor(level).map_or(-1, |f| f.tid)
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_team_size(level: i32) -> i32 {
    current().ancestor(level).map_or(-1, |f| f.team.nproc)
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_in_final() -> i32 {
    0
}

/// `omp_proc_bind_false`: threads are not bound (no affinity support).
#[unsafe(no_mangle)]
pub extern "C" fn omp_get_proc_bind() -> i32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_set_default_device(device: i32) {
    ICV.default_device.store(device, Ordering::Relaxed);
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_default_device() -> i32 {
    ICV.default_device.load(Ordering::Relaxed)
}

/// PORT-NOTE: host-only runtime: there are no offload devices.
#[unsafe(no_mangle)]
pub extern "C" fn omp_get_num_devices() -> i32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_num_teams() -> i32 {
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_team_num() -> i32 {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_is_initial_device() -> i32 {
    1
}

static EPOCH: Lazy<Instant> = Lazy::new(Instant::now);

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_wtime() -> f64 {
    EPOCH.elapsed().as_secs_f64()
}

#[unsafe(no_mangle)]
pub extern "C" fn omp_get_wtick() -> f64 {
    1e-9
}
