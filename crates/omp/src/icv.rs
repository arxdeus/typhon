//! Internal control variables (ICVs) and their environment initialization.
//!
//! libomp keeps most ICVs per task (data environment). Typhon only sets
//! them from the initial thread (`omp.set_num_threads`, `omp.set_schedule`, ...),
//! so this runtime keeps them process-global; values set inside a parallel region
//! are therefore visible to all threads, not just the setting task.

use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

use once_cell::sync::Lazy;

/// `omp_sched_t` values.
pub const SCHED_STATIC: i32 = 1;
pub const SCHED_DYNAMIC: i32 = 2;
pub const SCHED_GUIDED: i32 = 3;
pub const SCHED_AUTO: i32 = 4;

pub struct Icvs {
    pub nthreads: AtomicI32,
    pub dynamic: AtomicBool,
    pub max_active_levels: AtomicI32,
    pub run_sched_kind: AtomicI32,
    pub run_sched_chunk: AtomicI32,
    pub default_device: AtomicI32,
    pub num_procs: i32,
}

/// libomp's `KMP_MAX_ACTIVE_LEVELS_LIMIT`.
pub const MAX_ACTIVE_LEVELS_LIMIT: i32 = i32::MAX;
/// libomp's default thread limit (`KMP_MAX_NTH`-style "unlimited").
pub const THREAD_LIMIT: i32 = i32::MAX;

fn env_i32(name: &str) -> Option<i32> {
    let v = std::env::var(name).ok()?;
    // OMP_NUM_THREADS may be a nesting list ("4,2"); only the first level is used.
    v.split(',').next()?.trim().parse().ok()
}

fn env_bool(name: &str) -> Option<bool> {
    let v = std::env::var(name).ok()?;
    match v.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

/// Parse `OMP_SCHEDULE` (`[modifier:]kind[,chunk]`) into (`omp_sched_t`, chunk).
pub fn parse_schedule(s: &str) -> Option<(i32, i32)> {
    let s = s.trim().to_ascii_lowercase();
    let s = s.rsplit(':').next()?.to_string();
    let mut parts = s.splitn(2, ',');
    let kind = match parts.next()?.trim() {
        "static" => SCHED_STATIC,
        "dynamic" => SCHED_DYNAMIC,
        "guided" => SCHED_GUIDED,
        "auto" => SCHED_AUTO,
        _ => return None,
    };
    let chunk = match parts.next() {
        Some(c) => c.trim().parse::<i32>().ok()?.max(0),
        None => 0,
    };
    Some((kind, chunk))
}

pub static ICV: Lazy<Icvs> = Lazy::new(|| {
    let num_procs = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(1);
    let nthreads = env_i32("OMP_NUM_THREADS").filter(|&n| n > 0).unwrap_or(num_procs);
    let (kind, chunk) = std::env::var("OMP_SCHEDULE")
        .ok()
        .and_then(|s| parse_schedule(&s))
        .unwrap_or((SCHED_STATIC, 0));
    // PORT-NOTE: like recent libomp, nested parallelism is inactive by default
    // (max-active-levels = 1) unless OMP_MAX_ACTIVE_LEVELS / OMP_NESTED say otherwise.
    let mut mal = 1;
    if env_bool("OMP_NESTED") == Some(true) {
        mal = MAX_ACTIVE_LEVELS_LIMIT;
    }
    if let Some(v) = env_i32("OMP_MAX_ACTIVE_LEVELS").filter(|&v| v >= 0) {
        mal = v;
    }
    Icvs {
        nthreads: AtomicI32::new(nthreads),
        dynamic: AtomicBool::new(env_bool("OMP_DYNAMIC").unwrap_or(false)),
        max_active_levels: AtomicI32::new(mal),
        run_sched_kind: AtomicI32::new(kind),
        run_sched_chunk: AtomicI32::new(chunk),
        default_device: AtomicI32::new(0),
        num_procs,
    }
});

pub fn nthreads() -> i32 {
    ICV.nthreads.load(Ordering::Relaxed)
}

pub fn max_active_levels() -> i32 {
    ICV.max_active_levels.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_parsing() {
        assert_eq!(parse_schedule("dynamic,4"), Some((SCHED_DYNAMIC, 4)));
        assert_eq!(parse_schedule("nonmonotonic:guided"), Some((SCHED_GUIDED, 0)));
        assert_eq!(parse_schedule(" Static , 10"), Some((SCHED_STATIC, 10)));
        assert_eq!(parse_schedule("bogus"), None);
    }
}
