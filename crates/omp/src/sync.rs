//! Mutual exclusion and reductions: critical, single, master, `__kmpc_reduce*`,
//! the `__kmpc_atomic_*` entry points used by Typhon, and flush.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU32, AtomicU64, Ordering, fence};

use once_cell::sync::Lazy;
use parking_lot::lock_api::RawMutex as _;
use parking_lot::{Mutex, RawMutex};

use crate::Ident;
use crate::team::{barrier, current};

/* -------------------------------------------------------------- critical --- */

/// Locks for `critical` / reductions, keyed by the address of the compiler-provided
/// `kmp_critical_name` (the stdlib 32-byte `Lock`). The `Lock` memory itself is never
/// written.
///
/// PORT-NOTE: libomp lazily stores a lock pointer inside the `kmp_critical_name`
/// words; we key a side table by address instead (the stdlib `Lock` is only 4-aligned).
static CRIT_LOCKS: Lazy<Mutex<HashMap<usize, &'static RawMutex>>> = Lazy::new(|| Mutex::new(HashMap::new()));

fn crit_lock(key: *const ()) -> &'static RawMutex {
    let mut m = CRIT_LOCKS.lock();
    m.entry(key as usize).or_insert_with(|| Box::leak(Box::new(RawMutex::INIT)))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_critical(_loc: *mut Ident, _gtid: i32, crit: *mut [i32; 8]) {
    crit_lock(crit as *const ()).lock();
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_end_critical(_loc: *mut Ident, _gtid: i32, crit: *mut [i32; 8]) {
    unsafe { crit_lock(crit as *const ()).unlock() };
}

/* ------------------------------------------------------- omp simple locks --- */

/// `omp_lock_t` is one pointer wide (libomp); the 1-byte `RawMutex` lives in place
/// (all-zero = unlocked, so `omp_init_lock` just zeroes it). Used by OpenBLAS.
const _: () = assert!(size_of::<RawMutex>() <= size_of::<usize>());

#[unsafe(no_mangle)]
pub unsafe extern "C" fn omp_init_lock(l: *mut usize) {
    unsafe { l.write(0) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn omp_destroy_lock(_l: *mut usize) {}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn omp_set_lock(l: *mut usize) {
    unsafe { (*(l as *const RawMutex)).lock() };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn omp_unset_lock(l: *mut usize) {
    unsafe { (*(l as *const RawMutex)).unlock() };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn omp_test_lock(l: *mut usize) -> i32 {
    i32::from(unsafe { (*(l as *const RawMutex)).try_lock() })
}

/* -------------------------------------------------------- single / master --- */

/// Returns 1 for exactly one thread of the team per encountered `single`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_single(_loc: *mut Ident, _gtid: i32) -> i32 {
    let f = current();
    let seen = f.singles_seen.get() + 1;
    f.singles_seen.set(seen);
    if f.team.nproc == 1 {
        return 1;
    }
    // Team counter = number of singles already claimed; claim ours if not yet taken.
    let claimed = &f.team.single;
    let mut cur = claimed.load(Ordering::Acquire);
    while cur < seen {
        match claimed.compare_exchange_weak(cur, seen, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return i32::from(cur == seen - 1),
            Err(v) => cur = v,
        }
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_end_single(_loc: *mut Ident, _gtid: i32) {}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_master(_loc: *mut Ident, _gtid: i32) -> i32 {
    i32::from(current().tid == 0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_end_master(_loc: *mut Ident, _gtid: i32) {}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_flush(_loc: *mut Ident) {
    fence(Ordering::SeqCst);
}

/* ------------------------------------------------------------ reductions --- */

// PORT-NOTE: reduction method. libomp picks critical / atomic / tree per platform and
// team size. This runtime always uses the *critical* method: `__kmpc_reduce[_nowait]`
// returns 1 with the reduction lock held and the caller combines its partial result
// non-atomically, then `__kmpc_end_reduce[_nowait]` releases the lock. The stdlib's
// generated code handles 1 (and 2 = atomic, never returned here); with a team of one
// thread no lock is taken (libomp's empty_reduce_block is also 1).

fn reduce_begin(lck: *const ()) -> i32 {
    let f = current();
    if f.team.nproc > 1 {
        let l = crit_lock(lck);
        l.lock();
        f.reduce_lock.set(l as *const RawMutex);
    }
    1
}

fn reduce_end() {
    let f = current();
    let l = f.reduce_lock.replace(std::ptr::null());
    if !l.is_null() {
        unsafe { (*l).unlock() };
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_reduce_nowait(
    _loc: *mut Ident,
    _gtid: i32,
    _num_vars: i32,
    _reduce_size: i64,
    _reduce_data: *mut std::ffi::c_void,
    _reduce_func: *mut std::ffi::c_void,
    lck: *mut [i32; 8],
) -> i32 {
    reduce_begin(lck as *const ())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_end_reduce_nowait(_loc: *mut Ident, _gtid: i32, _lck: *mut [i32; 8]) {
    reduce_end();
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_reduce(
    _loc: *mut Ident,
    _gtid: i32,
    _num_vars: i32,
    _reduce_size: i64,
    _reduce_data: *mut std::ffi::c_void,
    _reduce_func: *mut std::ffi::c_void,
    lck: *mut [i32; 8],
) -> i32 {
    reduce_begin(lck as *const ())
}

/// Ends a blocking reduction: release the lock, then the team barrier.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_end_reduce(_loc: *mut Ident, _gtid: i32, _lck: *mut [i32; 8]) {
    reduce_end();
    barrier(current());
}

/* --------------------------------------------------------------- atomics --- */

fn update_u64(p: &AtomicU64, f: impl Fn(u64) -> u64) {
    let _ = p.fetch_update(Ordering::AcqRel, Ordering::Relaxed, |v| Some(f(v)));
}

fn update_u32(p: &AtomicU32, f: impl Fn(u32) -> u32) {
    let _ = p.fetch_update(Ordering::AcqRel, Ordering::Relaxed, |v| Some(f(v)));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_atomic_fixed8_mul(_loc: *mut Ident, _gtid: i32, lhs: *mut i64, rhs: i64) {
    let a = unsafe { AtomicI64::from_ptr(lhs) };
    let _ = a.fetch_update(Ordering::AcqRel, Ordering::Relaxed, |v| Some(v.wrapping_mul(rhs)));
}

macro_rules! atomic_float {
    ($name:ident, $t:ty, $atom:ty, $upd:ident, $op:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $name(_loc: *mut Ident, _gtid: i32, lhs: *mut $t, rhs: $t) {
            let a = unsafe { <$atom>::from_ptr(lhs.cast()) };
            let op: fn($t, $t) -> $t = $op;
            $upd(a, |bits| op(<$t>::from_bits(bits), rhs).to_bits());
        }
    };
}

// min/max follow libomp: `if (*lhs > rhs) *lhs = rhs` (NaN rhs never stored).
atomic_float!(__kmpc_atomic_float8_add, f64, AtomicU64, update_u64, |a, b| a + b);
atomic_float!(__kmpc_atomic_float8_mul, f64, AtomicU64, update_u64, |a, b| a * b);
atomic_float!(__kmpc_atomic_float8_min, f64, AtomicU64, update_u64, |a, b| if a > b { b } else { a });
atomic_float!(__kmpc_atomic_float8_max, f64, AtomicU64, update_u64, |a, b| if a < b { b } else { a });
atomic_float!(__kmpc_atomic_float4_add, f32, AtomicU32, update_u32, |a, b| a + b);
atomic_float!(__kmpc_atomic_float4_mul, f32, AtomicU32, update_u32, |a, b| a * b);
atomic_float!(__kmpc_atomic_float4_min, f32, AtomicU32, update_u32, |a, b| if a > b { b } else { a });
atomic_float!(__kmpc_atomic_float4_max, f32, AtomicU32, update_u32, |a, b| if a < b { b } else { a });
