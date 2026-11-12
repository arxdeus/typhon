//! Worksharing loops: `__kmpc_for_static_init_8` and the dynamic dispatcher
//! (`__kmpc_dispatch_{init,next,fini}_8`) including `ordered`.
//!
//! The dispatcher mirrors libomp's ring of shared dispatch buffers: the k-th
//! dynamically scheduled loop a team executes uses buffer `k % DISP_BUFS`, so
//! threads may run ahead into following `nowait` loops while slower threads finish.
//! Iterations are handed out as indices in `[0, trip_count)` from an atomic
//! counter (dynamic: fixed chunks; guided: CAS with shrinking chunks).

use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use parking_lot::Mutex;

use crate::team::{Spinner, current};
use crate::{Ident, icv};

/// libomp's `__kmp_dispatch_num_buffers` default.
pub const DISP_BUFS: usize = 7;

// enum sched_type (kmp.h)
const SCH_STATIC_CHUNKED: i32 = 33;
const SCH_STATIC: i32 = 34;
const SCH_DYNAMIC_CHUNKED: i32 = 35;
const SCH_GUIDED_CHUNKED: i32 = 36;
const SCH_RUNTIME: i32 = 37;
const SCH_AUTO: i32 = 38;
const SCH_TRAPEZOIDAL: i32 = 39;
const SCH_STATIC_GREEDY: i32 = 40;
const SCH_STATIC_BALANCED: i32 = 41;
const SCH_GUIDED_ITERATIVE: i32 = 42;
const SCH_GUIDED_ANALYTICAL: i32 = 43;
const SCH_STATIC_STEAL: i32 = 44;
const SCH_STATIC_BALANCED_CHUNKED: i32 = 45;
const SCH_GUIDED_SIMD: i32 = 46;
const SCH_RUNTIME_SIMD: i32 = 47;
const ORD_LOWER: i32 = 64;
const ORD_UPPER: i32 = 96;
const SCH_MODIFIER_MASK: i32 = (1 << 29) | (1 << 30);

/// Normalized dispatch algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Static,
    StaticChunked,
    Dynamic,
    Guided,
}

/// Strip modifiers and ordered offset; returns (base schedule, ordered).
fn decode(schedule: i32) -> (i32, bool) {
    let s = schedule & !SCH_MODIFIER_MASK;
    if s > ORD_LOWER && s <= ORD_UPPER { (s - ORD_LOWER + 32, true) } else { (s, false) }
}

fn normalize(base: i32, chunk: i64) -> (Kind, i64) {
    match base {
        SCH_RUNTIME | SCH_RUNTIME_SIMD => {
            let kind = icv::ICV.run_sched_kind.load(Ordering::Relaxed);
            let c = icv::ICV.run_sched_chunk.load(Ordering::Relaxed) as i64;
            match kind {
                icv::SCHED_DYNAMIC => (Kind::Dynamic, c.max(1)),
                icv::SCHED_GUIDED | icv::SCHED_AUTO => (Kind::Guided, c.max(1)),
                _ if c > 0 => (Kind::StaticChunked, c),
                _ => (Kind::Static, 0),
            }
        }
        SCH_STATIC_CHUNKED if chunk > 0 => (Kind::StaticChunked, chunk),
        SCH_STATIC_CHUNKED | SCH_STATIC | SCH_STATIC_GREEDY | SCH_STATIC_BALANCED => (Kind::Static, 0),
        SCH_STATIC_BALANCED_CHUNKED | SCH_STATIC_STEAL => {
            if chunk > 0 { (Kind::StaticChunked, chunk) } else { (Kind::Static, 0) }
        }
        SCH_DYNAMIC_CHUNKED => (Kind::Dynamic, chunk.max(1)),
        // PORT-NOTE: auto and all guided/trapezoidal variants use one guided algorithm
        // (chunk = max(chunk, remaining / (2 * nproc))), libomp's guided_iterative.
        SCH_GUIDED_CHUNKED | SCH_AUTO | SCH_TRAPEZOIDAL | SCH_GUIDED_ITERATIVE
        | SCH_GUIDED_ANALYTICAL | SCH_GUIDED_SIMD => (Kind::Guided, chunk.max(1)),
        _ => (Kind::Dynamic, chunk.max(1)),
    }
}

/// Trip count of `lb..=ub` by `st` (0 for an empty loop).
fn trip_count(lb: i64, ub: i64, st: i64) -> u64 {
    if st == 0 {
        return 0;
    }
    let (lb, ub, st) = (lb as i128, ub as i128, st as i128);
    let n = if st > 0 {
        if ub < lb { 0 } else { (ub - lb) / st + 1 }
    } else if lb < ub {
        0
    } else {
        (lb - ub) / (-st) + 1
    };
    n as u64
}

/* -------------------------------------------------------------- static ---- */

/// `void __kmpc_for_static_init_8(ident_t*, kmp_int32 gtid, kmp_int32 schedtype,
///   kmp_int32 *plastiter, kmp_int64 *plower, kmp_int64 *pupper, kmp_int64 *pstride,
///   kmp_int64 incr, kmp_int64 chunk)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_for_static_init_8(
    _loc: *mut Ident,
    _gtid: i32,
    schedtype: i32,
    plastiter: *mut i32,
    plower: *mut i64,
    pupper: *mut i64,
    pstride: *mut i64,
    incr: i64,
    chunk: i64,
) {
    let f = current();
    let (lower, upper) = unsafe { (*plower, *pupper) };
    let (last, lo, hi, stride) = static_bounds(schedtype, f.tid as i64, f.team.nproc as i64, lower, upper, incr, chunk);
    unsafe {
        if !plastiter.is_null() {
            *plastiter = last as i32;
        }
        *plower = lo;
        *pupper = hi;
        if !pstride.is_null() {
            *pstride = stride;
        }
    }
}

/// Pure static-schedule computation (libomp `__kmp_for_static_init`, greedy default).
/// Returns (is_last, lower, upper, stride).
pub fn static_bounds(
    schedtype: i32,
    tid: i64,
    nth: i64,
    lower: i64,
    upper: i64,
    incr: i64,
    chunk: i64,
) -> (bool, i64, i64, i64) {
    let incr = if incr == 0 { 1 } else { incr };
    let tc = trip_count(lower, upper, incr);
    if tc == 0 {
        // zero-trip loop: leave the (empty) bounds, nobody is last
        return (false, lower, upper, incr);
    }
    let full_stride = if incr > 0 { upper - lower + 1 } else { -(lower - upper + 1) };
    if nth <= 1 {
        return (true, lower, upper, full_stride);
    }
    let tc_i = tc as i64;
    let (base, _) = decode(schedtype);
    let chunked = matches!(base, SCH_STATIC_CHUNKED | SCH_STATIC_BALANCED_CHUNKED) && chunk > 0;
    if !chunked {
        // kmp_sch_static (greedy)
        if tc_i < nth {
            if tid < tc_i {
                let lo = lower + tid * incr;
                return (tid == tc_i - 1, lo, lo, tc_i);
            }
            return (false, upper + incr.signum(), upper, tc_i);
        }
        let big = (tc_i / nth + i64::from(tc_i % nth != 0)) * incr;
        let lo = lower.wrapping_add(tid * big);
        let mut hi = lo.wrapping_add(big - incr);
        let last;
        if incr > 0 {
            if hi < lo {
                hi = i64::MAX;
            }
            last = lo <= upper && hi > upper - incr;
            hi = hi.min(upper);
        } else {
            if hi > lo {
                hi = i64::MIN;
            }
            last = lo >= upper && hi < upper - incr;
            hi = hi.max(upper);
        }
        return (last, lo, hi, tc_i);
    }
    let chunk = chunk.min(tc_i).max(1);
    let nchunks = tc_i / chunk + i64::from(tc_i % chunk != 0);
    let span = chunk * incr;
    let last = tid == (nchunks - 1) % nth;
    if nchunks < nth {
        if tid < nchunks {
            let lo = lower + span * tid;
            (last, lo, lo + span - incr, span * nchunks)
        } else {
            (last, upper + incr.signum(), upper, span * nchunks)
        }
    } else {
        let lo = lower + span * tid;
        (last, lo, lo + span - incr, span * nth)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_for_static_fini(_loc: *mut Ident, _gtid: i32) {}

/* ------------------------------------------------------------- dynamic ---- */

/// Team-shared state of one dynamically scheduled loop.
pub struct LoopShared {
    kind: Kind,
    lb: i64,
    st: i64,
    tc: i64,
    chunk: i64,
    nproc: i64,
    next: AtomicI64,
    /// Number of iterations (in iteration-index order) whose ordered part is done.
    ordered_iter: AtomicI64,
}

struct Slot {
    loop_id: u64,
    shared: Option<Arc<LoopShared>>,
    done: i64,
}

/// One entry of the team's dispatch-buffer ring.
pub struct DispBuf {
    slot: Mutex<Slot>,
}

impl DispBuf {
    pub(crate) fn new(first_id: u64) -> Self {
        DispBuf { slot: Mutex::new(Slot { loop_id: first_id, shared: None, done: 0 }) }
    }
}

/// Thread-private dispatcher state (per team membership).
#[derive(Default)]
pub struct DispPriv {
    /// Number of dynamic loops this thread has started in the team.
    count: Cell<u64>,
    cur: RefCell<Option<Arc<LoopShared>>>,
    /// Static-in-dispatch: next chunk index for this thread.
    static_next: Cell<i64>,
    ordered: Cell<bool>,
    /// Iteration index whose ordered section is next for this thread.
    ord_cur: Cell<i64>,
    ord_end: Cell<i64>,
    ord_bumped: Cell<bool>,
}

/// `void __kmpc_dispatch_init_8(ident_t*, kmp_int32 gtid, enum sched_type,
///   kmp_int64 lb, kmp_int64 ub, kmp_int64 st, kmp_int64 chunk)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_dispatch_init_8(
    _loc: *mut Ident,
    _gtid: i32,
    schedule: i32,
    lb: i64,
    ub: i64,
    st: i64,
    chunk: i64,
) {
    let f = current();
    let p = &f.disp;
    let (base, ordered) = decode(schedule);
    let (kind, chunk) = normalize(base, chunk);
    let st = if st == 0 { 1 } else { st };
    let id = p.count.get();
    p.count.set(id + 1);
    let buf = &f.team.disp[(id % DISP_BUFS as u64) as usize];
    let mut spin = Spinner::new();
    let shared = loop {
        {
            let mut s = buf.slot.lock();
            if s.loop_id == id {
                let sh = s.shared.get_or_insert_with(|| {
                    Arc::new(LoopShared {
                        kind,
                        lb,
                        st,
                        tc: trip_count(lb, ub, st) as i64,
                        chunk,
                        nproc: f.team.nproc as i64,
                        next: AtomicI64::new(0),
                        ordered_iter: AtomicI64::new(0),
                    })
                });
                break sh.clone();
            }
        }
        // The buffer is still in use by a loop DISP_BUFS loops ago.
        if !spin.spin() {
            std::thread::yield_now();
        }
    };
    p.static_next.set(f.tid as i64);
    p.ordered.set(ordered);
    p.ord_cur.set(0);
    p.ord_end.set(0);
    p.ord_bumped.set(false);
    *p.cur.borrow_mut() = Some(shared);
}

/// Mark the current loop finished for this thread and release the buffer slot
/// once every team member is done with it.
fn finish_loop(f: &crate::team::Frame) {
    let p = &f.disp;
    if p.cur.borrow_mut().take().is_none() {
        return;
    }
    let id = p.count.get() - 1;
    let buf = &f.team.disp[(id % DISP_BUFS as u64) as usize];
    let mut s = buf.slot.lock();
    s.done += 1;
    if s.done >= f.team.nproc as i64 {
        s.done = 0;
        s.shared = None;
        s.loop_id += DISP_BUFS as u64;
    }
}

/// `kmp_int32 __kmpc_dispatch_next_8(ident_t*, kmp_int32 gtid, kmp_int32 *p_last,
///   kmp_int64 *p_lb, kmp_int64 *p_ub, kmp_int64 *p_st)`: 1 = chunk assigned, 0 = done.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_dispatch_next_8(
    _loc: *mut Ident,
    _gtid: i32,
    p_last: *mut i32,
    p_lb: *mut i64,
    p_ub: *mut i64,
    p_st: *mut i64,
) -> i32 {
    let f = current();
    let p = &f.disp;
    let Some(sh) = p.cur.borrow().clone() else {
        return 0;
    };
    let range = next_chunk(&sh, p);
    match range {
        Some((first, count)) => {
            if p.ordered.get() {
                p.ord_cur.set(first);
                p.ord_end.set(first + count);
                p.ord_bumped.set(false);
            }
            unsafe {
                if !p_last.is_null() {
                    *p_last = i32::from(first + count >= sh.tc);
                }
                *p_lb = sh.lb + first * sh.st;
                *p_ub = sh.lb + (first + count - 1) * sh.st;
                if !p_st.is_null() {
                    *p_st = sh.st;
                }
            }
            1
        }
        None => {
            finish_loop(f);
            0
        }
    }
}

/// Claim the next chunk: (first iteration index, iteration count).
fn next_chunk(sh: &LoopShared, p: &DispPriv) -> Option<(i64, i64)> {
    match sh.kind {
        Kind::Static => {
            // one contiguous greedy block per thread
            let k = p.static_next.get();
            if k >= sh.nproc || k < 0 {
                return None;
            }
            p.static_next.set(-1);
            let big = sh.tc / sh.nproc + i64::from(sh.tc % sh.nproc != 0);
            let first = k * big;
            if first >= sh.tc {
                return None;
            }
            Some((first, big.min(sh.tc - first)))
        }
        Kind::StaticChunked => {
            let k = p.static_next.get();
            let first = k.checked_mul(sh.chunk)?;
            if first >= sh.tc {
                return None;
            }
            p.static_next.set(k + sh.nproc);
            Some((first, sh.chunk.min(sh.tc - first)))
        }
        Kind::Dynamic => {
            // cheap pre-check avoids unbounded counter growth after exhaustion
            if sh.next.load(Ordering::Relaxed) >= sh.tc {
                return None;
            }
            let first = sh.next.fetch_add(sh.chunk, Ordering::Relaxed);
            if first >= sh.tc {
                return None;
            }
            Some((first, sh.chunk.min(sh.tc - first)))
        }
        Kind::Guided => {
            let mut cur = sh.next.load(Ordering::Relaxed);
            loop {
                let remaining = sh.tc - cur;
                if remaining <= 0 {
                    return None;
                }
                let size = (remaining / (2 * sh.nproc)).max(sh.chunk).min(remaining);
                match sh.next.compare_exchange_weak(cur, cur + size, Ordering::Relaxed, Ordering::Relaxed) {
                    Ok(_) => return Some((cur, size)),
                    Err(v) => cur = v,
                }
            }
        }
    }
}

fn wait_ordered(sh: &LoopShared, iter: i64) {
    let mut spin = Spinner::new();
    while sh.ordered_iter.load(Ordering::Acquire) < iter {
        if !spin.spin() {
            std::thread::yield_now();
        }
    }
}

/// End of one iteration of an `ordered` loop: if the iteration did not execute an
/// `ordered` region, still pass the ordered ticket on (libomp `__kmp_dispatch_finish`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_dispatch_fini_8(_loc: *mut Ident, _gtid: i32) {
    let p = &current().disp;
    let Some(sh) = p.cur.borrow().clone() else {
        return;
    };
    if !p.ordered.get() {
        return;
    }
    let it = p.ord_cur.get();
    if p.ord_bumped.get() {
        p.ord_bumped.set(false);
    } else {
        wait_ordered(&sh, it);
        sh.ordered_iter.store(it + 1, Ordering::Release);
    }
    if it + 1 < p.ord_end.get() {
        p.ord_cur.set(it + 1);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_ordered(_loc: *mut Ident, _gtid: i32) {
    let p = &current().disp;
    let sh = p.cur.borrow().clone();
    if let Some(sh) = sh.filter(|_| p.ordered.get()) {
        wait_ordered(&sh, p.ord_cur.get());
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_end_ordered(_loc: *mut Ident, _gtid: i32) {
    let p = &current().disp;
    let sh = p.cur.borrow().clone();
    if let Some(sh) = sh.filter(|_| p.ordered.get()) {
        sh.ordered_iter.store(p.ord_cur.get() + 1, Ordering::Release);
        p.ord_bumped.set(true);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cover(sched: i32, nth: i64, lb: i64, ub: i64, st: i64, chunk: i64) -> Vec<i64> {
        let mut seen = Vec::new();
        let mut lasts = 0;
        for tid in 0..nth {
            let (last, mut lo, mut hi, stride) = static_bounds(sched, tid, nth, lb, ub, st, chunk);
            lasts += i32::from(last);
            // chunked loops step by stride (stdlib _static_chunked)
            loop {
                let mut i = lo;
                while (st > 0 && i <= hi.min(ub)) || (st < 0 && i >= hi.max(ub)) {
                    seen.push(i);
                    i += st;
                }
                if sched != SCH_STATIC_CHUNKED {
                    break;
                }
                lo += stride * st;
                hi += stride * st;
                if (st > 0 && lo > ub) || (st < 0 && lo < ub) {
                    break;
                }
            }
        }
        if trip_count(lb, ub, st) > 0 {
            assert_eq!(lasts, 1, "exactly one thread is last");
        }
        seen.sort();
        seen
    }

    #[test]
    fn static_unchunked_covers() {
        for nth in 1..9 {
            for n in 0..40 {
                let want: Vec<i64> = (0..n).collect();
                assert_eq!(cover(SCH_STATIC, nth, 0, n - 1, 1, 0), want, "nth={nth} n={n}");
            }
        }
    }

    #[test]
    fn static_chunked_covers() {
        for nth in 1..6 {
            for chunk in 1..7 {
                for n in 0..30 {
                    let want: Vec<i64> = (0..n).collect();
                    assert_eq!(cover(SCH_STATIC_CHUNKED, nth, 0, n - 1, 1, chunk), want);
                }
            }
        }
    }

    #[test]
    fn decode_schedules() {
        assert_eq!(decode((1 << 30) | 35), (35, false));
        assert_eq!(decode((1 << 29) | 67), (35, true));
        assert_eq!(decode(65), (33, true));
    }
}
