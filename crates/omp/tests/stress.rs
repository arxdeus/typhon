//! Stress tests for races: ordered + dynamic/guided with nowait, dispatch-buffer
//! ring reuse with lagging threads, barrier generations, hot-team reuse and task
//! stealing during barriers.

use std::ffi::c_void;
use std::ptr::{null, null_mut};
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

use typhon_omp::*;

type Micro = unsafe extern "C" fn(*mut i32, *mut i32, *mut c_void);

fn loc() -> *mut Ident {
    static mut IDENT: Ident = Ident { reserved_1: 0, flags: 2, reserved_2: 0, reserved_3: 0, psource: null() };
    &raw mut IDENT
}

fn par<T>(n: i32, f: Micro, arg: &T) {
    unsafe {
        __kmpc_push_num_threads(loc(), __kmpc_global_thread_num(loc()), n);
        __kmpc_fork_call(loc(), 1, f as *mut c_void, arg as *const T as *mut c_void);
    }
}

fn arg<'a, T>(p: *mut c_void) -> &'a T {
    unsafe { &*(p as *const T) }
}

const NM: i32 = 1 << 30;

/// Run one dispatch loop over `0..n`; `body(i)` per iteration; returns lasts seen.
fn dyn_loop(g: i32, sched: i32, chunk: i64, n: i64, ordered: bool, mut body: impl FnMut(i64)) -> i32 {
    let mut lasts = 0;
    unsafe { __kmpc_dispatch_init_8(loc(), g, sched, 0, n - 1, 1, chunk) };
    loop {
        let (mut last, mut lo, mut hi, mut st) = (0, 0, 0, 0);
        if unsafe { __kmpc_dispatch_next_8(loc(), g, &mut last, &mut lo, &mut hi, &mut st) } == 0 {
            break;
        }
        lasts += last;
        for i in lo..=hi {
            body(i);
            if ordered {
                unsafe { __kmpc_dispatch_fini_8(loc(), g) };
            }
        }
    }
    lasts
}

/* ------------------------------------------------- ordered + nowait mixes -- */

struct OrdMix {
    loops: usize,
    n: i64,
    outs: Vec<Mutex<Vec<i64>>>,
    lasts: Vec<AtomicUsize>,
}

unsafe extern "C" fn ord_mix_body(gtid: *mut i32, btid: *mut i32, a: *mut c_void) {
    let a: &OrdMix = arg(a);
    let (g, tid) = unsafe { (*gtid, *btid) };
    for k in 0..a.loops {
        // alternate dynamic/guided ordered, chunks 1..4, all nowait
        let sched = if k % 2 == 0 { 67 | NM } else { 68 | NM };
        let chunk = 1 + (k as i64 % 4);
        if (k + tid as usize) % 5 == 0 {
            std::thread::yield_now(); // make some threads lag
        }
        let out = &a.outs[k];
        let l = dyn_loop(g, sched, chunk, a.n, true, |i| {
            if i % 2 == 0 {
                unsafe { __kmpc_ordered(loc(), g) };
                out.lock().unwrap().push(i);
                unsafe { __kmpc_end_ordered(loc(), g) };
            }
        });
        a.lasts[k].fetch_add(l as usize, Ordering::Relaxed);
    }
}

#[test]
fn stress_ordered_dynamic_guided_nowait() {
    for nth in [2, 3, 4, 8] {
        for _ in 0..20 {
            let loops = 20; // > DISP_BUFS consecutive nowait loops
            let a = OrdMix {
                loops,
                n: 97,
                outs: (0..loops).map(|_| Mutex::new(Vec::new())).collect(),
                lasts: (0..loops).map(|_| AtomicUsize::new(0)).collect(),
            };
            par(nth, ord_mix_body, &a);
            let want: Vec<i64> = (0..97).filter(|i| i % 2 == 0).collect();
            for k in 0..loops {
                assert_eq!(*a.outs[k].lock().unwrap(), want, "nth {nth} loop {k}");
                assert_eq!(a.lasts[k].load(Ordering::Relaxed), 1, "one last chunk");
            }
        }
    }
}

/* ------------------------------------------- dispatch ring with laggards -- */

struct Ring {
    loops: usize,
    hits: Vec<Vec<AtomicUsize>>,
}

unsafe extern "C" fn ring_body(gtid: *mut i32, btid: *mut i32, a: *mut c_void) {
    let a: &Ring = arg(a);
    let (g, tid) = unsafe { (*gtid, *btid) };
    for k in 0..a.loops {
        if tid == 1 && k % 9 == 0 {
            std::thread::sleep(std::time::Duration::from_micros(200));
        }
        let sched = [35, 36, 34, 33][k % 4] | NM;
        let n = a.hits[k].len() as i64;
        dyn_loop(g, sched, 1 + (k as i64 % 3), n, false, |i| {
            a.hits[k][i as usize].fetch_add(1, Ordering::Relaxed);
        });
    }
}

#[test]
fn stress_dispatch_ring_reuse() {
    for nth in [2, 4, 8] {
        for _ in 0..10 {
            let a = Ring { loops: 64, hits: (0..64).map(|k| (0..(10 + k)).map(|_| AtomicUsize::new(0)).collect()).collect() };
            par(nth, ring_body, &a);
            for (k, h) in a.hits.iter().enumerate() {
                assert!(h.iter().all(|x| x.load(Ordering::Relaxed) == 1), "nth {nth} loop {k}");
            }
        }
    }
}

/* ------------------------------------------------- barrier generations -- */

struct Gens {
    rounds: i64,
    counter: AtomicI64,
    errors: AtomicI64,
}

unsafe extern "C" fn gens_body(gtid: *mut i32, btid: *mut i32, a: *mut c_void) {
    let a: &Gens = arg(a);
    let (g, tid) = unsafe { (*gtid, *btid) };
    let n = omp_get_num_threads() as i64;
    for r in 1..=a.rounds {
        // lag past the spin window sometimes so waiters park
        if tid as i64 == r % n && r % 37 == 0 {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        a.counter.fetch_add(1, Ordering::SeqCst);
        unsafe { __kmpc_barrier(loc(), g) };
        if a.counter.load(Ordering::SeqCst) != r * n {
            a.errors.fetch_add(1, Ordering::SeqCst);
        }
        unsafe { __kmpc_barrier(loc(), g) };
    }
}

#[test]
fn stress_barrier_generations() {
    for nth in [2, 3, 5, 8, 16] {
        let a = Gens { rounds: 400, counter: AtomicI64::new(0), errors: AtomicI64::new(0) };
        par(nth, gens_body, &a);
        assert_eq!(a.errors.load(Ordering::SeqCst), 0, "nth {nth}");
        assert_eq!(a.counter.load(Ordering::SeqCst), 400 * nth as i64);
    }
}

/* ------------------------------------------------------ hot-team reuse -- */

unsafe extern "C" fn hot_body(gtid: *mut i32, _b: *mut i32, a: *mut c_void) {
    let a: &AtomicI64 = arg(a);
    let g = unsafe { *gtid };
    // a single + a short dynamic loop + reduction-ish critical per region
    if unsafe { __kmpc_single(loc(), g) } != 0 {
        a.fetch_add(1000, Ordering::Relaxed);
    }
    unsafe { __kmpc_barrier(loc(), g) };
    dyn_loop(g, 35 | NM, 1, 5, false, |_| {
        a.fetch_add(1, Ordering::Relaxed);
    });
}

#[test]
fn stress_hot_team_reuse() {
    let c = AtomicI64::new(0);
    let mut expect = 0;
    for k in 0..4000 {
        let n = [2, 3, 8, 1, 16, 4][k % 6];
        par(n, hot_body, &c);
        expect += 1005;
        if k % 500 == 0 {
            assert_eq!(c.load(Ordering::Relaxed), expect);
        }
    }
    assert_eq!(c.load(Ordering::Relaxed), expect);
}

/* ------------------------------------------- task stealing in barriers -- */

#[repr(C)]
struct Thunk {
    shareds: *mut c_void,
    routine: *mut c_void,
    flags: i32,
    x: *mut c_void,
    y: *mut c_void,
    i: i64,
    a: *const Steal,
}

struct Steal {
    hits: Vec<AtomicUsize>,
}

unsafe extern "C" fn steal_task(_g: i32, t: *mut c_void) -> i32 {
    let t = unsafe { &*(t as *const Thunk) };
    let a = unsafe { &*t.a };
    a.hits[t.i as usize].fetch_add(1, Ordering::SeqCst);
    if t.i % 7 == 0 {
        std::thread::yield_now();
    }
    0
}

unsafe extern "C" fn steal_body(gtid: *mut i32, btid: *mut i32, a: *mut c_void) {
    let a: &Steal = arg(a);
    let (g, tid) = unsafe { (*gtid, *btid) };
    // only thread 0 spawns; everyone else must steal while waiting in the barrier
    if tid == 0 {
        for i in 0..a.hits.len() as i64 {
            unsafe {
                let size = __kmpc_omp_task_alloc_size(size_of::<Thunk>() as i64, 0);
                let mem = Box::leak(vec![0u64; (size as usize).div_ceil(8)].into_boxed_slice()).as_mut_ptr().cast();
                let t = __kmpc_omp_task_alloc(loc(), g, 1, size_of::<Thunk>() as i64, 0, Some(steal_task), mem)
                    as *mut Thunk;
                (*t).i = i;
                (*t).a = a;
                __kmpc_omp_task(loc(), g, t.cast());
            }
        }
    }
    unsafe { __kmpc_barrier(loc(), g) };
    // after the barrier every task must be complete
    assert!(a.hits.iter().all(|h| h.load(Ordering::SeqCst) == 1));
}

#[test]
fn stress_task_stealing_in_barrier() {
    for nth in [2, 4, 8] {
        for _ in 0..30 {
            let a = Steal { hits: (0..300).map(|_| AtomicUsize::new(0)).collect() };
            par(nth, steal_body, &a);
            assert!(a.hits.iter().all(|h| h.load(Ordering::SeqCst) == 1));
        }
    }
    let _ = null_mut::<c_void>();
}
