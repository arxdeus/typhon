//! Integration tests: drive the C ABI exactly like compiled Typhon code does
//! (outlined microtasks as `extern "C" fn`, args passed as one pointer).

use std::collections::HashSet;
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

/// Run `f(arg)` as a parallel region with `n` threads.
fn par<T>(n: i32, f: Micro, arg: &T) {
    unsafe {
        __kmpc_push_num_threads(loc(), __kmpc_global_thread_num(loc()), n);
        __kmpc_fork_call(loc(), 1, f as *mut c_void, arg as *const T as *mut c_void);
    }
}

fn arg<'a, T>(p: *mut c_void) -> &'a T {
    unsafe { &*(p as *const T) }
}

/* ------------------------------------------------------------------ basic -- */

struct Ids {
    seen: Mutex<Vec<(i32, i32, i32, i32)>>,
}

unsafe extern "C" fn ids_body(gtid: *mut i32, btid: *mut i32, a: *mut c_void) {
    let a: &Ids = arg(a);
    let (g, b) = unsafe { (*gtid, *btid) };
    assert_eq!(g, unsafe { __kmpc_global_thread_num(loc()) });
    assert_eq!(b, omp_get_thread_num());
    // an inactive (1-thread) region is not "in parallel" (libomp: r_active)
    assert_eq!(omp_in_parallel(), i32::from(omp_get_num_threads() > 1));
    assert_eq!(omp_get_level(), 1);
    a.seen.lock().unwrap().push((g, b, omp_get_num_threads(), omp_get_team_size(1)));
}

#[test]
fn parallel_region_ids() {
    for n in [1, 2, 3, 4, 8] {
        let a = Ids { seen: Mutex::new(Vec::new()) };
        par(n, ids_body, &a);
        let v = a.seen.into_inner().unwrap();
        assert_eq!(v.len(), n as usize);
        let tids: HashSet<i32> = v.iter().map(|x| x.1).collect();
        assert_eq!(tids, (0..n).collect());
        let gtids: HashSet<i32> = v.iter().map(|x| x.0).collect();
        assert_eq!(gtids.len(), n as usize, "distinct gtids");
        assert!(v.iter().all(|x| x.2 == n && x.3 == n));
    }
    assert_eq!(omp_in_parallel(), 0);
    assert_eq!(omp_get_num_threads(), 1);
    assert_eq!(omp_get_thread_num(), 0);
}

/* -------------------------------------------------------------- schedules -- */

struct LoopArg {
    sched: i32,
    chunk: i64,
    n: i64,
    hits: Vec<AtomicUsize>,
    lasts: AtomicUsize,
}

impl LoopArg {
    fn new(sched: i32, chunk: i64, n: i64) -> Self {
        LoopArg { sched, chunk, n, hits: (0..n).map(|_| AtomicUsize::new(0)).collect(), lasts: AtomicUsize::new(0) }
    }
    fn check(&self) {
        for (i, h) in self.hits.iter().enumerate() {
            assert_eq!(h.load(Ordering::Relaxed), 1, "iteration {i} (sched {} chunk {})", self.sched, self.chunk);
        }
        if self.n > 0 {
            assert_eq!(self.lasts.load(Ordering::Relaxed), 1, "one last chunk");
        }
    }
}

/// Mirrors `_static_loop_outline_template` / `_static_chunked_loop_outline_template`.
unsafe extern "C" fn static_body(gtid: *mut i32, _b: *mut i32, a: *mut c_void) {
    let a: &LoopArg = arg(a);
    let g = unsafe { *gtid };
    let (mut last, mut lo, mut hi, mut stride) = (0i32, 0i64, a.n - 1, 1i64);
    unsafe {
        __kmpc_for_static_init_8(loc(), g, a.sched, &mut last, &mut lo, &mut hi, &mut stride, 1, a.chunk);
    }
    let ub = a.n - 1;
    if a.chunk > 0 {
        while lo <= ub {
            for i in lo..=hi.min(ub) {
                a.hits[i as usize].fetch_add(1, Ordering::Relaxed);
            }
            lo += stride;
            hi += stride;
        }
    } else {
        for i in lo..=hi.min(ub) {
            a.hits[i as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
    unsafe { __kmpc_for_static_fini(loc(), g) };
    if last != 0 {
        a.lasts.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn static_schedules_cover() {
    for n in [0i64, 1, 3, 7, 100, 1001] {
        for threads in [1, 3, 4] {
            let a = LoopArg::new(34, 0, n);
            par(threads, static_body, &a);
            a.check();
            for chunk in [1, 2, 5, 64] {
                let a = LoopArg::new(33, chunk, n);
                par(threads, static_body, &a);
                a.check();
            }
        }
    }
}

/// Mirrors `_dynamic_loop_outline_template`.
unsafe extern "C" fn dynamic_body(gtid: *mut i32, _b: *mut i32, a: *mut c_void) {
    let a: &LoopArg = arg(a);
    let g = unsafe { *gtid };
    unsafe { __kmpc_dispatch_init_8(loc(), g, a.sched, 0, a.n - 1, 1, a.chunk) };
    loop {
        let (mut last, mut lo, mut hi, mut st) = (0, 0, 0, 0);
        if unsafe { __kmpc_dispatch_next_8(loc(), g, &mut last, &mut lo, &mut hi, &mut st) } == 0 {
            break;
        }
        for i in lo..=hi {
            a.hits[i as usize].fetch_add(1, Ordering::Relaxed);
        }
        if last != 0 {
            a.lasts.fetch_add(1, Ordering::Relaxed);
        }
    }
}

#[test]
fn dynamic_guided_runtime_cover() {
    let nm = 1 << 30;
    for sched in [35 | nm, 36 | nm, 37 | nm, 38 | nm, 33, 34] {
        for n in [0i64, 1, 5, 333, 10_000] {
            for chunk in [1, 3, 17] {
                for threads in [1, 2, 4, 7] {
                    let a = LoopArg::new(sched, chunk, n);
                    par(threads, dynamic_body, &a);
                    a.check();
                }
            }
        }
    }
}

/// Several back-to-back nowait dynamic loops in one region (dispatch buffer ring).
unsafe extern "C" fn many_loops_body(g: *mut i32, b: *mut i32, a: *mut c_void) {
    let v: &Vec<LoopArg> = arg(a);
    for l in v {
        unsafe { dynamic_body(g, b, l as *const LoopArg as *mut c_void) };
    }
}

#[test]
fn many_nowait_dynamic_loops() {
    let v: Vec<LoopArg> = (0..50).map(|k| LoopArg::new(35, 1 + k % 4, 200 + k)).collect();
    par(4, many_loops_body, &v);
    v.iter().for_each(LoopArg::check);
}

/* ---------------------------------------------------------------- ordered -- */

struct Ord {
    n: i64,
    out: Mutex<Vec<i64>>,
    sched: i32,
    chunk: i64,
}

unsafe extern "C" fn ordered_body(gtid: *mut i32, _b: *mut i32, a: *mut c_void) {
    let a: &Ord = arg(a);
    let g = unsafe { *gtid };
    unsafe { __kmpc_dispatch_init_8(loc(), g, a.sched, 0, a.n - 1, 1, a.chunk) };
    loop {
        let (mut last, mut lo, mut hi, mut st) = (0, 0, 0, 0);
        if unsafe { __kmpc_dispatch_next_8(loc(), g, &mut last, &mut lo, &mut hi, &mut st) } == 0 {
            break;
        }
        for i in lo..=hi {
            // odd iterations skip the ordered region (ticket must still advance)
            if i % 3 != 1 {
                unsafe { __kmpc_ordered(loc(), g) };
                a.out.lock().unwrap().push(i);
                unsafe { __kmpc_end_ordered(loc(), g) };
            }
            unsafe { __kmpc_dispatch_fini_8(loc(), g) };
        }
    }
}

#[test]
fn ordered_sequencing() {
    let nm = 1 << 30;
    for (sched, chunk) in [(67 | nm, 1), (67 | nm, 4), (68 | nm, 2), (65, 3), (66, 0)] {
        let a = Ord { n: 500, out: Mutex::new(Vec::new()), sched, chunk };
        par(4, ordered_body, &a);
        let want: Vec<i64> = (0..500).filter(|i| i % 3 != 1).collect();
        assert_eq!(*a.out.lock().unwrap(), want, "sched {sched}");
    }
}

/* ------------------------------------------------------- barrier/critical -- */

struct Phases {
    counter: AtomicI64,
    errors: AtomicI64,
}

unsafe extern "C" fn barrier_body(gtid: *mut i32, _b: *mut i32, a: *mut c_void) {
    let a: &Phases = arg(a);
    let n = omp_get_num_threads() as i64;
    for phase in 1..=50 {
        a.counter.fetch_add(1, Ordering::SeqCst);
        unsafe { __kmpc_barrier(loc(), *gtid) };
        if a.counter.load(Ordering::SeqCst) != phase * n {
            a.errors.fetch_add(1, Ordering::SeqCst);
        }
        unsafe { __kmpc_barrier(loc(), *gtid) };
    }
}

#[test]
fn barrier_correctness() {
    let a = Phases { counter: AtomicI64::new(0), errors: AtomicI64::new(0) };
    par(6, barrier_body, &a);
    assert_eq!(a.errors.load(Ordering::SeqCst), 0);
    assert_eq!(a.counter.load(Ordering::SeqCst), 300);
}

struct Crit {
    lock: [i32; 8],
    value: std::cell::UnsafeCell<i64>,
    inside: AtomicI64,
}
unsafe impl Sync for Crit {}

unsafe extern "C" fn critical_body(gtid: *mut i32, _b: *mut i32, a: *mut c_void) {
    let a: &Crit = arg(a);
    let lck = &a.lock as *const [i32; 8] as *mut [i32; 8];
    for _ in 0..2000 {
        unsafe {
            __kmpc_critical(loc(), *gtid, lck);
            assert_eq!(a.inside.fetch_add(1, Ordering::SeqCst), 0);
            *a.value.get() += 1;
            a.inside.fetch_sub(1, Ordering::SeqCst);
            __kmpc_end_critical(loc(), *gtid, lck);
        }
    }
}

#[test]
fn critical_mutual_exclusion() {
    let a = Crit { lock: [0; 8], value: 0.into(), inside: AtomicI64::new(0) };
    par(8, critical_body, &a);
    assert_eq!(unsafe { *a.value.get() }, 16_000);
}

struct SM {
    singles: Vec<AtomicI64>,
    masters: AtomicI64,
    master_tid: AtomicI64,
}

unsafe extern "C" fn single_master_body(gtid: *mut i32, _b: *mut i32, a: *mut c_void) {
    let a: &SM = arg(a);
    let g = unsafe { *gtid };
    for s in &a.singles {
        if unsafe { __kmpc_single(loc(), g) } != 0 {
            s.fetch_add(1, Ordering::SeqCst);
            unsafe { __kmpc_end_single(loc(), g) };
        }
        // nowait on even singles
    }
    if unsafe { __kmpc_master(loc(), g) } != 0 {
        a.masters.fetch_add(1, Ordering::SeqCst);
        a.master_tid.store(omp_get_thread_num() as i64, Ordering::SeqCst);
        unsafe { __kmpc_end_master(loc(), g) };
    }
    unsafe { __kmpc_flush(loc()) };
}

#[test]
fn single_and_master() {
    for n in [1, 2, 5] {
        let a = SM { singles: (0..100).map(|_| AtomicI64::new(0)).collect(), masters: 0.into(), master_tid: (-1).into() };
        par(n, single_master_body, &a);
        assert!(a.singles.iter().all(|s| s.load(Ordering::SeqCst) == 1));
        assert_eq!(a.masters.load(Ordering::SeqCst), 1);
        assert_eq!(a.master_tid.load(Ordering::SeqCst), 0);
    }
}

/* ------------------------------------------------------------- reductions -- */

struct Red {
    lock: [i32; 8],
    sum: std::cell::UnsafeCell<i64>,
    prod: std::cell::UnsafeCell<f64>,
    min: std::cell::UnsafeCell<f64>,
    max: std::cell::UnsafeCell<f64>,
    atomic_sum: std::cell::UnsafeCell<f64>,
    atomic_prod: std::cell::UnsafeCell<i64>,
    amin: std::cell::UnsafeCell<f32>,
    amax: std::cell::UnsafeCell<f32>,
    aprod: std::cell::UnsafeCell<f32>,
    asum4: std::cell::UnsafeCell<f32>,
    amin8: std::cell::UnsafeCell<f64>,
    amax8: std::cell::UnsafeCell<f64>,
    amul8: std::cell::UnsafeCell<f64>,
    nowait: bool,
}
unsafe impl Sync for Red {}

unsafe extern "C" fn reduce_body(gtid: *mut i32, _b: *mut i32, a: *mut c_void) {
    let a: &Red = arg(a);
    let g = unsafe { *gtid };
    // static loop over 1..=1000 with private partials
    let (mut last, mut lo, mut hi, mut st) = (0, 0i64, 999i64, 1i64);
    unsafe { __kmpc_for_static_init_8(loc(), g, 34, &mut last, &mut lo, &mut hi, &mut st, 1, 1) };
    let (mut s, mut p, mut mn, mut mx) = (0i64, 1f64, f64::MAX, f64::MIN);
    for i in lo..=hi.min(999) {
        let x = (i + 1) as f64;
        s += i + 1;
        p *= 1.001;
        mn = mn.min(x);
        mx = mx.max(x);
        unsafe {
            __kmpc_atomic_float8_add(loc(), g, a.atomic_sum.get(), 0.5);
            __kmpc_atomic_float4_add(loc(), g, a.asum4.get(), 0.25);
            __kmpc_atomic_float4_min(loc(), g, a.amin.get(), x as f32);
            __kmpc_atomic_float4_max(loc(), g, a.amax.get(), x as f32);
            __kmpc_atomic_float8_min(loc(), g, a.amin8.get(), x);
            __kmpc_atomic_float8_max(loc(), g, a.amax8.get(), x);
        }
        if i < 20 {
            unsafe {
                __kmpc_atomic_fixed8_mul(loc(), g, a.atomic_prod.get(), 2);
                __kmpc_atomic_float4_mul(loc(), g, a.aprod.get(), 2.0);
                __kmpc_atomic_float8_mul(loc(), g, a.amul8.get(), 2.0);
            }
        }
    }
    unsafe { __kmpc_for_static_fini(loc(), g) };
    let lck = &a.lock as *const [i32; 8] as *mut [i32; 8];
    let code = unsafe {
        if a.nowait {
            __kmpc_reduce_nowait(loc(), g, 4, 32, null_mut(), null_mut(), lck)
        } else {
            __kmpc_reduce(loc(), g, 4, 32, null_mut(), null_mut(), lck)
        }
    };
    assert!(code == 1 || code == 2);
    unsafe {
        *a.sum.get() += s;
        *a.prod.get() *= p;
        *a.min.get() = (*a.min.get()).min(mn);
        *a.max.get() = (*a.max.get()).max(mx);
        if a.nowait {
            __kmpc_end_reduce_nowait(loc(), g, lck);
        } else {
            __kmpc_end_reduce(loc(), g, lck);
        }
    }
}

#[test]
fn reductions_and_atomics() {
    for nowait in [true, false] {
        for n in [1, 3, 8] {
            let a = Red {
                lock: [0; 8],
                sum: 0.into(),
                prod: 1.0.into(),
                min: f64::MAX.into(),
                max: f64::MIN.into(),
                atomic_sum: 0.0.into(),
                atomic_prod: 1.into(),
                amin: f32::MAX.into(),
                amax: f32::MIN.into(),
                aprod: 1.0.into(),
                asum4: 0.0.into(),
                amin8: f64::MAX.into(),
                amax8: f64::MIN.into(),
                amul8: 1.0.into(),
                nowait,
            };
            par(n, reduce_body, &a);
            unsafe {
                assert_eq!(*a.sum.get(), 500_500);
                assert!((*a.prod.get() - 1.001f64.powi(1000)).abs() < 1e-9);
                assert_eq!(*a.min.get(), 1.0);
                assert_eq!(*a.max.get(), 1000.0);
                assert_eq!(*a.atomic_sum.get(), 500.0);
                assert_eq!(*a.asum4.get(), 250.0);
                assert_eq!(*a.atomic_prod.get(), 1 << 20);
                assert_eq!(*a.aprod.get(), (1 << 20) as f32);
                assert_eq!(*a.amul8.get(), (1 << 20) as f64);
                assert_eq!(*a.amin.get(), 1.0);
                assert_eq!(*a.amax.get(), 1000.0);
                assert_eq!(*a.amin8.get(), 1.0);
                assert_eq!(*a.amax8.get(), 1000.0);
            }
        }
    }
}

/* ------------------------------------------------------------------ tasks -- */

/// The stdlib `TaskWithPrivates[(i64, *const TaskArg)]` with no shareds.
#[repr(C)]
struct TaskThunk {
    shareds: *mut c_void,
    routine: *mut c_void,
    flags: i32,
    x: *mut c_void,
    y: *mut c_void,
    i: i64,
    targ: *const TaskArg,
}

struct TaskArg {
    hits: Vec<AtomicUsize>,
    done_before_wait: AtomicUsize,
    tg: std::cell::UnsafeCell<*mut c_void>,
    red_sum: std::cell::UnsafeCell<i64>,
    shar: std::cell::UnsafeCell<*mut i64>,
    use_red: bool,
    threads_seen: Mutex<HashSet<i32>>,
}
unsafe impl Sync for TaskArg {}

unsafe extern "C" fn task_entry(gtid: i32, t: *mut c_void) -> i32 {
    let t = unsafe { &*(t as *const TaskThunk) };
    let a = unsafe { &*t.targ };
    a.hits[t.i as usize].fetch_add(1, Ordering::SeqCst);
    a.threads_seen.lock().unwrap().insert(omp_get_thread_num());
    if a.use_red {
        let p = unsafe { __kmpc_task_reduction_get_th_data(gtid, *a.tg.get(), (*a.shar.get()).cast()) };
        unsafe { *(p as *mut i64) += t.i };
    }
    std::thread::sleep(std::time::Duration::from_micros(50));
    0
}

fn spawn_task(g: i32, i: i64, a: &TaskArg) {
    unsafe {
        let size = __kmpc_omp_task_alloc_size(size_of::<TaskThunk>() as i64, 0);
        let mem = libc_calloc(size as usize);
        let t = __kmpc_omp_task_alloc(loc(), g, 1, size_of::<TaskThunk>() as i64, 0, Some(task_entry), mem)
            as *mut TaskThunk;
        (*t).i = i;
        (*t).targ = a;
        __kmpc_omp_task(loc(), g, t.cast());
    }
}

fn libc_calloc(n: usize) -> *mut c_void {
    // leaked on purpose (GC-allocated in Typhon)
    Box::leak(vec![0u64; n.div_ceil(8)].into_boxed_slice()).as_mut_ptr().cast()
}

unsafe extern "C" fn red_init(lhs: *mut c_void, _orig: *mut c_void) {
    unsafe { *(lhs as *mut i64) = 0 };
}
unsafe extern "C" fn red_comb(lhs: *mut c_void, rhs: *mut c_void) {
    unsafe { *(lhs as *mut i64) += *(rhs as *mut i64) };
}

/// Mirrors `_task_loop_outline_template`.
unsafe extern "C" fn tasks_body(gtid: *mut i32, _b: *mut i32, a: *mut c_void) {
    let a: &TaskArg = arg(a);
    let g = unsafe { *gtid };
    let n = a.hits.len() as i64;
    let mut local = 0i64; // per-thread shared reduction var (stdlib: info.local)
    if a.use_red {
        let input = TaskRedInput {
            reduce_shar: (&mut local as *mut i64).cast(),
            reduce_orig: (&mut local as *mut i64).cast(),
            reduce_size: 8,
            reduce_init: red_init as *mut c_void,
            reduce_fini: null_mut(),
            reduce_comb: red_comb as *mut c_void,
            flags: 0,
        };
        let tg = unsafe { __kmpc_taskred_modifier_init(loc(), g, 0, 1, (&input as *const TaskRedInput).cast_mut().cast()) };
        if unsafe { __kmpc_single(loc(), g) } != 0 {
            // Typhon passes &local of the spawning (single) thread as the shared item
            unsafe { *a.tg.get() = tg };
            unsafe { *a.shar.get() = &mut local as *mut i64 };
        }
    }
    if unsafe { __kmpc_single(loc(), g) } != 0 {
        unsafe { __kmpc_taskgroup(loc(), g) };
        for i in 0..n / 2 {
            spawn_task(g, i, a);
        }
        unsafe { __kmpc_omp_taskwait(loc(), g) };
        let done = a.hits.iter().take((n / 2) as usize).filter(|h| h.load(Ordering::SeqCst) == 1).count();
        a.done_before_wait.store(done, Ordering::SeqCst);
        for i in n / 2..n {
            spawn_task(g, i, a);
        }
        unsafe { __kmpc_end_taskgroup(loc(), g) };
        assert!(a.hits.iter().all(|h| h.load(Ordering::SeqCst) == 1), "taskgroup waits for all");
        unsafe { __kmpc_end_single(loc(), g) };
    }
    if a.use_red {
        unsafe { __kmpc_task_reduction_modifier_fini(loc(), g, 0) };
        // combine per-thread `local` into the result (stdlib _loop_reductions)
        static LCK: [i32; 8] = [0; 8];
        unsafe {
            __kmpc_critical(loc(), g, &LCK as *const _ as *mut _);
            *a.red_sum.get() += local;
            __kmpc_end_critical(loc(), g, &LCK as *const _ as *mut _);
        }
    }
    unsafe { __kmpc_barrier(loc(), g) };
}

#[test]
fn tasks_taskwait_taskgroup() {
    for threads in [1, 4] {
        let a = TaskArg {
            hits: (0..200).map(|_| AtomicUsize::new(0)).collect(),
            done_before_wait: 0.into(),
            tg: null_mut::<c_void>().into(),
            red_sum: 0.into(),
            shar: null_mut::<i64>().into(),
            use_red: false,
            threads_seen: Mutex::new(HashSet::new()),
        };
        par(threads, tasks_body, &a);
        assert!(a.hits.iter().all(|h| h.load(Ordering::SeqCst) == 1));
        assert_eq!(a.done_before_wait.load(Ordering::SeqCst), 100, "taskwait waits for children");
        if threads > 1 {
            assert!(a.threads_seen.lock().unwrap().len() > 1, "tasks are stolen by other threads");
        }
    }
}

#[test]
fn task_reductions() {
    for threads in [1, 2, 4] {
        let a = TaskArg {
            hits: (0..300).map(|_| AtomicUsize::new(0)).collect(),
            done_before_wait: 0.into(),
            tg: null_mut::<c_void>().into(),
            red_sum: 0.into(),
            shar: null_mut::<i64>().into(),
            use_red: true,
            threads_seen: Mutex::new(HashSet::new()),
        };
        par(threads, tasks_body, &a);
        assert_eq!(unsafe { *a.red_sum.get() }, (0..300).sum::<i64>(), "threads {threads}");
    }
}

/* ----------------------------------------------------------------- nested -- */

struct Nest {
    inner_threads: i32,
    records: Mutex<Vec<(i32, i32, i32, i32, i32, i32)>>,
}

unsafe extern "C" fn inner_body(_g: *mut i32, _b: *mut i32, a: *mut c_void) {
    let a: &Nest = arg(a);
    a.records.lock().unwrap().push((
        omp_get_level(),
        omp_get_active_level(),
        omp_get_ancestor_thread_num(1),
        omp_get_thread_num(),
        omp_get_num_threads(),
        unsafe { __kmpc_global_thread_num(loc()) },
    ));
}

unsafe extern "C" fn outer_body(_g: *mut i32, _b: *mut i32, a: *mut c_void) {
    let a: &Nest = arg(a);
    par(a.inner_threads, inner_body, a);
}

#[test]
fn nested_parallelism() {
    let saved = omp_get_max_active_levels();
    // default max-active-levels = 1: inner regions serialized
    omp_set_max_active_levels(1);
    let a = Nest { inner_threads: 3, records: Mutex::new(Vec::new()) };
    par(2, outer_body, &a);
    let r = a.records.into_inner().unwrap();
    assert_eq!(r.len(), 2);
    assert!(r.iter().all(|x| x.0 == 2 && x.1 == 1 && x.3 == 0 && x.4 == 1));
    // active nesting
    omp_set_max_active_levels(2);
    let a = Nest { inner_threads: 3, records: Mutex::new(Vec::new()) };
    par(2, outer_body, &a);
    omp_set_max_active_levels(saved);
    let r = a.records.into_inner().unwrap();
    assert_eq!(r.len(), 6);
    assert!(r.iter().all(|x| x.0 == 2 && x.1 == 2 && x.4 == 3));
    let anc: HashSet<i32> = r.iter().map(|x| x.2).collect();
    assert_eq!(anc, HashSet::from([0, 1]));
    let gt: HashSet<i32> = r.iter().map(|x| x.5).collect();
    assert_eq!(gt.len(), 6, "distinct gtids across nested teams");
}

/* -------------------------------------------------------------------- api -- */

#[test]
fn omp_api_getters() {
    assert!(omp_get_num_procs() >= 1);
    assert!(omp_get_max_threads() >= 1);
    assert_eq!(omp_get_level(), 0);
    assert_eq!(omp_get_active_level(), 0);
    assert_eq!(omp_get_ancestor_thread_num(0), 0);
    assert_eq!(omp_get_team_size(0), 1);
    assert_eq!(omp_get_ancestor_thread_num(5), -1);
    assert_eq!(omp_get_team_size(-1), -1);
    assert_eq!(omp_in_final(), 0);
    assert_eq!(omp_get_proc_bind(), 0);
    assert_eq!(omp_get_num_devices(), 0);
    assert_eq!(omp_get_num_teams(), 1);
    assert_eq!(omp_get_team_num(), 0);
    assert_eq!(omp_is_initial_device(), 1);
    assert_eq!(omp_get_cancellation(), 0);
    assert!(omp_get_thread_limit() >= omp_get_max_threads());
    omp_set_default_device(0);
    assert_eq!(omp_get_default_device(), 0);
    let t0 = omp_get_wtime();
    std::thread::sleep(std::time::Duration::from_millis(5));
    assert!(omp_get_wtime() - t0 >= 0.004);
    assert!(omp_get_wtick() > 0.0);
    let d = omp_get_dynamic();
    omp_set_dynamic(1);
    assert_eq!(omp_get_dynamic(), 1);
    omp_set_dynamic(d);
    let (mut k, mut c) = (0, 0);
    omp_set_schedule(2, 7);
    unsafe { omp_get_schedule(&mut k, &mut c) };
    assert_eq!((k, c), (2, 7));
    omp_set_schedule(1, 0);
    unsafe { omp_get_schedule(&mut k, &mut c) };
    assert_eq!((k, c), (1, 0));
}

/* ----------------------------------------------------------------- stress -- */

unsafe extern "C" fn tiny_body(_g: *mut i32, _b: *mut i32, a: *mut c_void) {
    let a: &AtomicI64 = arg(a);
    a.fetch_add(1, Ordering::Relaxed);
}

#[test]
fn stress_many_regions() {
    let c = AtomicI64::new(0);
    let mut expect = 0;
    for k in 0..3000 {
        let n = 1 + (k % 6);
        par(n, tiny_body, &c);
        expect += n as i64;
    }
    assert_eq!(c.load(Ordering::Relaxed), expect);
}

/// Several application threads forking concurrently (each has its own hot team).
#[test]
fn concurrent_roots() {
    let hs: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(|| {
                let c = AtomicI64::new(0);
                for _ in 0..200 {
                    par(3, tiny_body, &c);
                }
                c.load(Ordering::Relaxed)
            })
        })
        .collect();
    for h in hs {
        assert_eq!(h.join().unwrap(), 600);
    }
}

#[test]
fn omp_simple_lock() {
    use typhon_omp::sync::*;
    let mut l: usize = 0xdead;
    let p = &mut l as *mut usize as usize;
    unsafe {
        omp_init_lock(p as *mut usize);
        assert_eq!(omp_test_lock(p as *mut usize), 1);
        assert_eq!(omp_test_lock(p as *mut usize), 0);
        omp_unset_lock(p as *mut usize);
    }
    let n = std::sync::Arc::new(AtomicUsize::new(0));
    let hs: Vec<_> = (0..4)
        .map(|_| {
            let n = n.clone();
            std::thread::spawn(move || {
                for _ in 0..1000 {
                    unsafe { omp_set_lock(p as *mut usize) };
                    let v = n.load(Ordering::Relaxed);
                    n.store(v + 1, Ordering::Relaxed);
                    unsafe { omp_unset_lock(p as *mut usize) };
                }
            })
        })
        .collect();
    hs.into_iter().for_each(|h| h.join().unwrap());
    assert_eq!(n.load(Ordering::Relaxed), 4000);
}
