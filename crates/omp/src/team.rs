//! Teams, the persistent worker pool, fork/join and the team barrier.
//!
//! * Every OS thread that touches the runtime gets a global thread id (gtid) from a
//!   process-wide counter (the first thread is normally the main thread, gtid 0).
//! * Each thread has a stack of [`Frame`]s (one per enclosing parallel region it is a
//!   member of); the current frame is in TLS. A thread's outermost frame is an
//!   implicit serial "root" team of size 1.
//! * A thread that forks keeps a *hot team* of persistent workers in its TLS (like
//!   libomp's hot teams); consecutive regions reuse the same OS threads. Workers wait
//!   for jobs by spinning briefly and then sleeping on a condvar.
//! * The barrier is centralized and generation-based (sense reversing). Waiting
//!   threads execute queued explicit tasks and the barrier is only released when
//!   every member has arrived *and* the team has no pending tasks, so the implicit
//!   join barrier also completes all tasks of the region.

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use crossbeam_deque::{Stealer, Worker};
use crossbeam_utils::CachePadded;
use parking_lot::{Condvar, Mutex};

use crate::sched::{DispBuf, DispPriv, DISP_BUFS};
use crate::task::{self, TaskCtx, TaskRec, TeamRed};
use crate::{Ident, Microtask, gc, icv};

unsafe extern "C" {
    /// csrc/fork_call.c: spreads `args` into the variadic microtask call.
    fn typhon_omp_invoke_microtask(
        f: Microtask,
        gtid: *mut i32,
        tid: *mut i32,
        argc: i32,
        args: *mut *mut c_void,
    );
}

/// How long idle workers / barrier waiters spin before sleeping.
const SPIN_TIME: Duration = Duration::from_micros(200);

/// Spin-then-park helper: returns `true` while the caller should keep spinning.
pub(crate) struct Spinner {
    iters: u32,
    start: Option<Instant>,
}

impl Spinner {
    pub(crate) fn new() -> Self {
        Spinner { iters: 0, start: None }
    }

    pub(crate) fn spin(&mut self) -> bool {
        // Busy-wait with `spin_loop` hints only: `sched_yield` and reading the clock
        // on every iteration cost microseconds per fork/join on Android, which made
        // small threaded BLAS/LAPACK calls slower than serial ones.
        std::hint::spin_loop();
        self.iters = self.iters.wrapping_add(1);
        if self.iters % 256 != 0 {
            return true;
        }
        let start = *self.start.get_or_insert_with(Instant::now);
        start.elapsed() < SPIN_TIME
    }

    pub(crate) fn reset(&mut self) {
        self.iters = 0;
        self.start = None;
    }
}

/* ---------------------------------------------------------------- barrier --- */

pub struct Barrier {
    arrived: CachePadded<AtomicUsize>,
    generation: CachePadded<AtomicUsize>,
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    cv: Condvar,
}

impl Barrier {
    fn new() -> Self {
        Barrier {
            arrived: CachePadded::new(AtomicUsize::new(0)),
            generation: CachePadded::new(AtomicUsize::new(0)),
            sleepers: AtomicUsize::new(0),
            lock: Mutex::new(()),
            cv: Condvar::new(),
        }
    }

    /// Wake threads sleeping in the barrier (new tasks, release).
    pub(crate) fn wake(&self) {
        if self.sleepers.load(Ordering::SeqCst) > 0 {
            let _g = self.lock.lock();
            self.cv.notify_all();
        }
    }
}

/* ------------------------------------------------------------------- team --- */

pub struct Team {
    pub nproc: i32,
    microtask: Microtask,
    args: Vec<*mut c_void>,
    pub(crate) barrier: Barrier,
    /// Number of `single` constructs claimed so far (libomp's t_construct).
    pub(crate) single: CachePadded<AtomicU64>,
    pub(crate) disp: Box<[CachePadded<DispBuf>]>,
    /// Explicit tasks created in this team and not yet completed.
    pub(crate) pending_tasks: CachePadded<AtomicI64>,
    pub(crate) stealers: Box<[OnceLock<Stealer<Box<TaskRec>>>]>,
    /// Shared data of a `task_reduction` modifier (libomp's t_tg_reduce_data[0]).
    pub(crate) red: Mutex<Option<Arc<TeamRed>>>,
    pub(crate) red_fini: AtomicI32,
}

unsafe impl Send for Team {}
unsafe impl Sync for Team {}

impl Team {
    fn new(nproc: i32, microtask: Microtask, args: Vec<*mut c_void>) -> Arc<Team> {
        Arc::new(Team {
            nproc,
            microtask,
            args,
            barrier: Barrier::new(),
            single: CachePadded::new(AtomicU64::new(0)),
            disp: (0..DISP_BUFS).map(|i| CachePadded::new(DispBuf::new(i as u64))).collect(),
            pending_tasks: CachePadded::new(AtomicI64::new(0)),
            stealers: (0..nproc).map(|_| OnceLock::new()).collect(),
            red: Mutex::new(None),
            red_fini: AtomicI32::new(0),
        })
    }
}

/* ------------------------------------------------------------------ frame --- */

/// Per-thread state for one team membership (one "implicit task").
pub struct Frame {
    pub team: Arc<Team>,
    pub tid: i32,
    /// Nesting depth of parallel regions (active or not), `omp_get_level`.
    pub level: i32,
    /// Number of enclosing active (nproc > 1) regions, `omp_get_active_level`.
    pub active_level: i32,
    pub parent: *const Frame,
    /// `__kmpc_push_num_threads` value for the next fork from this frame (0 = none).
    pub next_nthreads: Cell<i32>,
    pub(crate) singles_seen: Cell<u64>,
    pub(crate) disp: DispPriv,
    pub(crate) reduce_lock: Cell<*const parking_lot::RawMutex>,
    pub(crate) implicit_task: TaskCtx,
    /// Created on first task push (most regions create no tasks).
    deque: std::cell::OnceCell<Worker<Box<TaskRec>>>,
}

impl Frame {
    fn new(team: Arc<Team>, tid: i32, level: i32, active_level: i32, parent: *const Frame) -> Self {
        Frame {
            team,
            tid,
            level,
            active_level,
            parent,
            next_nthreads: Cell::new(0),
            singles_seen: Cell::new(0),
            disp: DispPriv::default(),
            reduce_lock: Cell::new(std::ptr::null()),
            implicit_task: TaskCtx::new(None),
            deque: std::cell::OnceCell::new(),
        }
    }

    pub(crate) fn deque(&self) -> &Worker<Box<TaskRec>> {
        self.deque.get_or_init(|| {
            let w = Worker::new_lifo();
            let _ = self.team.stealers[self.tid as usize].set(w.stealer());
            w
        })
    }

    pub(crate) fn pop_own(&self) -> Option<Box<TaskRec>> {
        self.deque.get().and_then(|d| d.pop())
    }

    /// Walk up to the frame at nesting `level` (0 = the root frame).
    pub fn ancestor(&self, level: i32) -> Option<&Frame> {
        let mut f: &Frame = self;
        loop {
            if f.level == level {
                return Some(f);
            }
            if f.parent.is_null() || f.level < level {
                return None;
            }
            f = unsafe { &*f.parent };
        }
    }
}

static NEXT_GTID: AtomicI32 = AtomicI32::new(0);

thread_local! {
    static GTID: Cell<i32> = const { Cell::new(-1) };
    static CUR_FRAME: Cell<*const Frame> = const { Cell::new(std::ptr::null()) };
    static ROOT_FRAME: Frame = Frame::new(Team::new(1, std::ptr::null_mut(), Vec::new()), 0, 0, 0, std::ptr::null());
    /// Hot teams of this thread, one per nesting level it forks from: a worker of the
    /// level-L team is busy for the whole level-L region, so a nested fork by this
    /// thread must use a different set of workers.
    static HOT_TEAMS: RefCell<Vec<HotTeam>> = const { RefCell::new(Vec::new()) };
    static GC_REGISTERED: Cell<bool> = const { Cell::new(false) };
}

/// Global thread id of the calling thread (assigned on first use).
pub fn gtid() -> i32 {
    GTID.with(|g| {
        let v = g.get();
        if v >= 0 {
            v
        } else {
            let v = NEXT_GTID.fetch_add(1, Ordering::Relaxed);
            g.set(v);
            v
        }
    })
}

/// The calling thread's current frame.
///
/// The returned reference is valid for as long as the caller stays inside the
/// construct it is executing (frames outlive every call made from within them).
pub fn current<'a>() -> &'a Frame {
    let p = CUR_FRAME.with(|c| c.get());
    if !p.is_null() {
        return unsafe { &*p };
    }
    ROOT_FRAME.with(|f| unsafe { &*(f as *const Frame) })
}

fn ensure_gc_registered() {
    GC_REGISTERED.with(|r| {
        if !r.get() && gc::callbacks_installed() {
            gc::register_current_thread();
            r.set(true);
        }
    });
}

/* ------------------------------------------------------------ worker pool --- */

struct Job {
    team: Arc<Team>,
    tid: i32,
    level: i32,
    active_level: i32,
    parent: *const Frame,
}

unsafe impl Send for Job {}

enum Mail {
    Run(Job),
    Exit,
}

struct WorkerHandle {
    has_mail: AtomicBool,
    mail: Mutex<Option<Mail>>,
    cv: Condvar,
}

impl WorkerHandle {
    fn post(&self, m: Mail) {
        let mut g = self.mail.lock();
        *g = Some(m);
        self.has_mail.store(true, Ordering::Release);
        self.cv.notify_one();
    }

    fn take(&self) -> Mail {
        let mut spin = Spinner::new();
        while !self.has_mail.load(Ordering::Acquire) && spin.spin() {}
        let mut out = None;
        let mut park = || {
            let mut g = self.mail.lock();
            loop {
                if let Some(m) = g.take() {
                    self.has_mail.store(false, Ordering::Relaxed);
                    out = Some(m);
                    return;
                }
                self.cv.wait(&mut g);
            }
        };
        if self.has_mail.load(Ordering::Acquire) || !GC_REGISTERED.with(|r| r.get()) {
            park();
        } else {
            gc::blocking(park);
        }
        out.unwrap()
    }
}

/// Hot team of the forking thread; exiting threads shut their workers down.
struct HotTeam(Vec<Arc<WorkerHandle>>);

impl Drop for HotTeam {
    fn drop(&mut self) {
        for w in &self.0 {
            w.post(Mail::Exit);
        }
    }
}

fn stack_size() -> usize {
    // PORT-NOTE: honors OMP_STACKSIZE ("<n>[B|K|M|G]", default K); default 8 MiB.
    std::env::var("OMP_STACKSIZE")
        .ok()
        .and_then(|s| {
            let s = s.trim().to_ascii_uppercase();
            let (num, mult) = match s.chars().last()? {
                'B' => (&s[..s.len() - 1], 1),
                'K' => (&s[..s.len() - 1], 1 << 10),
                'M' => (&s[..s.len() - 1], 1 << 20),
                'G' => (&s[..s.len() - 1], 1 << 30),
                _ => (&s[..], 1 << 10),
            };
            num.trim().parse::<usize>().ok().map(|n| n * mult)
        })
        .filter(|&n| n >= 64 << 10)
        .unwrap_or(8 << 20)
}

fn spawn_worker() -> Arc<WorkerHandle> {
    let h = Arc::new(WorkerHandle {
        has_mail: AtomicBool::new(false),
        mail: Mutex::new(None),
        cv: Condvar::new(),
    });
    let h2 = h.clone();
    std::thread::Builder::new()
        .name("typhon-omp-worker".into())
        .stack_size(stack_size())
        .spawn(move || {
            gtid();
            loop {
                match h2.take() {
                    Mail::Exit => break,
                    Mail::Run(job) => run_member(job.team, job.tid, job.level, job.active_level, job.parent),
                }
            }
            if GC_REGISTERED.with(|r| r.get()) {
                gc::unregister_current_thread();
            }
        })
        .expect("typhon-omp: failed to spawn worker thread");
    h
}

/// Execute one team member's implicit task (microtask + join barrier).
fn run_member(team: Arc<Team>, tid: i32, level: i32, active_level: i32, parent: *const Frame) {
    ensure_gc_registered();
    let frame = Frame::new(team, tid, level, active_level, parent);
    let prev = CUR_FRAME.with(|c| c.replace(&frame as *const Frame));
    let prev_task = task::enter_task(&frame.implicit_task);
    let mut g = gtid();
    let mut t = tid;
    let team = &frame.team;
    unsafe {
        typhon_omp_invoke_microtask(
            team.microtask,
            &mut g,
            &mut t,
            team.args.len() as i32,
            team.args.as_ptr() as *mut *mut c_void,
        );
    }
    barrier(&frame);
    task::leave_task(prev_task);
    CUR_FRAME.with(|c| c.set(prev));
}

/* -------------------------------------------------------------- fork/join --- */

/// Rust side of the variadic `__kmpc_fork_call` (csrc/fork_call.c).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn typhon_omp_fork_call(
    _loc: *mut Ident,
    argc: i32,
    microtask: Microtask,
    args: *const *mut c_void,
) {
    let parent = current();
    let args: Vec<*mut c_void> = if argc > 0 && !args.is_null() {
        unsafe { std::slice::from_raw_parts(args, argc as usize) }.to_vec()
    } else {
        Vec::new()
    };
    let mut n = match parent.next_nthreads.replace(0) {
        0 => icv::nthreads(),
        k => k,
    }
    .max(1);
    // PORT-NOTE: OMP_DYNAMIC / omp_set_dynamic only caps the team at the number of
    // processors (libomp's load-balance heuristics are not reproduced).
    if icv::ICV.dynamic.load(Ordering::Relaxed) {
        n = n.min(icv::ICV.num_procs);
    }
    if parent.active_level >= icv::max_active_levels() {
        n = 1; // serialized nested region
    }
    let team = Team::new(n, microtask, args);
    let level = parent.level + 1;
    let active_level = parent.active_level + i32::from(n > 1);
    if n > 1 {
        HOT_TEAMS.with(|teams| {
            let mut teams = teams.borrow_mut();
            let lvl = parent.level as usize;
            while teams.len() <= lvl {
                teams.push(HotTeam(Vec::new()));
            }
            let h = &mut teams[lvl];
            while h.0.len() < (n - 1) as usize {
                h.0.push(spawn_worker());
            }
            for tid in 1..n {
                h.0[(tid - 1) as usize].post(Mail::Run(Job {
                    team: team.clone(),
                    tid,
                    level,
                    active_level,
                    parent: parent as *const Frame,
                }));
            }
        });
    }
    run_member(team, 0, level, active_level, parent as *const Frame);
}

/* --------------------------------------------------------------- barrier --- */

/// Team barrier: also a task scheduling point; completes all team tasks.
pub(crate) fn barrier(frame: &Frame) {
    let team = &*frame.team;
    let b = &team.barrier;
    let n = team.nproc as usize;
    let g = b.generation.load(Ordering::Acquire);
    b.arrived.fetch_add(1, Ordering::AcqRel);
    let mut spin = Spinner::new();
    loop {
        if b.generation.load(Ordering::Acquire) != g {
            return;
        }
        if task::run_one(frame) {
            spin.reset();
            continue;
        }
        if b.arrived.load(Ordering::Acquire) == n
            && team.pending_tasks.load(Ordering::Acquire) == 0
            && b.arrived.compare_exchange(n, 0, Ordering::AcqRel, Ordering::Relaxed).is_ok()
        {
            b.generation.fetch_add(1, Ordering::SeqCst);
            b.wake();
            return;
        }
        if spin.spin() {
            continue;
        }
        let mut guard = b.lock.lock();
        b.sleepers.fetch_add(1, Ordering::SeqCst);
        if b.generation.load(Ordering::SeqCst) == g {
            // Timed wait: task arrival / release both notify; the timeout is a safety net.
            b.cv.wait_for(&mut guard, Duration::from_millis(1));
        }
        b.sleepers.fetch_sub(1, Ordering::SeqCst);
        drop(guard);
        spin.reset();
    }
}

/* --------------------------------------------------------------- C ABI ----- */

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_global_thread_num(_loc: *mut Ident) -> i32 {
    gtid()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_push_num_threads(_loc: *mut Ident, _gtid: i32, num_threads: i32) {
    if num_threads > 0 {
        current().next_nthreads.set(num_threads);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_barrier(_loc: *mut Ident, _gtid: i32) {
    barrier(current());
}
