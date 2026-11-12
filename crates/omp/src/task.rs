//! Explicit tasks: allocation, deferred execution through per-thread
//! `crossbeam-deque` work-stealing queues, taskwait, taskgroup and task
//! reductions (`__kmpc_taskred_modifier_init` & co).
//!
//! Task memory layout (Typhon allocates it with the GC, sized by the Typhon-specific
//! `__kmpc_omp_task_alloc_size`): `[TaskHdr][kmp_task_t + privates][shareds]`.
//! While a task is queued or running its address is kept in a GC root slot, so the
//! collector never frees a task whose only reference lives in a runtime queue.

use std::cell::{Cell, RefCell};
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};

use crossbeam_deque::Steal;
use parking_lot::Mutex;

use crate::gc::{self, RootedBlock};
use crate::team::{Frame, Spinner, current, gtid};
use crate::{Ident, TaskRoutine};

/// `kmp_task_t` prefix (the stdlib `Task`).
#[repr(C)]
pub struct KmpTask {
    pub shareds: *mut c_void,
    pub routine: Option<TaskRoutine>,
    pub part_id: i32,
    pub data1: *mut c_void,
    pub data2: *mut c_void,
}

/// Runtime header in front of every task.
#[repr(C)]
struct TaskHdr {
    /// 1 if the runtime malloc'ed the block itself (standard 6-arg allocation path).
    malloced: u64,
    _reserved: u64,
}

const HDR: usize = size_of::<TaskHdr>();

fn round8(n: usize) -> usize {
    (n + 7) & !7
}

/* ---------------------------------------------------------- task contexts --- */

pub struct TaskGroup {
    count: AtomicI64,
    parent: Option<Arc<TaskGroup>>,
    red: Mutex<Option<TgRed>>,
}

/// Execution context of a task (implicit or explicit).
pub struct TaskCtx {
    /// Incomplete child tasks (for taskwait).
    children: Arc<AtomicI64>,
    taskgroup: RefCell<Option<Arc<TaskGroup>>>,
}

impl TaskCtx {
    pub(crate) fn new(taskgroup: Option<Arc<TaskGroup>>) -> Self {
        TaskCtx { children: Arc::new(AtomicI64::new(0)), taskgroup: RefCell::new(taskgroup) }
    }
}

thread_local! {
    static CUR_TASK: Cell<*const TaskCtx> = const { Cell::new(std::ptr::null()) };
}

pub(crate) fn enter_task(ctx: &TaskCtx) -> *const TaskCtx {
    CUR_TASK.with(|c| c.replace(ctx as *const TaskCtx))
}

pub(crate) fn leave_task(prev: *const TaskCtx) {
    CUR_TASK.with(|c| c.set(prev));
}

fn cur_task<'a>() -> &'a TaskCtx {
    let p = CUR_TASK.with(|c| c.get());
    if p.is_null() { &current().implicit_task } else { unsafe { &*p } }
}

/// A deferred task in a queue.
pub struct TaskRec {
    task: *mut KmpTask,
    root_slot: usize,
    parent_children: Arc<AtomicI64>,
    taskgroup: Option<Arc<TaskGroup>>,
}

unsafe impl Send for TaskRec {}

/* -------------------------------------------------------------- execution --- */

fn execute(rec: Box<TaskRec>, frame: &Frame) {
    let ctx = TaskCtx::new(rec.taskgroup.clone());
    let prev = enter_task(&ctx);
    unsafe {
        if let Some(routine) = (*rec.task).routine {
            routine(gtid(), rec.task.cast());
        }
    }
    leave_task(prev);
    complete(&rec, frame);
}

fn complete(rec: &TaskRec, frame: &Frame) {
    unsafe {
        let hdr = rec.task.cast::<u8>().sub(HDR).cast::<TaskHdr>();
        if (*hdr).malloced == 1 {
            libc::free(hdr.cast());
        }
    }
    gc::unroot_pointer(rec.root_slot);
    rec.parent_children.fetch_sub(1, Ordering::AcqRel);
    if let Some(tg) = &rec.taskgroup {
        tg.count.fetch_sub(1, Ordering::AcqRel);
    }
    frame.team.pending_tasks.fetch_sub(1, Ordering::AcqRel);
    frame.team.barrier.wake();
}

/// Run one queued task of the current team if any (own deque first, then steal).
pub(crate) fn run_one(frame: &Frame) -> bool {
    if frame.team.pending_tasks.load(Ordering::Acquire) == 0 {
        return false;
    }
    if let Some(t) = frame.pop_own() {
        execute(t, frame);
        return true;
    }
    let n = frame.team.stealers.len();
    let start = frame.tid as usize;
    for k in 1..n {
        let Some(s) = frame.team.stealers[(start + k) % n].get() else {
            continue;
        };
        loop {
            match s.steal() {
                Steal::Success(t) => {
                    execute(t, frame);
                    return true;
                }
                Steal::Retry => continue,
                Steal::Empty => break,
            }
        }
    }
    false
}

fn wait_until(frame: &Frame, done: impl Fn() -> bool) {
    let mut spin = Spinner::new();
    while !done() {
        if run_one(frame) {
            spin.reset();
        } else if !spin.spin() {
            std::thread::yield_now();
        }
    }
}

/* ------------------------------------------------------------------ C ABI --- */

/// Typhon-specific: bytes to allocate for a task with the given sizes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_omp_task_alloc_size(size_of_task: i64, size_of_shareds: i64) -> i64 {
    (HDR + round8(size_of_task.max(0) as usize) + size_of_shareds.max(0) as usize) as i64
}

/// `kmp_task_t *__kmpc_omp_task_alloc(ident_t*, gtid, flags, sizeof_kmp_task_t,
///   sizeof_shareds, task_entry, taskdata)`: Typhon passes a GC block of
/// `__kmpc_omp_task_alloc_size` bytes as the extra `taskdata` argument; when it is
/// NULL (standard libomp ABI) the runtime mallocs and frees the block itself.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_omp_task_alloc(
    _loc: *mut Ident,
    _gtid: i32,
    _flags: i32,
    sizeof_kmp_task_t: i64,
    sizeof_shareds: i64,
    task_entry: Option<TaskRoutine>,
    taskdata: *mut c_void,
) -> *mut c_void {
    let size = unsafe { __kmpc_omp_task_alloc_size(sizeof_kmp_task_t, sizeof_shareds) } as usize;
    let (base, malloced) = if taskdata.is_null() {
        (unsafe { libc::calloc(1, size) }.cast::<u8>(), 1)
    } else {
        (taskdata.cast::<u8>(), 0)
    };
    assert!(!base.is_null(), "typhon-omp: out of memory");
    unsafe {
        let hdr = base.cast::<TaskHdr>();
        (*hdr).malloced = malloced;
        (*hdr)._reserved = 0;
        let task = base.add(HDR).cast::<KmpTask>();
        (*task).shareds = if sizeof_shareds > 0 {
            base.add(HDR + round8(sizeof_kmp_task_t as usize)).cast()
        } else {
            std::ptr::null_mut()
        };
        (*task).routine = task_entry;
        (*task).part_id = 0;
        task.cast()
    }
}

/// Submit a task; it is deferred (queued) when the team has more than one thread.
/// Returns 0 (`TASK_CURRENT_NOT_QUEUED`) like libomp.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_omp_task(_loc: *mut Ident, _gtid: i32, new_task: *mut c_void) -> i32 {
    let frame = current();
    let parent = cur_task();
    let tg = parent.taskgroup.borrow().clone();
    parent.children.fetch_add(1, Ordering::AcqRel);
    if let Some(tg) = &tg {
        tg.count.fetch_add(1, Ordering::AcqRel);
    }
    frame.team.pending_tasks.fetch_add(1, Ordering::AcqRel);
    let rec = Box::new(TaskRec {
        task: new_task.cast(),
        root_slot: gc::root_pointer(new_task),
        parent_children: parent.children.clone(),
        taskgroup: tg,
    });
    if frame.team.nproc == 1 {
        // PORT-NOTE: serialized teams execute tasks immediately (undeferred).
        execute(rec, frame);
    } else {
        frame.deque().push(rec);
        frame.team.barrier.wake();
    }
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_omp_taskwait(_loc: *mut Ident, _gtid: i32) -> i32 {
    let frame = current();
    let children = cur_task().children.clone();
    wait_until(frame, || children.load(Ordering::Acquire) == 0);
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_omp_taskyield(_loc: *mut Ident, _gtid: i32, _end_part: i32) -> i32 {
    run_one(current());
    0
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_taskgroup(_loc: *mut Ident, _gtid: i32) {
    let t = cur_task();
    let parent = t.taskgroup.borrow().clone();
    *t.taskgroup.borrow_mut() =
        Some(Arc::new(TaskGroup { count: AtomicI64::new(0), parent, red: Mutex::new(None) }));
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_end_taskgroup(_loc: *mut Ident, _gtid: i32) {
    let frame = current();
    let t = cur_task();
    let Some(tg) = t.taskgroup.borrow().clone() else {
        return;
    };
    wait_until(frame, || tg.count.load(Ordering::Acquire) == 0);
    if let Some(red) = tg.red.lock().take() {
        finish_reduction(frame, red);
    }
    *t.taskgroup.borrow_mut() = tg.parent.clone();
}

/* -------------------------------------------------------- task reductions --- */

/// `kmp_taskred_input_t` (the stdlib `TaskReductionInput`).
#[repr(C)]
pub struct TaskRedInput {
    pub reduce_shar: *mut c_void,
    pub reduce_orig: *mut c_void,
    pub reduce_size: usize,
    pub reduce_init: *mut c_void,
    pub reduce_fini: *mut c_void,
    pub reduce_comb: *mut c_void,
    pub flags: u32,
}

type InitFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
type CombFn = unsafe extern "C" fn(*mut c_void, *mut c_void);
type FiniFn = unsafe extern "C" fn(*mut c_void);

/// Team-shared reduction item: one private copy per team thread.
struct RedItem {
    /// Per-thread slot size (rounded up to a cache line).
    size: usize,
    comb: Option<CombFn>,
    fini: Option<FiniFn>,
    privs: RootedBlock,
}

pub struct TeamRed {
    items: Vec<RedItem>,
    nth: usize,
}

unsafe impl Send for TeamRed {}
unsafe impl Sync for TeamRed {}

/// A taskgroup's view of reduction data: shared items + this thread's shared vars.
pub(crate) struct TgRed {
    data: Arc<TeamRed>,
    shars: Vec<*mut c_void>,
    /// Created by `__kmpc_taskred_modifier_init` (team-wide) vs a plain taskgroup.
    team_wide: bool,
}

unsafe impl Send for TgRed {}

fn build_red(inputs: &[TaskRedInput], nth: usize) -> TeamRed {
    let items = inputs
        .iter()
        .map(|inp| {
            let size = inp.reduce_size.max(1).div_ceil(64) * 64;
            let privs = RootedBlock::new(size * nth);
            let init: Option<InitFn> = unsafe { std::mem::transmute(inp.reduce_init) };
            let orig = if inp.reduce_orig.is_null() { inp.reduce_shar } else { inp.reduce_orig };
            if let Some(f) = init {
                for j in 0..nth {
                    unsafe { f(privs.ptr().add(j * size).cast(), orig) };
                }
            }
            RedItem {
                size,
                comb: unsafe { std::mem::transmute::<*mut c_void, Option<CombFn>>(inp.reduce_comb) },
                fini: unsafe { std::mem::transmute::<*mut c_void, Option<FiniFn>>(inp.reduce_fini) },
                privs,
            }
        })
        .collect();
    TeamRed { items, nth }
}

fn combine_all(red: &TgRed) {
    for (item, &shar) in red.data.items.iter().zip(&red.shars) {
        for j in 0..red.data.nth {
            let p = unsafe { item.privs.ptr().add(j * item.size) }.cast::<c_void>();
            if let Some(c) = item.comb {
                unsafe { c(shar, p) };
            }
            if let Some(f) = item.fini {
                unsafe { f(p) };
            }
        }
    }
}

fn finish_reduction(frame: &Frame, red: TgRed) {
    if !red.team_wide {
        combine_all(&red);
        return;
    }
    let team = &frame.team;
    let n = team.red_fini.fetch_add(1, Ordering::AcqRel) + 1;
    if n == team.nproc {
        // last thread out combines every private copy into its own shared variable
        combine_all(&red);
        *team.red.lock() = None;
        team.red_fini.store(0, Ordering::Release);
    }
}

fn inputs<'a>(num: i32, data: *mut c_void) -> &'a [TaskRedInput] {
    if num <= 0 || data.is_null() {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(data.cast::<TaskRedInput>(), num as usize) }
    }
}

/// Plain taskgroup reduction init (`__kmpc_taskred_init`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_taskred_init(_gtid: i32, num: i32, data: *mut c_void) -> *mut c_void {
    let frame = current();
    let t = cur_task();
    let Some(tg) = t.taskgroup.borrow().clone() else {
        return std::ptr::null_mut();
    };
    let inp = inputs(num, data);
    if frame.team.nproc > 1 && !inp.is_empty() {
        let red = Arc::new(build_red(inp, frame.team.nproc as usize));
        *tg.red.lock() =
            Some(TgRed { data: red, shars: inp.iter().map(|i| i.reduce_shar).collect(), team_wide: false });
    }
    Arc::as_ptr(&tg) as *mut c_void
}

/// `void *__kmpc_taskred_modifier_init(ident_t*, gtid, is_ws, num, data)`: opens a
/// taskgroup in every team thread and sets up team-shared private copies.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_taskred_modifier_init(
    loc: *mut Ident,
    gtid: i32,
    _is_ws: i32,
    num: i32,
    data: *mut c_void,
) -> *mut c_void {
    unsafe { __kmpc_taskgroup(loc, gtid) };
    let frame = current();
    let tg = cur_task().taskgroup.borrow().clone().unwrap_or_else(|| unreachable!("taskgroup just created"));
    let inp = inputs(num, data);
    if frame.team.nproc > 1 && !inp.is_empty() {
        // libomp keeps separate parallel/worksharing slots (is_ws); Typhon
        // only uses is_ws = 0, so one team slot is kept.
        let shared = {
            let mut slot = frame.team.red.lock();
            slot.get_or_insert_with(|| Arc::new(build_red(inp, frame.team.nproc as usize))).clone()
        };
        *tg.red.lock() =
            Some(TgRed { data: shared, shars: inp.iter().map(|i| i.reduce_shar).collect(), team_wide: true });
    }
    Arc::as_ptr(&tg) as *mut c_void
}

/// Old-interface alias (single-parameter initializer); same data layout here.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_task_reduction_modifier_init(
    loc: *mut Ident,
    gtid: i32,
    is_ws: i32,
    num: i32,
    data: *mut c_void,
) -> *mut c_void {
    unsafe { __kmpc_taskred_modifier_init(loc, gtid, is_ws, num, data) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_task_reduction_modifier_fini(loc: *mut Ident, gtid: i32, _is_ws: i32) {
    unsafe { __kmpc_end_taskgroup(loc, gtid) };
}

/// Thread-specific location of reduction item `data` (shared address or any private
/// copy). `tskgrp` may be NULL: the current task's taskgroup chain is searched.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_task_reduction_get_th_data(
    _gtid: i32,
    tskgrp: *mut c_void,
    data: *mut c_void,
) -> *mut c_void {
    let frame = current();
    if frame.team.nproc == 1 {
        return data;
    }
    let tid = frame.tid as usize;
    let mut tg: Option<Arc<TaskGroup>> = cur_task().taskgroup.borrow().clone();
    if !tskgrp.is_null() {
        // Prefer the explicit group when it is in our chain (it always is for Typhon).
        let mut g = tg.clone();
        while let Some(x) = g {
            if Arc::as_ptr(&x) as *mut c_void == tskgrp {
                tg = Some(x);
                break;
            }
            g = x.parent.clone();
        }
    }
    while let Some(g) = tg {
        if let Some(red) = g.red.lock().as_ref() {
            for (item, &shar) in red.data.items.iter().zip(&red.shars) {
                if data == shar || item.privs.contains(data.cast()) {
                    return unsafe { item.privs.ptr().add(tid * item.size) }.cast();
                }
            }
        }
        tg = g.parent.clone();
    }
    // PORT-NOTE: libomp asserts on unknown items; we fall back to the shared address.
    data
}
