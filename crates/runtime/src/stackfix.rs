//! Stop-the-world stack ranges when the suspend handler runs on a sigaltstack.
//!
//! In an Android app ART's libsigchain interposes `sigaction` and registers its own
//! dispatcher for every claimed signal with SA_ONSTACK; every ART/Dart-attached thread
//! has a sigaltstack. bdwgc's suspend handler then runs on the altstack and records
//! *its* sp as the thread's stack pointer, so marking scans [altstack sp, stack
//! bottom): from a small mmap'ed altstack across unrelated mappings to the thread's
//! stack bottom. For the main thread that walks into the guard page below `[stack]`
//! (SIGSEGV SEGV_ACCERR in GC_mark_from, fault just below the main `[stack]`).
//! bdwgc's `GC_register_altstack` only replaces the range by the altstack (its own
//! FIXME: the normal stack is then not scanned), which would lose live roots.
//!
//! Fix: for each thread known to the collector record its normal stack [lo, hi)
//! (pthread attrs; for the grow-down main stack lo is probed with mincore at
//! collection time) and its altstack. A `GC_set_sp_corrector` callback moves a recorded sp
//! that lies outside the normal stack to `lo`, so the whole (mapped) normal stack is
//! scanned conservatively, and a push-other-roots hook scans those altstacks, which
//! hold the interrupted register context saved by the kernel.
//! The table is only touched under the allocator lock.

use core::ffi::{c_int, c_void};

use crate::gc;

#[derive(Clone, Copy)]
struct Entry {
    tid: libc::pthread_t,
    lo: usize,
    hi: usize,
    alt_lo: usize,
    alt_hi: usize,
}

static mut TABLE: Vec<Entry> = Vec::new();
static mut PREV_PUSH: Option<unsafe extern "C" fn()> = None;

unsafe extern "C" {
    fn GC_set_sp_corrector(f: Option<unsafe extern "C" fn(*mut *mut c_void, *mut c_void)>);
    fn GC_set_push_other_roots(f: Option<unsafe extern "C" fn()>);
    fn GC_get_push_other_roots() -> Option<unsafe extern "C" fn()>;
    fn GC_push_all_eager(lo: *mut c_void, hi: *mut c_void);
    fn GC_call_with_alloc_lock(f: unsafe extern "C" fn(*mut c_void) -> *mut c_void, d: *mut c_void) -> *mut c_void;
}

#[allow(static_mut_refs)]
unsafe extern "C" fn sp_corrector(sp: *mut *mut c_void, tid: *mut c_void) {
    let tid = tid as libc::pthread_t;
    unsafe {
        if let Some(e) = TABLE.iter().find(|e| e.tid == tid) {
            let s = *sp as usize;
            let lo = if e.lo != 0 { e.lo } else { mapped_low(e.hi) };
            if s < lo || s >= e.hi {
                *sp = lo as *mut c_void;
            }
        }
    }
}

#[allow(static_mut_refs)]
unsafe extern "C" fn push_altstacks() {
    unsafe {
        for e in TABLE.iter() {
            if e.alt_lo < e.alt_hi {
                GC_push_all_eager(e.alt_lo as *mut c_void, e.alt_hi as *mut c_void);
            }
        }
        if let Some(p) = PREV_PUSH {
            p();
        }
    }
}

/// Install the hooks (call once, right after GC_init).
pub fn install() {
    unsafe {
        let mut fds = [-1; 2];
        if libc::pipe2(fds.as_mut_ptr(), libc::O_CLOEXEC | libc::O_NONBLOCK) == 0 {
            PROBE = fds;
        }
        PREV_PUSH = GC_get_push_other_roots();
        GC_set_push_other_roots(Some(push_altstacks));
        GC_set_sp_corrector(Some(sp_corrector));
    }
}

/// Lowest address of the contiguous *readable* range ending at `hi` (the main thread's
/// grow-down `[stack]`, which can still grow, so it is probed at collection time; ART
/// puts a PROT_NONE guard right below it). Probed by writing one byte from each page
/// to a pipe: the kernel returns EFAULT for unreadable memory instead of raising a
/// signal, and nothing allocates or takes a lock (the world is stopped).
fn mapped_low(hi: usize) -> usize {
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let (r, w) = unsafe { (PROBE[0], PROBE[1]) };
    let mut lo = (hi - 1) & !(page - 1);
    if w < 0 {
        return lo;
    }
    let mut sink = [0u8; 1];
    while lo >= page {
        let p = lo - page;
        if unsafe { libc::write(w, p as *const c_void, 1) } != 1 {
            break;
        }
        unsafe { libc::read(r, sink.as_mut_ptr().cast(), 1) };
        lo = p;
    }
    lo
}

static mut PROBE: [c_int; 2] = [-1, -1];

/// Mapped low end of a pthread stack (fixed size; excludes the guard page).
fn pthread_stack_low() -> usize {
    unsafe {
        let mut attr: libc::pthread_attr_t = core::mem::zeroed();
        if libc::pthread_getattr_np(libc::pthread_self(), &mut attr) != 0 {
            return 0;
        }
        let (mut addr, mut size) = (core::ptr::null_mut(), 0usize);
        let r = libc::pthread_attr_getstack(&attr, &mut addr, &mut size);
        libc::pthread_attr_destroy(&mut attr);
        if r == 0 { addr as usize } else { 0 }
    }
}

#[allow(static_mut_refs)]
unsafe extern "C" fn update(p: *mut c_void) -> *mut c_void {
    unsafe {
        let e = &*(p as *const Entry);
        TABLE.retain(|x| x.tid != e.tid);
        if e.hi != 0 {
            TABLE.push(*e);
        }
    }
    core::ptr::null_mut()
}

/// Record the calling (registered) thread's normal stack and altstack.
pub fn record_current_thread() {
    let mut sb = gc::GC_stack_base { mem_base: core::ptr::null_mut() };
    if unsafe { gc::GC_get_my_stackbottom(&mut sb) }.is_null() || sb.mem_base.is_null() {
        return;
    }
    let hi = sb.mem_base as usize;
    // 0 = main thread: probe its grow-down stack when needed.
    let main = unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t == libc::getpid() };
    let lo = if main { 0 } else { pthread_stack_low() };
    if !main && lo == 0 {
        return;
    }
    let mut ss: libc::stack_t = unsafe { core::mem::zeroed() };
    let (mut alt_lo, mut alt_hi) = (0, 0);
    if unsafe { libc::sigaltstack(core::ptr::null(), &mut ss) } == 0 && ss.ss_flags & libc::SS_DISABLE == 0 && !ss.ss_sp.is_null() {
        alt_lo = ss.ss_sp as usize;
        alt_hi = alt_lo + ss.ss_size;
    }
    let mut e = Entry { tid: unsafe { libc::pthread_self() }, lo, hi, alt_lo, alt_hi };
    unsafe { GC_call_with_alloc_lock(update, (&mut e as *mut Entry).cast()) };
}

/// Drop the calling thread's entry (before it exits; pthread_t values are reused).
pub fn forget_current_thread() {
    let mut e = Entry { tid: unsafe { libc::pthread_self() }, lo: 0, hi: 0, alt_lo: 0, alt_hi: 0 };
    unsafe { GC_call_with_alloc_lock(update, (&mut e as *mut Entry).cast()) };
}
