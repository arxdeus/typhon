//! Runtime core: init, GC allocation, time, OS helpers, locale,
//! float parsing, printing / output capture, and locks/condition variables.

use core::ffi::{c_char, c_int, c_void};
use std::ffi::CStr;
use std::sync::Once;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::gc;
use parking_lot::{Condvar, Mutex, ReentrantMutex};

use crate::types::*;

/// `int seq_flags` (exported data symbol, same name as C++).
#[unsafe(no_mangle)]
pub static seq_flags: AtomicI32 = AtomicI32::new(0);

#[inline]
pub(crate) fn flags() -> i32 {
    seq_flags.load(Ordering::Relaxed)
}

/*
 * General
 */

type GcSetupCallback = unsafe extern "C" fn(*mut gc::GC_stack_base) -> c_int;
#[cfg_attr(feature = "omp-rust", allow(dead_code))]
type GcRootsCallback = unsafe extern "C" fn(*mut c_void, *mut c_void);

#[cfg(feature = "openmp-libomp")]
unsafe extern "C" {
    // Provided by the GC-patched libomp: registers bdwgc thread/roots callbacks so
    // @par worker threads allocate safely.
    fn __kmpc_set_gc_callbacks(
        get_stack_base: GcSetupCallback,
        register_thread: GcSetupCallback,
        add_roots: GcRootsCallback,
        del_roots: GcRootsCallback,
    );
}

unsafe extern "C" fn register_thread_shim(sb: *mut gc::GC_stack_base) -> c_int {
    unsafe { gc::GC_register_my_thread(sb) }
}

thread_local! {
    static DISPATCHING: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

unsafe extern "C" fn run_finalizers() {
    // Allocating callbacks may notify recursively; only the outer invocation drains.
    if DISPATCHING.with(|d| d.get()) || unsafe { gc::GC_should_invoke_finalizers() } == 0 {
        return;
    }
    DISPATCHING.with(|d| d.set(true));
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            DISPATCHING.with(|d| d.set(false));
        }
    }
    let _reset = Reset;
    unsafe { gc::GC_invoke_finalizers() };
}

static GC_DIVISOR: AtomicUsize = AtomicUsize::new(3);
/// Heap size seen at the last collection we allowed (detects heap growth).
static LAST_HEAP: AtomicUsize = AtomicUsize::new(0);

/// bdwgc 8.3 (`bdwgc-alloc-sys`) `GC_should_collect()` forces a full collection
/// right after every heap growth (`GC_last_heap_growth_gc_no == GC_gc_no`), which
/// bdwgc 8.2.12 does not; on allocation-heavy programs that is ~4x more
/// collections (float.py: 22 vs 6). Allocation-triggered collections go through
/// `GC_default_stop_func` (`GC_collect_or_expand`); refusing to start one makes the
/// collector expand the heap instead. We only veto that forced post-growth GC (the
/// heap size changed since the last collection we allowed) and only while fewer
/// bytes were allocated since the last GC than `in_use / (2 * free_space_divisor)`.
/// Ordinary threshold-triggered GCs (8.3's own `min_bytes_allocd`, which already
/// weights atomic data at 1/4) are never vetoed, so the heap stays as tight as
/// the C++ baseline's (binary_trees 20: 263 MB vs 540 MB with the earlier `2*in_use/div` veto).
/// Called with the allocator lock held; once a GC is allowed LAST_HEAP is updated so
/// repeated calls during marking keep answering "continue".
unsafe extern "C" fn gc_growth_stop_func() -> c_int {
    let mut s = gc::GC_prof_stats_s::default();
    unsafe { gc::GC_get_prof_stats_unsafe(&mut s, core::mem::size_of::<gc::GC_prof_stats_s>()) };
    let in_use = s.heapsize_full.saturating_sub(s.free_bytes_full);
    let grew = s.heapsize_full != LAST_HEAP.load(Ordering::Relaxed);
    let min_allocd = in_use / 2 / GC_DIVISOR.load(Ordering::Relaxed).max(1);
    let stop = s.gc_no > 1 && grew && s.bytes_allocd_since_gc < min_allocd;
    if !stop {
        LAST_HEAP.store(s.heapsize_full, Ordering::Relaxed);
    }
    stop as c_int
}

extern "C" fn print_gc_stats() {
    let mut s = gc::GC_prof_stats_s::default();
    unsafe { gc::GC_get_prof_stats_unsafe(&mut s, core::mem::size_of::<gc::GC_prof_stats_s>()) };
    eprintln!(
        "typhon gc: collections={} heap={} KiB free={} KiB unmapped={} KiB obtained_from_os={} KiB",
        s.gc_no, s.heapsize_full >> 10, s.free_bytes_full >> 10, s.unmapped_bytes >> 10, s.obtained_from_os_bytes >> 10
    );
}

static GC_INIT: Once = Once::new();

/// Initialize the collector once (idempotent; used by tests too).
pub fn gc_init() {
    GC_INIT.call_once(|| unsafe {
        gc::GC_set_finalize_on_demand(1);
        gc::GC_init();
        gc::GC_set_warn_proc(Some(gc::GC_ignore_warn_proc));
        gc::GC_set_finalizer_notifier(Some(run_finalizers));
        GC_DIVISOR.store(gc::GC_get_free_space_divisor(), Ordering::Relaxed);
        gc::GC_set_stop_func(Some(gc_growth_stop_func));
        // `TYPHON_GC_STATS=1`: print collection count and heap stats at exit.
        if std::env::var_os("TYPHON_GC_STATS").is_some() {
            unsafe extern "C" {
                fn atexit(f: extern "C" fn()) -> c_int;
            }
            atexit(print_gc_stats);
        }
        #[cfg(feature = "omp-rust")]
        // Same C ABI; typhon-omp spells the stack-base pointer as `void *`.
        typhon_omp::__kmpc_set_gc_callbacks(
            core::mem::transmute::<GcSetupCallback, typhon_omp::gc::GcSetupCallback>(gc::GC_get_stack_base),
            core::mem::transmute::<GcSetupCallback, typhon_omp::gc::GcSetupCallback>(register_thread_shim),
            gc::GC_add_roots,
            gc::GC_remove_roots,
        );
        #[cfg(feature = "omp-rust")]
        typhon_omp::gc::set_unregister_callback(gc::GC_unregister_my_thread);
        #[cfg(feature = "openmp-libomp")]
        __kmpc_set_gc_callbacks(
            gc::GC_get_stack_base,
            register_thread_shim,
            gc::GC_add_roots,
            gc::GC_remove_roots,
        );
        #[cfg(not(any(feature = "omp-rust", feature = "openmp-libomp")))]
        let _ = (register_thread_shim as GcSetupCallback, gc::GC_add_roots as GcRootsCallback);
    });
    // libtest runs each test on its own thread; register it so the collector scans
    // its stack (unregistered threads' roots are invisible and objects get freed).
    #[cfg(test)]
    {
        seq_allow_register_threads();
        let mut sb = gc::GC_stack_base { mem_base: core::ptr::null_mut() };
        unsafe {
            if gc::GC_get_stack_base(&mut sb) == 0 {
                gc::GC_register_my_thread(&sb); // returns GC_DUPLICATE if already registered
            }
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_init(flags: c_int) {
    gc_init();
    // Libraries (`-lib`) may be entered from foreign threads (e.g. Dart isolates):
    // allow those to register themselves via `seq_register_current_thread`.
    seq_allow_register_threads();
    crate::exc::seq_exc_init(flags);
    seq_flags.store(flags, Ordering::Relaxed);
}

/// Register the calling thread with the collector if it is not yet known
/// (idempotent; cheap when already registered). Initializes the runtime if needed.
/// Called at the entry of every exported function in `-lib` builds.
#[unsafe(no_mangle)]
pub extern "C" fn seq_register_current_thread() {
    // Hot path (every exported call): one TLS load + branch, no GC lock.
    if THREAD_REGISTERED.get() {
        return;
    }
    register_current_thread_slow();
}

#[cold]
#[inline(never)]
fn register_current_thread_slow() {
    if !GC_INIT.is_completed() {
        seq_init(0);
    }
    unsafe {
        if gc::GC_thread_is_registered() != 0 {
            // Main/initial thread (registered by GC_init) or registered elsewhere.
            THREAD_REGISTERED.set(true);
            return;
        }
        seq_allow_register_threads();
        let mut sb = gc::GC_stack_base { mem_base: core::ptr::null_mut() };
        if gc::GC_get_stack_base(&mut sb) == 0 && gc::GC_register_my_thread(&sb) == 0 {
            // A foreign thread that exits while registered leaves a dead entry the
            // collector would signal (pthread_kill on a stale pthread_t aborts on
            // bionic): unregister from a TLS destructor at thread exit.
            FOREIGN_THREAD_GUARD.with(|g| g.set(true));
            THREAD_REGISTERED.set(true);
        }
    }
}

struct ForeignThreadGuard(core::cell::Cell<bool>);
impl ForeignThreadGuard {
    fn set(&self, v: bool) {
        self.0.set(v)
    }
}
impl Drop for ForeignThreadGuard {
    fn drop(&mut self) {
        if self.0.get() {
            // Later TLS destructors calling exports must re-register.
            let _ = THREAD_REGISTERED.try_with(|r| r.set(false));
            unsafe { gc::GC_unregister_my_thread() };
        }
    }
}
thread_local! {
    /// Set once this thread is known to the collector; const-initialized and
    /// without a destructor, so reading it is a plain TLS load.
    static THREAD_REGISTERED: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
    static FOREIGN_THREAD_GUARD: ForeignThreadGuard = const { ForeignThreadGuard(core::cell::Cell::new(false)) };
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_allow_register_threads() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe { gc::GC_allow_register_threads() });
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_pid() -> SeqInt {
    std::process::id() as SeqInt
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_time() -> SeqInt {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as SeqInt)
        .unwrap_or_else(|e| -(e.duration().as_nanos() as SeqInt))
}

fn monotonic_nanos() -> SeqInt {
    // PORT-NOTE: std::chrono::steady_clock epoch is unspecified; we use CLOCK_MONOTONIC
    // directly (same clock libc++ uses on macOS/Linux).
    let mut ts = libc::timespec { tv_sec: 0, tv_nsec: 0 };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    ts.tv_sec as SeqInt * 1_000_000_000 + ts.tv_nsec as SeqInt
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_time_monotonic() -> SeqInt {
    monotonic_nanos()
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_time_highres() -> SeqInt {
    // libc++ high_resolution_clock == steady_clock.
    monotonic_nanos()
}

fn copy_time_c_to_seq(x: &libc::tm, out: &mut SeqTime) {
    out.year = x.tm_year as i16;
    out.yday = x.tm_yday as i16;
    out.sec = x.tm_sec as i8;
    out.min = x.tm_min as i8;
    out.hour = x.tm_hour as i8;
    out.mday = x.tm_mday as i8;
    out.mon = x.tm_mon as i8;
    out.wday = x.tm_wday as i8;
    out.isdst = x.tm_isdst as i8;
}

fn copy_time_seq_to_c(x: &SeqTime) -> libc::tm {
    let mut t: libc::tm = unsafe { core::mem::zeroed() };
    t.tm_year = x.year as c_int;
    t.tm_yday = x.yday as c_int;
    t.tm_sec = x.sec as c_int;
    t.tm_min = x.min as c_int;
    t.tm_hour = x.hour as c_int;
    t.tm_mday = x.mday as c_int;
    t.tm_mon = x.mon as c_int;
    t.tm_wday = x.wday as c_int;
    t.tm_isdst = x.isdst as c_int;
    t
}

/// # Safety
/// `output` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_localtime(secs: SeqInt, output: *mut SeqTime) -> bool {
    unsafe {
        let now: libc::time_t = if secs >= 0 { secs as _ } else { libc::time(core::ptr::null_mut()) };
        let mut r: libc::tm = core::mem::zeroed();
        if now == -1 || libc::localtime_r(&now, &mut r).is_null() {
            return false;
        }
        copy_time_c_to_seq(&r, &mut *output);
        true
    }
}

/// # Safety
/// `output` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_gmtime(secs: SeqInt, output: *mut SeqTime) -> bool {
    unsafe {
        let now: libc::time_t = if secs >= 0 { secs as _ } else { libc::time(core::ptr::null_mut()) };
        let mut r: libc::tm = core::mem::zeroed();
        if now == -1 || libc::gmtime_r(&now, &mut r).is_null() {
            return false;
        }
        copy_time_c_to_seq(&r, &mut *output);
        true
    }
}

/// # Safety
/// `time` must be a valid pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_mktime(time: *mut SeqTime) -> SeqInt {
    unsafe {
        let mut t = copy_time_seq_to_c(&*time);
        libc::mktime(&mut t) as SeqInt
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_sleep(secs: f64) {
    if secs > 0.0 && secs.is_finite() {
        std::thread::sleep(Duration::from_secs_f64(secs));
    }
}

unsafe extern "C" {
    #[cfg(target_vendor = "apple")]
    fn _NSGetEnviron() -> *mut *mut *mut c_char;
    #[cfg(not(target_vendor = "apple"))]
    static mut environ: *mut *mut c_char;
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_env() -> *mut *mut c_char {
    #[cfg(target_vendor = "apple")]
    unsafe {
        *_NSGetEnviron()
    }
    #[cfg(not(target_vendor = "apple"))]
    unsafe {
        environ
    }
}

/// # Safety
/// `path` must be a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_os_open(path: *const c_char, flags: i32, mode: u32) -> i32 {
    unsafe { libc::open(path, flags | libc::O_CLOEXEC, mode as libc::c_uint) }
}

/// # Safety
/// `path` is NULL or NUL-terminated; `fields` has room for 14 i64.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_os_stat(
    path: *const c_char,
    descriptor: i32,
    follow: bool,
    fields: *mut i64,
) -> i32 {
    unsafe {
        let mut info: libc::stat = core::mem::zeroed();
        let r = if !path.is_null() {
            if follow { libc::stat(path, &mut info) } else { libc::lstat(path, &mut info) }
        } else {
            libc::fstat(descriptor, &mut info)
        };
        if r != 0 {
            return r;
        }
        let f = core::slice::from_raw_parts_mut(fields, 14);
        f[0] = info.st_mode as i64;
        f[1] = info.st_ino as i64;
        f[2] = info.st_dev as i64;
        f[3] = info.st_nlink as i64;
        f[4] = info.st_uid as i64;
        f[5] = info.st_gid as i64;
        f[6] = info.st_size as i64;
        f[7] = info.st_atime as i64;
        f[8] = info.st_mtime as i64;
        f[9] = info.st_ctime as i64;
        f[10] = info.st_atime_nsec as i64;
        f[11] = info.st_mtime_nsec as i64;
        f[12] = info.st_ctime_nsec as i64;
        f[13] = info.st_rdev as i64;
        0
    }
}

fn set_errno(v: c_int) {
    #[cfg(target_vendor = "apple")]
    unsafe {
        *libc::__error() = v
    };
    #[cfg(target_os = "linux")]
    unsafe {
        *libc::__errno_location()
    };
    #[cfg(target_os = "android")]
    unsafe {
        *libc::__errno() = v
    };
}

fn get_errno() -> c_int {
    std::io::Error::last_os_error().raw_os_error().unwrap_or(0)
}

/// # Safety
/// `directory` must be a valid `DIR*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_os_readdir(directory: *mut c_void) -> *const c_char {
    set_errno(0);
    unsafe {
        let e = libc::readdir(directory as *mut libc::DIR);
        if e.is_null() { core::ptr::null() } else { (*e).d_name.as_ptr() }
    }
}

/// # Safety
/// `name` is NULL or NUL-terminated. Result is `malloc`ed (strdup) or NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_os_home(name: *const c_char) -> *mut c_char {
    let mut buffer = vec![0 as c_char; 1024];
    unsafe {
        let mut record: libc::passwd = core::mem::zeroed();
        let mut found: *mut libc::passwd = core::ptr::null_mut();
        loop {
            let error = if !name.is_null() {
                libc::getpwnam_r(name, &mut record, buffer.as_mut_ptr(), buffer.len(), &mut found)
            } else {
                libc::getpwuid_r(libc::getuid(), &mut record, buffer.as_mut_ptr(), buffer.len(), &mut found)
            };
            if error == libc::ERANGE && buffer.len() < 1024 * 1024 {
                let n = buffer.len() * 2;
                buffer.resize(n, 0);
                continue;
            }
            set_errno(error);
            return if error != 0 || found.is_null() {
                core::ptr::null_mut()
            } else {
                libc::strdup(record.pw_dir)
            };
        }
    }
}

/*
 * GC
 */

#[unsafe(no_mangle)]
pub extern "C" fn seq_alloc(n: usize) -> *mut c_void {
    unsafe { gc::GC_malloc(n) }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_alloc_atomic(n: usize) -> *mut c_void {
    unsafe { gc::GC_malloc_atomic(n) }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_alloc_uncollectable(n: usize) -> *mut c_void {
    unsafe { gc::GC_malloc_uncollectable(n) }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_alloc_atomic_uncollectable(n: usize) -> *mut c_void {
    unsafe { gc::GC_malloc_atomic_uncollectable(n) }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_realloc(p: *mut c_void, newsize: usize, _oldsize: usize) -> *mut c_void {
    unsafe { gc::GC_realloc(p, newsize) }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_free(p: *mut c_void) {
    unsafe { gc::GC_free(p) }
}

pub type Finalizer = unsafe extern "C" fn(obj: *mut c_void, data: *mut c_void);

#[unsafe(no_mangle)]
pub extern "C" fn seq_register_finalizer(p: *mut c_void, f: Option<Finalizer>) {
    unsafe {
        gc::GC_register_finalizer(p, f, core::ptr::null_mut(), core::ptr::null_mut(), core::ptr::null_mut())
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_gc_add_roots(start: *mut c_void, end: *mut c_void) {
    unsafe { gc::GC_add_roots(start, end) }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_gc_remove_roots(start: *mut c_void, end: *mut c_void) {
    unsafe { gc::GC_remove_roots(start, end) }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_gc_clear_roots() {
    unsafe { gc::GC_clear_roots() }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_gc_exclude_static_roots(start: *mut c_void, end: *mut c_void) {
    unsafe { gc::GC_exclude_static_roots(start, end) }
}

/*
 * Locale / string conversion
 */

unsafe fn cstr_bytes<'a>(p: *const c_char) -> &'a [u8] {
    if p.is_null() { &[] } else { unsafe { CStr::from_ptr(p).to_bytes() } }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_locale_decimal_point() -> SeqStr {
    unsafe { string_conv(cstr_bytes((*libc::localeconv()).decimal_point)) }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_locale_thousands_sep() -> SeqStr {
    unsafe { string_conv(cstr_bytes((*libc::localeconv()).thousands_sep)) }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_locale_grouping() -> SeqBytes {
    unsafe {
        let g = (*libc::localeconv()).grouping;
        SeqBytes { ptr: g as *mut u8, len: cstr_bytes(g).len() as SeqInt }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_locale_numeric() -> SeqStr {
    unsafe { string_conv(cstr_bytes(libc::setlocale(libc::LC_NUMERIC, core::ptr::null()))) }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_set_locale_numeric(locale: SeqStr) -> bool {
    let mut s = locale.encode();
    s.push(0);
    unsafe { !libc::setlocale(libc::LC_NUMERIC, s.as_ptr() as *const c_char).is_null() }
}

/// Parse a float prefix like fast_float with `FASTFLOAT_ALLOWS_LEADING_PLUS` and
/// `FASTFLOAT_SKIP_WHITE_SPACE`. Returns (value, bytes consumed) or None.
pub fn parse_float_prefix(s: &[u8]) -> Option<(f64, usize)> {
    let ws = s.iter().take_while(|c| matches!(c, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')).count();
    let rest = &s[ws..];
    match fast_float2::parse_partial::<f64, _>(rest) {
        Ok((v, n)) if n > 0 => Some((v, ws + n)),
        _ => None,
    }
}

/// # Safety
/// `s` must be a valid string and `e` a valid out-pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_float_from_str(s: SeqStr, e: *mut *const c_char) -> f64 {
    unsafe {
        if s.kind() != SEQ_STR_KIND_ASCII {
            *e = s.ptr as *const c_char;
            return 0.0;
        }
        let bytes = if s.ptr.is_null() { &[][..] } else { core::slice::from_raw_parts(s.ptr, s.len()) };
        match parse_float_prefix(bytes) {
            Some((v, n)) => {
                *e = s.ptr.add(n) as *const c_char;
                v
            }
            None => {
                *e = s.ptr as *const c_char;
                0.0
            }
        }
    }
}

/*
 * General I/O
 */

#[unsafe(no_mangle)]
pub extern "C" fn seq_check_errno() -> SeqStr {
    let err = get_errno();
    if err != 0 {
        unsafe { string_conv(cstr_bytes(libc::strerror(err))) }
    } else {
        SeqStr::EMPTY
    }
}

unsafe extern "C" {
    #[cfg(target_vendor = "apple")]
    #[link_name = "__stdinp"]
    static mut c_stdin: *mut libc::FILE;
    #[cfg(target_vendor = "apple")]
    #[link_name = "__stdoutp"]
    static mut c_stdout: *mut libc::FILE;
    #[cfg(target_vendor = "apple")]
    #[link_name = "__stderrp"]
    static mut c_stderr: *mut libc::FILE;
    #[cfg(not(target_vendor = "apple"))]
    #[link_name = "stdin"]
    static mut c_stdin: *mut libc::FILE;
    #[cfg(not(target_vendor = "apple"))]
    #[link_name = "stdout"]
    static mut c_stdout: *mut libc::FILE;
    #[cfg(not(target_vendor = "apple"))]
    #[link_name = "stderr"]
    static mut c_stderr: *mut libc::FILE;
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_stdin() -> *mut c_void {
    unsafe { c_stdin.cast() }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_stdout() -> *mut c_void {
    unsafe { c_stdout.cast() }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_stderr() -> *mut c_void {
    unsafe { c_stderr.cast() }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_print(s: SeqStr) {
    unsafe { seq_print_full(s, c_stdout) }
}

/// # Safety
/// `fo` must be a valid `FILE*`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_print_full(s: SeqStr, fo: *mut libc::FILE) {
    let utf8 = s.encode();
    unsafe {
        if !utf8.is_empty() {
            libc::fwrite(utf8.as_ptr().cast(), 1, utf8.len(), fo);
        }
    }
}


/*
 * Threading
 */

// PORT-NOTE: std::timed_mutex / std::recursive_timed_mutex / std::condition_variable_any
// are replaced by parking_lot equivalents placed in GC memory. parking_lot primitives
// are word-sized and need no destructor, so no finalizer is registered for them.

type Lock = Mutex<()>;
type RLock = ReentrantMutex<()>;

fn timeout_dur(t: f64) -> Duration {
    if t.is_finite() { Duration::from_secs_f64(t.max(0.0)) } else { Duration::from_secs(u32::MAX as u64) }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_lock_new() -> *mut c_void {
    let p = seq_alloc_atomic(core::mem::size_of::<Lock>()) as *mut Lock;
    unsafe { p.write(Lock::new(())) };
    p.cast()
}

/// # Safety
/// `lock` must come from `seq_lock_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_lock_acquire(lock: *mut c_void, block: bool, timeout: f64) -> bool {
    let m = unsafe { &*(lock as *const Lock) };
    let g = if timeout < 0.0 {
        if block { Some(m.lock()) } else { m.try_lock() }
    } else {
        m.try_lock_for(timeout_dur(timeout))
    };
    match g {
        Some(g) => {
            core::mem::forget(g);
            true
        }
        None => false,
    }
}

/// # Safety
/// `lock` must be held.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_lock_release(lock: *mut c_void) {
    unsafe { (*(lock as *const Lock)).force_unlock() }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_cond_new() -> *mut c_void {
    let p = seq_alloc_atomic(core::mem::size_of::<Condvar>()) as *mut Condvar;
    unsafe { p.write(Condvar::new()) };
    p.cast()
}

/// # Safety
/// `lock` must be a held `seq_lock_new` lock.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_cond_wait(condition: *mut c_void, lock: *mut c_void, timeout: f64) {
    unsafe {
        let cv = &*(condition as *const Condvar);
        let m = &*(lock as *const Lock);
        let mut guard = m.make_guard_unchecked();
        if timeout < 0.0 {
            cv.wait(&mut guard);
        } else {
            let _ = cv.wait_until(&mut guard, Instant::now() + timeout_dur(timeout));
        }
        core::mem::forget(guard);
    }
}

/// # Safety
/// `condition` must come from `seq_cond_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_cond_notify(condition: *mut c_void) {
    unsafe { (*(condition as *const Condvar)).notify_one() };
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_rlock_new() -> *mut c_void {
    let p = seq_alloc_atomic(core::mem::size_of::<RLock>()) as *mut RLock;
    unsafe { p.write(RLock::new(())) };
    p.cast()
}

/// # Safety
/// `lock` must come from `seq_rlock_new`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_rlock_acquire(lock: *mut c_void, block: bool, timeout: f64) -> bool {
    let m = unsafe { &*(lock as *const RLock) };
    let g = if timeout < 0.0 {
        if block { Some(m.lock()) } else { m.try_lock() }
    } else {
        m.try_lock_for(timeout_dur(timeout))
    };
    match g {
        Some(g) => {
            core::mem::forget(g);
            true
        }
        None => false,
    }
}

/// # Safety
/// `lock` must be held by this thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_rlock_release(lock: *mut c_void) {
    unsafe { (*(lock as *const RLock)).force_unlock() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_prefix() {
        assert_eq!(parse_float_prefix(b"  +1.5e3xyz"), Some((1500.0, 8)));
        assert_eq!(parse_float_prefix(b"-inf"), Some((f64::NEG_INFINITY, 4)));
        assert_eq!(parse_float_prefix(b"abc"), None);
        assert_eq!(parse_float_prefix(b"1e999").map(|x| x.0), Some(f64::INFINITY));
    }

    #[test]
    fn string_roundtrip() {
        gc_init();
        for s in ["", "hello", "caf\u{e9}", "\u{3b1}\u{3b2}", "x\u{1F600}"] {
            let st = string_conv(s.as_bytes());
            assert_eq!(st.encode(), s.as_bytes());
            assert!(st.eq_utf8(s.as_bytes()));
        }
        assert_eq!(string_conv("caf\u{e9}".as_bytes()).kind(), SEQ_STR_KIND_LATIN1);
        assert_eq!(string_conv("\u{3b1}".as_bytes()).kind(), SEQ_STR_KIND_UCS2);
        assert_eq!(string_conv("\u{1F600}".as_bytes()).kind(), SEQ_STR_KIND_UCS4);
        assert!(!string_conv(b"abc").eq_utf8(b"abd"));
    }

    #[test]
    fn utf8_invalid() {
        let mut p = 0;
        assert!(utf8_read_checked(&[0xC0, 0x80], &mut p).is_err());
        p = 0;
        assert!(utf8_read_checked(&[0xED, 0xA0, 0x80], &mut p).is_err());
    }

    #[test]
    fn float_from_str_abi() {
        gc_init();
        let s = string_conv(b"3.25rest");
        let mut e: *const c_char = core::ptr::null();
        let v = unsafe { seq_float_from_str(s, &mut e) };
        assert_eq!(v, 3.25);
        assert_eq!(e as usize - s.ptr as usize, 4);
    }

    #[test]
    fn time_roundtrip() {
        let mut t = SeqTime::default();
        assert!(unsafe { seq_gmtime(86400 * 365, &mut t) });
        assert_eq!((t.year, t.mon, t.mday), (71, 0, 1));
        assert!(seq_time() > 0);
        let a = seq_time_monotonic();
        assert!(seq_time_monotonic() >= a);
    }

    #[test]
    fn locks() {
        gc_init();
        let l = seq_lock_new();
        unsafe {
            assert!(seq_lock_acquire(l, true, -1.0));
            assert!(!seq_lock_acquire(l, false, -1.0));
            assert!(!seq_lock_acquire(l, true, 0.01));
            let c = seq_cond_new();
            seq_cond_wait(c, l, 0.01);
            seq_lock_release(l);
            assert!(seq_lock_acquire(l, false, -1.0));
            seq_lock_release(l);
            let r = seq_rlock_new();
            assert!(seq_rlock_acquire(r, true, -1.0));
            assert!(seq_rlock_acquire(r, false, -1.0));
            seq_rlock_release(r);
            seq_rlock_release(r);
        }
    }
}
