//! Boehm GC integration.
//!
//! A GC-patched libomp exports `__kmpc_set_gc_callbacks`, which libtyphonrt's
//! `seq_init` calls with `GC_get_stack_base`, `GC_register_my_thread`,
//! `GC_add_roots` and `GC_remove_roots`. Every worker thread registers itself
//! before running any microtask, and runtime-owned memory that may hold the only
//! reference to GC objects (queued task descriptors, task-reduction private
//! copies) lives in malloc'ed blocks registered as GC roots.

use std::ffi::{c_int, c_void};
use std::sync::OnceLock;

use parking_lot::Mutex;

pub type GcSetupCallback = unsafe extern "C" fn(*mut c_void) -> c_int;
pub type GcRootsCallback = unsafe extern "C" fn(*mut c_void, *mut c_void);

#[derive(Clone, Copy)]
struct Callbacks {
    get_stack_base: GcSetupCallback,
    register_thread: GcSetupCallback,
    add_roots: GcRootsCallback,
    del_roots: GcRootsCallback,
}

static CALLBACKS: OnceLock<Callbacks> = OnceLock::new();
static UNREGISTER: OnceLock<unsafe extern "C" fn() -> c_int> = OnceLock::new();

/// `GC_unregister_my_thread`, called by a worker before it exits. Workers of a hot team
/// whose forking thread exits (e.g. a foreign thread calling a `-lib`) do exit; a dead
/// thread left registered makes the next collection `pthread_kill` it.
pub fn set_unregister_callback(f: unsafe extern "C" fn() -> c_int) {
    let _ = UNREGISTER.set(f);
}

pub fn unregister_current_thread() {
    if let Some(f) = UNREGISTER.get() {
        unsafe { f() };
    }
}

/// Install the GC hooks (GC-aware libomp extension).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __kmpc_set_gc_callbacks(
    get_stack_base: GcSetupCallback,
    register_thread: GcSetupCallback,
    add_roots: GcRootsCallback,
    del_roots: GcRootsCallback,
) {
    let _ = CALLBACKS.set(Callbacks { get_stack_base, register_thread, add_roots, del_roots });
    // Root blocks allocated before the hooks were installed must become visible now.
    ROOTS.lock().register_all();
}

pub fn callbacks_installed() -> bool {
    CALLBACKS.get().is_some()
}

/// Register the calling (freshly spawned) thread with the collector.
///
/// PORT-NOTE: workers live for the whole process (persistent pool), so they are
/// never unregistered, matching the patched libomp (which had no unregister hook).
pub fn register_current_thread() {
    if let Some(cb) = CALLBACKS.get() {
        // struct GC_stack_base { void *mem_base; [void *reg_base;] }: oversize it.
        let mut sb = [std::ptr::null_mut::<c_void>(); 4];
        unsafe {
            if (cb.get_stack_base)(sb.as_mut_ptr().cast()) == 0 {
                (cb.register_thread)(sb.as_mut_ptr().cast());
            }
        }
    }
}

pub fn add_roots(lo: *mut u8, hi: *mut u8) {
    if let Some(cb) = CALLBACKS.get() {
        unsafe { (cb.add_roots)(lo.cast(), hi.cast()) }
    }
}

pub fn del_roots(lo: *mut u8, hi: *mut u8) {
    if let Some(cb) = CALLBACKS.get() {
        unsafe { (cb.del_roots)(lo.cast(), hi.cast()) }
    }
}

/// Zeroed malloc'ed memory that the GC scans conservatively while alive.
pub struct RootedBlock {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for RootedBlock {}
unsafe impl Sync for RootedBlock {}

impl RootedBlock {
    pub fn new(len: usize) -> Self {
        let len = len.max(8);
        let ptr = unsafe { libc::calloc(1, len) }.cast::<u8>();
        assert!(!ptr.is_null(), "typhon-omp: out of memory");
        add_roots(ptr, unsafe { ptr.add(len) });
        RootedBlock { ptr, len }
    }

    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn contains(&self, p: *const u8) -> bool {
        let p = p as usize;
        p >= self.ptr as usize && p < self.ptr as usize + self.len
    }
}

impl Drop for RootedBlock {
    fn drop(&mut self) {
        del_roots(self.ptr, unsafe { self.ptr.add(self.len) });
        unsafe { libc::free(self.ptr.cast()) };
    }
}

const SLAB_BLOCK: usize = 1024;

/// Slot allocator for pointers that must stay reachable for the GC (queued tasks).
/// Blocks are malloc'ed once, registered as roots and never freed.
struct RootSlab {
    blocks: Vec<*mut usize>,
    registered: usize,
    free: Vec<usize>,
}

unsafe impl Send for RootSlab {}

impl RootSlab {
    fn register_all(&mut self) {
        if CALLBACKS.get().is_none() {
            return;
        }
        while self.registered < self.blocks.len() {
            let b = self.blocks[self.registered].cast::<u8>();
            add_roots(b, unsafe { b.add(SLAB_BLOCK * size_of::<usize>()) });
            self.registered += 1;
        }
    }
}

static ROOTS: Mutex<RootSlab> =
    Mutex::new(RootSlab { blocks: Vec::new(), registered: 0, free: Vec::new() });

/// Store `p` in a GC-visible slot; returns the slot handle.
pub fn root_pointer(p: *mut c_void) -> usize {
    let mut slab = ROOTS.lock();
    if slab.free.is_empty() {
        let block = unsafe { libc::calloc(SLAB_BLOCK, size_of::<usize>()) }.cast::<usize>();
        assert!(!block.is_null(), "typhon-omp: out of memory");
        let base = slab.blocks.len() * SLAB_BLOCK;
        slab.blocks.push(block);
        slab.free.extend((base..base + SLAB_BLOCK).rev());
        slab.register_all();
    }
    let slot = slab.free.pop().unwrap_or_default();
    unsafe { *slab.blocks[slot / SLAB_BLOCK].add(slot % SLAB_BLOCK) = p as usize };
    slot
}

/// Release a slot obtained from [`root_pointer`].
pub fn unroot_pointer(slot: usize) {
    let mut slab = ROOTS.lock();
    unsafe { *slab.blocks[slot / SLAB_BLOCK].add(slot % SLAB_BLOCK) = 0 };
    slab.free.push(slot);
}
