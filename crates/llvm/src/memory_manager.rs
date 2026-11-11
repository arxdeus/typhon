//! Port of `codon/compiler/memory_manager.{h,cpp}`.
//!
//! The C++ `BoehmGCJITLinkMemoryManager` is a JITLink memory manager that
//! registers writable JIT segments as GC roots (`seq_gc_add_roots`) and removes
//! them on deallocation.
//!
//! PORT-NOTE: the LLVM C API cannot install a custom JITLink memory manager, but
//! it can create an RTDyld object linking layer driven by MCJIT-style memory
//! manager callbacks
//! (`LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks`).
//! This module implements those callbacks: sections are bump-allocated from a
//! single large mapping (keeping all JIT code within branch range on AArch64),
//! writable data sections are registered as GC roots, and code is made
//! executable on finalization. The Windows SEH registration plugin has no C API
//! equivalent and is not ported (Windows is not a supported host here).

use std::ffi::CStr;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::debug_listener::{DebugListener, LoadedSection};

/// GC root registration hooks (`seq_gc_add_roots` / `seq_gc_remove_roots`).
#[derive(Clone, Copy)]
pub struct GcRootCallbacks {
    pub add_roots: unsafe extern "C" fn(*mut libc::c_void, *mut libc::c_void),
    pub remove_roots: unsafe extern "C" fn(*mut libc::c_void, *mut libc::c_void),
}

const SLAB_SIZE: usize = 512 << 20; // virtual reservation; pages are committed lazily

fn page_size() -> usize {
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

/// Process-wide bump allocator over one reserved mapping.
struct Slab {
    base: usize,
    size: usize,
    next: usize,
}

impl Slab {
    fn new() -> Option<Self> {
        let p = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                SLAB_SIZE,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        if p == libc::MAP_FAILED {
            return None;
        }
        Some(Slab { base: p as usize, size: SLAB_SIZE, next: p as usize })
    }

    /// Page-aligned allocation so that protections can be changed per section.
    fn alloc(&mut self, size: usize, align: usize) -> Option<usize> {
        let ps = page_size();
        let align = align.max(ps);
        let start = (self.next + align - 1) & !(align - 1);
        let end = start + ((size.max(1) + ps - 1) & !(ps - 1));
        if end > self.base + self.size {
            return None;
        }
        self.next = end;
        Some(start)
    }
}

#[derive(Clone, Copy, Debug)]
struct Allocation {
    addr: usize,
    size: usize,
    code: bool,
    read_only: bool,
}

/// Per-object state (one `CreateContext` per linked object).
pub(crate) struct ObjectContext {
    shared: Arc<Shared>,
    allocations: Vec<Allocation>,
    sections: Vec<LoadedSection>,
    object: Option<Vec<u8>>,
    roots: Vec<(usize, usize)>,
}

/// State shared between the engine and the RTDyld callbacks.
pub(crate) struct Shared {
    slab: Mutex<Option<Slab>>,
    pub(crate) gc: Option<GcRootCallbacks>,
    pub(crate) debug: Arc<DebugListener>,
    /// Objects captured by the object transform layer, awaiting linking.
    pub(crate) pending_objects: Mutex<std::collections::VecDeque<Vec<u8>>>,
    next_key: Mutex<u64>,
}

impl Shared {
    pub(crate) fn new(gc: Option<GcRootCallbacks>, debug: Arc<DebugListener>) -> Self {
        Shared {
            slab: Mutex::new(None),
            gc,
            debug,
            pending_objects: Mutex::new(Default::default()),
            next_key: Mutex::new(1),
        }
    }

    fn allocate(&self, size: usize, align: usize) -> *mut u8 {
        let mut s = self.slab.lock();
        if s.is_none() {
            *s = Slab::new();
        }
        let addr = s.as_mut().and_then(|slab| slab.alloc(size, align));
        match addr {
            Some(a) => {
                // `memset(Slab.base(), 0, ...)`: anonymous mappings are already zeroed.
                a as *mut u8
            }
            None => std::ptr::null_mut(),
        }
    }
}

unsafe extern "C" {
    #[cfg(target_os = "macos")]
    fn sys_icache_invalidate(start: *mut libc::c_void, len: usize);
    #[cfg(not(target_os = "macos"))]
    fn __clear_cache(start: *mut libc::c_char, end: *mut libc::c_char);
}

fn invalidate_icache(addr: usize, size: usize) {
    unsafe {
        #[cfg(target_os = "macos")]
        sys_icache_invalidate(addr as *mut _, size);
        #[cfg(not(target_os = "macos"))]
        __clear_cache(addr as *mut _, (addr + size) as *mut _);
    }
}

fn section_name(p: *const libc::c_char) -> String {
    if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() }
}

// ---------------------------------------------------------------- C callbacks

pub(crate) extern "C" fn create_context(ctx_ctx: *mut libc::c_void) -> *mut libc::c_void {
    // SAFETY: `ctx_ctx` is the `Arc<Shared>` raw pointer owned by the Engine.
    // Each object context keeps its own strong reference.
    unsafe { Arc::<Shared>::increment_strong_count(ctx_ctx as *const Shared) };
    let shared = unsafe { Arc::from_raw(ctx_ctx as *const Shared) };
    let object = shared.pending_objects.lock().pop_front();
    let ctx = Box::new(ObjectContext { shared, allocations: vec![], sections: vec![], object, roots: vec![] });
    Box::into_raw(ctx) as *mut libc::c_void
}

pub(crate) extern "C" fn notify_terminating(_ctx_ctx: *mut libc::c_void) {}

pub(crate) extern "C" fn allocate_code(
    opaque: *mut libc::c_void,
    size: libc::uintptr_t,
    alignment: libc::c_uint,
    _section_id: libc::c_uint,
    name: *const libc::c_char,
) -> *mut u8 {
    let ctx = unsafe { &mut *(opaque as *mut ObjectContext) };
    let p = ctx.shared.allocate(size, alignment as usize);
    if !p.is_null() {
        ctx.allocations.push(Allocation { addr: p as usize, size, code: true, read_only: true });
        ctx.sections.push(LoadedSection { name: section_name(name), addr: p as u64, size: size as u64 });
    }
    p
}

pub(crate) extern "C" fn allocate_data(
    opaque: *mut libc::c_void,
    size: libc::uintptr_t,
    alignment: libc::c_uint,
    _section_id: libc::c_uint,
    name: *const libc::c_char,
    read_only: llvm_sys::prelude::LLVMBool,
) -> *mut u8 {
    let ctx = unsafe { &mut *(opaque as *mut ObjectContext) };
    let p = ctx.shared.allocate(size, alignment as usize);
    if !p.is_null() {
        let ro = read_only != 0;
        ctx.allocations.push(Allocation { addr: p as usize, size, code: false, read_only: ro });
        ctx.sections.push(LoadedSection { name: section_name(name), addr: p as u64, size: size as u64 });
        if !ro {
            // Writable segment: register with the GC (C++: MemProt::Write).
            if let Some(gc) = ctx.shared.gc {
                let (s, e) = (p as usize, p as usize + size);
                unsafe { (gc.add_roots)(s as *mut _, e as *mut _) };
                ctx.roots.push((s, e));
            }
        }
    }
    p
}

pub(crate) extern "C" fn finalize(opaque: *mut libc::c_void, _err: *mut *mut libc::c_char) -> llvm_sys::prelude::LLVMBool {
    let ctx = unsafe { &mut *(opaque as *mut ObjectContext) };
    let ps = page_size();
    for a in &ctx.allocations {
        let len = (a.size.max(1) + ps - 1) & !(ps - 1);
        let prot = if a.code {
            libc::PROT_READ | libc::PROT_EXEC
        } else if a.read_only {
            libc::PROT_READ
        } else {
            libc::PROT_READ | libc::PROT_WRITE
        };
        unsafe { libc::mprotect(a.addr as *mut _, len, prot) };
        if a.code {
            invalidate_icache(a.addr, a.size);
        }
    }
    if let Some(obj) = ctx.object.take() {
        let key = {
            let mut k = ctx.shared.next_key.lock();
            *k += 1;
            *k
        };
        ctx.shared.debug.register(key, obj, std::mem::take(&mut ctx.sections), opaque as usize);
    }
    0
}

pub(crate) extern "C" fn destroy(opaque: *mut libc::c_void) {
    let ctx = unsafe { Box::from_raw(opaque as *mut ObjectContext) };
    if let Some(gc) = ctx.shared.gc {
        for &(s, e) in &ctx.roots {
            unsafe { (gc.remove_roots)(s as *mut _, e as *mut _) };
        }
    }
    ctx.shared.debug.unregister_owner(opaque as usize);
    // PORT-NOTE: memory stays reserved in the slab (bump allocator); C++ releases
    // the mapping. JIT code lives for the process lifetime in Codon anyway.
}
