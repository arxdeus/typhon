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

pub(crate) extern "C" fn finalize(
    opaque: *mut libc::c_void,
    _err: *mut *mut libc::c_char,
) -> llvm_sys::prelude::LLVMBool {
    let ctx = unsafe { &mut *(opaque as *mut ObjectContext) };
    #[cfg(target_os = "macos")]
    fix_pointer_to_got(ctx);
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
        let r = unsafe { libc::mprotect(a.addr as *mut _, len, prot) };
        if std::env::var_os("TYPHON_JIT_DEBUG").is_some() {
            eprintln!("mprotect {:#x}+{:#x} code={} ro={} -> {}", a.addr, len, a.code, a.read_only, r);
        }
        if a.code {
            invalidate_icache(a.addr, a.size);
        }
    }
    // Register DWARF unwind info (RTDyld has already applied relocations).
    let code = ctx.allocations.iter().filter(|a| a.code);
    let lo = code.clone().map(|a| a.addr).min().unwrap_or(0);
    let hi = code.map(|a| a.addr + a.size).max().unwrap_or(0);
    if let Some(eh) = ctx.sections.iter().find(|s| {
        let n = s.name.rsplit(',').next().unwrap_or(&s.name);
        n == "__eh_frame" || n == ".eh_frame"
    }) {
        #[cfg(target_os = "macos")]
        if let Some(obj) = ctx.object.as_deref() {
            use object::{Object, ObjectSection};
            if let Ok(f) = object::File::parse(obj) {
                let addr_of = |n: &str| f.section_by_name(n).map(|s| s.address() as isize);
                let load_of = |n: &str| ctx.sections.iter().find(|s| s.name.ends_with(n)).map(|s| s.addr as isize);
                if let (Some(t_obj), Some(e_obj), Some(t_load)) =
                    (addr_of("__text"), addr_of("__eh_frame"), load_of("__text"))
                {
                    let eh_slide = eh.addr as isize - e_obj;
                    let pc_delta = eh_slide - (t_load - t_obj);
                    let lsda_delta = match (addr_of("__gcc_except_tab"), load_of("__gcc_except_tab")) {
                        (Some(g_obj), Some(g_load)) => eh_slide - (g_load - g_obj),
                        _ => 0,
                    };
                    if std::env::var_os("TYPHON_JIT_DEBUG").is_some() {
                        eprintln!("eh_frame fixup pc_delta={pc_delta:#x} lsda_delta={lsda_delta:#x}");
                    }
                    unsafe {
                        crate::eh_frames::adjust_fde_pcs(eh.addr as usize, eh.size as usize, pc_delta, lsda_delta)
                    };
                }
            }
        }
        crate::eh_frames::register(lo, hi, eh.addr as usize, eh.size as usize, opaque as usize);
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

/// PORT-NOTE: works around an RTDyld bug (absent in C++, which uses JITLink).
/// `RuntimeDyldMachOAArch64` handles `ARM64_RELOC_POINTER_TO_GOT` by zeroing the
/// target offset before creating the GOT stub. Local symbols (e.g. the private
/// `typhon.typeidx.*` globals referenced from the LSDA type table) resolve to
/// section+offset, so every such GOT entry points at the *section base*, and all
/// of them share one stub. Typed `except` clauses then read a wrong type index
/// (search phase matched, cleanup phase did not -> libunwind abort).
/// Here every 32-bit pc-relative POINTER_TO_GOT field gets a fresh, correct
/// 8-byte slot and is re-pointed at it.
#[cfg(target_os = "macos")]
fn fix_pointer_to_got(ctx: &mut ObjectContext) {
    use object::{Object, ObjectSection, ObjectSymbol, RelocationTarget};
    let Some(obj) = ctx.object.as_deref() else { return };
    let Ok(f) = object::File::parse(obj) else { return };
    if f.architecture() != object::Architecture::Aarch64 {
        return;
    }
    // Load address of an object section: match by name and size (names repeat,
    // e.g. two `__const`); stubs may make the loaded copy larger.
    let load_of = |s: &object::Section| -> Option<u64> {
        let n = s.name().ok()?;
        let seg = s.segment_name().ok().flatten().unwrap_or("");
        let full = format!("{seg},{n}");
        let mut best: Option<&LoadedSection> = None;
        for l in &ctx.sections {
            if (l.name == n || l.name == full || l.name.ends_with(&format!(",{n}"))) && l.size >= s.size() {
                if best.is_none_or(|b| l.size < b.size) {
                    best = Some(l);
                }
            }
        }
        best.map(|l| l.addr)
    };
    let mut fixes: Vec<(usize, u64)> = vec![]; // (field address, target address)
    for sec in f.sections() {
        let Some(sec_load) = load_of(&sec) else { continue };
        for (off, r) in sec.relocations() {
            // ARM64_RELOC_POINTER_TO_GOT == 7
            let object::RelocationFlags::MachO { r_type, r_pcrel: true, r_length: 2 } = r.flags() else { continue };
            if r_type != object::macho::ARM64_RELOC_POINTER_TO_GOT {
                continue;
            }
            let RelocationTarget::Symbol(si) = r.target() else { continue };
            let Ok(sym) = f.symbol_by_index(si) else { continue };
            if !sym.is_local() || sym.is_undefined() {
                continue; // external symbols are resolved correctly by name
            }
            let Some(ts) = sym.section_index().and_then(|i| f.section_by_index(i).ok()) else { continue };
            let Some(t_load) = load_of(&ts) else { continue };
            fixes.push(((sec_load + off) as usize, t_load + (sym.address() - ts.address())));
        }
    }
    if fixes.is_empty() {
        return;
    }
    let slots = ctx.shared.allocate(fixes.len() * 8, 8) as usize;
    if slots == 0 {
        return;
    }
    ctx.allocations.push(Allocation { addr: slots, size: fixes.len() * 8, code: false, read_only: true });
    for (i, &(field, target)) in fixes.iter().enumerate() {
        let slot = slots + i * 8;
        let rel = slot as i64 - field as i64;
        if rel < i32::MIN as i64 || rel > i32::MAX as i64 {
            continue;
        }
        unsafe {
            (slot as *mut u64).write(target);
            (field as *mut i32).write_unaligned(rel as i32);
        }
    }
    if std::env::var_os("TYPHON_JIT_DEBUG").is_some() {
        eprintln!("fixed {} POINTER_TO_GOT relocations", fixes.len());
    }
}

pub(crate) extern "C" fn destroy(opaque: *mut libc::c_void) {
    let ctx = unsafe { Box::from_raw(opaque as *mut ObjectContext) };
    if let Some(gc) = ctx.shared.gc {
        for &(s, e) in &ctx.roots {
            unsafe { (gc.remove_roots)(s as *mut _, e as *mut _) };
        }
    }
    ctx.shared.debug.unregister_owner(opaque as usize);
    crate::eh_frames::deregister_owner(opaque as usize);
    // PORT-NOTE: memory stays reserved in the slab (bump allocator); C++ releases
    // the mapping. JIT code lives for the process lifetime in Codon anyway.
}
