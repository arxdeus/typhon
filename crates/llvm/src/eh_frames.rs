//! Registration of JIT-emitted DWARF unwind info with the system unwinder.
//!
//! C++ Codon uses JITLink, whose `EHFrameRegistrationPlugin` (and on Darwin the
//! `UnwindInfoManager`) registers `.eh_frame` for each linked graph.
//!
//! PORT-NOTE: the C API only offers an RTDyld linking layer with MCJIT-like
//! memory manager callbacks and no `registerEHFrames` hook, so the memory
//! manager's finalize callback calls into this module with the relocated
//! `__eh_frame` / `.eh_frame` section:
//! - macOS: a callback installed with `__unw_add_find_dynamic_unwind_sections`
//!   maps code addresses to the DWARF section of the object containing them
//!   (macOS `__register_frame` only accepts single FDEs).
//! - other targets: libgcc/libunwind `__register_frame` on the whole section.

use parking_lot::Mutex;

#[derive(Clone, Copy, Debug)]
struct Entry {
    code_start: usize,
    code_end: usize,
    eh_frame: usize,
    eh_frame_len: usize,
    owner: usize,
}

static ENTRIES: Mutex<Vec<Entry>> = Mutex::new(Vec::new());

#[cfg(target_os = "macos")]
mod apple {
    use super::ENTRIES;

    /// llvm/llvm-project#49036: libunwind needs a mach header as DSO base.
    #[repr(C)]
    pub struct MachHeader64 {
        magic: u32,
        cputype: i32,
        cpusubtype: i32,
        filetype: u32,
        ncmds: u32,
        sizeofcmds: u32,
        flags: u32,
        reserved: u32,
    }

    pub static FAKE_MACH_HEADER: MachHeader64 = MachHeader64 {
        magic: 0xfeed_facf,
        #[cfg(target_arch = "aarch64")]
        cputype: 0x0100_000c,
        #[cfg(not(target_arch = "aarch64"))]
        cputype: 0x0100_0007,
        cpusubtype: 0,
        filetype: 0x6,
        ncmds: 0,
        sizeofcmds: 0,
        flags: 0,
        reserved: 0,
    };

    #[repr(C)]
    pub struct UnwDynamicUnwindSections {
        dso_base: usize,
        dwarf_section: usize,
        dwarf_section_length: usize,
        compact_unwind_section: usize,
        compact_unwind_section_length: usize,
    }

    pub unsafe extern "C" fn find(addr: usize, info: *mut UnwDynamicUnwindSections) -> libc::c_int {
        let entries = ENTRIES.lock();
        let Some(e) = entries.iter().find(|e| e.code_start <= addr && addr < e.code_end) else {
            return 0;
        };
        unsafe {
            if std::env::var_os("TYPHON_JIT_DEBUG").is_some() {
                eprintln!("find {addr:#x}: {:?}", super::fdes(e.eh_frame, e.eh_frame_len));
            }
            (*info).dso_base = &FAKE_MACH_HEADER as *const _ as usize;
            (*info).dwarf_section = e.eh_frame;
            (*info).dwarf_section_length = e.eh_frame_len;
            (*info).compact_unwind_section = 0;
            (*info).compact_unwind_section_length = 0;
        }
        1
    }

    pub fn install() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| unsafe {
            let sym = libc::dlsym(libc::RTLD_DEFAULT, c"__unw_add_find_dynamic_unwind_sections".as_ptr());
            if !sym.is_null() {
                type FindFn = unsafe extern "C" fn(usize, *mut UnwDynamicUnwindSections) -> libc::c_int;
                let add: unsafe extern "C" fn(FindFn) -> libc::c_int = std::mem::transmute(sym);
                add(find);
            }
        });
    }
}

#[cfg(not(target_os = "macos"))]
unsafe extern "C" {
    fn __register_frame(begin: *const u8);
    fn __deregister_frame(begin: *const u8);
}

/// Size in bytes of an FDE pointer with DWARF encoding `enc` (low nibble).
fn ptr_size(enc: u8) -> Option<usize> {
    match enc & 0x0f {
        0x00 | 0x04 | 0x0c => Some(8),
        0x03 | 0x0b => Some(4),
        0x02 | 0x0a => Some(2),
        _ => None,
    }
}

unsafe fn read_uleb(p: &mut *const u8) -> u64 {
    let (mut r, mut s) = (0u64, 0);
    loop {
        let b = unsafe { **p };
        *p = unsafe { p.add(1) };
        r |= ((b & 0x7f) as u64) << s;
        s += 7;
        if b & 0x80 == 0 {
            return r;
        }
    }
}

/// FDE pointer and LSDA encodings from a CIE at `cie` (`R`/`L` augmentation).
unsafe fn cie_encodings(cie: *const u8) -> (u8, Option<u8>) {
    unsafe {
        let mut p = cie.add(9); // length(4) id(4) version(1)
        let aug = std::ffi::CStr::from_ptr(p.cast()).to_bytes().to_vec();
        p = p.add(aug.len() + 1);
        read_uleb(&mut p); // code align
        read_uleb(&mut p); // data align (sleb, only skipping)
        read_uleb(&mut p); // return register
        let (mut r, mut l) = (0u8, None);
        if aug.first() != Some(&b'z') {
            return (r, l);
        }
        read_uleb(&mut p); // augmentation length
        for &c in &aug[1..] {
            match c {
                b'R' => {
                    r = *p;
                    p = p.add(1);
                }
                b'L' => {
                    l = Some(*p);
                    p = p.add(1);
                }
                b'P' => {
                    let e = *p;
                    p = p.add(1 + ptr_size(e).unwrap_or(8));
                }
                _ => {}
            }
        }
        (r, l)
    }
}

unsafe fn add_pcrel(p: *mut u8, enc: u8, delta: isize) {
    if enc & 0x70 != 0x10 {
        return;
    }
    unsafe {
        match ptr_size(enc) {
            Some(8) => {
                let v = p.cast::<i64>().read_unaligned();
                p.cast::<i64>().write_unaligned(v.wrapping_add(delta as i64));
            }
            Some(4) => {
                let v = p.cast::<i32>().read_unaligned();
                p.cast::<i32>().write_unaligned(v.wrapping_add(delta as i32));
            }
            _ => {}
        }
    }
}

/// Undo a second section-delta adjustment of every FDE's pcrel `pc_begin`.
///
/// PORT-NOTE: on MachO, RTDyld resolves the explicit SUBTRACTOR relocations of
/// `__eh_frame` and then `RuntimeDyldMachO::processFDE` (run from
/// `registerEHFrames`, before the memory manager's finalize) adds
/// `(text_load - text_obj) - (eh_load - eh_obj)` again, so PCs end up shifted.
/// JITLink (C++ Codon) has no such step. `delta` is the amount to add back.
pub(crate) unsafe fn adjust_fde_pcs(base: usize, len: usize, pc_delta: isize, lsda_delta: isize) {
    if pc_delta == 0 && lsda_delta == 0 {
        return;
    }
    let mut off = 0usize;
    while off + 8 <= len {
        let rec = (base + off) as *const u8;
        let l = unsafe { rec.cast::<u32>().read_unaligned() } as usize;
        if l == 0 || l == 0xffff_ffff {
            break;
        }
        let id = unsafe { rec.add(4).cast::<u32>().read_unaligned() } as usize;
        if id != 0 {
            let cie = unsafe { rec.add(4).sub(id) };
            let (enc, lsda_enc) = unsafe { cie_encodings(cie) };
            let p = unsafe { rec.add(8) } as *mut u8;
            unsafe { add_pcrel(p, enc, pc_delta) };
            if let (Some(le), Some(sz)) = (lsda_enc, ptr_size(enc)) {
                // pc_begin, pc_range, augmentation length, LSDA pointer
                let mut q = unsafe { p.add(2 * sz) } as *const u8;
                unsafe { read_uleb(&mut q) };
                let lp = q as *mut u8;
                let is_null = match ptr_size(le) {
                    Some(8) => unsafe { lp.cast::<u64>().read_unaligned() == 0 },
                    Some(4) => unsafe { lp.cast::<u32>().read_unaligned() == 0 },
                    _ => true,
                };
                if le != 0xff && !is_null {
                    unsafe { add_pcrel(lp, le, lsda_delta) };
                }
            }
        }
        off += 4 + l;
    }
}

/// Debug helper: walk FDEs assuming pcrel|absptr (8-byte) pointer encoding,
/// which LLVM emits for the Large code model used by the JIT. Returns `(offset, pc_begin, pc_range)`.
unsafe fn fdes(base: usize, len: usize) -> Vec<(usize, usize, usize)> {
    let mut out = vec![];
    let mut off = 0usize;
    while off + 8 <= len {
        let l = unsafe { (base as *const u8).add(off).cast::<u32>().read_unaligned() } as usize;
        if l == 0 || l == 0xffff_ffff {
            break;
        }
        let id = unsafe { (base as *const u8).add(off + 4).cast::<u32>().read_unaligned() };
        if id != 0 {
            let p = base + off + 8;
            let rel = unsafe { (p as *const i64).read_unaligned() } as isize;
            let range = unsafe { ((p + 8) as *const u64).read_unaligned() } as usize;
            out.push((off, (p as isize + rel) as usize, range));
        }
        off += 4 + l;
    }
    out
}

/// Register the (relocated) eh_frame section of one linked object whose code
/// spans `[code_start, code_end)`.
pub(crate) fn register(code_start: usize, code_end: usize, eh_frame: usize, eh_frame_len: usize, owner: usize) {
    if eh_frame == 0 || eh_frame_len == 0 {
        return;
    }
    if std::env::var_os("TYPHON_JIT_DEBUG").is_some() {
        eprintln!("eh_frame {eh_frame:#x}+{eh_frame_len:#x} code {code_start:#x}..{code_end:#x}");
        for (off, pc, len) in unsafe { fdes(eh_frame, eh_frame_len) } {
            eprintln!("  fde@{off:#x} pc={pc:#x} len={len:#x}");
        }
    }
    #[cfg(target_os = "macos")]
    apple::install();
    #[cfg(not(target_os = "macos"))]
    unsafe {
        __register_frame(eh_frame as *const u8)
    };
    ENTRIES.lock().push(Entry { code_start, code_end, eh_frame, eh_frame_len, owner });
}

/// Deregister all frames owned by one object context.
pub(crate) fn deregister_owner(owner: usize) {
    let mut entries = ENTRIES.lock();
    entries.retain(|e| {
        if e.owner != owner {
            return true;
        }
        #[cfg(not(target_os = "macos"))]
        unsafe {
            __deregister_frame(e.eh_frame as *const u8)
        };
        false
    });
}
