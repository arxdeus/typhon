//! Native exceptions for compiled Typhon code: SEH on MSVC, Itanium elsewhere.
//! (`seq_alloc_exc`, `seq_throw`, `seq_personality` with hand-written LSDA parsing,
//! `seq_terminate`) and backtraces.

use core::ffi::{c_int, c_void};

use crate::lib_rt::{flags, seq_alloc, seq_alloc_atomic, seq_free};
use crate::rtti::{BASE_EXCEPTION_SELECTOR, CATCH_ALL_SELECTOR, TypeInfo, isinstance};
use crate::types::*;

/*
 * System unwinder ABI (libunwind in libSystem on macOS, libgcc_s on Linux).
 */

pub type UnwindReasonCode = c_int;
pub const URC_NO_REASON: UnwindReasonCode = 0;
pub const URC_FATAL_PHASE1_ERROR: UnwindReasonCode = 3;
pub const URC_END_OF_STACK: UnwindReasonCode = 5;
pub const URC_HANDLER_FOUND: UnwindReasonCode = 6;
pub const URC_INSTALL_CONTEXT: UnwindReasonCode = 7;
pub const URC_CONTINUE_UNWIND: UnwindReasonCode = 8;

pub type UnwindAction = c_int;
pub const UA_SEARCH_PHASE: UnwindAction = 1;
pub const UA_CLEANUP_PHASE: UnwindAction = 2;
pub const UA_HANDLER_FRAME: UnwindAction = 4;
pub const UA_FORCE_UNWIND: UnwindAction = 8;

pub type UnwindCleanupFn = unsafe extern "C" fn(UnwindReasonCode, *mut UnwindException);

/// `struct _Unwind_Exception` (declared `__attribute__((aligned))`, i.e. 16 bytes).
#[repr(C, align(16))]
pub struct UnwindException {
    pub exception_class: u64,
    pub exception_cleanup: Option<UnwindCleanupFn>,
    pub private_1: usize,
    pub private_2: usize,
}

#[repr(C)]
pub struct UnwindContext {
    _private: [u8; 0],
}

#[cfg(not(all(target_os = "windows", target_env = "msvc")))]
unsafe extern "C" {
    fn _Unwind_RaiseException(exc: *mut UnwindException) -> UnwindReasonCode;
    fn _Unwind_GetIP(ctx: *mut UnwindContext) -> usize;
    fn _Unwind_GetRegionStart(ctx: *mut UnwindContext) -> usize;
    fn _Unwind_GetLanguageSpecificData(ctx: *mut UnwindContext) -> *const u8;
    fn _Unwind_SetGR(ctx: *mut UnwindContext, reg: c_int, value: usize);
    fn _Unwind_SetIP(ctx: *mut UnwindContext, value: usize);
}

// Keep the envelope layout and seq_exc_offset identical on both ABIs. On MSVC
// the embedded UnwindException is an opaque payload, not a system unwind record.
// This follows codon/runtime/exc.cpp (Windows support commit 494bbe8d7).
#[cfg(any(test, all(target_os = "windows", target_env = "msvc")))]
mod seh {
    use super::*;
    use core::cell::Cell;

    const CODON_SEH_CODE: u32 = 0xE0C0_D047;
    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    const EXCEPTION_NONCONTINUABLE: u32 = 1;

    std::thread_local! {
        static CURRENT: Cell<*mut c_void> = const { Cell::new(core::ptr::null_mut()) };
    }

    // Only ExceptionCode is inspected, but describe the complete Win32 layout so
    // tests also validate the pointer indirection used by __C_specific_handler.
    #[repr(C)]
    struct ExceptionRecord {
        code: u32,
        flags: u32,
        record: *mut ExceptionRecord,
        address: *mut c_void,
        parameter_count: u32,
        information: [usize; 15],
    }

    #[repr(C)]
    struct ExceptionPointers {
        record: *const ExceptionRecord,
        context: *mut c_void,
    }

    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    #[link(name = "kernel32")]
    unsafe extern "system-unwind" {
        fn RaiseException(code: u32, flags: u32, count: u32, arguments: *const usize);
    }

    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    pub(super) unsafe fn raise(exc: *mut c_void) {
        CURRENT.set(exc);
        // SEH must be allowed to unwind this Rust frame into generated funclets.
        unsafe { RaiseException(CODON_SEH_CODE, EXCEPTION_NONCONTINUABLE, 0, core::ptr::null()) };
    }

    #[cfg_attr(all(target_os = "windows", target_env = "msvc"), unsafe(no_mangle))]
    pub extern "C" fn seq_exc_current() -> *mut c_void {
        CURRENT.get()
    }

    /// # Safety
    /// `info` is null or a valid Win32 EXCEPTION_POINTERS supplied by SEH.
    #[cfg_attr(all(target_os = "windows", target_env = "msvc"), unsafe(no_mangle))]
    pub unsafe extern "C" fn seq_exc_filter(info: *mut c_void, _frame: *mut c_void) -> i32 {
        let info = info.cast::<ExceptionPointers>();
        if info.is_null() {
            return 0;
        }
        unsafe {
            let record = (*info).record;
            i32::from(!record.is_null() && (*record).code == CODON_SEH_CODE)
        }
    }

    /// Return the first matching clause's selector, exactly as the Itanium
    /// personality does. Zero represents either catch-all or no typed match.
    /// # Safety
    /// `exc` is null or an allocated Typhon exception. `ids` has `n` entries.
    #[cfg_attr(all(target_os = "windows", target_env = "msvc"), unsafe(no_mangle))]
    pub unsafe extern "C" fn seq_exc_match(exc: *mut c_void, ids: *const i32, n: i32) -> i32 {
        if exc.is_null() || n <= 0 {
            return 0;
        }
        unsafe {
            let base = &*base_of(exc);
            for &ty in core::slice::from_raw_parts(ids, n as usize) {
                if ty == CATCH_ALL_SELECTOR || ty == BASE_EXCEPTION_SELECTOR || isinstance(base.obj, ty as SeqInt) {
                    return ty;
                }
            }
        }
        0
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn filter_accepts_only_private_exception_code() {
            unsafe {
                assert_eq!(seq_exc_filter(core::ptr::null_mut(), core::ptr::null_mut()), 0);
                let mut info = ExceptionPointers { record: core::ptr::null(), context: core::ptr::null_mut() };
                let p = (&mut info as *mut ExceptionPointers).cast();
                assert_eq!(seq_exc_filter(p, core::ptr::null_mut()), 0);
                // Access violations and C++ exceptions are not language exceptions.
                for code in [0, 0xC000_0005, 0xE06D_7363, CODON_SEH_CODE] {
                    let record = ExceptionRecord { code, ..core::mem::zeroed() };
                    let mut info = ExceptionPointers { record: &record, context: core::ptr::null_mut() };
                    let p = (&mut info as *mut ExceptionPointers).cast();
                    assert_eq!(seq_exc_filter(p, core::ptr::null_mut()), i32::from(code == CODON_SEH_CODE));
                }
            }
        }

        #[test]
        fn current_payload_is_thread_local_and_replaceable() {
            assert!(seq_exc_current().is_null());
            let mut payload = 42u8;
            let p = (&mut payload as *mut u8).cast();
            CURRENT.set(p);
            assert_eq!(seq_exc_current(), p);
            std::thread::spawn(|| {
                assert!(seq_exc_current().is_null());
                let mut nested = 1u8;
                let p = (&mut nested as *mut u8).cast();
                CURRENT.set(p);
                assert_eq!(seq_exc_current(), p);
                CURRENT.set(core::ptr::null_mut());
            }).join().unwrap();
            assert_eq!(seq_exc_current(), p);
            CURRENT.set(core::ptr::null_mut());
        }

        #[test]
        fn selector_obeys_clause_order_and_inheritance() {
            let mro = [5, 3, 1];
            let info = TypeInfo {
                id: 5, n_mro: 3, mro: mro.as_ptr(), offsets: core::ptr::null(),
                sorted_mro: core::ptr::null(), n_slots: 0, slots: core::ptr::null(),
                n_methods: 0, methods: core::ptr::null(), rtti: true,
                raw_name: SeqStr::EMPTY, nice_name: SeqStr::EMPTY, base_name: SeqStr::EMPTY,
                n_params: 0, params: core::ptr::null(), size: 0, reference: true,
                inline_value: false, exception_offset: -1, system_exit_offset: -1,
            };
            let mut obj = &info as *const TypeInfo;
            let mut envelope: BaseExceptionData = unsafe { core::mem::zeroed() };
            envelope.obj = (&mut obj as *mut *const TypeInfo).cast();
            envelope.unwind_exception.exception_class = SEQ_EXCEPTION_CLASS;
            let exc = (&mut envelope.unwind_exception as *mut UnwindException).cast();
            unsafe {
                assert_eq!((*base_of(exc)).obj, envelope.obj);
                for (ids, expected) in [
                    (&[7, 5][..], 5), (&[3, 5][..], 3), (&[5, 3][..], 5),
                    (&[7, 1][..], 1), (&[7, 8][..], 0),
                    (&[BASE_EXCEPTION_SELECTOR, 5][..], BASE_EXCEPTION_SELECTOR),
                    (&[CATCH_ALL_SELECTOR, 5][..], CATCH_ALL_SELECTOR),
                ] {
                    assert_eq!(seq_exc_match(exc, ids.as_ptr(), ids.len() as i32), expected);
                }
                assert_eq!(seq_exc_match(core::ptr::null_mut(), core::ptr::null(), 0), 0);
                assert_eq!(seq_exc_match(exc, core::ptr::null(), 0), 0);
                assert_eq!(seq_exc_match(exc, core::ptr::null(), -1), 0);
            }
        }
    }
}

#[cfg(all(target_os = "windows", target_env = "msvc"))]
pub use seh::{seq_exc_current, seq_exc_filter, seq_exc_match};

/// `__builtin_eh_return_data_regno(0/1)`.
#[cfg(not(all(target_os = "windows", target_env = "msvc")))]
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64"))]
const EH_DATA_REG: [c_int; 2] = if cfg!(target_arch = "riscv64") { [10, 11] } else { [0, 1] };
#[cfg(not(all(target_os = "windows", target_env = "msvc")))]
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64", target_arch = "riscv64")))]
const EH_DATA_REG: [c_int; 2] = [0, 1];

/*
 * Backtraces
 */

#[repr(C)]
#[derive(Clone, Copy)]
pub struct BacktraceFrame {
    pub function: *mut u8,
    pub filename: *mut u8,
    pub pc: usize,
    pub lineno: i32,
}

#[repr(C)]
pub struct Backtrace {
    pub frames: *mut BacktraceFrame,
    pub count: usize,
}

impl Backtrace {
    pub const LIMIT: usize = 20;

    fn gc_cstr(s: &[u8]) -> *mut u8 {
        let p = seq_alloc_atomic(s.len() + 1) as *mut u8;
        unsafe {
            core::ptr::copy_nonoverlapping(s.as_ptr(), p, s.len());
            *p.add(s.len()) = 0;
        }
        p
    }

    fn push_back(&mut self, function: Option<&[u8]>, filename: Option<&[u8]>, pc: usize, lineno: i32) {
        let (Some(function), Some(filename)) = (function, filename) else { return };
        if self.count >= Self::LIMIT {
            return;
        }
        if self.count == 0 {
            self.frames = seq_alloc(Self::LIMIT * core::mem::size_of::<BacktraceFrame>()) as *mut _;
        }
        let f = BacktraceFrame { function: Self::gc_cstr(function), filename: Self::gc_cstr(filename), pc, lineno };
        unsafe { self.frames.add(self.count).write(f) };
        self.count += 1;
    }

    fn free(&mut self) {
        for i in 0..self.count {
            unsafe {
                let f = &*self.frames.add(i);
                seq_free(f.function.cast());
                seq_free(f.filename.cast());
            }
        }
        seq_free(self.frames.cast());
        self.frames = core::ptr::null_mut();
        self.count = 0;
    }

    fn frames(&self) -> &[BacktraceFrame] {
        if self.count == 0 { &[] } else { unsafe { core::slice::from_raw_parts(self.frames, self.count) } }
    }

    /// Capture with symbolization (libbacktrace `backtrace_full` equivalent).
    fn capture_full(&mut self, skip: usize) {
        // PORT-NOTE: libbacktrace -> `backtrace` crate. Frames lacking a function or
        // file name are dropped, exactly like the C++ full callback.
        let mut skipped = 0;
        backtrace::trace(|frame| {
            if skipped < skip + 1 {
                skipped += 1;
                return true;
            }
            let pc = frame.ip() as usize;
            backtrace::resolve_frame(frame, |sym| {
                let name = sym.name().map(|n| n.to_string());
                let file = sym.filename().map(|f| f.to_string_lossy().into_owned());
                self.push_back(
                    name.as_deref().map(str::as_bytes),
                    file.as_deref().map(str::as_bytes),
                    pc,
                    sym.lineno().unwrap_or(0) as i32,
                );
            });
            self.count < Self::LIMIT
        });
    }

    /// Capture raw PCs only (libbacktrace `backtrace_simple` equivalent).
    fn capture_simple(&mut self, skip: usize) {
        let mut skipped = 0;
        backtrace::trace(|frame| {
            if skipped < skip + 1 {
                skipped += 1;
                return true;
            }
            self.push_back(Some(b"<invalid>"), Some(b"<invalid>"), frame.ip() as usize, 0);
            self.count < Self::LIMIT
        });
    }
}

/*
 * Exception objects
 */

#[repr(C)]
pub struct BaseExceptionTypeData {
    pub ty: c_int,
}

#[repr(C)]
pub struct BaseExceptionData {
    pub obj: *mut c_void,
    pub bt: Backtrace,
    pub unwind_exception: UnwindException,
}

#[repr(C)]
pub struct ExceptionHeader {
    pub msg: SeqStr,
    pub func: SeqStr,
    pub file: SeqStr,
    pub line: SeqInt,
    pub col: SeqInt,
    pub cause: *mut c_void,
}

#[repr(C)]
pub struct SystemExitData {
    pub status: SeqInt,
    pub message_exit: bool,
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_exc_offset() -> i64 {
    -(core::mem::offset_of!(BaseExceptionData, unwind_exception) as i64)
}

#[inline]
unsafe fn base_of(exc: *mut c_void) -> *mut BaseExceptionData {
    unsafe { (exc as *mut u8).offset(seq_exc_offset() as isize) as *mut BaseExceptionData }
}

/// Executables carry their own unwind tables, so there is nothing to register.
pub fn seq_exc_init(_flags: c_int) {}

unsafe fn seq_delete_exc(exp: *mut UnwindException) {
    unsafe {
        if exp.is_null() || (*exp).exception_class != SEQ_EXCEPTION_CLASS {
            return;
        }
        let exc = base_of(exp.cast());
        if flags() & SEQ_FLAG_DEBUG != 0 {
            (*exc).bt.free();
        }
        seq_free(exc.cast());
    }
}

unsafe extern "C" fn seq_delete_unwind_exc(_reason: UnwindReasonCode, exp: *mut UnwindException) {
    unsafe { seq_delete_exc(exp) }
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_alloc_exc(obj: *mut c_void) -> *mut c_void {
    let size = core::mem::size_of::<BaseExceptionData>();
    let e = seq_alloc(size) as *mut BaseExceptionData;
    assert!(!e.is_null());
    unsafe {
        core::ptr::write_bytes(e as *mut u8, 0, size);
        (*e).obj = obj;
        (*e).unwind_exception.exception_class = SEQ_EXCEPTION_CLASS;
        (*e).unwind_exception.exception_cleanup = Some(seq_delete_unwind_exc);
        let f = flags();
        if f & SEQ_FLAG_DEBUG != 0 {
            (*e).bt.frames = core::ptr::null_mut();
            (*e).bt.count = 0;
            if f & SEQ_FLAG_STANDALONE != 0 {
                (*e).bt.capture_full(1);
            } else {
                (*e).bt.capture_simple(1);
            }
        }
        (&mut (*e).unwind_exception as *mut UnwindException).cast()
    }
}

/// Format one backtrace frame.
pub fn make_backtrace_frame_string(pc: usize, func: &str, file: &str, line: i32, col: i32) -> String {
    let mut buf = format!("[\x1b[33m0x{pc:x}\x1b[0m]");
    if !func.is_empty() {
        buf += &format!(" \x1b[32m{func}\x1b[0m");
        if !file.is_empty() {
            buf += &format!(" at \x1b[36m{file}\x1b[0m");
            if line != 0 {
                buf += &format!(":\x1b[33m{line}\x1b[0m");
                if col != 0 {
                    buf += &format!(":\x1b[33m{col}\x1b[0m");
                }
            }
        }
    }
    buf
}

fn from_last_dot(name: &str) -> &str {
    name.rfind('.').map_or(name, |i| &name[i + 1..])
}

fn write_stderr(b: &[u8]) {
    use std::io::Write;
    let mut e = std::io::stderr();
    let _ = e.write_all(b);
    let _ = e.flush();
}

/// # Safety
/// `exc` must be a pointer returned by `seq_alloc_exc`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_terminate(exc: *mut c_void) {
    unsafe {
        let base = &*base_of(exc);
        let obj = base.obj;
        let info = &**(obj as *const *const TypeInfo);
        assert!(info.exception_offset >= 0);
        let hdr = &*((obj as *mut u8).add(info.exception_offset as usize) as *const ExceptionHeader);

        if info.system_exit_offset >= 0 {
            let data = &*((obj as *mut u8).add(info.system_exit_offset as usize) as *const SystemExitData);
            if data.message_exit {
                let mut m = hdr.msg.encode();
                m.push(b'\n');
                write_stderr(&m);
            }
            libc::exit(data.status as c_int);
        }

        let ty = info.nice_name.to_string_lossy();
        let msg = hdr.msg.to_string_lossy();
        let func = hdr.func.to_string_lossy();
        let file = hdr.file.to_string_lossy();
        let f = flags();

        let mut buf = String::new();
        buf += "\x1b[1m";
        buf += from_last_dot(&ty);
        if !msg.is_empty() {
            buf += ": \x1b[0m";
            buf += &msg;
        } else {
            buf += "\x1b[0m";
        }
        buf += "\n\n\x1b[1mRaised from:\x1b[0m \x1b[32m";
        buf += &func;
        buf += "\x1b[0m\n";
        buf += &file;
        if hdr.line > 0 {
            buf += &format!(":{}", hdr.line);
            if hdr.col > 0 {
                buf += &format!(":{}", hdr.col);
            }
        }
        buf += "\n";

        if f & SEQ_FLAG_DEBUG != 0 && f & SEQ_FLAG_STANDALONE != 0 && base.bt.count > 0 {
            buf += "\n\x1b[1mBacktrace:\x1b[0m\n";
            for fr in base.bt.frames() {
                let fname = std::ffi::CStr::from_ptr(fr.function.cast()).to_string_lossy();
                let fl = std::ffi::CStr::from_ptr(fr.filename.cast()).to_string_lossy();
                buf += "  ";
                buf += &make_backtrace_frame_string(fr.pc, &fname, &fl, fr.lineno, 0);
                buf += "\n";
            }
        }

        write_stderr(buf.as_bytes());
        libc::abort();
    }
}

/// # Safety
/// `exc` must be a pointer returned by `seq_alloc_exc`.
#[cfg(not(all(target_os = "windows", target_env = "msvc")))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_throw(exc: *mut c_void) {
    unsafe {
        let _ = _Unwind_RaiseException(exc as *mut UnwindException);
        seq_terminate(exc);
    }
}

/// # Safety
/// `exc` must be a pointer returned by `seq_alloc_exc`.
#[cfg(all(target_os = "windows", target_env = "msvc"))]
#[unsafe(no_mangle)]
pub unsafe extern "C-unwind" fn seq_throw(exc: *mut c_void) {
    unsafe {
        seh::raise(exc);
        seq_terminate(exc);
    }
}

/*
 * LSDA parsing (based on llvm/examples/ExceptionDemo)
 */

pub const DW_EH_PE_ABSPTR: u8 = 0x00;
pub const DW_EH_PE_ULEB128: u8 = 0x01;
pub const DW_EH_PE_UDATA2: u8 = 0x02;
pub const DW_EH_PE_UDATA4: u8 = 0x03;
pub const DW_EH_PE_UDATA8: u8 = 0x04;
pub const DW_EH_PE_SLEB128: u8 = 0x09;
pub const DW_EH_PE_SDATA2: u8 = 0x0A;
pub const DW_EH_PE_SDATA4: u8 = 0x0B;
pub const DW_EH_PE_SDATA8: u8 = 0x0C;
pub const DW_EH_PE_PCREL: u8 = 0x10;
pub const DW_EH_PE_INDIRECT: u8 = 0x80;
pub const DW_EH_PE_OMIT: u8 = 0xFF;

/// # Safety
/// `*p` must point to a valid ULEB128.
pub unsafe fn read_uleb128(p: &mut *const u8) -> usize {
    let mut result = 0usize;
    let mut shift = 0u32;
    loop {
        let byte = unsafe { **p };
        *p = unsafe { p.add(1) };
        if shift < usize::BITS {
            result |= ((byte & 0x7f) as usize) << shift;
        }
        shift += 7;
        if byte & 0x80 == 0 {
            break;
        }
    }
    result
}

/// # Safety
/// `*p` must point to a valid SLEB128.
pub unsafe fn read_sleb128(p: &mut *const u8) -> usize {
    let mut result = 0usize;
    let mut shift = 0u32;
    let mut byte;
    loop {
        byte = unsafe { **p };
        *p = unsafe { p.add(1) };
        if shift < usize::BITS {
            result |= ((byte & 0x7f) as usize) << shift;
        }
        shift += 7;
        if byte & 0x80 == 0 {
            break;
        }
    }
    if byte & 0x40 != 0 && shift < usize::BITS {
        result |= !0usize << shift;
    }
    result
}

pub fn encoding_size(enc: u8) -> usize {
    if enc == DW_EH_PE_OMIT {
        return 0;
    }
    match enc & 0x0F {
        DW_EH_PE_ABSPTR => core::mem::size_of::<usize>(),
        DW_EH_PE_UDATA2 | DW_EH_PE_SDATA2 => 2,
        DW_EH_PE_UDATA4 | DW_EH_PE_SDATA4 => 4,
        DW_EH_PE_UDATA8 | DW_EH_PE_SDATA8 => 8,
        _ => std::process::abort(),
    }
}

/// # Safety
/// `*data` must point to a valid encoded pointer.
pub unsafe fn read_encoded_pointer(data: &mut *const u8, enc: u8) -> usize {
    if enc == DW_EH_PE_OMIT {
        return 0;
    }
    let start = *data;
    let mut p = *data;
    unsafe {
        let mut result = match enc & 0x0F {
            DW_EH_PE_ABSPTR => {
                let v = (p as *const usize).read_unaligned();
                p = p.add(8.min(core::mem::size_of::<usize>()));
                v
            }
            DW_EH_PE_ULEB128 => read_uleb128(&mut p),
            DW_EH_PE_SLEB128 => read_sleb128(&mut p),
            DW_EH_PE_UDATA2 => {
                let v = (p as *const u16).read_unaligned() as usize;
                p = p.add(2);
                v
            }
            DW_EH_PE_UDATA4 => {
                let v = (p as *const u32).read_unaligned() as usize;
                p = p.add(4);
                v
            }
            DW_EH_PE_UDATA8 => {
                let v = (p as *const u64).read_unaligned() as usize;
                p = p.add(8);
                v
            }
            DW_EH_PE_SDATA2 => {
                let v = (p as *const i16).read_unaligned() as isize as usize;
                p = p.add(2);
                v
            }
            DW_EH_PE_SDATA4 => {
                let v = (p as *const i32).read_unaligned() as isize as usize;
                p = p.add(4);
                v
            }
            DW_EH_PE_SDATA8 => {
                let v = (p as *const i64).read_unaligned() as isize as usize;
                p = p.add(8);
                v
            }
            _ => std::process::abort(),
        };
        match enc & 0x70 {
            DW_EH_PE_ABSPTR => {}
            DW_EH_PE_PCREL => result = result.wrapping_add(start as usize),
            _ => std::process::abort(),
        }
        if enc & DW_EH_PE_INDIRECT != 0 {
            result = *(result as *const usize);
        }
        *data = p;
        result
    }
}

#[cfg(not(all(target_os = "windows", target_env = "msvc")))]
unsafe fn handle_action_value(
    result_action: &mut i64,
    ttype_encoding: u8,
    class_info: *const u8,
    action_entry: usize,
    exception_class: u64,
    exception_object: *mut UnwindException,
) -> bool {
    if exception_object.is_null() || exception_class != SEQ_EXCEPTION_CLASS {
        return false;
    }
    unsafe {
        let excp = &*base_of(exception_object.cast());
        let mut action_pos = action_entry as *const u8;
        loop {
            let type_offset = read_sleb128(&mut action_pos) as i64;
            let mut temp = action_pos;
            let action_offset = read_sleb128(&mut temp) as i64;
            debug_assert!(type_offset >= 0);
            if type_offset > 0 {
                let enc_size = encoding_size(ttype_encoding);
                let mut entry = class_info.offset(-(type_offset as isize) * enc_size as isize);
                let p = read_encoded_pointer(&mut entry, ttype_encoding);
                let this_type = (*(p as *const BaseExceptionTypeData)).ty;
                if this_type == CATCH_ALL_SELECTOR
                    || this_type == BASE_EXCEPTION_SELECTOR
                    || isinstance(excp.obj, this_type as SeqInt)
                {
                    *result_action = this_type as i64;
                    return true;
                }
            }
            if action_offset == 0 {
                return false;
            }
            action_pos = action_pos.offset(action_offset as isize);
        }
    }
}

#[cfg(not(all(target_os = "windows", target_env = "msvc")))]
unsafe fn handle_lsda(
    mut lsda: *const u8,
    actions: UnwindAction,
    exception_class: u64,
    exception_object: *mut UnwindException,
    context: *mut UnwindContext,
) -> UnwindReasonCode {
    let mut ret = URC_CONTINUE_UNWIND;
    if lsda.is_null() {
        return ret;
    }
    unsafe {
        let pc = _Unwind_GetIP(context) - 1;
        let func_start = _Unwind_GetRegionStart(context);
        let pc_offset = pc.wrapping_sub(func_start);
        let mut class_info: *const u8 = core::ptr::null();

        let lp_start_encoding = *lsda;
        lsda = lsda.add(1);
        if lp_start_encoding != DW_EH_PE_OMIT {
            read_encoded_pointer(&mut lsda, lp_start_encoding);
        }
        let ttype_encoding = *lsda;
        lsda = lsda.add(1);
        if ttype_encoding != DW_EH_PE_OMIT {
            let off = read_uleb128(&mut lsda);
            class_info = lsda.add(off);
        }

        let call_site_encoding = *lsda;
        lsda = lsda.add(1);
        let table_len = read_uleb128(&mut lsda) as u32 as usize;
        let table_end = lsda.add(table_len);
        let action_table_start = table_end;
        let mut csp = lsda;

        while csp < table_end {
            let start = read_encoded_pointer(&mut csp, call_site_encoding);
            let length = read_encoded_pointer(&mut csp, call_site_encoding);
            let landing_pad = read_encoded_pointer(&mut csp, call_site_encoding);
            let mut action_entry = read_uleb128(&mut csp);

            if exception_class != SEQ_EXCEPTION_CLASS {
                // Foreign exception: only run cleanup landing pads.
                action_entry = 0;
            }
            if landing_pad == 0 {
                continue;
            }
            if action_entry != 0 {
                action_entry += action_table_start as usize - 1;
            }

            if start <= pc_offset && pc_offset < start + length {
                let mut action_value = 0i64;
                let mut matched = false;
                if action_entry != 0 {
                    matched = handle_action_value(
                        &mut action_value,
                        ttype_encoding,
                        class_info,
                        action_entry,
                        exception_class,
                        exception_object,
                    );
                }
                if actions & UA_SEARCH_PHASE == 0 {
                    _Unwind_SetGR(context, EH_DATA_REG[0], exception_object as usize);
                    let sel = if action_entry == 0 || !matched { 0 } else { action_value as usize };
                    _Unwind_SetGR(context, EH_DATA_REG[1], sel);
                    _Unwind_SetIP(context, func_start + landing_pad);
                    ret = URC_INSTALL_CONTEXT;
                } else if matched {
                    ret = URC_HANDLER_FOUND;
                }
                break;
            }
        }
    }
    ret
}

/// # Safety
/// Called only by the system unwinder.
#[unsafe(no_mangle)]
#[cfg(not(all(target_os = "windows", target_env = "msvc")))]
pub unsafe extern "C" fn seq_personality(
    _version: c_int,
    actions: UnwindAction,
    exception_class: u64,
    exception_object: *mut UnwindException,
    context: *mut UnwindContext,
) -> UnwindReasonCode {
    unsafe {
        let lsda = _Unwind_GetLanguageSpecificData(context);
        handle_lsda(lsda, actions, exception_class, exception_object, context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout() {
        assert_eq!(core::mem::size_of::<UnwindException>(), 32);
        assert_eq!(seq_exc_offset(), -32);
    }

    #[test]
    fn leb128() {
        let d = [0xE5u8, 0x8E, 0x26];
        let mut p = d.as_ptr();
        assert_eq!(unsafe { read_uleb128(&mut p) }, 624485);
        let d = [0x7Fu8];
        let mut p = d.as_ptr();
        assert_eq!(unsafe { read_sleb128(&mut p) } as isize, -1);
        let d = [0x80u8, 0x7F];
        let mut p = d.as_ptr();
        assert_eq!(unsafe { read_sleb128(&mut p) } as isize, -128);
    }

    #[test]
    fn encoded_pointers() {
        let d = [0x10u8, 0, 0, 0];
        let mut p = d.as_ptr();
        assert_eq!(unsafe { read_encoded_pointer(&mut p, DW_EH_PE_UDATA4) }, 16);
        assert_eq!(p as usize - d.as_ptr() as usize, 4);
        let d = [0xFCu8, 0xFF, 0xFF, 0xFF];
        let mut p = d.as_ptr();
        let v = unsafe { read_encoded_pointer(&mut p, DW_EH_PE_SDATA4 | DW_EH_PE_PCREL) };
        assert_eq!(v, d.as_ptr() as usize - 4);
        assert_eq!(encoding_size(DW_EH_PE_UDATA4), 4);
    }

    #[test]
    fn frame_string() {
        let s = make_backtrace_frame_string(0xab, "f", "x.py", 3, 0);
        assert_eq!(s, "[\x1b[33m0xab\x1b[0m] \x1b[32mf\x1b[0m at \x1b[36mx.py\x1b[0m:\x1b[33m3\x1b[0m");
    }
}
