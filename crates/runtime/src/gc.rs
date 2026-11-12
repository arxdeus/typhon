//! Minimal FFI bindings for the vendored bdwgc v8.2.12 (`vendor/bdwgc/include/gc.h`),
//! built by `build.rs` with the C++ Codon CMake options. Only the API used by the
//! runtime is declared; the full `GC_*` ABI is exported from the cdylib.
#![allow(non_camel_case_types, non_snake_case)]

use core::ffi::{c_char, c_int, c_void};

pub type GC_word = usize;

/// `struct GC_stack_base` (no `reg_base` on supported targets).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GC_stack_base {
    pub mem_base: *mut c_void,
}

pub type GC_finalization_proc = Option<unsafe extern "C" fn(obj: *mut c_void, client_data: *mut c_void)>;
pub type GC_warn_proc = Option<unsafe extern "C" fn(msg: *mut c_char, arg: GC_word)>;
pub type GC_finalizer_notifier_proc = Option<unsafe extern "C" fn()>;

unsafe extern "C" {
    pub fn GC_init();
    pub fn GC_malloc(n: usize) -> *mut c_void;
    pub fn GC_malloc_atomic(n: usize) -> *mut c_void;
    pub fn GC_malloc_uncollectable(n: usize) -> *mut c_void;
    pub fn GC_malloc_atomic_uncollectable(n: usize) -> *mut c_void;
    pub fn GC_realloc(p: *mut c_void, n: usize) -> *mut c_void;
    pub fn GC_free(p: *mut c_void);
    pub fn GC_register_finalizer(
        obj: *mut c_void,
        f: GC_finalization_proc,
        cd: *mut c_void,
        ofn: *mut GC_finalization_proc,
        ocd: *mut *mut c_void,
    );
    pub fn GC_add_roots(low: *mut c_void, high_plus_1: *mut c_void);
    pub fn GC_remove_roots(low: *mut c_void, high_plus_1: *mut c_void);
    pub fn GC_clear_roots();
    pub fn GC_exclude_static_roots(low: *mut c_void, high_plus_1: *mut c_void);
    pub fn GC_gcollect();
    pub fn GC_get_heap_size() -> usize;
    pub fn GC_get_stack_base(sb: *mut GC_stack_base) -> c_int;
    pub fn GC_register_my_thread(sb: *const GC_stack_base) -> c_int;
    pub fn GC_allow_register_threads();
    pub fn GC_set_finalize_on_demand(v: c_int);
    pub fn GC_set_warn_proc(p: GC_warn_proc);
    pub fn GC_ignore_warn_proc(msg: *mut c_char, arg: GC_word);
    pub fn GC_set_finalizer_notifier(p: GC_finalizer_notifier_proc);
    pub fn GC_should_invoke_finalizers() -> c_int;
    pub fn GC_invoke_finalizers() -> c_int;
}
