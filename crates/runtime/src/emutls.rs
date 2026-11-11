//! Port of `codon/runtime/emutls.cpp`: LLVM emulated-TLS runtime entry point
//! (`__emutls_get_address`) backed by uncollectable GC memory.

use core::ffi::c_void;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::lib_rt::{seq_alloc_uncollectable, seq_free};

/// LLVM's emulated TLS control block: size, alignment, runtime index, initializer.
#[repr(C)]
pub struct TlsControl {
    pub size: usize,
    pub alignment: usize,
    pub index: AtomicUsize,
    pub initializer: *const c_void,
}

struct TlsValues(HashMap<usize, (*mut c_void, *mut c_void)>);

impl Drop for TlsValues {
    fn drop(&mut self) {
        for (storage, _) in self.0.values() {
            seq_free(*storage);
        }
    }
}

static NEXT_INDEX: AtomicUsize = AtomicUsize::new(1);

thread_local! {
    static VALUES: RefCell<TlsValues> = RefCell::new(TlsValues(HashMap::new()));
}

/// # Safety
/// `control` must be a valid emutls control block.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn __emutls_get_address(control: *mut TlsControl) -> *mut c_void {
    let control = unsafe { &*control };
    let mut index = control.index.load(Ordering::Acquire);
    if index == 0 {
        let candidate = NEXT_INDEX.fetch_add(1, Ordering::Relaxed);
        index = match control.index.compare_exchange(0, candidate, Ordering::Release, Ordering::Acquire) {
            Ok(_) => candidate,
            Err(existing) => existing,
        };
    }
    VALUES.with(|v| {
        let mut v = v.borrow_mut();
        if let Some(&(_, addr)) = v.0.get(&index) {
            return addr;
        }
        // Uncollectable GC memory keeps TLS values and the objects they reference alive.
        let alignment = if control.alignment == 0 { 1 } else { control.alignment };
        let storage = seq_alloc_uncollectable(control.size + alignment - 1);
        if storage.is_null() {
            std::process::abort();
        }
        let addr = ((storage as usize + alignment - 1) & !(alignment - 1)) as *mut c_void;
        unsafe {
            if control.initializer.is_null() {
                core::ptr::write_bytes(addr as *mut u8, 0, control.size);
            } else {
                core::ptr::copy_nonoverlapping(control.initializer as *const u8, addr as *mut u8, control.size);
            }
        }
        v.0.insert(index, (storage, addr));
        addr
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn emutls_per_thread() {
        crate::lib_rt::gc_init();
        static INIT: u64 = 42;
        let ctl = Box::leak(Box::new(TlsControl {
            size: 8,
            alignment: 8,
            index: AtomicUsize::new(0),
            initializer: &INIT as *const u64 as *const c_void,
        })) as *mut TlsControl as usize;
        let a = unsafe { __emutls_get_address(ctl as *mut _) } as *mut u64;
        unsafe {
            assert_eq!(*a, 42);
            *a = 7;
            assert_eq!(__emutls_get_address(ctl as *mut _) as *mut u64, a);
        }
    }
}
