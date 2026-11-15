//! Regression: a `-lib` first entered from a non-main thread (Dart isolate worker)
//! initializes the collector on that thread. bdwgc's GC_init registers the caller
//! with the *main* thread's stack bottom on Linux/Android, so marking scanned from
//! the worker's sp up to the main stack and faulted (SIGSEGV SEGV_ACCERR in
//! GC_mark_from). The runtime must give GC_init the caller's own stack base.
//! Own test binary: the collector must not be initialized before the first call.

use typhonrt::lib_rt::*;

fn churn() {
    // Keep live pointers on this (small) stack and force collections.
    let mut keep: Vec<*mut core::ffi::c_void> = Vec::new();
    for i in 0..200_000usize {
        let p = seq_alloc(32 + (i % 512));
        if i % 1000 == 0 {
            keep.push(p);
        }
    }
    unsafe { typhonrt::gc::GC_gcollect() };
    assert!(!keep.is_empty());
}

fn on_small_thread(f: fn()) {
    std::thread::Builder::new().stack_size(512 << 10).spawn(f).unwrap().join().unwrap();
}

#[test]
fn first_call_from_small_worker_thread() {
    on_small_thread(|| {
        // Like the `@export` wrapper: first call initializes the runtime here.
        seq_register_current_thread();
        let mine = seq_gc_my_stackbottom();
        let os = seq_gc_os_stackbase();
        assert_eq!(mine, os, "GC stack bottom of the GC_init thread is not its own stack base");
        churn();
    });
    for _ in 0..4 {
        on_small_thread(|| {
            seq_register_current_thread();
            assert_eq!(seq_gc_my_stackbottom(), seq_gc_os_stackbase());
            churn();
        });
    }
}
