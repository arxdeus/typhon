//! Smoke test: initialize the runtime and allocate through the Boehm GC via the C ABI.

use codonrt::lib_rt::*;
use codonrt::types::string_conv;

#[test]
fn gc_alloc_smoke() {
    seq_init(0);
    unsafe {
        let p = seq_alloc(64) as *mut u64;
        assert!(!p.is_null());
        for i in 0..8 {
            *p.add(i) = i as u64;
        }
        let q = seq_realloc(p.cast(), 4096, 64) as *mut u64;
        assert_eq!(*q.add(7), 7);
        let a = seq_alloc_atomic(1 << 20);
        assert!(!a.is_null());
        let u = seq_alloc_uncollectable(32);
        seq_free(u);
        assert!(codonrt::gc::GC_get_heap_size() > 0);
        // churn the collector
        for _ in 0..10_000 {
            let _ = seq_alloc(256);
        }
        codonrt::gc::GC_gcollect();
        let s = string_conv("héllo".as_bytes());
        assert_eq!(s.encode(), "héllo".as_bytes());
    }
}
