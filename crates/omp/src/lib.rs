//! Rust implementation of the OpenMP runtime ABI (`__kmpc_*`, `omp_*`) that
//! Codon-compiled code and `stdlib/openmp.codon` call. Replaces the GC-patched
//! LLVM `libomp` that the C++ build links (omp lane: see ../../DESIGN.md).
//!
//! Layout:
//! * [`icv`]   - internal control variables + environment (`OMP_*`).
//! * [`gc`]    - Boehm GC integration (`__kmpc_set_gc_callbacks`, root sets).
//! * [`team`]  - thread pool, teams, member contexts, fork/join, barrier.
//! * [`sched`] - worksharing loops: static init, dynamic dispatch, ordered.
//! * [`sync`]  - critical/single/master/reductions/atomics/flush.
//! * [`task`]  - explicit tasks, taskwait, taskgroup, task reductions.
//! * [`api`]   - `omp_*` user API.
//!
//! The only C code is `csrc/fork_call.c` (variadic `__kmpc_fork_call` and the
//! microtask trampoline), since stable Rust cannot define C-variadic functions.

#![allow(clippy::missing_safety_doc)]

pub mod api;
pub mod gc;
pub mod icv;
pub mod sched;
pub mod sync;
pub mod task;
pub mod team;

use std::ffi::{c_char, c_void};

/// `ident_t` from kmp.h (source location descriptor). Only `flags` is ever read.
#[repr(C)]
#[derive(Debug)]
pub struct Ident {
    pub reserved_1: i32,
    pub flags: i32,
    pub reserved_2: i32,
    pub reserved_3: i32,
    pub psource: *const c_char,
}

/// Microtask entry: `void(int32_t *gtid, int32_t *btid, void *arg0, ...)`.
pub type Microtask = *mut c_void;

/// Task entry: `int32_t(int32_t gtid, kmp_task_t *task)`.
pub type TaskRoutine = unsafe extern "C" fn(i32, *mut c_void) -> i32;

pub use api::*;
pub use gc::__kmpc_set_gc_callbacks;
pub use sched::*;
pub use sync::*;
pub use task::*;
pub use team::{__kmpc_barrier, __kmpc_global_thread_num, __kmpc_push_num_threads, typhon_omp_fork_call};

unsafe extern "C" {
    /// Defined in csrc/fork_call.c (variadic).
    pub fn __kmpc_fork_call(loc: *mut Ident, argc: i32, microtask: Microtask, ...);
}

/// Address of the C shim's `__kmpc_fork_call`; referencing it keeps the object
/// file alive in every final link even without whole-archive.
#[used]
pub static KEEP_FORK_CALL: unsafe extern "C" fn(*mut Ident, i32, Microtask, ...) =
    __kmpc_fork_call;
