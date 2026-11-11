//! Helpers for tests (ports of the C++ test helpers in `test/cir/llvm/optimize.cpp`).
//! Not part of the stable API.

use inkwell::module::Module;
use llvm_sys::LLVMOpcode as Op;

use crate::analysis::{Cfg, DomTree, LoopInfo};
use crate::ffi;

/// Port of `countFixedAllocations`.
pub fn count_fixed_allocations(module: &Module, size: u64, in_loop_only: bool) -> usize {
    let mut count = 0;
    for f in ffi::functions(module.as_mut_ptr()) {
        if ffi::is_declaration(f) {
            continue;
        }
        let cfg = Cfg::new(f);
        let dt = DomTree::dominators(&cfg);
        let li = LoopInfo::new(&cfg, &dt);
        for (bi, &b) in cfg.blocks.iter().enumerate() {
            if in_loop_only && li.loop_for(bi).is_none() {
                continue;
            }
            for i in ffi::instructions(b) {
                if !ffi::is_call(i) || ffi::num_args(i) == 0 {
                    continue;
                }
                let Some(callee) = ffi::called_function(i) else { continue };
                if ffi::name(callee) == "seq_alloc_atomic" && ffi::const_zext(ffi::arg(i, 0)) == Some(size) {
                    count += 1;
                }
            }
        }
    }
    count
}

/// Port of `countLazyFixedAllocationCaches`.
pub fn count_lazy_fixed_allocation_caches(module: &Module, size: u64) -> usize {
    let mut count = 0;
    for f in ffi::functions(module.as_mut_ptr()) {
        if ffi::is_declaration(f) {
            continue;
        }
        let cfg = Cfg::new(f);
        let dt = DomTree::dominators(&cfg);
        let li = LoopInfo::new(&cfg, &dt);
        for (bi, &b) in cfg.blocks.iter().enumerate() {
            let Some(alloc_loop) = li.loop_for(bi) else { continue };
            for i in ffi::instructions(b) {
                if !ffi::is_call(i) || ffi::num_args(i) == 0 {
                    continue;
                }
                let Some(callee) = ffi::called_function(i) else { continue };
                if ffi::name(callee) != "seq_alloc_atomic" || ffi::const_zext(ffi::arg(i, 0)) != Some(size) {
                    continue;
                }
                for user in ffi::unique_users(i) {
                    if !ffi::is_instruction(user) || !ffi::is_phi(user) {
                        continue;
                    }
                    let mut found = false;
                    for (inc, _) in ffi::phi_incoming(user) {
                        if !ffi::is_instruction(inc) || !ffi::is_phi(inc) {
                            continue;
                        }
                        if li.loop_for(cfg.id(ffi::parent_block(inc))) != Some(alloc_loop) {
                            continue;
                        }
                        if ffi::phi_incoming(inc).iter().any(|&(v, _)| ffi::is_null_ptr_constant(v)) {
                            found = true;
                            break;
                        }
                    }
                    if found {
                        count += 1;
                    }
                }
            }
        }
    }
    count
}

/// Names of called functions in `func`, with whether the call block is inside a loop.
pub fn calls_in(module: &Module, func: &str) -> Vec<(String, bool)> {
    let Some(f) = ffi::get_function(module.as_mut_ptr(), func) else { return vec![] };
    let cfg = Cfg::new(f);
    let dt = DomTree::dominators(&cfg);
    let li = LoopInfo::new(&cfg, &dt);
    let mut out = Vec::new();
    for (bi, &b) in cfg.blocks.iter().enumerate() {
        for i in ffi::instructions(b) {
            if let Some(c) = ffi::called_function(i) {
                out.push((ffi::name(c), li.loop_for(bi).is_some()));
            }
        }
    }
    out
}

/// Alignments of all `alloca` instructions in `func`.
pub fn alloca_alignments(module: &Module, func: &str) -> Vec<u64> {
    let Some(f) = ffi::get_function(module.as_mut_ptr(), func) else { return vec![] };
    ffi::fn_instructions(f).into_iter().filter(|&i| ffi::opcode(i) == Op::LLVMAlloca).map(ffi::alignment).collect()
}

/// Names of the loop-metadata hints (operand 0 of each sub-node) on the latch
/// of the first loop in `func`.
pub fn first_loop_hints(module: &Module, func: &str) -> Option<Vec<String>> {
    let f = ffi::get_function(module.as_mut_ptr(), func)?;
    let cfg = Cfg::new(f);
    let dt = DomTree::dominators(&cfg);
    let li = LoopInfo::new(&cfg, &dt);
    let l = li.loops.first()?;
    let kind = ffi::md_kind(ffi::value_context(f), "llvm.loop");
    for &p in &cfg.preds[l.header] {
        if !l.contains(p) {
            continue;
        }
        let t = ffi::terminator(cfg.blocks[p])?;
        let id = unsafe { llvm_sys::core::LLVMGetMetadata(t, kind) };
        if id.is_null() {
            continue;
        }
        let ops = ffi::md_operands(id);
        let names = ops
            .iter()
            .skip(1)
            .filter(|o| ffi::is_md_node(**o))
            .filter_map(|&o| ffi::md_operands(o).first().and_then(|&s| ffi::md_string(s)))
            .collect();
        return Some(names);
    }
    None
}

/// Constant integer returned by every `ret` in `func` (None if any is non-constant).
pub fn constant_returns(module: &Module, func: &str) -> Option<Vec<u64>> {
    let f = ffi::get_function(module.as_mut_ptr(), func)?;
    let mut out = Vec::new();
    for b in ffi::blocks(f) {
        if let Some(t) = ffi::terminator(b) {
            if ffi::opcode(t) == Op::LLVMRet {
                if ffi::num_operands(t) == 0 {
                    return None;
                }
                out.push(ffi::const_zext(ffi::operand(t, 0))?);
            }
        }
    }
    Some(out)
}

/// Number of loops in `func`.
pub fn loop_count(module: &Module, func: &str) -> usize {
    let Some(f) = ffi::get_function(module.as_mut_ptr(), func) else { return 0 };
    let cfg = Cfg::new(f);
    let dt = DomTree::dominators(&cfg);
    LoopInfo::new(&cfg, &dt).loops.len()
}
