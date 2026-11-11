//! Port of the coroutine / OpenMP / debug / fast-math passes in
//! `codon/cir/llvm/optimize.cpp`: `CoroBranchSimplifier`, `CoroUnrollControl`,
//! `CoroResumeInliner`, `OpenMPThreadIdOptimizer`, `applyDebugTransformations`,
//! `applyFastMathTransformations`.

use llvm_sys::LLVMIntPredicate;
use llvm_sys::LLVMOpcode as Op;
use llvm_sys::core::*;
use llvm_sys::prelude::LLVMMetadataRef;

use crate::analysis::{Cfg, DomTree, LoopInfo};
use crate::ffi::{self, BB, V};
use crate::optimize::alloc::{Intrinsics, pointer_may_be_captured, underlying_object};

/// Name of the global constructor emitted by the visitor
/// (`LLVMVisitor::getGlobalCtorName()`).
pub const GLOBAL_CTOR_NAME: &str = ".main.ctor";

// ------------------------------------------------------------------ debug / fast-math

/// Port of `applyDebugTransformations`.
pub fn apply_debug_transformations(module: llvm_sys::prelude::LLVMModuleRef, debug: bool, jit: bool) {
    if debug {
        for f in ffi::functions(module) {
            if ffi::is_declaration(f) {
                // Attributes on declarations are harmless but the C++ loop touches
                // every function; keep identical behaviour for definitions only for
                // linkage (declarations must stay external anyway).
            }
            if !jit && ffi::name(f) != GLOBAL_CTOR_NAME && !ffi::is_declaration(f) {
                ffi::set_linkage(f, llvm_sys::LLVMLinkage::LLVMExternalLinkage);
            }
            if ffi::intrinsic_id(f) != 0 {
                // Intrinsics cannot carry arbitrary attributes (verifier rejects
                // e.g. uwtable on them); the C++ code relies on LLVM tolerating it.
                continue;
            }
            if !ffi::fn_has_attr(f, "alwaysinline") {
                ffi::add_fn_enum_attr(f, "noinline", 0);
            }
            // UWTableKind::Default == Async (2)
            ffi::add_fn_enum_attr(f, "uwtable", 2);
            ffi::add_fn_string_attr(f, "no-frame-pointer-elim", "true");
            ffi::add_fn_string_attr(f, "no-frame-pointer-elim-non-leaf", "");
            ffi::add_fn_string_attr(f, "no-jump-tables", "false");
            for i in ffi::fn_instructions(f) {
                if ffi::opcode(i) == Op::LLVMCall {
                    ffi::set_tail_call(i, false);
                }
            }
        }
    } else {
        unsafe { llvm_sys::debuginfo::LLVMStripModuleDebugInfo(module) };
    }
}

/// Port of `applyFastMathTransformations`.
pub fn apply_fast_math_transformations(module: llvm_sys::prelude::LLVMModuleRef) {
    const FAST: u32 = llvm_sys::LLVMFastMathAll;
    for f in ffi::functions(module) {
        for i in ffi::fn_instructions(f) {
            let ty = ffi::type_of(i);
            if !ffi::is_fp_ty(ty) {
                continue;
            }
            let is_binop = matches!(
                ffi::opcode(i),
                Op::LLVMFAdd | Op::LLVMFSub | Op::LLVMFMul | Op::LLVMFDiv | Op::LLVMFRem
            );
            let is_intrinsic = ffi::opcode(i) == Op::LLVMCall && ffi::call_intrinsic_id(i).is_some();
            if (is_binop || is_intrinsic) && unsafe { LLVMCanValueUseFastMathFlags(i) } != 0 {
                unsafe { LLVMSetFastMathFlags(i, FAST) };
            }
        }
    }
}

// ------------------------------------------------------------------ helpers

/// `getNonNullOperand(op1, op2)`.
fn non_null_operand(op1: V, op2: V) -> Option<V> {
    if !ffi::is_pointer_ty(ffi::type_of(op1)) {
        return None;
    }
    let n1 = ffi::is_null_value(op1);
    let n2 = ffi::is_null_value(op2);
    if !(n1 ^ n2) {
        return None;
    }
    Some(if n1 { op2 } else { op1 })
}

/// Conservative `llvm::isKnownNonZero` for pointers.
///
/// PORT-NOTE: ValueTracking is not in the C API; we recognise the common cases
/// (functions, globals, allocas, `nonnull` returns/arguments, non-null
/// constants, GEP inbounds of non-null, select/phi of non-null).
fn is_known_non_null(v: V, depth: u32) -> bool {
    if depth > 6 {
        return false;
    }
    if ffi::is_function(v) || ffi::is_global_value(v) {
        return !ffi::is_null_value(v);
    }
    if ffi::is_constant(v) {
        return !ffi::is_null_value(v) && unsafe { LLVMIsAUndefValue(v).is_null() };
    }
    if ffi::is_argument(v) {
        let f = unsafe { LLVMGetParamParent(v) };
        let idx = (0..ffi::count_params(f)).find(|&i| ffi::param(f, i) == v);
        return idx.is_some_and(|i| ffi::fn_param_has_attr(f, i, "nonnull") || ffi::fn_param_has_attr(f, i, "dereferenceable"));
    }
    if !ffi::is_instruction(v) {
        return false;
    }
    match ffi::opcode(v) {
        Op::LLVMAlloca => true,
        Op::LLVMCall | Op::LLVMInvoke => ffi::call_ret_has_attr(v, "nonnull"),
        Op::LLVMBitCast | Op::LLVMAddrSpaceCast => is_known_non_null(ffi::operand(v, 0), depth + 1),
        Op::LLVMGetElementPtr => {
            (unsafe { LLVMIsInBounds(v) } != 0) && is_known_non_null(ffi::operand(v, 0), depth + 1)
        }
        Op::LLVMSelect => {
            is_known_non_null(ffi::operand(v, 1), depth + 1) && is_known_non_null(ffi::operand(v, 2), depth + 1)
        }
        Op::LLVMLoad => {
            let k = ffi::md_kind(ffi::value_context(v), "nonnull");
            unsafe { !LLVMGetMetadata(v, k).is_null() }
        }
        _ => false,
    }
}

fn int_const(ty: ffi::Ty, v: u64) -> V {
    unsafe { LLVMConstInt(ty, v, 0) }
}

// ------------------------------------------------------------------ CoroBranchSimplifier

/// Port of `CoroBranchSimplifier::mergeGuardedState`.
fn merge_guarded_state(cfg: &Cfg, header: BB) -> bool {
    let _ = cfg;
    let states: Vec<V> = ffi::phis(header).into_iter().filter(|&p| ffi::is_pointer_ty(ffi::type_of(p))).collect();
    if states.len() > 8 {
        return false;
    }
    for first in 0..states.len() {
        for second in first + 1..states.len() {
            let left_root = states[first];
            let right_root = states[second];
            let mut visited: Vec<(V, V)> = Vec::new();
            fn equivalent(
                left: V,
                right: V,
                pred: Option<BB>,
                succ: Option<BB>,
                lr: V,
                rr: V,
                visited: &mut Vec<(V, V)>,
            ) -> bool {
                if left == right {
                    return true;
                }
                if let Some(p) = pred {
                    if let Some(br) = ffi::terminator(p) {
                        if ffi::is_conditional_br(br) {
                            let cmp = ffi::br_condition(br);
                            if ffi::is_instruction(cmp) && ffi::is_equality_icmp(cmp) {
                                let tested = non_null_operand(ffi::operand(cmp, 0), ffi::operand(cmp, 1));
                                let other = non_null_operand(left, right);
                                let same = match (tested, other) {
                                    (Some(t), Some(o)) => t == o || (t == lr && o == rr) || (t == rr && o == lr),
                                    _ => false,
                                };
                                let null_edge = if ffi::icmp_predicate(cmp) == LLVMIntPredicate::LLVMIntEQ { 0 } else { 1 };
                                if same
                                    && Some(ffi::successor(br, null_edge)) == succ
                                    && Some(ffi::successor(br, 1 - null_edge)) != succ
                                {
                                    return true;
                                }
                            }
                        }
                    }
                }
                if !ffi::is_instruction(left) || !ffi::is_instruction(right) || !ffi::is_phi(left) || !ffi::is_phi(right) {
                    return false;
                }
                if ffi::parent_block(left) != ffi::parent_block(right) {
                    return false;
                }
                if visited.contains(&(left, right)) {
                    return true;
                }
                if visited.len() == 16 {
                    return false;
                }
                visited.push((left, right));
                let rinc = ffi::phi_incoming(right);
                for (lv, block) in ffi::phi_incoming(left) {
                    let Some(&(rv, _)) = rinc.iter().find(|&&(_, b)| b == block) else { return false };
                    if !equivalent(lv, rv, Some(block), Some(ffi::parent_block(left)), lr, rr, visited) {
                        return false;
                    }
                }
                true
            }
            if equivalent(left_root, right_root, None, None, left_root, right_root, &mut visited) {
                ffi::replace_all_uses(right_root, left_root);
                ffi::erase(right_root);
                return true;
            }
        }
    }
    false
}

/// Port of `CoroBranchSimplifier::run` on one loop (identified by header block).
fn coro_branch_simplify_loop(func: V, header: BB) -> bool {
    let cfg = Cfg::new(func);
    let dt = DomTree::dominators(&cfg);
    let li = LoopInfo::new(&cfg, &dt);
    let Some(&h) = cfg.index.get(&header) else { return false };
    let Some(lid) = li.loops.iter().position(|l| l.header == h) else { return false };
    let mut changed = merge_guarded_state(&cfg, header);
    let blocks: Vec<BB> = li.loops[lid].blocks.iter().map(|&b| cfg.blocks[b]).collect();
    for &block in &blocks {
        for inst in ffi::instructions(block) {
            if ffi::opcode(inst) != Op::LLVMICmp || !ffi::is_equality_icmp(inst) {
                continue;
            }
            let Some(phi) = non_null_operand(ffi::operand(inst, 0), ffi::operand(inst, 1)) else { continue };
            if !ffi::is_instruction(phi) || !ffi::is_phi(phi) {
                continue;
            }
            let inc = ffi::phi_incoming(phi);
            if inc.len() > 8 {
                continue;
            }
            let ty = ffi::type_of(inst);
            let eq = ffi::icmp_predicate(inst) == LLVMIntPredicate::LLVMIntEQ;
            let mut values = Vec::new();
            for &(value, _pred) in &inc {
                let is_null = ffi::is_null_ptr_constant(value);
                if !is_null && !is_known_non_null(value, 0) {
                    break;
                }
                values.push(int_const(ty, (is_null == eq) as u64));
            }
            if values.len() != inc.len() {
                continue;
            }
            let b = ffi::Builder::new(ffi::value_context(inst));
            b.before(ffi::first_instruction(ffi::parent_block(phi)).expect("phi block"));
            unsafe {
                let cond = LLVMBuildPhi(b.0, ty, c"coro.done".as_ptr());
                let mut blks: Vec<BB> = inc.iter().map(|&(_, bb)| bb).collect();
                LLVMAddIncoming(cond, values.as_mut_ptr(), blks.as_mut_ptr(), values.len() as u32);
                ffi::replace_all_uses(inst, cond);
            }
            ffi::erase(inst);
            changed = true;
        }
    }
    if changed {
        return true;
    }
    let Some(exit_id) = li.exiting_block(&cfg, lid) else { return false };
    let exit = cfg.blocks[exit_id];
    let Some(br) = ffi::terminator(exit) else { return false };
    if ffi::opcode(br) != Op::LLVMBr || !ffi::is_conditional_br(br) || ffi::num_successors(br) != 2 {
        return false;
    }
    let lp = &li.loops[lid];
    let in_loop = |b: BB| cfg.index.get(&b).is_some_and(|&x| lp.contains(x));
    if in_loop(ffi::successor(br, 0)) || !in_loop(ffi::successor(br, 1)) {
        return false;
    }
    let cond = ffi::br_condition(br);
    if !ffi::is_instruction(cond) || ffi::opcode(cond) != Op::LLVMICmp {
        return false;
    }
    if ffi::icmp_predicate(cond) != LLVMIntPredicate::LLVMIntEQ {
        return false;
    }
    let Some(f) = non_null_operand(ffi::operand(cond, 0), ffi::operand(cond, 1)) else { return false };
    if !ffi::is_instruction(f) || ffi::opcode(f) != Op::LLVMSelect {
        return false;
    }
    let Some(g) = non_null_operand(ffi::operand(f, 1), ffi::operand(f, 2)) else { return false };
    if !ffi::is_instruction(g) || !ffi::is_phi(g) {
        return false;
    }
    for (value, block) in ffi::phi_incoming(g) {
        if ffi::is_constant(value) {
            if ffi::is_null_value(value) {
                return false;
            }
        } else if block != exit || value != f {
            return false;
        }
    }
    unsafe { LLVMSetCondition(br, ffi::operand(f, 0)) };
    true
}

/// Port of `CoroBranchSimplifier` applied to every loop of `func`.
pub fn coro_branch_simplifier(func: V) -> bool {
    if ffi::is_declaration(func) {
        return false;
    }
    let headers: Vec<BB> = {
        let cfg = Cfg::new(func);
        let dt = DomTree::dominators(&cfg);
        let li = LoopInfo::new(&cfg, &dt);
        li.worklist_order().into_iter().map(|l| cfg.blocks[li.loops[l].header]).collect()
    };
    let mut changed = false;
    for h in headers {
        changed |= coro_branch_simplify_loop(func, h);
    }
    changed
}

// ------------------------------------------------------------------ CoroUnrollControl

fn has_suspension(block: BB, intr: &Intrinsics) -> bool {
    ffi::instructions(block)
        .into_iter()
        .any(|i| ffi::opcode(i) == Op::LLVMCall && ffi::call_intrinsic_id(i) == Some(intr.coro_suspend))
}

fn md_name_of(op: V) -> String {
    if !ffi::is_md_node(op) {
        return String::new();
    }
    let ops = ffi::md_operands(op);
    ops.first().and_then(|&o| ffi::md_string(o)).unwrap_or_default()
}

/// Port of `CoroUnrollControl::updateMetadata`. `identifier` is a
/// MetadataAsValue (or null). Returns the new loop ID (MetadataAsValue or null)
/// and whether it changed.
fn update_loop_metadata(ctx: llvm_sys::prelude::LLVMContextRef, identifier: V, suspends: bool) -> (V, bool) {
    let ops = if identifier.is_null() { Vec::new() } else { ffi::md_operands(identifier) };
    let mut protected_loop = false;
    let mut disabled = false;
    for &op in ops.iter().skip(1) {
        let n = md_name_of(op);
        protected_loop |= n == "codon.coro.unroll";
        disabled |= n == "llvm.loop.unroll.disable";
    }
    if (suspends && disabled) || (!suspends && !protected_loop) {
        return (identifier, false);
    }
    let mut md: Vec<LLVMMetadataRef> = Vec::new();
    for &op in ops.iter().skip(1) {
        let n = md_name_of(op);
        if n != "codon.coro.unroll" && n != "llvm.loop.unroll.disable" {
            md.push(unsafe { LLVMValueAsMetadata(op) });
        }
    }
    if suspends {
        for name in ["codon.coro.unroll", "llvm.loop.unroll.disable"] {
            unsafe {
                let s = LLVMMDStringInContext2(ctx, name.as_ptr() as *const _, name.len());
                let mut a = [s];
                md.push(LLVMMDNodeInContext2(ctx, a.as_mut_ptr(), 1));
            }
        }
    }
    if md.is_empty() {
        return (std::ptr::null_mut(), true);
    }
    unsafe {
        // Distinct self-referential node: build with a temporary operand 0 and RAUW
        // it with the node itself (uniqued self-cycles become distinct in LLVM).
        let temp = llvm_sys::debuginfo::LLVMTemporaryMDNode(ctx, std::ptr::null_mut(), 0);
        let mut all = vec![temp];
        all.extend(md);
        let node = LLVMMDNodeInContext2(ctx, all.as_mut_ptr(), all.len());
        llvm_sys::debuginfo::LLVMMetadataReplaceAllUsesWith(temp, node);
        (LLVMMetadataAsValue(ctx, node), true)
    }
}

/// Port of `CoroUnrollControl::run` for every loop in `func`.
///
/// PORT-NOTE: the C++ pass is a loop pass at the LateLoopOptimizations extension
/// point (before full unroll). Through the C API it runs before the O3 pipeline;
/// the loops with suspension points are still present at that time (coroutines
/// are lowered inside the pipeline), so the same loops get protected.
pub fn coro_unroll_control(func: V) -> bool {
    if ffi::is_declaration(func) {
        return false;
    }
    let intr = Intrinsics::get();
    let ctx = ffi::value_context(func);
    let kind = ffi::md_kind(ctx, "llvm.loop");
    let cfg = Cfg::new(func);
    let dt = DomTree::dominators(&cfg);
    let li = LoopInfo::new(&cfg, &dt);
    let mut changed = false;
    for l in &li.loops {
        let suspends = l.blocks.iter().any(|&b| has_suspension(cfg.blocks[b], &intr));
        // latches: in-loop predecessors of the header
        let latches: Vec<V> = cfg.preds[l.header]
            .iter()
            .filter(|p| l.contains(**p))
            .filter_map(|&p| ffi::terminator(cfg.blocks[p]))
            .collect();
        if latches.is_empty() {
            continue;
        }
        // Loop::getLoopID: all latches must agree
        let ids: Vec<V> = latches.iter().map(|&t| unsafe { LLVMGetMetadata(t, kind) }).collect();
        let id = if ids.iter().all(|&x| x == ids[0]) { ids[0] } else { std::ptr::null_mut() };
        let (new, ch) = update_loop_metadata(ctx, id, suspends);
        if ch {
            for &t in &latches {
                unsafe { LLVMSetMetadata(t, kind, new) };
            }
            changed = true;
        }
    }
    changed
}

/// Port of `CoroUnrollControl::cleanup`.
pub fn coro_unroll_cleanup(module: llvm_sys::prelude::LLVMModuleRef) {
    let intr = Intrinsics::get();
    let ctx = ffi::module_context(module);
    let kind = ffi::md_kind(ctx, "llvm.loop");
    for f in ffi::functions(module) {
        let blocks = ffi::blocks(f);
        if blocks.iter().any(|&b| has_suspension(b, &intr)) {
            continue;
        }
        for i in ffi::fn_instructions(f) {
            let id = unsafe { LLVMGetMetadata(i, kind) };
            if id.is_null() {
                continue;
            }
            let (new, ch) = update_loop_metadata(ctx, id, false);
            if ch {
                unsafe { LLVMSetMetadata(i, kind, new) };
            }
        }
    }
}

// ------------------------------------------------------------------ CoroResumeInliner

/// Port of `CoroResumeInliner` candidate selection. Returns the selected call
/// sites, which are marked `alwaysinline` (call-site attribute) so that LLVM's
/// `always-inline` pass inlines exactly them.
///
/// PORT-NOTE: `llvm::InlineFunction` is not exposed by the C API. The caller
/// (`optimize`) runs the `always-inline` pass after this marking, then removes
/// the marker from any call that survived.
pub fn coro_resume_inliner_mark(module: llvm_sys::prelude::LLVMModuleRef) -> Vec<V> {
    let mut marked = Vec::new();
    for function in ffi::functions(module) {
        let mut calls = Vec::new();
        for i in ffi::fn_instructions(function) {
            if !ffi::is_call(i) {
                continue;
            }
            let Some(callee) = ffi::called_function(i) else { continue };
            if callee != function
                && ffi::has_local_linkage(callee)
                && ffi::fn_has_string_attr(callee, "codon.generator")
                && ffi::name(callee).ends_with(".resume")
                && ffi::num_args(i) == 1
                && !ffi::call_has_fn_attr(i, "noinline")
                && ffi::instruction_count(callee) <= 128
            {
                calls.push(i);
            }
        }
        let mut size = ffi::instruction_count(function);
        for call in calls {
            let callee = ffi::called_function(call).expect("direct");
            let csize = ffi::instruction_count(callee);
            if size + csize > 8192 {
                break;
            }
            let frame = underlying_object(ffi::arg(call, 0));
            let is_alloca = ffi::is_instruction(frame) && ffi::opcode(frame) == Op::LLVMAlloca;
            let allocator = if ffi::is_instruction(frame) && ffi::is_call(frame) {
                ffi::called_function(frame).map(ffi::name)
            } else {
                None
            };
            if !is_alloca && allocator.as_deref() != Some("seq_alloc") {
                continue;
            }
            if pointer_may_be_captured(frame, true, 8192) {
                continue;
            }
            ffi::add_call_enum_attr(call, ffi::FN_INDEX, "alwaysinline", 0);
            size += csize;
            marked.push(call);
        }
    }
    marked
}

// ------------------------------------------------------------------ OpenMPThreadIdOptimizer

/// Port of `OpenMPThreadIdOptimizer::run`.
pub fn openmp_thread_id_optimizer(function: V) -> bool {
    if ffi::is_declaration(function)
        || !ffi::has_local_linkage(function)
        || ffi::use_empty(function)
        || ffi::count_params(function) < 2
    {
        return false;
    }
    let a0 = ffi::type_of(ffi::param(function, 0));
    let a1 = ffi::type_of(ffi::param(function, 1));
    let fork = ffi::is_pointer_ty(a0) && ffi::is_pointer_ty(a1);
    let task = ffi::is_int_ty(a0, 32) && ffi::is_pointer_ty(a1);
    if !fork && !task {
        return false;
    }
    for u in ffi::uses(function) {
        let call = ffi::use_user(u);
        if !ffi::is_instruction(call) || !ffi::is_call(call) {
            return false;
        }
        let Some(callee) = ffi::called_function(call) else { return false };
        let Some(opno) = ffi::use_operand_no(u) else { return false };
        if opno >= ffi::num_args(call) {
            return false;
        }
        if fork {
            if ffi::name(callee) != "__kmpc_fork_call" || opno != 2 {
                return false;
            }
        } else {
            if ffi::name(callee) != "__kmpc_omp_task_alloc" || opno != 5 {
                return false;
            }
            match ffi::const_zext(ffi::arg(call, 2)) {
                Some(flags) if flags & 1 != 0 => {}
                _ => return false,
            }
        }
    }
    let mut global_q = Vec::new();
    let mut team_q = Vec::new();
    for i in ffi::fn_instructions(function) {
        if ffi::opcode(i) != Op::LLVMCall || !ffi::is_int_ty(ffi::type_of(i), 32) {
            continue;
        }
        let Some(callee) = ffi::called_function(i) else { continue };
        let n = ffi::name(callee);
        if n == "__kmpc_global_thread_num" && ffi::num_args(i) == 1 {
            global_q.push(i);
        } else if fork && n == "omp_get_thread_num" && ffi::num_args(i) == 0 {
            team_q.push(i);
        }
    }
    if global_q.is_empty() && team_q.is_empty() {
        return false;
    }
    let ctx = ffi::value_context(function);
    let b = ffi::Builder::new(ctx);
    let entry = ffi::entry_block(function).expect("definition");
    let replace = |queries: &[V], argument: u32, name: &std::ffi::CStr| {
        if queries.is_empty() {
            return;
        }
        match ffi::first_insertion_pt(entry) {
            Some(p) => b.before(p),
            None => b.at_end(entry),
        }
        let mut tid = ffi::param(function, argument);
        if fork {
            tid = unsafe { LLVMBuildLoad2(b.0, LLVMInt32TypeInContext(ctx), tid, name.as_ptr()) };
        }
        for &q in queries {
            ffi::replace_all_uses(q, tid);
            ffi::erase(q);
        }
    };
    replace(&global_q, 0, c"omp.gtid");
    replace(&team_q, 1, c"omp.btid");
    true
}
