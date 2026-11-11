//! Port of the allocation passes in `codon/cir/llvm/optimize.cpp`:
//! `AllocInfo`, `AllocationRemover`, `AllocationHoister`, `AllocationAutoFree`.

use llvm_sys::LLVMOpcode as Op;
use llvm_sys::core::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::analysis::{self, Cfg, DomTree, LoopInfo};
use crate::ffi::{self, BB, V};

/// Intrinsic IDs used by the passes (looked up by name once per run).
pub(crate) struct Intrinsics {
    pub memmove: u32,
    pub memcpy: u32,
    pub memset: u32,
    pub assume: u32,
    pub invariant_start: u32,
    pub invariant_end: u32,
    pub lifetime_start: u32,
    pub lifetime_end: u32,
    pub launder: u32,
    pub strip: u32,
    pub coro_suspend: u32,
}

impl Intrinsics {
    pub fn get() -> Self {
        let l = ffi::lookup_intrinsic;
        Intrinsics {
            memmove: l("llvm.memmove"),
            memcpy: l("llvm.memcpy"),
            memset: l("llvm.memset"),
            assume: l("llvm.assume"),
            invariant_start: l("llvm.invariant.start"),
            invariant_end: l("llvm.invariant.end"),
            lifetime_start: l("llvm.lifetime.start"),
            lifetime_end: l("llvm.lifetime.end"),
            launder: l("llvm.launder.invariant.group"),
            strip: l("llvm.strip.invariant.group"),
            coro_suspend: l("llvm.coro.suspend"),
        }
    }

    fn is_mem(&self, id: u32) -> bool {
        id == self.memmove || id == self.memcpy || id == self.memset
    }

    fn is_noop(&self, id: u32) -> bool {
        id == self.assume
            || id == self.invariant_start
            || id == self.invariant_end
            || id == self.lifetime_start
            || id == self.lifetime_end
    }

    fn is_group(&self, id: u32) -> bool {
        id == self.launder || id == self.strip
    }
}

/// `MemIntrinsic::isVolatile()`: the trailing `i1 isvolatile` argument.
fn mem_intrinsic_volatile(call: V) -> bool {
    let n = ffi::num_args(call);
    n >= 4 && ffi::const_zext(ffi::arg(call, 3)).is_some_and(|v| v != 0)
}

type Path = Vec<u32>;

fn starts_with(path: &[u32], prefix: &[u32]) -> bool {
    path.len() >= prefix.len() && path[..prefix.len()] == *prefix
}

/// Port of `AllocInfo`.
pub struct AllocInfo {
    pub allocators: Vec<String>,
    pub realloc: String,
    pub free: String,
    intr: Intrinsics,
}

impl AllocInfo {
    pub fn new(allocators: &[&str], realloc: &str, free: &str) -> Self {
        AllocInfo {
            allocators: allocators.iter().map(|s| s.to_string()).collect(),
            realloc: realloc.into(),
            free: free.into(),
            intr: Intrinsics::get(),
        }
    }

    fn get_fixed_arg(call: V, idx: u32) -> Option<u64> {
        if ffi::num_args(call) <= idx {
            return None;
        }
        ffi::const_zext(ffi::arg(call, idx))
    }

    /// `getCalledFunction`: callee of a non-intrinsic call.
    fn called(value: V) -> Option<V> {
        if !ffi::is_instruction(value) {
            return None;
        }
        let f = ffi::called_function(value)?;
        (ffi::intrinsic_id(f) == 0).then_some(f)
    }

    pub fn is_alloc(&self, value: V) -> bool {
        Self::called(value).is_some_and(|f| ffi::count_params(f) == 1 && self.allocators.contains(&ffi::name(f)))
    }

    pub fn is_realloc(&self, value: V) -> bool {
        Self::called(value).is_some_and(|f| ffi::count_params(f) == 3 && ffi::name(f) == self.realloc)
    }

    pub fn is_free(&self, value: V) -> bool {
        Self::called(value).is_some_and(|f| ffi::count_params(f) == 1 && ffi::name(f) == self.free)
    }

    fn is_never_equal_to_unescaped_alloc(&self, value: V, ai: V) -> bool {
        if ffi::is_null_ptr_constant(value) {
            return true;
        }
        if ffi::is_instruction(value) && ffi::opcode(value) == Op::LLVMLoad {
            return ffi::is_global_variable(ffi::operand(value, 0));
        }
        self.is_alloc(value) && value != ai
    }

    fn other_cmp_operand(cmp: V, pi: V) -> V {
        if ffi::operand(cmp, 0) == pi { ffi::operand(cmp, 1) } else { ffi::operand(cmp, 0) }
    }

    pub fn is_alloc_site_removable(&self, ai: V, users: &mut Vec<V>) -> bool {
        if ffi::opcode(ai) == Op::LLVMInvoke {
            return false;
        }
        let mut worklist = vec![ai];
        while let Some(pi) = worklist.pop() {
            for instr in ffi::users(pi) {
                if !ffi::is_instruction(instr) {
                    return false;
                }
                match ffi::opcode(instr) {
                    Op::LLVMAddrSpaceCast | Op::LLVMBitCast | Op::LLVMGetElementPtr => {
                        users.push(instr);
                        worklist.push(instr);
                    }
                    Op::LLVMICmp => {
                        if !ffi::is_equality_icmp(instr) {
                            return false;
                        }
                        if !self.is_never_equal_to_unescaped_alloc(Self::other_cmp_operand(instr, pi), ai) {
                            return false;
                        }
                        users.push(instr);
                    }
                    Op::LLVMCall => {
                        if let Some(id) = ffi::call_intrinsic_id(instr) {
                            let i = &self.intr;
                            if i.is_mem(id) {
                                if mem_intrinsic_volatile(instr) || ffi::arg(instr, 0) != pi {
                                    return false;
                                }
                                users.push(instr);
                            } else if i.is_noop(id) {
                                users.push(instr);
                            } else if i.is_group(id) {
                                users.push(instr);
                                worklist.push(instr);
                            } else {
                                return false;
                            }
                            continue;
                        }
                        if self.is_free(instr) {
                            users.push(instr);
                            continue;
                        }
                        if self.is_realloc(instr) {
                            users.push(instr);
                            worklist.push(instr);
                            continue;
                        }
                        return false;
                    }
                    Op::LLVMStore => {
                        if ffi::is_volatile(instr) || ffi::operand(instr, 1) != pi {
                            return false;
                        }
                        users.push(instr);
                    }
                    _ => return false,
                }
            }
        }
        true
    }

    /// Port of `GEPOperator::accumulateConstantOffset`.
    fn gep_constant_offset(gep: V) -> Option<i64> {
        let m = ffi::module_of_inst(gep);
        let dl = ffi::data_layout(m);
        let mut ty = unsafe { LLVMGetGEPSourceElementType(gep) };
        let n = ffi::num_operands(gep);
        let mut offset: i64 = 0;
        for i in 1..n {
            let idx = ffi::operand(gep, i);
            let c = ffi::const_sext(idx)?;
            if i == 1 {
                let size = unsafe { llvm_sys::target::LLVMABISizeOfType(dl, ty) } as i64;
                offset = offset.wrapping_add(c.wrapping_mul(size));
                continue;
            }
            match ffi::type_kind(ty) {
                llvm_sys::LLVMTypeKind::LLVMStructTypeKind => {
                    let off = unsafe { llvm_sys::target::LLVMOffsetOfElement(dl, ty, c as u32) } as i64;
                    offset = offset.wrapping_add(off);
                    ty = unsafe { LLVMStructGetTypeAtIndex(ty, c as u32) };
                }
                llvm_sys::LLVMTypeKind::LLVMArrayTypeKind | llvm_sys::LLVMTypeKind::LLVMVectorTypeKind => {
                    ty = unsafe { LLVMGetElementType(ty) };
                    let size = unsafe { llvm_sys::target::LLVMABISizeOfType(dl, ty) } as i64;
                    offset = offset.wrapping_add(c.wrapping_mul(size));
                }
                _ => return None,
            }
        }
        Some(offset)
    }

    fn return_alignment(call: V) -> u64 {
        let mut a = ffi::call_ret_align(call).unwrap_or(1);
        if let Some(callee) = ffi::called_function(call) {
            a = a.max(ffi::fn_ret_align(callee).unwrap_or(1));
        }
        a
    }

    pub fn is_alloc_site_demotable(
        &self,
        ai: V,
        size: &mut u64,
        users: &mut Vec<V>,
        stack_alignment: &mut u64,
        max_size: u64,
    ) -> bool {
        if ffi::opcode(ai) == Op::LLVMInvoke {
            return false;
        }
        match Self::get_fixed_arg(ai, 0) {
            Some(s) if 0 < s && s <= max_size => *size = s,
            _ => return false,
        }
        if ffi::pointer_address_space(ffi::type_of(ai)) != 0 {
            return false;
        }
        let m = ffi::module_of_inst(ai);
        let allocator = Self::called(ai).map(ffi::name).unwrap_or_default();
        let atomic = allocator == "seq_alloc_atomic" || allocator == "seq_alloc_atomic_uncollectable";
        let explicit = Self::return_alignment(ai);
        let heap_alignment = 2 * ffi::pointer_size(m);
        *stack_alignment = (if atomic { 1 } else { ffi::pointer_abi_align(m) }).max(explicit);

        #[derive(Clone)]
        struct PointerInfo {
            instruction: V,
            offset: i64,
            allocation_alignment: u64,
            constant_offset: bool,
        }
        let mut worklist =
            vec![PointerInfo { instruction: ai, offset: 0, allocation_alignment: heap_alignment.max(explicit), constant_offset: true }];
        let mut visited: FxHashSet<V> = FxHashSet::default();

        while let Some(pointer) = worklist.pop() {
            let pi = pointer.instruction;
            if !visited.insert(pi) {
                continue;
            }
            for instr in ffi::users(pi) {
                if !ffi::is_instruction(instr) {
                    return false;
                }
                let require = |required: Option<u64>, available: u64, sa: &mut u64| -> bool {
                    match required {
                        None => true,
                        Some(r) if r > available => false,
                        Some(r) => {
                            *sa = (*sa).max(r);
                            true
                        }
                    }
                };
                match ffi::opcode(instr) {
                    Op::LLVMAddrSpaceCast => return false,
                    Op::LLVMBitCast => {
                        worklist.push(PointerInfo { instruction: instr, ..pointer.clone() });
                    }
                    Op::LLVMGetElementPtr => {
                        let mut derived = pointer.clone();
                        derived.instruction = instr;
                        match (pointer.constant_offset, Self::gep_constant_offset(instr)) {
                            (true, Some(d)) => derived.offset = derived.offset.wrapping_add(d),
                            _ => derived.constant_offset = false,
                        }
                        worklist.push(derived);
                    }
                    Op::LLVMICmp => {
                        if !ffi::is_equality_icmp(instr) {
                            return false;
                        }
                        if !self.is_never_equal_to_unescaped_alloc(Self::other_cmp_operand(instr, pi), ai) {
                            return false;
                        }
                    }
                    Op::LLVMCall => {
                        let call = instr;
                        if ffi::num_operand_bundles(call) > 0 {
                            return false;
                        }
                        let callee = ffi::called_function(call);
                        for index in 0..ffi::num_args(call) {
                            if ffi::arg(call, index) != pi {
                                continue;
                            }
                            if !require(ffi::call_param_align(call, index), pointer.allocation_alignment, stack_alignment)
                                || callee.is_some_and(|f| {
                                    !require(ffi::fn_param_align(f, index), pointer.allocation_alignment, stack_alignment)
                                })
                            {
                                return false;
                            }
                        }
                        if let Some(id) = ffi::call_intrinsic_id(call) {
                            let i = &self.intr;
                            if i.is_mem(id) {
                                if mem_intrinsic_volatile(call) {
                                    return false;
                                }
                                users.push(instr);
                            } else if i.is_noop(id) {
                                users.push(instr);
                            } else if i.is_group(id) {
                                if !require(Some(Self::return_alignment(call)), pointer.allocation_alignment, stack_alignment) {
                                    return false;
                                }
                                users.push(instr);
                                worklist.push(PointerInfo { instruction: instr, ..pointer.clone() });
                            } else {
                                return false;
                            }
                            continue;
                        }
                        if self.is_free(instr) {
                            users.push(instr);
                            continue;
                        }
                        if self.is_realloc(instr) {
                            if ffi::arg(call, 0) != pi || !pointer.constant_offset || pointer.offset != 0 {
                                return false;
                            }
                            match Self::get_fixed_arg(call, 1) {
                                Some(ns) if 0 < ns && ns <= max_size => *size = (*size).max(ns),
                                _ => return false,
                            }
                            users.push(instr);
                            let ra = Self::return_alignment(call);
                            *stack_alignment = (*stack_alignment).max(ra);
                            worklist.push(PointerInfo {
                                instruction: instr,
                                offset: 0,
                                allocation_alignment: heap_alignment.max(ra),
                                constant_offset: true,
                            });
                            continue;
                        }
                        if callee.is_none() || ffi::is_must_tail_call(call) || ffi::is_pointer_ty(ffi::type_of(call)) {
                            return false;
                        }
                        for index in 0..ffi::num_args(call) {
                            if ffi::arg(call, index) != pi {
                                continue;
                            }
                            let no_free =
                                ffi::call_has_fn_attr(call, "nofree") || ffi::call_param_has_attr(call, index, "nofree");
                            if !ffi::call_param_has_attr(call, index, "nocapture") || !no_free {
                                return false;
                            }
                        }
                        *stack_alignment = (*stack_alignment).max(pointer.allocation_alignment);
                        users.push(instr);
                    }
                    Op::LLVMStore => {
                        if ffi::is_volatile(instr)
                            || ffi::operand(instr, 1) != pi
                            || !require(Some(ffi::alignment(instr)), pointer.allocation_alignment, stack_alignment)
                        {
                            return false;
                        }
                    }
                    Op::LLVMLoad => {
                        if ffi::is_volatile(instr)
                            || !require(Some(ffi::alignment(instr)), pointer.allocation_alignment, stack_alignment)
                        {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
    }

    pub fn is_aggregate_field_unused(
        &self,
        argument: V,
        indices: &[u32],
        budget: &mut u32,
        active: &mut Vec<V>,
    ) -> bool {
        let function = unsafe { LLVMGetParamParent(argument) };
        if !ffi::has_exact_definition(function) || active.contains(&function) || active.len() >= 8 {
            return false;
        }
        active.push(function);
        let result = self.aggregate_field_unused_inner(argument, indices, budget, active);
        active.retain(|&f| f != function);
        result
    }

    fn aggregate_field_unused_inner(&self, argument: V, indices: &[u32], budget: &mut u32, active: &mut Vec<V>) -> bool {
        let mut worklist: Vec<(V, Path)> = Vec::new();
        let mut visited: FxHashMap<V, Vec<Path>> = FxHashMap::default();
        let mut enqueue = |worklist: &mut Vec<(V, Path)>, value: V, path: Path| {
            let paths = visited.entry(value).or_default();
            if paths.contains(&path) {
                return;
            }
            paths.push(path.clone());
            worklist.push((value, path));
        };
        enqueue(&mut worklist, argument, indices.to_vec());
        while let Some((value, path)) = worklist.pop() {
            for user in ffi::users(value) {
                if *budget == 0 {
                    return false;
                }
                *budget -= 1;
                if !ffi::is_instruction(user) {
                    return false;
                }
                match ffi::opcode(user) {
                    Op::LLVMPHI | Op::LLVMFreeze => enqueue(&mut worklist, user, path.clone()),
                    Op::LLVMSelect => {
                        if ffi::operand(user, 0) == value {
                            return false;
                        }
                        enqueue(&mut worklist, user, path.clone());
                    }
                    Op::LLVMInsertValue => {
                        let idx = ffi::agg_indices(user);
                        if ffi::operand(user, 0) == value && !starts_with(&path, &idx) {
                            enqueue(&mut worklist, user, path.clone());
                        }
                        if ffi::operand(user, 1) == value {
                            let mut nested = idx.clone();
                            nested.extend_from_slice(&path);
                            enqueue(&mut worklist, user, nested);
                        }
                    }
                    Op::LLVMExtractValue => {
                        let idx = ffi::agg_indices(user);
                        if starts_with(&path, &idx) {
                            enqueue(&mut worklist, user, path[idx.len()..].to_vec());
                        }
                    }
                    Op::LLVMCall | Op::LLVMInvoke => {
                        let callee = ffi::called_function(user);
                        let Some(callee) = callee else { return false };
                        if path.is_empty() || ffi::num_operand_bundles(user) > 0 {
                            return false;
                        }
                        for index in 0..ffi::num_args(user) {
                            if ffi::arg(user, index) == value
                                && (index >= ffi::count_params(callee)
                                    || !self.is_aggregate_field_unused(ffi::param(callee, index), &path, budget, active))
                            {
                                return false;
                            }
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
    }

    /// `Loop::hasLoopInvariantOperands`.
    fn has_loop_invariant_operands(ai: V, cfg: &Cfg, lp: &analysis::Loop) -> bool {
        ffi::operands(ai).into_iter().all(|op| {
            !ffi::is_instruction(op) || !cfg.index.get(&ffi::parent_block(op)).is_some_and(|&b| lp.contains(b))
        })
    }

    pub fn is_alloc_site_hoistable(
        &self,
        ai: V,
        lid: usize,
        cfg: &Cfg,
        li: &LoopInfo,
        irreducible: &FxHashSet<usize>,
        releases: &mut Vec<V>,
    ) -> bool {
        let lp = &li.loops[lid];
        let parent = cfg.id(ffi::parent_block(ai));
        let any_sub_loop_contains = lp.subloops.iter().any(|&s| li.loops[s].contains(parent));
        if ffi::opcode(ai) == Op::LLVMInvoke
            || !Self::has_loop_invariant_operands(ai, cfg, lp)
            || any_sub_loop_contains
            || irreducible.contains(&parent)
        {
            return false;
        }
        let in_loop = |i: V| cfg.index.get(&ffi::parent_block(i)).is_some_and(|&b| lp.contains(b));

        let mut provenance: FxHashMap<V, Vec<Path>> = FxHashMap::default();
        let mut worklist: std::collections::VecDeque<V> = Default::default();
        let mut pending: FxHashSet<V> = FxHashSet::default();
        fn add(
            provenance: &mut FxHashMap<V, Vec<Path>>,
            worklist: &mut std::collections::VecDeque<V>,
            pending: &mut FxHashSet<V>,
            instr: V,
            path: Path,
        ) {
            let paths = provenance.entry(instr).or_default();
            if paths.contains(&path) {
                return;
            }
            paths.push(path);
            if pending.insert(instr) {
                worklist.push_front(instr);
            }
        }
        add(&mut provenance, &mut worklist, &mut pending, ai, vec![]);

        while let Some(pi) = worklist.pop_back() {
            pending.remove(&pi);
            for instr in ffi::users(pi) {
                if !ffi::is_instruction(instr) || !in_loop(instr) {
                    return false;
                }
                let pi_paths = provenance.get(&pi).cloned().unwrap_or_default();
                macro_rules! propagate {
                    () => {
                        for p in pi_paths.iter() {
                            add(&mut provenance, &mut worklist, &mut pending, instr, p.clone());
                        }
                    };
                }
                match ffi::opcode(instr) {
                    Op::LLVMPHI => {
                        if ffi::parent_block(instr) == cfg.blocks[lp.header] {
                            return false;
                        }
                        propagate!();
                    }
                    Op::LLVMSelect | Op::LLVMFreeze => {
                        propagate!();
                    }
                    Op::LLVMPtrToInt
                    | Op::LLVMIntToPtr
                    | Op::LLVMAdd
                    | Op::LLVMSub
                    | Op::LLVMAddrSpaceCast
                    | Op::LLVMBitCast
                    | Op::LLVMGetElementPtr => {
                        if pi_paths.iter().any(|p| !p.is_empty()) {
                            return false;
                        }
                        add(&mut provenance, &mut worklist, &mut pending, instr, vec![]);
                    }
                    Op::LLVMInsertValue => {
                        let indices = ffi::agg_indices(instr);
                        if ffi::operand(instr, 0) == pi {
                            for p in &pi_paths {
                                if !starts_with(p, &indices) {
                                    add(&mut provenance, &mut worklist, &mut pending, instr, p.clone());
                                }
                            }
                        }
                        if ffi::operand(instr, 1) == pi {
                            for p in &pi_paths {
                                let mut r = indices.clone();
                                r.extend_from_slice(p);
                                add(&mut provenance, &mut worklist, &mut pending, instr, r);
                            }
                        }
                    }
                    Op::LLVMExtractValue => {
                        let indices = ffi::agg_indices(instr);
                        for p in &pi_paths {
                            if starts_with(p, &indices) {
                                add(&mut provenance, &mut worklist, &mut pending, instr, p[indices.len()..].to_vec());
                            }
                        }
                    }
                    Op::LLVMICmp => {}
                    Op::LLVMCall | Op::LLVMInvoke => {
                        if let Some(id) = ffi::call_intrinsic_id(instr) {
                            let i = &self.intr;
                            if i.is_mem(id) {
                                if mem_intrinsic_volatile(instr) {
                                    return false;
                                }
                            } else if i.is_noop(id) {
                            } else if i.is_group(id) {
                                add(&mut provenance, &mut worklist, &mut pending, instr, vec![]);
                            } else {
                                return false;
                            }
                            continue;
                        }
                        let call = instr;
                        if self.is_free(call) {
                            if ffi::opcode(call) != Op::LLVMCall
                                || ffi::is_must_tail_call(call)
                                || ffi::num_operand_bundles(call) > 0
                                || ffi::num_args(call) != 1
                                || !ffi::use_empty(call)
                                || strip_pointer_casts(ffi::arg(call, 0)) != ai
                            {
                                return false;
                            }
                            if !releases.contains(&call) {
                                releases.push(call);
                            }
                            continue;
                        }
                        for i in 0..ffi::num_args(call) {
                            if ffi::arg(call, i) != pi {
                                continue;
                            }
                            let mut aggregate = false;
                            let mut budget = 256u32;
                            let mut active = Vec::new();
                            for p in &pi_paths {
                                if p.is_empty() {
                                    continue;
                                }
                                let callee = ffi::called_function(call);
                                let ok = callee.is_some_and(|c| {
                                    i < ffi::count_params(c)
                                        && ffi::num_operand_bundles(call) == 0
                                        && self.is_aggregate_field_unused(ffi::param(c, i), p, &mut budget, &mut active)
                                });
                                if !ok {
                                    return false;
                                }
                                aggregate = true;
                            }
                            if aggregate {
                                continue;
                            }
                            if ffi::call_param_has_attr(call, i, "byval") {
                                continue;
                            }
                            if ffi::is_pointer_ty(ffi::type_of(call)) {
                                return false;
                            }
                            let no_free = ffi::call_has_fn_attr(call, "nofree") || ffi::call_param_has_attr(call, i, "nofree");
                            if !ffi::call_param_has_attr(call, i, "nocapture") || !no_free {
                                return false;
                            }
                        }
                    }
                    Op::LLVMStore => {
                        if ffi::is_volatile(instr) || ffi::operand(instr, 1) != pi {
                            return false;
                        }
                    }
                    Op::LLVMLoad => {
                        if ffi::is_volatile(instr) {
                            return false;
                        }
                    }
                    _ => return false,
                }
            }
        }
        true
    }
}

/// `Value::stripPointerCasts`: bitcasts, address space casts and all-zero GEPs.
pub(crate) fn strip_pointer_casts(mut v: V) -> V {
    loop {
        if !ffi::is_instruction(v) {
            // constant expressions
            if unsafe { !LLVMIsAConstantExpr(v).is_null() } {
                let op = unsafe { LLVMGetConstOpcode(v) };
                if matches!(op, Op::LLVMBitCast | Op::LLVMAddrSpaceCast) {
                    v = ffi::operand(v, 0);
                    continue;
                }
            }
            return v;
        }
        match ffi::opcode(v) {
            Op::LLVMBitCast | Op::LLVMAddrSpaceCast => v = ffi::operand(v, 0),
            Op::LLVMGetElementPtr
                if (1..ffi::num_operands(v)).all(|i| ffi::const_zext(ffi::operand(v, i)) == Some(0)) =>
            {
                v = ffi::operand(v, 0)
            }
            _ => return v,
        }
    }
}

/// `llvm::getUnderlyingObject` (default max lookup 6).
pub(crate) fn underlying_object(mut v: V) -> V {
    for _ in 0..6 {
        if ffi::is_instruction(v) {
            match ffi::opcode(v) {
                Op::LLVMGetElementPtr | Op::LLVMBitCast | Op::LLVMAddrSpaceCast => {
                    v = ffi::operand(v, 0);
                    continue;
                }
                _ => {}
            }
            // `returned` argument / launder-like intrinsics
            if ffi::is_call(v) {
                if let Some(id) = ffi::call_intrinsic_id(v) {
                    let i = Intrinsics::get();
                    if i.is_group(id) {
                        v = ffi::arg(v, 0);
                        continue;
                    }
                }
            }
            return v;
        }
        return strip_pointer_casts(v);
    }
    v
}

/// Port of `llvm::PointerMayBeCaptured(V, ReturnCaptures, StoreCaptures, MaxUses)`.
///
/// PORT-NOTE: LLVM's CaptureTracking is not in the C API. This is a conservative
/// re-implementation of `DetermineUseCaptureKind`: loads, stores *to* the pointer,
/// `nocapture` call arguments, read-only nounwind void calls and null comparisons
/// do not capture; GEP/casts/PHI/select are followed; everything else captures.
pub(crate) fn pointer_may_be_captured(v: V, return_captures: bool, max_uses: usize) -> bool {
    let mut visited: FxHashSet<V> = FxHashSet::default();
    let mut worklist = vec![v];
    let mut count = 0usize;
    while let Some(cur) = worklist.pop() {
        if !visited.insert(cur) {
            continue;
        }
        for u in ffi::uses(cur) {
            count += 1;
            if count > max_uses {
                return true;
            }
            let user = ffi::use_user(u);
            if !ffi::is_instruction(user) {
                return true;
            }
            let opno = ffi::use_operand_no(u).unwrap_or(u32::MAX);
            match ffi::opcode(user) {
                Op::LLVMCall | Op::LLVMInvoke => {
                    let nargs = ffi::num_args(user);
                    if opno >= nargs {
                        // callee operand (or bundle): calling a pointer does not capture it
                        if ffi::called_value(user) == cur {
                            continue;
                        }
                        return true;
                    }
                    let void_ret = ffi::is_void_ty(ffi::type_of(user));
                    let readonly = ffi::call_has_fn_attr(user, "readonly")
                        || ffi::called_function(user).is_some_and(|f| {
                            crate::target::string_attr_value(f, "memory").is_none() && ffi::fn_has_attr(f, "readnone")
                        });
                    let nounwind = ffi::call_has_fn_attr(user, "nounwind");
                    if readonly && nounwind && void_ret {
                        continue;
                    }
                    if ffi::call_param_has_attr(user, opno, "nocapture") {
                        continue;
                    }
                    if let Some(id) = ffi::call_intrinsic_id(user) {
                        let i = Intrinsics::get();
                        if i.is_noop(id) {
                            continue;
                        }
                        if i.is_group(id) {
                            worklist.push(user);
                            continue;
                        }
                    }
                    return true;
                }
                Op::LLVMLoad => {
                    if ffi::is_volatile(user) {
                        return true;
                    }
                }
                Op::LLVMVAArg => {}
                Op::LLVMStore => {
                    if opno == 0 || ffi::is_volatile(user) {
                        return true;
                    }
                }
                Op::LLVMAtomicRMW | Op::LLVMAtomicCmpXchg => {
                    if opno != 0 || ffi::is_volatile(user) {
                        return true;
                    }
                }
                Op::LLVMBitCast | Op::LLVMGetElementPtr | Op::LLVMPHI | Op::LLVMSelect | Op::LLVMAddrSpaceCast => {
                    worklist.push(user);
                }
                Op::LLVMICmp => {
                    let other = if opno == 0 { ffi::operand(user, 1) } else { ffi::operand(user, 0) };
                    if ffi::is_null_ptr_constant(other) {
                        continue;
                    }
                    return true;
                }
                Op::LLVMRet => {
                    if return_captures {
                        return true;
                    }
                }
                _ => return true,
            }
        }
    }
    false
}

/// Port of `AllocationRemover::run`. Lowers small fixed-size allocations to
/// `alloca` and removes unused allocations.
pub fn allocation_remover(func: V) -> bool {
    let info = AllocInfo::new(
        &["seq_alloc", "seq_alloc_atomic", "seq_alloc_uncollectable", "seq_alloc_atomic_uncollectable"],
        "seq_realloc",
        "seq_free",
    );
    allocation_remover_with(&info, func)
}

pub fn allocation_remover_with(info: &AllocInfo, func: V) -> bool {
    if ffi::is_declaration(func) {
        return false;
    }
    let ctx = ffi::value_context(func);
    let mut erase: Vec<V> = Vec::new();
    let mut replace: Vec<(V, V)> = Vec::new();
    let mut allocas: Vec<(u64, u64)> = Vec::new(); // (size, align) -> created later
    let mut alloca_replace: Vec<(V, usize)> = Vec::new();
    let mut untail: Vec<V> = Vec::new();
    let push_erase = |erase: &mut Vec<V>, i: V| {
        if !erase.contains(&i) {
            erase.push(i);
        }
    };

    for mi in ffi::fn_instructions(func) {
        if !ffi::is_call(mi) || !info.is_alloc(mi) {
            continue;
        }
        let mut users = Vec::new();
        if info.is_alloc_site_removable(mi, &mut users) {
            for &instr in &users {
                if ffi::opcode(instr) == Op::LLVMICmp {
                    // isFalseWhenEqual: ne (true when not equal) -> false-when-equal
                    let false_when_equal = ffi::icmp_predicate(instr) == llvm_sys::LLVMIntPredicate::LLVMIntNE;
                    replace.push((instr, ffi::i1_const(ctx, false_when_equal)));
                } else if ffi::opcode(instr) != Op::LLVMStore {
                    replace.push((instr, unsafe { LLVMGetPoison(ffi::type_of(instr)) }));
                }
                push_erase(&mut erase, instr);
            }
            push_erase(&mut erase, mi);
            continue;
        }
        users.clear();
        let mut size = 0u64;
        let mut stack_alignment = 1u64;
        if info.is_alloc_site_demotable(mi, &mut size, &mut users, &mut stack_alignment, 1024) {
            allocas.push((size, stack_alignment));
            alloca_replace.push((mi, allocas.len() - 1));
            push_erase(&mut erase, mi);
            for &instr in &users {
                if info.is_free(instr) {
                    push_erase(&mut erase, instr);
                } else if info.is_realloc(instr) {
                    alloca_replace.push((instr, allocas.len() - 1));
                    push_erase(&mut erase, instr);
                } else if ffi::opcode(instr) == Op::LLVMCall && (ffi::is_tail_call(instr) || ffi::is_must_tail_call(instr)) {
                    untail.push(instr);
                }
            }
        }
    }

    // Create allocas at the entry block's first non-PHI.
    let mut created = Vec::new();
    if !allocas.is_empty() {
        let entry = ffi::entry_block(func).expect("definition has entry");
        let pos = ffi::first_non_phi(entry);
        let b = ffi::Builder::new(ctx);
        match pos {
            Some(p) => b.before(p),
            None => b.at_end(entry),
        }
        for &(size, align) in &allocas {
            unsafe {
                let i8t = LLVMInt8TypeInContext(ctx);
                let n = LLVMConstInt(LLVMInt64TypeInContext(ctx), size, 0);
                let a = LLVMBuildArrayAlloca(b.0, i8t, n, c"".as_ptr());
                LLVMSetAlignment(a, align as u32);
                created.push(a);
            }
        }
    }
    for (i, idx) in alloca_replace {
        replace.push((i, created[idx]));
    }
    for &c in &untail {
        ffi::set_tail_call(c, false);
    }
    for &(a, b) in &replace {
        ffi::replace_all_uses(a, b);
    }
    for &i in &erase {
        ffi::drop_all_references(i);
    }
    for &i in &erase {
        ffi::erase(i);
    }
    !erase.is_empty() || !replace.is_empty() || !created.is_empty() || !untail.is_empty()
}

/// Port of `AllocationHoister`: hoists allocations out of loops.
pub fn allocation_hoister(func: V) -> bool {
    let info = AllocInfo::new(&["seq_alloc", "seq_alloc_atomic"], "seq_realloc", "seq_free");
    allocation_hoister_with(&info, func)
}

pub fn allocation_hoister_with(info: &AllocInfo, func: V) -> bool {
    if ffi::is_declaration(func) {
        return false;
    }
    // Loop headers in worklist order; analyses are recomputed per loop since the
    // transformation splits blocks (C++ updates them incrementally).
    let headers: Vec<BB> = {
        let cfg = Cfg::new(func);
        let dt = DomTree::dominators(&cfg);
        let li = LoopInfo::new(&cfg, &dt);
        li.worklist_order().into_iter().rev().map(|l| cfg.blocks[li.loops[l].header]).collect()
    };
    let mut changed = false;
    // Analyses are only recomputed after a loop was actually transformed; on
    // unchanged IR they are deterministic, so reusing them is equivalent.
    let mut cached: Option<(Cfg, LoopInfo, DomTree, FxHashSet<usize>)> = None;
    // `SmallPriorityWorklist::pop_back_val` processes the last appended loop first.
    for header in headers.into_iter().rev() {
        if cached.is_none() {
            let cfg = Cfg::new(func);
            let dt = DomTree::dominators(&cfg);
            let li = LoopInfo::new(&cfg, &dt);
            let pdt = DomTree::post_dominators(&cfg);
            let irreducible = analysis::irreducible_blocks(&cfg, &dt, &li);
            cached = Some((cfg, li, pdt, irreducible));
        }
        let (cfg, li, pdt, irreducible) = cached.as_ref().unwrap();
        let Some(&h) = cfg.index.get(&header) else { continue };
        let Some(lid) = li.loops.iter().position(|l| l.header == h) else { continue };
        if process_loop(info, func, lid, cfg, li, pdt, irreducible) {
            changed = true;
            cached = None;
        }
    }
    changed
}

fn process_loop(
    info: &AllocInfo,
    func: V,
    lid: usize,
    cfg: &Cfg,
    li: &LoopInfo,
    pdt: &DomTree,
    irreducible: &FxHashSet<usize>,
) -> bool {
    let Some(preheader) = li.preheader(cfg, lid) else { return false };
    let exits: Vec<BB> = li.exit_blocks(cfg, lid).into_iter().map(|e| cfg.blocks[e]).collect();
    let can_release = li.has_dedicated_exits(cfg, lid) && !exits.is_empty() && !exits.iter().any(|&b| ffi::is_eh_pad(b));

    struct Candidate {
        allocation: V,
        releases: Vec<V>,
    }
    let mut hoist = Vec::new();
    for &b in &li.loops[lid].blocks {
        for ins in ffi::instructions(cfg.blocks[b]) {
            if !ffi::is_call(ins) || !info.is_alloc(ins) {
                continue;
            }
            let mut c = Candidate { allocation: ins, releases: vec![] };
            if info.is_alloc_site_hoistable(ins, lid, cfg, li, irreducible, &mut c.releases)
                && (c.releases.is_empty() || can_release)
            {
                hoist.push(c);
            }
        }
    }
    if hoist.is_empty() {
        return false;
    }

    let preheader_bb = cfg.blocks[preheader];
    let terminator = ffi::terminator(preheader_bb).expect("preheader terminator");
    let ctx = ffi::value_context(func);
    let ptr = ffi::ptr_ty(ctx);
    let b = ffi::Builder::new(ctx);
    let mut changed = false;
    // post-dominance must be evaluated on the original CFG
    let simple: Vec<bool> = hoist
        .iter()
        .map(|c| pdt.dominates(cfg.id(ffi::parent_block(c.allocation)), preheader))
        .collect();

    for (c, simple) in hoist.into_iter().zip(simple) {
        let ins = c.allocation;
        let mut cache: Option<V> = None;
        if simple {
            ffi::move_before(ins, terminator);
        } else {
            if ffi::const_zext(ffi::arg(ins, 0)).is_some_and(|n| n <= 1024) {
                continue;
            }
            unsafe {
                // SetInsertPointPastAllocas(parent)
                let entry = ffi::entry_block(func).expect("entry");
                let pos = ffi::instructions(entry).into_iter().find(|&i| ffi::opcode(i) != Op::LLVMAlloca);
                match pos {
                    Some(p) => b.before(p),
                    None => b.at_end(entry),
                }
                let cache_v = LLVMBuildAlloca(b.0, ptr, c"alloc_hoist.cache".as_ptr());
                b.before(terminator);
                LLVMBuildStore(b.0, LLVMConstPointerNull(ptr), cache_v);
                b.before(ins);
                let cached = LLVMBuildLoad2(b.0, ptr, cache_v, c"".as_ptr());
                let is_null = LLVMBuildIsNull(b.0, cached, c"".as_ptr());

                // SplitBlockAndInsertIfThenElse(cond, ins, &yes, &no)
                let head = ffi::parent_block(ins);
                let tail = ffi::split_block(ins, "");
                let br = ffi::terminator(head).expect("split branch");
                ffi::erase(br);
                let yes = ffi::new_block_after(head, "");
                let no = ffi::new_block_after(yes, "");
                b.at_end(yes);
                LLVMBuildBr(b.0, tail);
                b.at_end(no);
                LLVMBuildBr(b.0, tail);
                b.at_end(head);
                LLVMBuildCondBr(b.0, is_null, yes, no);

                // phi at the tail front
                let first = ffi::first_instruction(tail).expect("tail non-empty");
                b.before(first);
                let phi = LLVMBuildPhi(b.0, ptr, c"".as_ptr());
                ffi::replace_all_uses(ins, phi);
                let yes_term = ffi::terminator(yes).expect("yes br");
                ffi::move_before(ins, yes_term);
                b.before(yes_term);
                LLVMBuildStore(b.0, ins, cache_v);
                let mut vals = [ins, cached];
                let mut blks = [yes, no];
                LLVMAddIncoming(phi, vals.as_mut_ptr(), blks.as_mut_ptr(), 2);
                cache = Some(cache_v);
            }
        }
        if let Some(&release) = c.releases.first() {
            let callee = ffi::called_function(release).expect("free callee");
            let cc = unsafe { LLVMGetInstructionCallConv(release) };
            for &exit in &exits {
                match ffi::first_insertion_pt(exit) {
                    Some(p) => b.before(p),
                    None => b.at_end(exit),
                }
                unsafe {
                    let owner = match cache {
                        Some(cv) => LLVMBuildLoad2(b.0, ptr, cv, c"".as_ptr()),
                        None => ins,
                    };
                    let mut a = [owner];
                    let call = LLVMBuildCall2(b.0, ffi::function_type(callee), callee, a.as_mut_ptr(), 1, c"".as_ptr());
                    LLVMSetInstructionCallConv(call, cc);
                }
            }
            for r in c.releases {
                ffi::erase(r);
            }
        }
        changed = true;
    }
    changed
}

/// Port of `AllocationAutoFree`: inserts `seq_free` for non-escaping allocations.
pub fn allocation_auto_free(func: V) -> bool {
    let info = AllocInfo::new(
        &["seq_alloc", "seq_alloc_atomic", "seq_alloc_uncollectable", "seq_alloc_atomic_uncollectable"],
        "seq_realloc",
        "seq_free",
    );
    if ffi::is_declaration(func) {
        return false;
    }
    let cfg = Cfg::new(func);
    let dt = DomTree::dominators(&cfg);
    let pdt = DomTree::post_dominators(&cfg);
    let li = LoopInfo::new(&cfg, &dt);
    let mut changed = false;
    for i in ffi::fn_instructions(func) {
        if ffi::opcode(i) != Op::LLVMCall {
            continue;
        }
        let Some(callee) = ffi::called_function(i) else { continue };
        if !ffi::is_declaration(callee) || !info.is_alloc(i) {
            continue;
        }
        if pointer_may_be_captured(i, true, 20) {
            continue;
        }
        changed |= insert_free(&info, i, func, &cfg, &dt, &pdt, &li);
    }
    changed
}

fn insert_free(info: &AllocInfo, alloc: V, func: V, cfg: &Cfg, dt: &DomTree, pdt: &DomTree, li: &LoopInfo) -> bool {
    let mut worklist = vec![alloc];
    let mut visited: FxHashSet<V> = FxHashSet::default();
    let mut use_blocks = vec![cfg.id(ffi::parent_block(alloc))];
    while let Some(cur) = worklist.pop() {
        if !visited.insert(cur) {
            continue;
        }
        for u in ffi::users(cur) {
            if !ffi::is_instruction(u) {
                continue;
            }
            // PORT-NOTE: C++ calls `info.isFree(call->getCalledFunction())`, passing the
            // Function itself; `isFree` expects a call so this check never fires. Kept as-is.
            if matches!(ffi::opcode(u), Op::LLVMGetElementPtr | Op::LLVMBitCast | Op::LLVMPHI | Op::LLVMSelect) {
                worklist.push(u);
            } else {
                use_blocks.push(cfg.id(ffi::parent_block(u)));
            }
        }
    }
    let mut post: Option<usize> = None;
    for b in use_blocks {
        post = match post {
            None => Some(b),
            Some(p) => match pdt.nearest_common(p, b) {
                Some(x) => Some(x),
                None => return false,
            },
        };
    }
    let Some(mut post) = post else { return false };
    let alloc_loop = li.loop_for(cfg.id(ffi::parent_block(alloc)));
    let mut free_loop = li.loop_for(post);
    while alloc_loop != free_loop {
        let Some(fl) = free_loop else { return false };
        let Some(e) = li.exit_block(cfg, fl) else { return false };
        post = e;
        free_loop = li.loop_for(post);
    }
    if !dt.dominates(cfg.id(ffi::parent_block(alloc)), post) {
        return false;
    }
    let m = ffi::global_module(func);
    let ctx = ffi::value_context(func);
    let free_fn = match ffi::get_function(m, &info.free) {
        Some(f) => f,
        None => unsafe {
            let mut params = [ffi::ptr_ty(ctx)];
            let fty = LLVMFunctionType(LLVMVoidTypeInContext(ctx), params.as_mut_ptr(), 1, 0);
            let n = ffi::cstr(&info.free);
            let f = LLVMAddFunction(m, n.as_ptr(), fty);
            ffi::add_fn_enum_attr(f, "willreturn", 0);
            ffi::add_fn_enum_attr(f, "nounwind", 0);
            f
        },
    };
    let Some(term) = ffi::terminator(cfg.blocks[post]) else { return false };
    let b = ffi::Builder::new(ctx);
    b.before(term);
    unsafe {
        let mut a = [alloc];
        LLVMBuildCall2(b.0, ffi::function_type(free_fn), free_fn, a.as_mut_ptr(), 1, c"".as_ptr());
    }
    true
}
