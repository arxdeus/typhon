//! Constants and instructions (port of llvisitor.cpp "Constants" / "Instructions").

use std::ptr::null_mut;

use typhon_ir::attribute::attr_id;
use typhon_ir::ids::ValueId;
use typhon_ir::module::Module;
use typhon_ir::types::TypeKind;
use typhon_ir::util::irtools;
use typhon_ir::var::FuncKind;

use super::ll::{self, Builder, V};
use super::{LlvmVisitor, TcState, win};

impl<'ctx> LlvmVisitor<'ctx> {
    /// `visit(const StringConst*)`: UTF-8 is decoded into a compact Latin-1/UCS-2/UCS-4
    /// array; the length word packs the kind in its top byte.
    pub(super) fn string_const(&mut self, utf8: &str) -> V {
        // `&str` is valid UTF-8, so decoding cannot fail (C++ validates by hand).
        // PORT-NOTE: C++ literals are WTF-8 (lone surrogates from `\ud800` allowed);
        // the parser stores such non-UTF-8 literals byte-per-char (Latin-1,
        // `bytes_to_string`). Recover the C++ decoding for that form.
        let codepoints: Vec<u32> = Self::wtf8_from_latin1_fallback(utf8)
            .unwrap_or_else(|| utf8.chars().map(|c| c as u32).collect());
        let maxchar = codepoints.iter().copied().max().unwrap_or(0);
        let (kind, elem) = if maxchar > 0xFFFF {
            (3u64, self.b.i32())
        } else if maxchar > 0xFF {
            (2, self.b.i16())
        } else if maxchar > 0x7F {
            (1, self.b.i8())
        } else {
            (0, self.b.i8())
        };
        let mut elements: Vec<V> = codepoints.iter().map(|&c| ll::const_int(elem, c as u64, false)).collect();
        elements.push(ll::const_int(elem, 0, false));
        let arr_ty = ll::array_ty(elem, elements.len() as u64);
        let str_var = ll::new_global(
            self.m(),
            arr_ty,
            true,
            llvm_sys::LLVMLinkage::LLVMPrivateLinkage,
            Some(ll::const_array(elem, &elements)),
            ".str",
        );
        ll::set_unnamed_addr(str_var, llvm_sys::LLVMUnnamedAddr::LLVMGlobalUnnamedAddr);
        let str_type = ll::struct_ty(self.c, &[self.b.ptr(), self.b.i64()]);
        const LENGTH_BITS: u64 = 56;
        assert!((codepoints.len() as u64) < (1u64 << LENGTH_BITS), "string constant too large");
        let meta = self.b.c_i64((codepoints.len() as u64 | (kind << LENGTH_BITS)) as i64);
        // IRBuilder constant-folds the insertvalues of constants: emit the constant directly.
        let _ = str_type;
        ll::const_struct(self.c, &[str_var, meta])
    }

    /// If every char is <= 0xFF and the bytes are invalid UTF-8 but valid WTF-8 (the
    /// C++ `visit(StringConst*)` grammar, which admits surrogates), decode them.
    fn wtf8_from_latin1_fallback(s: &str) -> Option<Vec<u32>> {
        if s.is_ascii() || s.chars().any(|c| c as u32 > 0xFF) {
            return None;
        }
        let b: Vec<u8> = s.chars().map(|c| c as u8).collect();
        if std::str::from_utf8(&b).is_ok() {
            return None;
        }
        let mut out = Vec::with_capacity(b.len());
        let mut pos = 0;
        while pos < b.len() {
            let byte = b[pos];
            let (mut cp, width) = match byte {
                0..=0x7F => (byte as u32, 1),
                0xC2..=0xDF => ((byte & 0x1F) as u32, 2),
                0xE0..=0xEF => ((byte & 0x0F) as u32, 3),
                0xF0..=0xF4 => ((byte & 0x07) as u32, 4),
                _ => return None,
            };
            if pos + width > b.len() {
                return None;
            }
            for &c in &b[pos + 1..pos + width] {
                if c & 0xC0 != 0x80 {
                    return None;
                }
                cp = (cp << 6) | (c & 0x3F) as u32;
            }
            let ok = (width == 1 || cp >= 0x80)
                && (width != 3 || cp >= 0x800)
                && (width != 4 || cp >= 0x10000)
                && cp <= 0x10FFFF;
            if !ok {
                return None;
            }
            out.push(cp);
            pos += width;
        }
        Some(out)
    }

    /// `visit(const BytesConst*)`.
    pub(super) fn bytes_const(&mut self, bytes: &[u8]) -> V {
        let i8t = self.b.i8();
        let mut elements: Vec<V> = bytes.iter().map(|&b| ll::const_int(i8t, b as u64, false)).collect();
        elements.push(ll::const_int(i8t, 0, false));
        let arr_ty = ll::array_ty(i8t, elements.len() as u64);
        let var = ll::new_global(
            self.m(),
            arr_ty,
            true,
            llvm_sys::LLVMLinkage::LLVMPrivateLinkage,
            Some(ll::const_array(i8t, &elements)),
            ".bytes",
        );
        ll::set_unnamed_addr(var, llvm_sys::LLVMUnnamedAddr::LLVMGlobalUnnamedAddr);
        let _ = ll::struct_ty(self.c, &[self.b.ptr(), self.b.i64()]);
        ll::const_struct(self.c, &[var, self.b.c_i64(bytes.len() as i64)])
    }

    /// `visit(const VarValue*)`.
    pub(super) fn visit_var_value(&mut self, m: &Module, v: ValueId) {
        let var = m.actual_var(m.as_var_value(v).unwrap().var);
        if m.is_func(var) {
            self.value = self.get_function_value(m, var);
            assert!(!self.value.is_null(), "function value not found");
            return;
        }
        let llvm_type = self.get_llvm_type(m, m.value_type(v));
        if !self.is_storable_type(llvm_type) {
            self.value = self.get_dummy_value(llvm_type);
            return;
        }
        let mut ptr = self.get_var(m, var);
        assert!(!ptr.is_null(), "{} value not found", m.var_name(var));
        self.b.at_end(self.block);
        if m.is_thread_local(var) {
            ptr = self.b.thread_local_address(self.m(), ptr);
        }
        self.value = self.b.load(llvm_type, ptr);
    }

    /// `visit(const PointerValue*)`.
    pub(super) fn visit_pointer_value(&mut self, m: &Module, v: ValueId) {
        let x = m.as_pointer_value(v).unwrap().clone();
        let var_id = m.actual_var(x.var);
        let mut var = self.get_var(m, var_id);
        assert!(!var.is_null(), "variable not found");
        self.b.at_end(self.block);
        if m.is_thread_local(var_id) {
            var = self.b.thread_local_address(self.m(), var);
        }
        if x.fields.is_empty() {
            self.value = var; // note: we don't load the pointer
            return;
        }
        let mut ty = m.var_type(var_id).unwrap();
        for field in &x.fields {
            let t = m.actual_type(ty);
            match m.type_kind(t) {
                TypeKind::Ref { .. } => {
                    let memb = m.member_type(t, field).unwrap_or_else(|| panic!("field {field} not found in reference type"));
                    let lt = self.get_llvm_type(m, t);
                    let object = self.b.load(lt, var);
                    var = self.get_field_address(m, object, t, field, None);
                    ty = memb;
                }
                TypeKind::Record { .. } => {
                    let idx = m.member_index(t, field);
                    let memb = m.member_type(t, field);
                    assert!(idx >= 0 && memb.is_some(), "field {field} not found in record type");
                    let lt = self.get_llvm_type(m, t);
                    var = self.b.struct_gep(lt, var, idx as u32);
                    ty = memb.unwrap();
                }
                _ => panic!("type in pointer value was not a record or reference type"),
            }
        }
        self.value = var;
    }

    /// `visit(const AssignInstr*)`.
    pub(super) fn visit_assign_instr(&mut self, m: &Module, v: ValueId) {
        let x = m.as_assign_instr(v).unwrap().clone();
        let lhs = m.actual_var(x.lhs);
        let mut var = self.get_var(m, lhs);
        assert!(!var.is_null(), "could not find {} var", m.var_name(lhs));
        self.process_value(m, x.rhs);
        let lt = self.get_llvm_type(m, m.var_type(lhs).unwrap());
        if self.is_storable_type(lt) {
            self.b.at_end(self.block);
            if m.is_thread_local(lhs) {
                var = self.b.thread_local_address(self.m(), var);
            }
            self.b.store(self.value, var);
        }
    }

    /// `visit(const ExtractInstr*)`.
    pub(super) fn visit_extract_instr(&mut self, m: &Module, v: ValueId) {
        let x = m.as_extract_instr(v).unwrap().clone();
        let t = m.actual_type(m.value_type(x.val));
        assert!(m.type_kind(t).is_membered(), "{} is not a membered type", m.type_name(t));
        let index = m.member_index(t, &x.field);
        assert!(index >= 0, "invalid index");
        self.process_value(m, x.val);
        self.b.at_end(self.block);
        if m.type_kind(t).is_ref() {
            let exact = m.get_attribute(v.into(), attr_id::EXACT_TYPE).and_then(|a| a.as_exact_type());
            let address = self.get_field_address(m, self.value, t, &x.field, exact);
            let rt = self.get_llvm_type(m, m.value_type(v));
            self.value = self.b.load(rt, address);
            return;
        }
        self.value = self.b.extract_value(self.value, index as u32);
    }

    /// `visit(const InsertInstr*)`.
    pub(super) fn visit_insert_instr(&mut self, m: &Module, v: ValueId) {
        let x = m.as_insert_instr(v).unwrap().clone();
        let t = m.actual_type(m.value_type(x.lhs));
        assert!(m.type_kind(t).is_ref(), "{} is not a reference type", m.type_name(t));
        assert!(m.member_index(t, &x.field) >= 0, "invalid index");
        self.process_value(m, x.lhs);
        let lhs = self.value;
        self.process_value(m, x.rhs);
        let rhs = self.value;
        self.b.at_end(self.block);
        let exact = m.get_attribute(v.into(), attr_id::EXACT_TYPE).and_then(|a| a.as_exact_type());
        let address = self.get_field_address(m, lhs, t, &x.field, exact);
        self.b.store(rhs, address);
        self.value = lhs;
    }

    /// `visit(const CallInstr*)`.
    pub(super) fn visit_call_instr(&mut self, m: &Module, v: ValueId) {
        let x = m.as_call_instr(v).unwrap().clone();
        self.b.at_end(self.block);
        let direct = irtools::get_func(m, m.actual_value(x.callee));
        if let Some(f) = direct
            && let FuncKind::Internal { intrinsic, .. } = m.func(f).unwrap().kind
            && intrinsic.is_type_query()
        {
            assert!(x.args.is_empty(), "type query cannot have runtime arguments");
            self.value = self.codegen_type_query(m, f);
            return;
        }
        // Function values use native entry points on Win64. A direct call keeps
        // the logical signature, so do not lower its callee as a first-class value.
        let mut f = match direct {
            Some(d) => self.get_func(m, d),
            None => {
                self.process_value(m, x.callee);
                self.value
            }
        };
        let mut native_indirect = direct.is_none();
        let func_type = self.get_llvm_func_type(m, m.value_type(x.callee));
        let mut targets: Vec<(i64, V)> = Vec::new();
        let mut dynamic_id: V = null_mut();
        // Virtual dispatch specialization.
        if ll::is_call(f) {
            let lookup = f;
            let called = ll::called_value(lookup);
            let receiver = self.virtual_lookup_types.get(&(called as usize)).copied();
            let thunk = if ll::num_arg_operands(lookup) == 2 {
                let t = ll::operand(lookup, 1);
                ll::is_const_int(t).then_some(t)
            } else {
                None
            };
            if let (Some(rtype), Some(thunk)) = (receiver, thunk)
                && ll::use_empty(lookup)
                && let Some(info) = m.runtime_info(m.actual_type(rtype))
            {
                let key = (info.id, ll::const_sext(thunk));
                if !self.virtual_targets.contains_key(&key) {
                    const MAX_DIRECT_TYPES: usize = 4;
                    const MAX_SINGLETON_TYPES: usize = 64;
                    let mut candidates: Vec<(i64, V)> = Vec::new();
                    let mut singleton = true;
                    let rt = m.actual_type(rtype);
                    for ty in m.types() {
                        let Some(tinfo) = m.runtime_info(ty) else { continue };
                        if !tinfo.mro.iter().any(|b| m.actual_type(*b) == rt) {
                            continue;
                        }
                        let Some(&target) = tinfo.methods.get(&(key.1 as usize)) else {
                            candidates.clear();
                            break;
                        };
                        let tid = tinfo.id;
                        let function = self.get_func(m, target);
                        if function.is_null() || ll::function_type(function) != func_type || candidates.len() == MAX_SINGLETON_TYPES {
                            candidates.clear();
                            break;
                        }
                        candidates.push((tid, function));
                        singleton &= function == candidates[0].1;
                        if !singleton && candidates.len() > MAX_DIRECT_TYPES {
                            candidates.clear();
                            break;
                        }
                    }
                    self.virtual_targets.insert(key, candidates);
                }
                targets = self.virtual_targets[&key].clone();
                if !targets.is_empty() {
                    let singleton = targets.iter().all(|t| t.1 == targets[0].1);
                    if singleton {
                        // Different concrete classes can share one implementation.
                        f = targets[0].1;
                        native_indirect = false;
                        targets.clear();
                    } else {
                        let header = self.b.load(self.b.ptr(), ll::operand(lookup, 0));
                        dynamic_id = self.load_runtime_field(header, rtti_field::TYPE_ID);
                        let valid = self.new_block_pub("dispatch.valid");
                        let missing = self.new_block_pub("dispatch.missing");
                        let selection = self.b.switch(dynamic_id, missing, targets.len() as u32);
                        for (id, _) in &targets {
                            Builder::add_case(selection, self.b.c_i64(*id), valid);
                        }
                        self.b.at_end(missing);
                        let (tt, tf) = ll::intrinsic(self.m(), "llvm.trap", &[]);
                        self.b.call(tt, tf, &[]);
                        self.b.unreachable();
                        self.b.at_end(valid);
                        self.block = valid;
                    }
                    ll::erase(lookup);
                }
            }
        }

        let mut args = Vec::new();
        for a in &x.args {
            self.b.at_end(self.block);
            self.process_value(m, *a);
            args.push(self.value);
        }

        if !targets.is_empty() {
            let result_type = self.get_llvm_type(m, m.value_type(v));
            let done = self.new_block_pub("dispatch.done");
            let missing = self.new_block_pub("dispatch.unreachable");
            self.b.at_end(self.block);
            let selection = self.b.switch(dynamic_id, missing, targets.len() as u32);
            let mut results = Vec::new();
            for (id, target) in &targets {
                let dest = self.new_block_pub("dispatch.call");
                Builder::add_case(selection, self.b.c_i64(*id), dest);
                self.b.at_end(dest);
                self.block = dest;
                let r = self.call(func_type, *target, &args);
                // call() can emit an invoke and move block to its normal continuation.
                results.push((r, self.block));
                self.b.br(done);
            }
            self.b.at_end(missing);
            self.b.unreachable();
            self.b.at_end(done);
            self.block = done;
            if self.is_storable_type(result_type) {
                let phi = self.b.phi(result_type);
                for (val, pred) in results {
                    Builder::add_incoming(phi, val, pred);
                }
                self.value = phi;
            } else {
                self.value = self.get_dummy_value(result_type);
            }
            return;
        }

        let callee_is_external = direct.is_some_and(|d| m.func(d).unwrap().is_external());
        if self.is_win_msvc() && (native_indirect || callee_is_external) {
            self.b.at_end(self.block);
            let module = self.m();
            // Emit through `self.call` so try/catch invokes are preserved.
            let prepared = win::prepare(module, &self.b, func_type, &args);
            let c = self.call(prepared.sig, f, &prepared.args);
            self.b.at_end(self.block);
            let result = win::finish(module, &self.b, &prepared, c);
            let rt = self.get_llvm_type(m, m.value_type(v));
            self.value = match result {
                Some(r) if self.is_storable_type(rt) => r,
                _ => self.get_dummy_value(rt),
            };
            return;
        }

        self.b.at_end(self.block);
        let call_result = self.call(func_type, f, &args);
        let rt = self.get_llvm_type(m, m.value_type(v));
        if !self.is_storable_type(rt) {
            self.value = self.get_dummy_value(rt);
            return;
        }
        self.value = call_result;
    }

    pub(super) fn new_block_pub(&self, name: &str) -> ll::BB {
        ll::append_block(self.c, self.func, name)
    }

    /// `visit(const YieldInInstr*)`.
    pub(super) fn visit_yield_in_instr(&mut self, m: &Module, v: ValueId) {
        let x = m.as_yield_in_instr(v).unwrap().clone();
        self.b.at_end(self.block);
        if x.suspend {
            let (st, sf) = ll::intrinsic(self.m(), "llvm.coro.suspend", &[]);
            let susp = self.b.call(st, sf, &[ll::token_none(self.c), self.b.c_i1(false)]);
            self.block = self.new_block_pub("yieldin.new");
            let sw = self.b.switch(susp, self.coro.suspend, 2);
            Builder::add_case(sw, self.b.c_i8(0), self.block);
            Builder::add_case(sw, self.b.c_i8(1), self.coro.cleanup);
            self.check_pending_generator_exc();
            self.b.at_end(self.block);
        }
        let t = self.get_llvm_type(m, m.value_type(v));
        self.value = self.b.load(t, self.coro.promise);
    }

    /// `visit(const StackAllocInstr*)`.
    pub(super) fn visit_stack_alloc_instr(&mut self, m: &Module, v: ValueId) {
        let x = m.as_stack_alloc_instr(v).unwrap().clone();
        let pt = m.actual_type(m.value_type(v));
        let TypeKind::Pointer { base } = *m.type_kind(pt) else {
            panic!("stack alloc did not have ptr type");
        };
        let term = ll::terminator(ll::entry_block(self.func)).expect("entry block has no terminator");
        self.b.before(term);
        let bt = self.get_llvm_type(m, base);
        self.b.before(term);
        self.value = self.b.array_alloca(bt, self.b.c_i64(x.count));
    }

    /// `visit(const TernaryInstr*)`.
    pub(super) fn visit_ternary_instr(&mut self, m: &Module, v: ValueId) {
        let x = m.as_ternary_instr(v).unwrap().clone();
        let mut true_block = self.new_block_pub("ternary.true");
        let mut false_block = self.new_block_pub("ternary.false");
        let exit_block = self.new_block_pub("ternary.exit");
        let value_type = self.get_llvm_type(m, m.value_type(v));
        self.process_value(m, x.cond);
        let cond = self.value;
        self.b.at_end(self.block);
        let cond = self.b.trunc(cond, self.b.i1());
        self.b.cond_br(cond, true_block, false_block);

        self.block = true_block;
        self.process_value(m, x.true_value);
        let true_value = self.value;
        true_block = self.block;
        self.b.at_end(true_block);
        self.b.br(exit_block);

        self.block = false_block;
        self.process_value(m, x.false_value);
        let false_value = self.value;
        false_block = self.block;
        self.b.at_end(false_block);
        self.b.br(exit_block);

        self.b.at_end(exit_block);
        let phi = self.b.phi(value_type);
        Builder::add_incoming(phi, true_value, true_block);
        Builder::add_incoming(phi, false_value, false_block);
        self.value = phi;
        self.block = exit_block;
    }

    fn break_or_continue(&mut self, m: &Module, loop_: Option<ValueId>, is_break: bool) {
        assert!(!self.loops.is_empty(), "not in a loop");
        self.b.at_end(self.block);
        let l = match loop_ {
            None => self.loops.last().unwrap().clone(),
            Some(l) => self.get_loop_data(m.value_uid(l)).expect("loop not found"),
        };
        let fin = self.finally.last().cloned();
        match fin {
            Some(tc) if tc.sequence_number >= l.sequence_number => {
                let state = if is_break { TcState::Break } else { TcState::Continue };
                self.b.store(self.b.c_i8(state as i64), tc.exc_flag);
                self.b.store(self.b.c_i64(l.sequence_number), tc.loop_sequence);
                self.b.br(tc.finally_block);
            }
            _ => {
                self.b.br(if is_break { l.break_block } else { l.continue_block });
            }
        }
        self.block = self.new_block_pub(if is_break { "break.new" } else { "continue.new" });
    }

    /// `visit(const BreakInstr*)`.
    pub(super) fn visit_break_instr(&mut self, m: &Module, v: ValueId) {
        let l = m.as_break_instr(v).unwrap().loop_;
        self.break_or_continue(m, l, true);
    }

    /// `visit(const ContinueInstr*)`.
    pub(super) fn visit_continue_instr(&mut self, m: &Module, v: ValueId) {
        let l = m.as_continue_instr(v).unwrap().loop_;
        self.break_or_continue(m, l, false);
    }

    /// `visit(const ReturnInstr*)`.
    pub(super) fn visit_return_instr(&mut self, m: &Module, v: ValueId) {
        let val = m.as_return_instr(v).unwrap().value;
        if let Some(x) = val {
            self.process_value(m, x);
        }
        self.b.at_end(self.block);
        let ret_state = self.b.c_i8(TcState::Return as i64);
        if !self.coro.exit.is_null() {
            if self.coro.is_async {
                self.b.store(self.value, self.coro.promise);
            }
            if let Some(tc) = self.innermost_try_catch() {
                self.b.store(ret_state, tc.exc_flag);
                self.b.br(tc.finally_block);
            } else {
                self.b.br(self.coro.exit);
            }
        } else if let Some(tc) = self.innermost_try_catch() {
            self.b.store(ret_state, tc.exc_flag);
            if !tc.ret_store.is_null() {
                assert!(!self.value.is_null(), "no return value storage");
                self.b.store(self.value, tc.ret_store);
            }
            self.b.br(tc.finally_block);
        } else if val.is_some() {
            self.b.ret(self.value);
        } else {
            self.b.ret_void();
        }
        self.block = self.new_block_pub("return.new");
    }

    /// `visit(const YieldInstr*)`.
    pub(super) fn visit_yield_instr(&mut self, m: &Module, v: ValueId) {
        let x = m.as_yield_instr(v).unwrap().clone();
        if x.final_ {
            if let Some(val) = x.value {
                assert!(!self.coro.promise.is_null(), "no coroutine promise");
                self.process_value(m, val);
                self.b.at_end(self.block);
                self.b.store(self.value, self.coro.promise);
            }
            self.b.at_end(self.block);
            if let Some(tc) = self.innermost_try_catch() {
                self.b.store(self.b.c_i8(TcState::Return as i64), tc.exc_flag);
                self.b.br(tc.finally_block);
            } else {
                self.b.br(self.coro.exit);
            }
            self.block = self.new_block_pub("yield.new");
        } else if let Some(val) = x.value {
            self.process_value(m, val);
            let r = self.value;
            self.make_yield(Some(r), false);
        } else {
            self.make_yield(None, false);
        }
    }

    /// `visit(const ThrowInstr*)`.
    pub(super) fn visit_throw_instr(&mut self, m: &Module, v: ValueId) {
        let x = m.as_throw_instr(v).unwrap().clone();
        if self.options.noexc {
            self.b.at_end(self.block);
            self.b.unreachable();
            self.block = self.new_block_pub("throw_unreachable.new");
            return;
        }
        // note: exception header should be set in the frontend
        let (eat, eaf) = self.make_exc_alloc_func();
        let (tt, tf) = self.make_throw_func();
        let obj;
        if let Some(val) = x.value {
            self.process_value(m, val);
            obj = self.value;
        } else {
            obj = self.catches.last().map(|c| c.exception).unwrap_or_else(|| self.b.null_ptr());
            self.b.at_end(self.block);
            let no_exception = self.new_block_pub("raise.empty");
            let raise_exception = self.new_block_pub("raise.active");
            let is_null = self.b.is_null(obj);
            self.b.cond_br(is_null, no_exception, raise_exception);
            self.block = no_exception;
            let fb = x.fallback.expect("empty raise instruction not lowered");
            self.process_value(m, fb);
            self.b.at_end(self.block);
            self.b.unreachable();
            self.block = raise_exception;
        }
        self.b.at_end(self.block);
        let exc = self.b.call(eat, eaf, &[obj]);
        self.call(tt, tf, &[exc]);
    }
}

/// `runtime::TypeInfoField` indices (runtime rtti layout).
pub(super) mod rtti_field {
    pub const TYPE_ID: u32 = 0;
    pub const MRO_COUNT: u32 = 1;
    pub const MRO_OFFSETS: u32 = 3;
    pub const METHOD_COUNT: u32 = 7;
    pub const METHODS: u32 = 8;
}
