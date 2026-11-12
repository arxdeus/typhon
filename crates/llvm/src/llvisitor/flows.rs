//! Control-flow codegen (port of llvisitor.cpp "Control flow").

use std::ptr::null_mut;

use llvm_sys::LLVMIntPredicate::*;
use typhon_ir::ids::{TypeId, ValueId};
use typhon_ir::module::Module;
use typhon_ir::types::TypeKind;
use typhon_ir::value::{Stage, ValueKind};

use super::ll::{self, BB, Builder, V};
use super::{CatchData, LlvmVisitor, LoopData, SEQ_EXCEPTION_CLASS, TcState, TryCatchData, seq_exc_offset};

/// `anyMatch`.
fn any_match(m: &Module, t: Option<TypeId>, types: &[Option<TypeId>]) -> bool {
    match t {
        Some(t) => types.iter().any(|x| x.is_some_and(|x| m.type_name(x) == m.type_name(t))),
        None => types.iter().any(|x| x.is_none()),
    }
}

impl<'ctx> LlvmVisitor<'ctx> {
    fn new_block(&self, name: &str) -> BB {
        ll::append_block(self.c, self.func, name)
    }
    fn trunc_i1(&self, v: V) -> V {
        self.b.trunc(v, self.b.i1())
    }

    /// `visit(const IfFlow*)`.
    pub(super) fn visit_if_flow(&mut self, m: &Module, v: ValueId) {
        let Some(x) = m.as_if_flow(v).cloned() else { unreachable!() };
        let true_block = self.new_block("if.true");
        let false_block = self.new_block("if.false");
        let exit_block = self.new_block("if.exit");

        self.process_value(m, x.cond);
        let cond = self.value;
        self.b.at_end(self.block);
        let cond = self.trunc_i1(cond);
        self.b.cond_br(cond, true_block, false_block);

        self.block = true_block;
        self.process_value(m, x.true_branch);
        self.b.at_end(self.block);
        self.b.br(exit_block);

        self.block = false_block;
        if let Some(f) = x.false_branch {
            self.process_value(m, f);
        }
        self.b.at_end(self.block);
        self.b.br(exit_block);

        self.block = exit_block;
    }

    /// `visit(const WhileFlow*)`.
    pub(super) fn visit_while_flow(&mut self, m: &Module, v: ValueId) {
        let Some(x) = m.as_while_flow(v).cloned() else { unreachable!() };
        let cond_block = self.new_block("while.cond");
        let body_block = self.new_block("while.body");
        let exit_block = self.new_block("while.exit");

        self.b.at_end(self.block);
        self.b.br(cond_block);

        self.block = cond_block;
        self.process_value(m, x.cond);
        let cond = self.value;
        self.b.at_end(self.block);
        let cond = self.trunc_i1(cond);
        self.b.cond_br(cond, body_block, exit_block);

        self.block = body_block;
        self.enter_loop(LoopData { sequence_number: -1, break_block: exit_block, continue_block: cond_block, loop_id: m.value_uid(v) });
        self.process_value(m, x.body);
        self.exit_loop();
        self.b.at_end(self.block);
        self.b.br(cond_block);

        self.block = exit_block;
    }

    /// Generator iteration shared by `ForFlow` and generator pipeline stages.
    fn coro_intrinsics(&self) -> [(ll::Ty, V); 4] {
        let mm = self.m();
        [
            ll::intrinsic(mm, "llvm.coro.resume", &[]),
            ll::intrinsic(mm, "llvm.coro.done", &[]),
            ll::intrinsic(mm, "llvm.coro.promise", &[]),
            ll::intrinsic(mm, "llvm.coro.destroy", &[]),
        ]
    }

    /// `visit(const ForFlow*)`.
    pub(super) fn visit_for_flow(&mut self, m: &Module, v: ValueId) {
        let Some(x) = m.as_for_flow(v).cloned() else { unreachable!() };
        assert!(x.schedule.is_none(), "parallel for-loop not lowered");
        let loop_var_type = self.get_llvm_type(m, m.var_type(x.var).unwrap());
        let loop_var = self.get_var(m, x.var);
        assert!(!loop_var.is_null(), "loop variable not found");

        let cond_block = self.new_block("for.cond");
        let body_block = self.new_block("for.body");
        let check_block = self.new_block("for.check");
        let cleanup_block = self.new_block("for.cleanup");
        let exit_block = self.new_block("for.exit");

        let [resume, done_f, promise_f, destroy] = self.coro_intrinsics();

        self.process_value(m, x.iter);
        let iter = self.value;
        self.b.at_end(self.block);
        let d = self.b.call(done_f.0, done_f.1, &[iter]);
        self.b.cond_br(d, exit_block, cond_block);

        self.b.at_end(check_block);
        let d = self.b.call(done_f.0, done_f.1, &[iter]);
        self.b.cond_br(d, exit_block, cond_block);

        self.block = cond_block;
        self.call(resume.0, resume.1, &[iter]);
        self.b.at_end(self.block);
        let done = self.b.call(done_f.0, done_f.1, &[iter]);
        self.b.cond_br(done, cleanup_block, body_block);

        if self.is_storable_type(loop_var_type) {
            self.b.at_end(body_block);
            let alignment = self.b.c_i32(ll::pref_align(self.m(), loop_var_type) as i64);
            let from = self.b.c_i1(false);
            let promise = self.b.call(promise_f.0, promise_f.1, &[iter, alignment, from]);
            let generated = self.b.load(loop_var_type, promise);
            self.b.store(generated, loop_var);
        }

        self.block = body_block;
        self.enter_loop(LoopData { sequence_number: -1, break_block: exit_block, continue_block: check_block, loop_id: m.value_uid(v) });
        self.process_value(m, x.body);
        self.exit_loop();
        self.b.at_end(self.block);
        self.b.br(check_block);

        self.b.at_end(cleanup_block);
        self.b.call(destroy.0, destroy.1, &[iter]);
        self.b.br(exit_block);

        self.block = exit_block;
    }

    /// `visit(const ImperativeForFlow*)`.
    pub(super) fn visit_imperative_for_flow(&mut self, m: &Module, v: ValueId) {
        let Some(x) = m.as_imperative_for_flow(v).cloned() else { unreachable!() };
        assert!(x.schedule.is_none(), "parallel for-loop not lowered");
        let loop_var = self.get_var(m, x.var);
        assert!(!loop_var.is_null(), "loop variable not found");
        assert!(x.step != 0, "step cannot be 0");

        let cond_block = self.new_block("imp_for.cond");
        let body_block = self.new_block("imp_for.body");
        let update_block = self.new_block("imp_for.update");
        let exit_block = self.new_block("imp_for.exit");

        self.process_value(m, x.start);
        let start = self.value;
        self.process_value(m, x.end);
        let end = self.value;

        self.b.at_end(self.block);
        self.b.br(cond_block);
        self.b.at_end(cond_block);
        let phi = self.b.phi(self.b.i64());
        Builder::add_incoming(phi, start, self.block);
        let done = if x.step > 0 { self.b.icmp(LLVMIntSGE, phi, end) } else { self.b.icmp(LLVMIntSLE, phi, end) };
        self.b.cond_br(done, exit_block, body_block);

        self.b.at_end(body_block);
        self.b.store(phi, loop_var);
        self.block = body_block;

        self.enter_loop(LoopData { sequence_number: -1, break_block: exit_block, continue_block: update_block, loop_id: m.value_uid(v) });
        self.process_value(m, x.body);
        self.exit_loop();
        self.b.at_end(self.block);
        self.b.br(update_block);

        self.b.at_end(update_block);
        let next = self.b.add(phi, self.b.c_i64(x.step));
        Builder::add_incoming(phi, next, update_block);
        self.b.br(cond_block);

        self.block = exit_block;
    }

    /// Windows: catchswitch + catchpad recovering the exception into `catch_store`.
    fn win_catchpad_to_store(&mut self, name: &str, catch_store: V) -> BB {
        let pad_type = self.get_pad_type();
        let cs = self.b.catch_switch(ll::token_none(self.c), None, 1);
        let catch_block = self.new_block(name);
        Builder::add_handler(cs, catch_block);
        self.b.at_end(catch_block);
        let (_, filter) = self.make_win_eh_filter();
        let cp = self.b.catch_pad(cs, &[filter]);
        let (ect, ecf) = self.exc_current_func();
        let exc = self.b.call_funclet(ect, ecf, &[], cp);
        let mut pad = ll::undef(pad_type);
        pad = self.b.insert_value(pad, exc, 0);
        pad = self.b.insert_value(pad, self.b.c_i32(0), 1);
        self.b.store(pad, catch_store);
        let cont = self.new_block(&format!("{name}.ret"));
        self.b.catch_ret(cp, cont);
        self.b.at_end(cont);
        cont
    }

    /// `visit(const TryCatchFlow*)`.
    pub(super) fn visit_try_catch_flow(&mut self, m: &Module, v: ValueId) {
        let Some(x) = m.as_try_catch_flow(v).cloned() else { unreachable!() };
        let is_root = self.innermost_try_catch().is_none();
        let parent_finally_sequence = self.finally.last().map(|f| f.sequence_number).unwrap_or(-1);
        let support_break_and_continue = !self.loops.is_empty();
        let have_finally = x.finally.is_some_and(|f| m.as_series_flow(m.actual_value(f)).is_some_and(|s| !s.series.is_empty()));
        self.b.at_end(self.block);
        let entry_block = self.new_block("trycatch.entry");
        self.b.br(entry_block);

        let mut tc = TryCatchData {
            exception_block: self.new_block("trycatch.exception"),
            exception_route_block: self.new_block("trycatch.exception_route"),
            finally_block: self.new_block("trycatch.finally"),
            finally_exception_block: self.new_block("trycatch.finally.exception"),
            ..Default::default()
        };

        let external_exc_block = self.new_block("trycatch.exception_external");
        let unwind_resume_block = self.new_block("trycatch.unwind_resume");
        let rethrow_block = self.new_block("trycatch.rethrow");
        let end_block = self.new_block("trycatch.end");

        let entry_term = ll::terminator(ll::entry_block(self.func)).expect("entry block has no terminator");
        self.b.before(entry_term);
        let st = |s: TcState| s as i64;
        let pad_type = self.get_pad_type();
        let unwind_type = ll::struct_ty(self.c, &[self.b.i64()]); // header only
        let ret_ty = ll::fn_ret_ty(ll::function_type(self.func));

        if is_root {
            tc.exc_flag = self.b.alloca(self.b.i8());
            tc.catch_store = self.b.alloca(pad_type);
            tc.delegate_depth = self.b.alloca(self.b.i64());
            tc.ret_store = if !self.coro.exit.is_null() || ll::is_void(ret_ty) { null_mut() } else { self.b.alloca(ret_ty) };
            tc.loop_sequence = self.b.alloca(self.b.i64());
            self.b.store(self.b.c_i8(st(TcState::NotThrown)), tc.exc_flag);
            self.b.store(ll::const_null(pad_type), tc.catch_store);
            self.b.store(self.b.c_i64(0), tc.delegate_depth);
            self.b.store(self.b.c_i64(-1), tc.loop_sequence);
        } else {
            let parent = self.innermost_try_catch().unwrap();
            tc.exc_flag = parent.exc_flag;
            tc.catch_store = parent.catch_store;
            tc.delegate_depth = parent.delegate_depth;
            tc.ret_store = parent.ret_store;
            tc.loop_sequence = parent.loop_sequence;
        }

        // translate finally
        self.block = tc.finally_block;
        self.b.at_end(self.block);
        let mut saved_state: Vec<(V, V)> = Vec::new();
        if have_finally {
            let save = |this: &Self, saved: &mut Vec<(V, V)>, storage: V, ty: ll::Ty| {
                let s = this.b.load(ty, storage);
                saved.push((storage, s));
                s
            };
            let pending_state = save(self, &mut saved_state, tc.exc_flag, self.b.i8());
            let pending_exception = save(self, &mut saved_state, tc.catch_store, pad_type);
            save(self, &mut saved_state, tc.delegate_depth, self.b.i64());
            save(self, &mut saved_state, tc.loop_sequence, self.b.i64());
            if !tc.ret_store.is_null() {
                save(self, &mut saved_state, tc.ret_store, ret_ty);
            }
            self.b.store(self.b.c_i8(st(TcState::NotThrown)), tc.exc_flag);
            self.b.store(self.b.c_i64(0), tc.delegate_depth);

            let exception_active = self.new_block("trycatch.finally.active");
            let finally_body = self.new_block("trycatch.finally.body");
            let has_exception = self.b.or(
                self.b.icmp_eq(pending_state, self.b.c_i8(st(TcState::Thrown))),
                self.b.icmp_eq(pending_state, self.b.c_i8(st(TcState::Rethrow))),
            );
            self.b.cond_br(has_exception, exception_active, finally_body);
            self.b.at_end(exception_active);
            let unwind_exception = self.b.extract_value(pending_exception, 0);
            let exception_ptr = self.b.const_gep_i8(unwind_exception, seq_exc_offset(ll::pointer_size(self.m())));
            let exception = self.b.load(self.b.ptr(), exception_ptr);
            self.b.br(finally_body);
            self.b.at_end(finally_body);
            let active = self.b.phi(self.b.ptr());
            Builder::add_incoming(active, exception, exception_active);
            let outer = self.catches.last().map(|c| c.exception).unwrap_or_else(|| self.b.null_ptr());
            Builder::add_incoming(active, outer, tc.finally_block);
            self.enter_catch(CatchData { sequence_number: -1, exception: active, type_id: null_mut() });
            self.block = finally_body;
        }
        if let Some(f) = x.finally {
            self.process_value(m, f);
        }
        if have_finally {
            self.exit_catch();
        }
        let mut finally_block = self.block;
        self.b.at_end(finally_block);
        for (storage, value) in &saved_state {
            self.b.store(*value, *storage);
        }
        let exc_flag_read = self.b.load(self.b.i8(), tc.exc_flag);

        if let Some(parent_tc) = self.trycatch.last().cloned() {
            let depth_read = self.b.load(self.b.i64(), tc.delegate_depth);
            let delegate = self.b.icmp(LLVMIntSGT, depth_read, self.b.c_i64(0));
            let finally_normal = self.new_block("trycatch.finally.normal");
            let finally_delegate = self.new_block("trycatch.finally.delegate");
            self.b.cond_br(delegate, finally_delegate, finally_normal);

            self.b.at_end(finally_delegate);
            let depth_new = self.b.sub(depth_read, self.b.c_i64(1));
            let delegate_new = self.b.icmp(LLVMIntSGT, depth_new, self.b.c_i64(0));
            self.b.store(depth_new, tc.delegate_depth);
            self.b.cond_br(delegate_new, parent_tc.finally_block, parent_tc.exception_route_block);

            finally_block = finally_normal;
            self.b.at_end(finally_normal);
        }

        // handle exceptions that must route through 'finally'
        self.b.at_end(tc.finally_exception_block);
        if !self.options.noexc {
            if self.is_win_msvc() {
                self.win_catchpad_to_store("trycatch.finally.catchpad", tc.catch_store);
                self.b.store(self.b.c_i8(st(TcState::Rethrow)), tc.exc_flag);
                self.b.store(self.b.c_i64(0), tc.delegate_depth);
                self.b.br(tc.finally_block);
            } else {
                let lp = self.b.landing_pad(pad_type, 1);
                Builder::set_cleanup(lp, true);
                Builder::add_clause(lp, self.get_type_idx_var(m, None));
                self.b.store(lp, tc.catch_store);
                self.b.store(self.b.c_i8(st(TcState::Rethrow)), tc.exc_flag);
                self.b.store(self.b.c_i64(0), tc.delegate_depth);
                self.b.br(tc.finally_block);
            }
        } else {
            self.b.unreachable();
        }

        self.b.at_end(finally_block);
        let the_switch = self.b.switch(exc_flag_read, end_block, if support_break_and_continue { 6 } else { 4 });
        Builder::add_case(the_switch, self.b.c_i8(st(TcState::Caught)), end_block);
        Builder::add_case(the_switch, self.b.c_i8(st(TcState::Thrown)), unwind_resume_block);
        Builder::add_case(the_switch, self.b.c_i8(st(TcState::Rethrow)), rethrow_block);

        if is_root {
            let finally_return = self.new_block("trycatch.finally.return");
            Builder::add_case(the_switch, self.b.c_i8(st(TcState::Return)), finally_return);
            self.b.at_end(finally_return);
            if !self.coro.exit.is_null() {
                self.b.br(self.coro.exit);
            } else if !tc.ret_store.is_null() {
                let r = self.b.load(ret_ty, tc.ret_store);
                self.b.ret(r);
            } else {
                self.b.ret_void();
            }
        } else {
            let inner = self.innermost_try_catch().unwrap();
            Builder::add_case(the_switch, self.b.c_i8(st(TcState::Return)), inner.finally_block);
        }

        if support_break_and_continue {
            let inner = self.innermost_try_catch();
            let prev_seq = if is_root { -1 } else { inner.as_ref().unwrap().sequence_number };
            let finally_break = self.new_block("trycatch.finally.break");
            let finally_break_done = self.new_block("trycatch.finally.break.done");
            let finally_continue = self.new_block("trycatch.finally.continue");
            let finally_continue_done = self.new_block("trycatch.finally.continue.done");

            self.b.at_end(finally_break);
            let break_switch = self.b.switch(self.b.load(self.b.i64(), tc.loop_sequence), end_block, 0);
            self.b.at_end(finally_break_done);
            self.b.store(self.b.c_i8(st(TcState::NotThrown)), tc.exc_flag);
            let break_done_switch = self.b.switch(self.b.load(self.b.i64(), tc.loop_sequence), end_block, 0);

            self.b.at_end(finally_continue);
            let continue_switch = self.b.switch(self.b.load(self.b.i64(), tc.loop_sequence), end_block, 0);
            self.b.at_end(finally_continue_done);
            self.b.store(self.b.c_i8(st(TcState::NotThrown)), tc.exc_flag);
            let continue_done_switch = self.b.switch(self.b.load(self.b.i64(), tc.loop_sequence), end_block, 0);

            for l in self.loops.clone() {
                let seq = self.b.c_i64(l.sequence_number);
                if !is_root && l.sequence_number < prev_seq {
                    let fb = inner.as_ref().unwrap().finally_block;
                    Builder::add_case(break_switch, seq, fb);
                    Builder::add_case(continue_switch, seq, fb);
                } else {
                    Builder::add_case(break_switch, seq, finally_break_done);
                    Builder::add_case(break_done_switch, seq, l.break_block);
                    Builder::add_case(continue_switch, seq, finally_continue_done);
                    Builder::add_case(continue_done_switch, seq, l.continue_block);
                }
            }
            Builder::add_case(the_switch, self.b.c_i8(st(TcState::Break)), finally_break);
            Builder::add_case(the_switch, self.b.c_i8(st(TcState::Continue)), finally_continue);
        }

        // try and catch translate
        let catches = x.catches.clone();
        let mut catch_all: Option<BB> = None;
        for c in &catches {
            let catch_block = self.new_block("trycatch.catch");
            tc.catch_types.push(c.ty);
            tc.handlers.push(catch_block);
            if c.ty.is_none() {
                assert!(catch_all.is_none(), "cannot be catch all");
                catch_all = Some(catch_block);
            }
        }

        // translate try
        if !self.loops.is_empty() {
            // make sure we reset the state to avoid issues with 'break'/'continue'
            self.b.at_end(entry_block);
            self.b.store(self.b.c_i8(st(TcState::NotThrown)), tc.exc_flag);
        }
        self.block = entry_block;
        if have_finally {
            self.enter_finally(tc.clone());
        }
        self.enter_try(tc.clone()); // this is last so as to have larger sequence number
        self.process_value(m, x.body);
        self.exit_try();

        // translate else
        if let Some(e) = x.else_ {
            self.b.at_end(self.block);
            let else_block = self.new_block("trycatch.else");
            self.b.br(else_block);
            self.block = else_block;
            self.process_value(m, e);
        }

        // make sure we always get to finally block
        self.b.at_end(self.block);
        self.b.br(tc.finally_block);

        // resume if uncaught
        self.b.at_end(unwind_resume_block);
        if self.options.noexc {
            self.b.unreachable();
        } else if !is_root {
            self.b.br(rethrow_block);
        } else if self.is_win_msvc() {
            // `resume` is invalid under WinEH: re-raise via seq_throw after dropping our
            // own finally entry so the re-raise propagates to the enclosing handler.
            self.block = unwind_resume_block;
            let pad = self.b.load(pad_type, tc.catch_store);
            let exc = self.b.extract_value(pad, 0);
            let pop_finally = have_finally && !self.finally.is_empty();
            let saved = if pop_finally { self.finally.pop() } else { None };
            let (tt, tf) = self.make_throw_func();
            self.call(tt, tf, &[exc]);
            if let Some(s) = saved {
                self.finally.push(s);
            }
            self.b.at_end(self.block);
            self.b.unreachable();
        } else {
            let pad = self.b.load(pad_type, tc.catch_store);
            self.b.resume(pad);
        }

        // make sure we delegate to parent try-catch if necessary
        let mut catch_types_full = tc.catch_types.clone();
        let mut handlers_full = tc.handlers.clone();
        let mut depths: Vec<u64> = vec![0; tc.catch_types.len()];
        let mut depth = 1u64;
        let mut catch_all_depth = 0u64;
        for it in self.trycatch.clone().iter().rev() {
            if it.sequence_number < parent_finally_sequence {
                break;
            }
            if catch_all.is_some() {
                // can't ever delegate past catch-all
                break;
            }
            assert_eq!(it.catch_types.len(), it.handlers.len(), "handler mismatch");
            for i in 0..it.catch_types.len() {
                if !any_match(m, it.catch_types[i], &catch_types_full) {
                    catch_types_full.push(it.catch_types[i]);
                    depths.push(depth);
                    if it.catch_types[i].is_none() && catch_all.is_none() {
                        // catch-all is in parent; set finally depth
                        let ca = self.new_block("trycatch.fdepth_catchall");
                        self.b.at_end(ca);
                        self.b.store(self.b.c_i64(depth as i64), tc.delegate_depth);
                        self.b.br(it.handlers[i]);
                        handlers_full.push(ca);
                        catch_all = Some(ca);
                        catch_all_depth = depth;
                    } else {
                        handlers_full.push(it.handlers[i]);
                    }
                }
            }
            depth += 1;
        }

        // exception handling
        self.b.at_end(tc.exception_block);
        let mut unwind_exception;
        if self.is_win_msvc() && !self.options.noexc {
            self.win_catchpad_to_store("trycatch.catchpad", tc.catch_store);
            // funclet-local SSA can't escape the funclet; reload from catchStore.
            let pad = self.b.load(pad_type, tc.catch_store);
            unwind_exception = self.b.extract_value(pad, 0);
            let id_consts: Vec<V> =
                catch_types_full.iter().flatten().map(|t| self.b.c_i32(Self::get_type_idx(m, Some(*t)) as i64)).collect();
            let mut selector = self.b.c_i32(0);
            if !id_consts.is_empty() {
                let arr_ty = ll::array_ty(self.b.i32(), id_consts.len() as u64);
                let gv = ll::new_global(
                    self.m(),
                    arr_ty,
                    true,
                    llvm_sys::LLVMLinkage::LLVMPrivateLinkage,
                    Some(ll::const_array(self.b.i32(), &id_consts)),
                    "typhon.win.catchids",
                );
                let (mt, mf) = self.decl("seq_exc_match", self.b.i32(), &[self.b.ptr(), self.b.ptr(), self.b.i32()]);
                selector = self.b.call(mt, mf, &[unwind_exception, gv, self.b.c_i32(id_consts.len() as i64)]);
            }
            let pad2 = self.b.load(pad_type, tc.catch_store);
            let pad2 = self.b.insert_value(pad2, selector, 1);
            self.b.store(pad2, tc.catch_store);
        } else {
            let caught = if !self.options.noexc {
                let lp = self.b.landing_pad(pad_type, catches.len() as u32);
                Builder::set_cleanup(lp, true);
                Some(lp)
            } else {
                None
            };
            for ct in &catch_types_full {
                assert!(ct.is_none_or(|t| m.type_kind(m.actual_type(t)).is_ref()), "invalid catch type");
                let tidx = self.get_type_idx_var(m, *ct);
                if let Some(lp) = caught {
                    Builder::add_clause(lp, tidx);
                }
            }
            let c = caught.unwrap_or_else(|| ll::undef(pad_type));
            unwind_exception = self.b.extract_value(c, 0);
            self.b.store(c, tc.catch_store);
        }
        self.b.store(self.b.c_i8(st(TcState::Thrown)), tc.exc_flag);
        self.b.store(self.b.c_i64(0), tc.delegate_depth);

        let class_ptr = self.b.struct_gep(unwind_type, unwind_exception, 0);
        let unwind_exception_class = self.b.load(self.b.i64(), class_ptr);
        // check for foreign exceptions
        let is_ours = self.b.icmp_eq(unwind_exception_class, self.b.c_i64(SEQ_EXCEPTION_CLASS as i64));
        self.b.cond_br(is_ours, tc.exception_route_block, external_exc_block);

        // external exception (currently assumed to be unreachable)
        self.b.at_end(external_exc_block);
        self.b.unreachable();

        // reroute Typhon exceptions
        self.b.at_end(tc.exception_route_block);
        let pad = self.b.load(pad_type, tc.catch_store);
        unwind_exception = self.b.extract_value(pad, 0);
        let exc_val = self.b.const_gep_i8(unwind_exception, seq_exc_offset(ll::pointer_size(self.m())));
        let loaded_exc = self.b.load(self.b.ptr(), exc_val);

        // set depth when catch-all entered
        let default_route_block = self.new_block("trycatch.fdepth");
        self.b.at_end(default_route_block);
        if catch_all.is_some() {
            self.b.store(self.b.c_i64(catch_all_depth as i64), tc.delegate_depth);
        }
        let target = match catch_all {
            Some(ca) if catch_all_depth == 0 => ca,
            _ => tc.finally_block,
        };
        self.b.br(target);

        self.b.at_end(tc.exception_route_block);
        let pad = self.b.load(pad_type, tc.catch_store);
        let obj_type = self.b.extract_value(pad, 1);
        let switch_to_catch = self.b.switch(obj_type, default_route_block, handlers_full.len() as u32);
        for i in 0..handlers_full.len() {
            // set finally depth
            let depth_set = self.new_block("trycatch.fdepth");
            self.b.at_end(depth_set);
            self.b.store(self.b.c_i64(depths[i] as i64), tc.delegate_depth);
            self.b.br(if i < tc.handlers.len() { handlers_full[i] } else { tc.finally_block });

            if let Some(t) = catch_types_full[i] {
                Builder::add_case(switch_to_catch, self.b.c_i32(Self::get_type_idx(m, Some(t)) as i64), depth_set);
            }

            // translate catch body if this block is ours (vs. a parent's)
            if i < catches.len() {
                self.block = handlers_full[i];
                self.b.at_end(self.block);
                if let Some(var) = catches[i].var {
                    let var_ptr = self.get_var(m, var);
                    assert!(!var_ptr.is_null(), "could not get catch var");
                    self.b.store(loaded_exc, var_ptr);
                }
                self.b.store(self.b.c_i8(st(TcState::Caught)), tc.exc_flag);
                self.enter_catch(CatchData { sequence_number: -1, exception: loaded_exc, type_id: obj_type });
                self.process_value(m, catches[i].handler);
                self.exit_catch();
                self.b.at_end(self.block);
                self.b.br(tc.finally_block);
            }
        }

        if have_finally {
            self.exit_finally();
        }

        // rethrow if handling 'finally' after exception raised from 'except'/'else'
        self.b.at_end(rethrow_block);
        if (!have_finally && is_root) || self.options.noexc {
            self.b.unreachable();
        } else {
            let (tt, tf) = self.make_throw_func();
            let pad = self.b.load(pad_type, tc.catch_store);
            let ue = self.b.extract_value(pad, 0);
            self.block = rethrow_block;
            self.call(tt, tf, &[ue]);
            self.b.at_end(self.block);
            self.b.unreachable();
        }

        self.block = end_block;
    }

    /// `callStage`.
    fn call_stage(&mut self, m: &Module, stage: &Stage) {
        let output = self.value;
        self.process_value(m, stage.callee);
        let f = self.value;
        let mut args = Vec::new();
        for arg in &stage.args {
            match arg {
                Some(a) => {
                    self.process_value(m, *a);
                    args.push(self.value);
                }
                None => args.push(output),
            }
        }
        let ft = self.get_llvm_func_type(m, m.value_type(stage.callee));
        self.value = self.call(ft, f, &args);
    }

    /// `Stage::getOutputType`.
    fn stage_output_type(m: &Module, stage: &Stage) -> TypeId {
        let ct = m.value_type(stage.callee);
        if stage.args.is_empty() { ct } else { m.func_type_ret(ct).expect("not a function type") }
    }

    /// `codegenPipeline`.
    fn codegen_pipeline(&mut self, m: &Module, stages: &[Stage], where_: usize) {
        if where_ >= stages.len() {
            return;
        }
        let stage = &stages[where_];
        if where_ == 0 {
            self.process_value(m, stage.callee);
            self.codegen_pipeline(m, stages, where_ + 1);
            return;
        }
        let prev = &stages[where_ - 1];
        if prev.generator {
            let out = m.actual_type(Self::stage_output_type(m, prev));
            let TypeKind::Generator { base } = *m.type_kind(out) else {
                panic!("{} is not a generator type", m.type_name(out));
            };
            let base_type = self.get_llvm_type(m, base);

            let cond_block = self.new_block("pipeline.cond");
            let body_block = self.new_block("pipeline.body");
            let check_block = self.new_block("pipeline.check");
            let cleanup_block = self.new_block("pipeline.cleanup");
            let exit_block = self.new_block("pipeline.exit");

            let [resume, done_f, promise_f, destroy] = self.coro_intrinsics();
            let iter = self.value;
            self.b.at_end(self.block);
            let d = self.b.call(done_f.0, done_f.1, &[iter]);
            self.b.cond_br(d, exit_block, cond_block);

            self.b.at_end(check_block);
            let d = self.b.call(done_f.0, done_f.1, &[iter]);
            self.b.cond_br(d, exit_block, cond_block);

            self.block = cond_block;
            self.call(resume.0, resume.1, &[iter]);
            self.b.at_end(self.block);
            let done = self.b.call(done_f.0, done_f.1, &[iter]);
            self.b.cond_br(done, cleanup_block, body_block);

            self.b.at_end(body_block);
            let alignment = self.b.c_i32(ll::pref_align(self.m(), base_type) as i64);
            let promise = self.b.call(promise_f.0, promise_f.1, &[iter, alignment, self.b.c_i1(false)]);
            self.value = self.b.load(base_type, promise);

            self.block = body_block;
            self.call_stage(m, stage);
            self.codegen_pipeline(m, stages, where_ + 1);

            self.b.at_end(self.block);
            self.b.br(check_block);

            self.b.at_end(cleanup_block);
            self.b.call(destroy.0, destroy.1, &[iter]);
            self.b.br(exit_block);

            self.block = exit_block;
        } else {
            self.call_stage(m, stage);
            self.codegen_pipeline(m, stages, where_ + 1);
        }
    }

    /// `visit(const PipelineFlow*)`.
    pub(super) fn visit_pipeline_flow(&mut self, m: &Module, v: ValueId) {
        let ValueKind::PipelineFlow(x) = m.kind(v) else { unreachable!() };
        let stages = x.stages.clone();
        self.codegen_pipeline(m, &stages, 0);
    }
}
