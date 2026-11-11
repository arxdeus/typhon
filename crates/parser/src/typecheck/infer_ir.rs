//! Port of codon/parser/visitors/typecheck/infer.cpp, part 3: `makeIRType`,
//! `makeIRFunction`, `realizeIRFunc`.

use std::collections::BTreeMap;

use typhon_common::mangle::{self, stdlib};
use typhon_common::{CompilerError, ErrorKind, Result, SrcInfo};
use typhon_ir::{Attribute, FuncId, FuncKind, Intrinsic, Module, NodeRef, RuntimeInfo, TypeKind};

use super::TypecheckVisitor;
use crate::ast::Attr;
use crate::cache::{AstType, ClassField, FunctionRealizationRef};
use crate::ids::TypeId;

type IrType = typhon_ir::TypeId;

impl<'a> TypecheckVisitor<'a> {
    fn module(&self) -> Result<std::rc::Rc<std::cell::RefCell<Module>>> {
        self.cache.module.clone().ok_or_else(|| CompilerError::custom("cache's module is not set", self.src))
    }

    fn force_find_ir_type(&self, t: TypeId) -> IrType {
        let types = &self.cache.types;
        let c = types.get_class(t).expect("not a class");
        let rn = types.class_realized_name(c);
        let cls = self.cache.get_class_by_type(c).expect("bad class");
        let r = cls.realizations.get(&rn).unwrap_or_else(|| panic!("{} not realized", types.debug_string(t, 2)));
        r.borrow().ir.unwrap_or_else(|| panic!("no LLVM type for {}", types.debug_string(t, 2)))
    }

    fn set_polymorphic(m: &mut Module, t: IrType) {
        if let TypeKind::Ref { polymorphic, .. } = &mut m.ty_mut(t).kind {
            *polymorphic = true;
        }
    }

    /// `makeIRType(t)`: IR type of a realized class (realizing it if necessary).
    pub fn make_ir_type(&mut self, t0: TypeId) -> Result<IrType> {
        let types = self.types();
        let st = stdlib();
        let mut t = t0;
        let mut rn = types.class_realized_name(t);
        let has = self.cache.get_class_by_type(t).is_some_and(|c| c.realizations.contains_key(&rn));
        if !has {
            let tc = types.get_class(t).unwrap();
            t = self.realize(tc)?.and_then(|r| types.get_class(r)).ok_or_else(|| CompilerError::custom("cannot realize", self.src))?;
            rn = types.class_realized_name(t);
        }
        let (realization, rtti) = {
            let cls = self.cache.get_class_by_type(t).expect("bad class");
            let cname = cls.ast.and_then(|a| self.cache.ast[a].kind.as_class().map(|c| c.name.clone())).unwrap_or_default();
            (cls.realizations.get(&rn).cloned().unwrap(), cls.has_rtti(&cname))
        };
        let module = self.module()?;
        if let Some(l) = realization.borrow().ir {
            if rtti {
                Self::set_polymorphic(&mut module.borrow_mut(), l);
            }
            return Ok(l);
        }
        let name = types.class_name(t).unwrap().to_string();
        // Prepare generics and statics
        let mut tys: Vec<Option<IrType>> = vec![];
        let mut statics: Vec<TypeId> = vec![];
        if types.is(t, &st.unrealized_type) {
            tys.push(None);
        } else {
            for g in types.generics(t) {
                if types.get_static(g.ty()).is_some() {
                    statics.push(g.ty());
                } else {
                    tys.push(Some(self.force_find_ir_type(g.ty())));
                }
            }
        }
        let handle: IrType = if name == st.bool_ {
            module.borrow_mut().get_bool_type()
        } else if name == st.float {
            module.borrow_mut().get_float_type()
        } else if name == "float32" {
            module.borrow_mut().get_float32_type()
        } else if name == st.float16 {
            module.borrow_mut().get_float16_type()
        } else if name == "bfloat16" {
            module.borrow_mut().get_bfloat16_type()
        } else if name == "float128" {
            module.borrow_mut().get_float128_type()
        } else if name == st.string {
            module.borrow_mut().get_string_type()
        } else if name == "bytes" {
            module.borrow_mut().get_bytes_type()
        } else if name == st.int || name == st.uint {
            let n = self.get_int_literal(statics[0], 0);
            module.borrow_mut().unsafe_get_int_type(n as u32, name == st.int)
        } else if name == st.ptr {
            module.borrow_mut().unsafe_get_pointer_type(tys[0])
        } else if name == st.generator || name == "AsyncGenerator" || name == st.coroutine {
            module.borrow_mut().unsafe_get_generator_type(tys[0].unwrap())
        } else if name == st.optional {
            module.borrow_mut().unsafe_get_optional_type(tys[0].unwrap())
        } else if name == st.none_type {
            let mut m = module.borrow_mut();
            let r = m.unsafe_get_membered_type(&rn, false);
            m.realize_membered(r, vec![], vec![]);
            r
        } else if name == st.union {
            let uv: Vec<IrType> = types.union_realization_types(t).into_iter().map(|u| self.force_find_ir_type(u)).collect();
            module.borrow_mut().unsafe_get_union_type(&uv)
        } else if name == st.function {
            let args = self.extract_class_generic(t, 0);
            let at: Vec<IrType> = types.generics(args).iter().map(|g| self.force_find_ir_type(g.ty())).collect();
            let ret = self.force_find_ir_type(self.extract_class_generic(t, 1));
            module.borrow_mut().unsafe_get_func_type(&rn, ret, at, false)
        } else if name == st.vec {
            let count = self.get_int_literal(statics[0], 0);
            if count <= 0 || count > u32::MAX as i64 {
                typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "SIMD vector width must be positive and fit in 32 bits");
            }
            let el = tys[0].unwrap();
            let ok = matches!(
                module.borrow().type_kind(el),
                TypeKind::Int { .. } | TypeKind::Float | TypeKind::Float32 | TypeKind::Float16 | TypeKind::BFloat16 | TypeKind::Float128
            );
            if !ok {
                typhon_common::bail_e!(ErrorKind::CUSTOM, self.src, "SIMD element type must be an integer or floating-point type");
            }
            module.borrow_mut().unsafe_get_vector_type(count as u32, el)
        } else if types.is_record(t) {
            // Record: fields populated right away
            typhon_common::seqassert!(!types.is(t, "__NTuple__"), "ntuple not inlined");
            let ft = self.get_class_field_types(t)?;
            let fields = self.cache.get_class_by_type(t).unwrap().fields.clone();
            let mut type_args = vec![];
            let mut names = vec![];
            let mut member_info = BTreeMap::new();
            for (i, f) in ft.iter().enumerate() {
                if self.realize(*f)?.is_none() {
                    let p = types.pretty_string(t);
                    typhon_common::bail_e!(ErrorKind::TYPE_CANNOT_REALIZE_ATTR, self.src, fields[i].name, p);
                }
                names.push(fields[i].name.clone());
                let fc = types.get_class(*f).unwrap();
                type_args.push(self.make_ir_type(fc)?);
                member_info.insert(fields[i].name.clone(), types.src(*f));
            }
            let mut m = module.borrow_mut();
            let r = m.unsafe_get_membered_type(&rn, false);
            m.realize_membered(r, type_args, names);
            m.set_attribute(NodeRef::Type(r), Attribute::Member(member_info));
            r
        } else {
            // Type arguments are populated afterwards (recursive reference types).
            let mut m = module.borrow_mut();
            let h = m.unsafe_get_membered_type(&rn, true);
            if rtti {
                Self::set_polymorphic(&mut m, h);
            }
            h
        };
        {
            let mut m = module.borrow_mut();
            m.set_src_info(NodeRef::Type(handle), types.src(t));
            m.set_ast_type(handle, Some(AstType::to_ref(t)));
        }
        // The realization must point at the IR type before bases are realized (recursion).
        realization.borrow_mut().ir = Some(handle);
        let mut info = RuntimeInfo {
            id: realization.borrow().id as i64,
            raw_name: types.realized_name(t),
            nice_name: types.pretty_string(t),
            base_name: name.clone(),
            mro: vec![handle],
            ..Default::default()
        };
        let bases = realization.borrow().bases.clone();
        for b in bases {
            let bt = self.realize(b)?.and_then(|x| types.get_class(x)).ok_or_else(|| CompilerError::custom("cannot realize base", self.src))?;
            info.mro.push(self.make_ir_type(bt)?);
        }
        {
            let m = module.borrow();
            let sysexit = mangle::class("std.internal.types.error", "SystemExit");
            for b in &info.mro {
                let bn = m.ast_type(*b).and_then(|a| AstType::from_ref(&a)).and_then(|a| types.class_name(a));
                if let Some(bn) = bn {
                    if bn == st.base_exception.as_str() {
                        info.exception_base = Some(*b);
                    }
                    if bn == sysexit.as_str() {
                        info.system_exit_base = Some(*b);
                    }
                }
            }
        }
        if !types.is(t, &st.unrealized_type) {
            if types.is(t, &st.int) {
                info.parameters.push(handle); // special case for Int[N] to avoid recursion
            } else {
                for g in types.generics(t) {
                    let mut p = g.ty();
                    if types.get_static(p).is_some() {
                        p = types.non_static_type(p).unwrap_or(p);
                    }
                    let pc = types.get_class(p).unwrap();
                    info.parameters.push(self.make_ir_type(pc)?);
                }
            }
        }
        module.borrow_mut().set_runtime_info(handle, info);
        Ok(handle)
    }

    /// Final part of `realizeType`: populate reference type contents and field owners.
    pub(crate) fn finish_ir_type_fields(
        &mut self,
        lt: IrType,
        type_args: Vec<IrType>,
        names: Vec<String>,
        member_info: BTreeMap<String, SrcInfo>,
        fields: &[ClassField],
    ) -> Result<()> {
        let module = self.module()?;
        let mut m = module.borrow_mut();
        if !names.is_empty() {
            if let Some(contents) = m.ref_contents(lt) {
                m.realize_membered(contents, type_args, names);
                m.set_attribute(NodeRef::Type(lt), Attribute::Member(member_info.clone()));
                m.set_attribute(NodeRef::Type(contents), Attribute::Member(member_info));
            }
        }
        let mro = m.runtime_info(lt).map(|r| r.mro.clone()).unwrap_or_default();
        let types = self.types();
        let mut owners = vec![];
        for f in fields {
            let mut owner = lt;
            for b in &mro {
                let bn = m.ast_type(*b).and_then(|a| AstType::from_ref(&a)).and_then(|a| types.class_name(a));
                if bn.is_some_and(|n| n == f.base_class.as_str()) {
                    owner = *b;
                    break;
                }
            }
            owners.push(owner);
        }
        if let Some(ri) = m.ty_mut(lt).runtime_info.as_mut() {
            ri.field_owners.extend(owners);
        }
        Ok(())
    }

    /// `makeIRFunction(r)`: IR function node for a realized function.
    pub fn make_ir_function(&mut self, r: &FunctionRealizationRef) -> Result<FuncId> {
        let types = self.types();
        let (rty, rast) = {
            let b = r.borrow();
            (b.ty.unwrap(), b.ast.unwrap())
        };
        let module = self.module()?;
        let fname = types.func_name(rty);
        let rname = types.realized_name(rty);
        let mut unmangled = self.get_unmangled_name(&fname);
        let attrs = self.cache.ast[rast].attrs.clone();
        let kind = if attrs.has(Attr::Internal) {
            FuncKind::Internal { intrinsic: Intrinsic::None, intrinsic_type: None }
        } else if attrs.has(Attr::LLVM) {
            FuncKind::Llvm { literals: vec![], declarations: String::new(), body: String::new() }
        } else if attrs.has(Attr::C) {
            if let Some(f) = attrs.get_key_value(Attr::FunctionAttributes) {
                if let Some(i) = f.get(&mangle::func("std.internal.c_stubs", "linkname")) {
                    if let Some(fp) = self.cache.find_function(i) {
                        if let Some(s) = types.get_str_static(self.extract_func_generic(fp, 0)) {
                            unmangled = s;
                        }
                    }
                }
            }
            FuncKind::External
        } else {
            FuncKind::Bodied { symbols: vec![], body: None, jit: false }
        };
        let f = module.borrow_mut().nr_func(rname.clone(), kind);
        if let Some(fd) = module.borrow_mut().func_mut(f) {
            fd.unmangled_name = unmangled.clone();
        }
        let mut parent = types.func_parent(rty);
        if let Some(pc) = attrs.get_str(Attr::ParentClass) {
            if !pc.is_empty() && !attrs.has(Attr::Method) {
                // Hack for non-generic methods
                parent = self.find(pc, 0).and_then(|i| i.borrow().ty);
            }
        }
        if let Some(p) = parent {
            if types.is_instantiated(p) && types.can_realize(p) {
                let p = self.extract_class_type(p).unwrap();
                self.realize(p)?;
                let pir = self.make_ir_type(p)?;
                if let Some(fd) = module.borrow_mut().func_mut(f) {
                    fd.parent_type = Some(pir);
                }
            }
        }
        module.borrow_mut().set_global(f, true);
        // Mark this realization as pending (i.e., realized but not translated)
        self.cache.pending_realizations.insert((fname.clone(), rname.clone()));

        let params = self.cache.ast[rast].kind.as_function().unwrap().items.clone();
        let mut names = vec![];
        let mut tys = vec![];
        let mut j = 0;
        for p in &params {
            if p.is_value() {
                let at = self.extract_func_arg_type(rty, j);
                if types.get_func(at).is_none() {
                    let c = types.get_class(at).unwrap();
                    tys.push(self.make_ir_type(c)?);
                    names.push(self.get_unmangled_name(&p.name));
                }
                j += 1;
            }
        }
        let variadic = attrs.has(Attr::CVarArg);
        if variadic {
            tys.pop();
            names.pop();
        }
        let ret = types.get_class(types.func_ret(rty)).unwrap();
        let ret_ir = self.make_ir_type(ret)?;
        let ir_type = module.borrow_mut().unsafe_get_func_type(&rname, ret_ir, tys.clone(), variadic);
        let mut m = module.borrow_mut();
        m.set_ast_type(ir_type, Some(AstType::to_ref(rty)));
        m.realize_func(f, ir_type, &names);
        drop(m);
        // Identify builtins at the frontend boundary; CIR consumers use intrinsic IDs.
        if attrs.has(Attr::Internal) {
            let mm = |c: &str, me: &str| mangle::mangled_method("", c, me, 0, 0);
            let ret_t = module.borrow().func_type_ret(ir_type);
            let parent_t = module.borrow().func(f).and_then(|fd| fd.parent_type);
            let set = |k: Intrinsic, t: Option<IrType>| module.borrow_mut().set_intrinsic(f, k, t);
            if fname == mm("type", "_ref_alloc") {
                set(Intrinsic::Allocate, ret_t);
            } else if fname == mm("type", "_force_cast") || fname == mm("type", "_ref_raw") || fname == mm("type", "_force_value_cast") {
                if fname != mm("type", "_force_value_cast") {
                    let m = module.borrow();
                    let pointer_like = |t: IrType| match m.type_kind(t) {
                        TypeKind::Optional { base } => matches!(m.type_kind(*base), TypeKind::Ref { .. }),
                        TypeKind::Ref { .. } | TypeKind::Pointer { .. } | TypeKind::Func { .. } | TypeKind::Generator { .. } => true,
                        _ => false,
                    };
                    let ok = tys.len() == 1 && pointer_like(tys[0]) && ret_t.is_some_and(pointer_like);
                    drop(m);
                    if !ok {
                        let src = self.cache.ast[rast].src;
                        typhon_common::bail_e!(ErrorKind::CUSTOM, src, "pointer cast requires pointer-like source and destination types");
                    }
                }
                set(Intrinsic::RepresentationCast, ret_t);
            } else if fname == mm("type", "_ref_size") {
                let g = types.get_class(self.extract_func_generic(rty, 0)).unwrap();
                let gi = self.make_ir_type(g)?;
                set(Intrinsic::AllocationSize, Some(gi));
            } else if fname == mm("TypeInfo", "_get") {
                let g = types.get_class(self.extract_func_generic(rty, 0)).unwrap();
                let gi = self.make_ir_type(g)?;
                set(Intrinsic::Typeinfo, Some(gi));
            } else if fname == mm("TypeInfo", "_lookup") {
                set(Intrinsic::TypeinfoLookup, None);
            } else if fname == mm("TypeInfo", "_lookup_method") {
                set(Intrinsic::TypeinfoMethodLookup, None);
            } else if (unmangled == "__new__" && parent_t.is_some_and(|p| matches!(module.borrow().type_kind(p), TypeKind::Record { .. })))
                || fname == mm("Partial", "__new__")
            {
                set(Intrinsic::RecordNew, ret_t);
            } else if fname == mm("Generator", "__promise__") {
                set(Intrinsic::GeneratorPromise, parent_t);
            }
        }
        Ok(f)
    }

    /// `realizeIRFunc(fn, generics)` (used by cythonization).
    pub fn realize_ir_func(&mut self, f: TypeId, generics: &[TypeId]) -> Result<Option<FuncId>> {
        let types = self.types();
        let ft = self.instantiate_type(f, None)?;
        let fg = types.func_generics(ft);
        let mut u = crate::types::Unification::new();
        for (i, g) in generics.iter().enumerate() {
            types.unify(fg[i].ty(), *g, Some(&mut u), self)?;
        }
        if self.realize(ft)?.is_none() {
            return Ok(None);
        }
        let fname = types.func_name(f);
        let rn = types.realized_name(ft);
        Ok(self.cache.functions.get(&fname).and_then(|x| x.realizations.get(&rn)).and_then(|r| r.borrow().ir))
    }
}
