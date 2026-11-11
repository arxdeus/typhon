//! `getLLVMType`, `getLLVMFuncType`, `getDIType` (port of llvisitor.cpp "Types").

use rustc_hash::FxHashMap;
use typhon_ir::ids::TypeId;
use typhon_ir::module::Module;
use typhon_ir::types::TypeKind;

use super::LlvmVisitor;
use super::ll::{self, Md, Ty};

impl<'ctx> LlvmVisitor<'ctx> {
    /// `getLLVMType`.
    pub fn get_llvm_type(&mut self, m: &Module, t: TypeId) -> Ty {
        let t = m.actual_type(t);
        let c = self.c;
        match m.type_kind(t).clone() {
            TypeKind::Int { len, .. } => ll::int_ty(c, len),
            TypeKind::Float => ll::double_ty(c),
            TypeKind::Float32 => ll::float_ty(c),
            TypeKind::Float16 => ll::half_ty(c),
            TypeKind::BFloat16 => ll::bfloat_ty(c),
            TypeKind::Float128 => ll::fp128_ty(c),
            TypeKind::Bool => ll::int_ty(c, 8),
            TypeKind::Record { fields } => {
                let body: Vec<Ty> = fields.iter().map(|f| self.get_llvm_type(m, f.ty)).collect();
                ll::struct_ty(c, &body)
            }
            TypeKind::Ref { .. } => ll::ptr_ty(c),
            TypeKind::Func { .. } => ll::ptr_ty(c),
            TypeKind::Optional { base } => {
                if m.type_kind(m.actual_type(base)).is_ref() {
                    self.get_llvm_type(m, base)
                } else {
                    let b = self.get_llvm_type(m, base);
                    ll::struct_ty(c, &[ll::int_ty(c, 1), b])
                }
            }
            TypeKind::Pointer { .. } => ll::ptr_ty(c),
            TypeKind::Generator { .. } => ll::ptr_ty(c),
            TypeKind::Vector { count, base } => {
                let b = self.get_llvm_type(m, base);
                ll::vector_ty(b, count)
            }
            TypeKind::Union { types } => {
                let i8t = ll::int_ty(c, 8);
                let mut alignment_type = i8t;
                let mut max_size = 0u64;
                for member in types {
                    let lt = self.get_llvm_type(m, member);
                    max_size = max_size.max(ll::alloc_size(self.m(), lt));
                    if ll::abi_align(self.m(), lt) > ll::abi_align(self.m(), alignment_type) {
                        alignment_type = lt;
                    }
                }
                let storage = ll::struct_ty(c, &[ll::array_ty(alignment_type, 0), ll::array_ty(i8t, max_size)]);
                ll::struct_ty(c, &[i8t, storage])
            }
            TypeKind::Custom(ct) => self.custom_type_builder_type(m, &ct),
        }
    }

    /// `getLLVMFuncType`.
    pub fn get_llvm_func_type(&mut self, m: &Module, t: TypeId) -> Ty {
        let t = m.actual_type(t);
        let TypeKind::Func { ret, args, variadic } = m.type_kind(t).clone() else {
            panic!("input type was not a func type");
        };
        let r = self.get_llvm_type(m, ret);
        let a: Vec<Ty> = args.iter().map(|x| self.get_llvm_type(m, *x)).collect();
        ll::fn_ty(r, &a, variadic)
    }

    /// `getDIType`.
    pub fn get_di_type(&mut self, m: &Module, t: TypeId) -> Md {
        let mut cache = FxHashMap::default();
        self.get_di_type_helper(m, t, &mut cache)
    }

    /// Subroutine type of a function type (`cast<DIDerivedType>(getDIType(t))->getBaseType()`).
    pub(super) fn get_di_subroutine_type(&mut self, m: &Module, t: TypeId) -> Md {
        let mut cache = FxHashMap::default();
        let t = m.actual_type(t);
        let TypeKind::Func { ret, args, .. } = m.type_kind(t).clone() else {
            panic!("not a function type");
        };
        let mut types = vec![self.get_di_type_helper(m, ret, &mut cache)];
        for a in args {
            types.push(self.get_di_type_helper(m, a, &mut cache));
        }
        let file = self.db.file("<internal>");
        self.db.b().subroutine_type(file, &types)
    }

    fn bits(&self, t: Ty) -> u64 {
        if ll::is_void(t) { 0 } else { ll::alloc_size(self.m(), t) * 8 }
    }

    /// `getDITypeHelper`.
    fn get_di_type_helper(&mut self, m: &Module, t: TypeId, cache: &mut FxHashMap<String, Md>) -> Md {
        let t = m.actual_type(t);
        let ty = self.get_llvm_type(m, t);
        let name = m.type_name(t).to_string();
        match m.type_kind(t).clone() {
            TypeKind::Int { sign, .. } => {
                self.db.b().basic_type(&name, self.bits(ty), if sign { ll::DW_ATE_SIGNED } else { ll::DW_ATE_UNSIGNED })
            }
            TypeKind::Float | TypeKind::Float32 | TypeKind::Float16 | TypeKind::BFloat16 => {
                self.db.b().basic_type(&name, self.bits(ty), ll::DW_ATE_FLOAT)
            }
            TypeKind::Float128 => self.db.b().basic_type(&name, self.bits(ty), ll::DW_ATE_HP_FLOAT128),
            TypeKind::Bool => self.db.b().basic_type(&name, self.bits(ty), ll::DW_ATE_BOOLEAN),
            TypeKind::Record { fields } => {
                if let Some(d) = cache.get(&name) {
                    return *d;
                }
                let src = Self::src_info(m, t.into());
                let member_info = m
                    .get_attribute(t.into(), typhon_ir::attribute::attr_id::MEMBER)
                    .and_then(|a| a.as_member())
                    .cloned();
                let file = self.db.file(&Self::src_file(&src));
                let size = self.bits(ty);
                let line = src.line.max(0) as u32;
                // prevent infinite recursion on recursive types
                let temp = self.db.b().temp_struct(file, &name, file, line, size);
                cache.insert(name.clone(), temp);
                let mut members = Vec::new();
                for (idx, field) in fields.iter().enumerate() {
                    let mut sub_line = line;
                    let mut sub_file = file;
                    if let Some(mi) = &member_info
                        && let Some(s) = mi.get(&field.name)
                    {
                        sub_line = s.line.max(0) as u32;
                        sub_file = self.db.file(&Self::src_file(s));
                    }
                    let ft = self.get_llvm_type(m, field.ty);
                    let fbits = self.bits(ft);
                    let off = ll::element_offset(self.m(), ty, idx as u32) * 8;
                    let fd = self.get_di_type_helper(m, field.ty, cache);
                    members.push(self.db.b().member_type(temp, &field.name, sub_file, sub_line, fbits, off, ll::DI_FLAG_ZERO, fd));
                }
                let real = self.db.b().finish_struct(temp, file, &name, file, line, size, &members);
                cache.insert(name, real);
                real
            }
            TypeKind::Ref { contents, polymorphic } => {
                if polymorphic {
                    let object = if let Some(d) = cache.get(&name) {
                        *d
                    } else {
                        let src = Self::src_info(m, t.into());
                        let file = self.db.file(&Self::src_file(&src));
                        let layout = self.get_object_layout(m, t);
                        let size = self.bits(layout);
                        let line = src.line.max(0) as u32;
                        let temp = self.db.b().temp_struct(file, &name, file, line, size);
                        cache.insert(name.clone(), temp);
                        let pbits = ll::pointer_size(self.m()) * 8;
                        let mut members = Vec::new();
                        let tptr = self.db.b().pointer_type(std::ptr::null_mut(), pbits);
                        members.push(self.db.b().member_type(temp, "typeinfo", file, line, pbits, 0, ll::DI_FLAG_ARTIFICIAL, tptr));
                        for (index, field) in m.type_fields(t).iter().enumerate() {
                            let ft = self.get_llvm_type(m, field.ty);
                            let fbits = self.bits(ft);
                            let off = self.get_object_field_offset(m, t, index) * 8;
                            let fd = self.get_di_type_helper(m, field.ty, cache);
                            members.push(self.db.b().member_type(temp, &field.name, file, line, fbits, off, ll::DI_FLAG_ZERO, fd));
                        }
                        let real = self.db.b().finish_struct(temp, file, &name, file, line, size, &members);
                        cache.insert(name, real);
                        real
                    };
                    return self.db.b().reference_type(ll::DW_TAG_REFERENCE_TYPE, object);
                }
                let inner = self.get_di_type_helper(m, contents, cache);
                self.db.b().reference_type(ll::DW_TAG_REFERENCE_TYPE, inner)
            }
            TypeKind::Func { ret, args, .. } => {
                let mut types = vec![self.get_di_type_helper(m, ret, cache)];
                for a in args {
                    types.push(self.get_di_type_helper(m, a, cache));
                }
                let file = self.db.file("<internal>");
                let sub = self.db.b().subroutine_type(file, &types);
                self.db.b().pointer_type(sub, self.bits(ty))
            }
            TypeKind::Optional { base } => {
                if m.type_kind(m.actual_type(base)).is_ref() {
                    return self.get_di_type_helper(m, base, cache);
                }
                let base_type = self.get_llvm_type(m, base);
                let struct_type = ll::struct_ty(self.c, &[ll::int_ty(self.c, 1), base_type]);
                let src = Self::src_info(m, t.into());
                let i1_bits = self.bits(ll::int_ty(self.c, 1));
                let i1_dt = self.db.b().basic_type("i1", i1_bits, ll::DW_ATE_BOOLEAN);
                let file = self.db.file(&Self::src_file(&src));
                let line = src.line.max(0) as u32;
                let size = self.bits(struct_type);
                let temp = self.db.b().temp_struct(file, &name, file, line, size);
                let off0 = ll::element_offset(self.m(), struct_type, 0) * 8;
                let off1 = ll::element_offset(self.m(), struct_type, 1) * 8;
                let has = self.db.b().member_type(temp, "has", file, line, i1_bits, off0, ll::DI_FLAG_ZERO, i1_dt);
                let bd = self.get_di_type_helper(m, base, cache);
                let val = self.db.b().member_type(temp, "val", file, line, self.bits(base_type), off1, ll::DI_FLAG_ZERO, bd);
                self.db.b().finish_struct(temp, file, &name, file, line, size, &[has, val])
            }
            TypeKind::Pointer { base } => {
                let bd = self.get_di_type_helper(m, base, cache);
                self.db.b().pointer_type(bd, self.bits(ty))
            }
            TypeKind::Generator { .. } => self.db.b().basic_type(&name, self.bits(ty), ll::DW_ATE_ADDRESS),
            TypeKind::Vector { .. } | TypeKind::Union { .. } => {
                self.db.b().basic_type(&name, self.bits(ty), ll::DW_ATE_UNSIGNED)
            }
            TypeKind::Custom(ct) => self.custom_type_builder_debug(m, &ct),
        }
    }
}
