//! Runtime type information, object layouts and compiler intrinsics
//! (port of the RTTI section of llvisitor.cpp).

use std::collections::BTreeMap;
use std::ptr::null_mut;

use llvm_sys::LLVMLinkage::*;
use typhon_ir::ids::{FuncId, TypeId};
use typhon_ir::module::Module;
use typhon_ir::types::TypeKind;
use typhon_ir::var::{FuncKind, Intrinsic};

use super::instrs::rtti_field;
use super::ll::{self, Builder, Ty, V};
use super::{FieldOffsetResolver, LlvmVisitor};

/// `runtime::METHOD_HASH_MULTIPLIER` / `METHOD_HASH_SHIFT` / `ANY_INLINE_BYTES`.
const METHOD_HASH_MULTIPLIER: u64 = 0x9e37_79b9_7f4a_7c15;
const METHOD_HASH_SHIFT: u64 = 32;
const ANY_INLINE_BYTES: u64 = 8;

/// `runtime::methodHash`.
fn method_hash(id: u64) -> u64 {
    let mixed = id.wrapping_mul(METHOD_HASH_MULTIPLIER);
    mixed ^ (mixed >> METHOD_HASH_SHIFT)
}

impl<'ctx> LlvmVisitor<'ctx> {
    /// `getRuntimeTypeInfoType` (must match runtime::TypeInfo).
    pub(super) fn get_runtime_type_info_type(&self) -> Ty {
        let i = self.b.i64();
        let p = self.b.ptr();
        let i8t = self.b.i8();
        let s = ll::struct_ty(self.c, &[p, i]);
        ll::struct_ty(self.c, &[i, i, p, p, p, i, p, i, p, i8t, s, s, s, i, p, i, i8t, i8t, i, i])
    }

    fn runtime_info_or_panic(m: &Module, t: TypeId) -> &typhon_ir::types::RuntimeInfo {
        m.runtime_info(m.actual_type(t)).unwrap_or_else(|| panic!("missing runtime metadata for {}", m.type_name(t)))
    }

    /// `getOwnLayout`.
    fn get_own_layout(&mut self, m: &Module, t: TypeId) -> Ty {
        let t = m.actual_type(t);
        let owners = Self::runtime_info_or_panic(m, t).field_owners.clone();
        let mut fields = Vec::new();
        for (i, f) in m.type_fields(t).iter().enumerate() {
            if m.actual_type(owners[i]) == t {
                fields.push(self.get_llvm_type(m, f.ty));
            }
        }
        ll::struct_ty(self.c, &fields)
    }

    fn is_polymorphic(m: &Module, t: TypeId) -> bool {
        matches!(m.type_kind(m.actual_type(t)), TypeKind::Ref { polymorphic: true, .. })
    }

    /// `getObjectLayout`.
    pub(super) fn get_object_layout(&mut self, m: &Module, t: TypeId) -> Ty {
        let t = m.actual_type(t);
        if !Self::is_polymorphic(m, t) {
            let contents = m.ref_contents(t).expect("not a reference type");
            return self.get_llvm_type(m, contents);
        }
        let mro = Self::runtime_info_or_panic(m, t).mro.clone();
        // {TypeInfo*, object-own, Root-own, ..., Leaf-own}
        let mut fields = vec![self.b.ptr()];
        for base in mro.iter().rev() {
            fields.push(self.get_own_layout(m, *base));
        }
        ll::struct_ty(self.c, &fields)
    }

    /// `getObjectOwnerOffset`.
    fn get_object_owner_offset(&mut self, m: &Module, t: TypeId, owner: TypeId) -> u64 {
        let mro = Self::runtime_info_or_panic(m, t).mro.clone();
        let owner = m.actual_type(owner);
        let pos = mro.iter().position(|b| m.actual_type(*b) == owner).expect("field owner missing from MRO");
        let layout = self.get_object_layout(m, t);
        ll::element_offset(self.m(), layout, (mro.len() - pos) as u32)
    }

    /// `getObjectFieldOffset`.
    pub(super) fn get_object_field_offset(&mut self, m: &Module, t: TypeId, index: usize) -> u64 {
        if !Self::is_polymorphic(m, t) {
            let layout = self.get_object_layout(m, t);
            return ll::element_offset(self.m(), layout, index as u32);
        }
        let owners = Self::runtime_info_or_panic(m, t).field_owners.clone();
        let owner = m.actual_type(owners[index]);
        let local = owners[..index].iter().filter(|o| m.actual_type(**o) == owner).count();
        let base = self.get_object_owner_offset(m, t, owner);
        let own = self.get_own_layout(m, owner);
        base + ll::element_offset(self.m(), own, local as u32)
    }

    fn const_array_global(&self, elem: Ty, values: &[V], name: &str) -> V {
        if values.is_empty() {
            return self.b.null_ptr();
        }
        let at = ll::array_ty(elem, values.len() as u64);
        ll::new_global(self.m(), at, true, LLVMPrivateLinkage, Some(ll::const_array(elem, values)), name)
    }

    /// `getRuntimeTypeInfo`.
    pub(super) fn get_runtime_type_info(&mut self, m: &Module, t: TypeId) -> V {
        let t = m.actual_type(t);
        let info = Self::runtime_info_or_panic(m, t).clone();
        let name = format!(".typhon.typeinfo.{}", info.id);
        if let Some(g) = ll::get_global(self.m(), &name) {
            return g;
        }
        let rtt = self.get_runtime_type_info_type();
        let result = ll::new_global(self.m(), rtt, true, LLVMPrivateLinkage, None, &name);

        let kind = m.type_kind(t).clone();
        let reference = kind.is_ref();
        let membered = kind.is_membered();
        let polymorphic = Self::is_polymorphic(m, t);
        let i64t = self.b.i64();
        let mut exception_offset: i64 = -1;
        let mut system_exit_offset: i64 = -1;
        let mut mro = Vec::new();
        let mut mro_ids = Vec::new();
        let mut offsets = Vec::new();
        let mut offset_vals = Vec::new();
        for base in &info.mro {
            let bid = Self::runtime_info_or_panic(m, *base).id;
            mro_ids.push(bid as u64);
            mro.push(self.b.c_i64(bid));
            let off = if polymorphic { self.get_object_owner_offset(m, t, *base) } else { 0 };
            offset_vals.push(off);
            offsets.push(self.b.c_i64(off as i64));
            let base = m.actual_type(*base);
            if info.exception_base.map(|x| m.actual_type(x)) == Some(base) {
                exception_offset = off as i64;
            }
            if info.system_exit_base.map(|x| m.actual_type(x)) == Some(base) {
                system_exit_offset = off as i64;
            }
        }
        let mut sorted_ids = mro_ids.clone();
        sorted_ids.sort();
        let sorted_mro: Vec<V> = sorted_ids.iter().map(|&i| self.b.c_i64(i as i64)).collect();
        if polymorphic {
            // Keep the public offsets[0:n_mro] prefix in MRO order; append mask + keyed
            // {owner ID, offset} slots using the method-table hash.
            let mut capacity = 1usize;
            while capacity < 2 * mro_ids.len() {
                capacity *= 2;
            }
            let mut index = vec![(0u64, 0u64); capacity];
            for (pos, &id) in mro_ids.iter().enumerate() {
                let mut slot = (method_hash(id) as usize) & (capacity - 1);
                while index[slot].1 != 0 {
                    slot = (slot + 1) & (capacity - 1);
                }
                index[slot] = (id, offset_vals[pos]);
            }
            offsets.push(self.b.c_i64(capacity as i64 - 1));
            for (id, off) in index {
                offsets.push(self.b.c_i64(id as i64));
                offsets.push(self.b.c_i64(off as i64));
            }
        }
        let string_type = ll::struct_ty(self.c, &[self.b.ptr(), i64t]);
        let slot_type = ll::struct_ty(self.c, &[string_type, i64t, i64t]);
        let mut slots = Vec::new();
        if membered {
            for (index, field) in m.type_fields(t).iter().enumerate() {
                let finfo = m
                    .runtime_info(m.actual_type(field.ty))
                    .unwrap_or_else(|| panic!("missing field metadata for {}", m.type_name(field.ty)));
                let fid = finfo.id;
                let offset = if reference {
                    self.get_object_field_offset(m, t, index)
                } else {
                    let lt = self.get_llvm_type(m, t);
                    ll::element_offset(self.m(), lt, index as u32)
                };
                let s = self.string_const(&field.name);
                slots.push(ll::const_struct(self.c, &[s, self.b.c_i64(fid), self.b.c_i64(offset as i64)]));
            }
        }
        if let TypeKind::Optional { base } = kind {
            let bid = Self::runtime_info_or_panic(m, base).id;
            let repr = self.get_llvm_type(m, t);
            let offset = if ll::is_struct(repr) { ll::element_offset(self.m(), repr, 1) } else { 0 };
            let s = self.string_const("value");
            slots.push(ll::const_struct(self.c, &[s, self.b.c_i64(bid), self.b.c_i64(offset as i64)]));
        }
        let parameters: Vec<V> = info.parameters.iter().map(|p| self.b.c_i64(Self::runtime_info_or_panic(m, *p).id)).collect();

        let mut method_count = if info.methods.is_empty() { 0 } else { 1 };
        while method_count < 2 * info.methods.len() {
            method_count *= 2;
        }
        let method_slot_type = ll::struct_ty(self.c, &[i64t, self.b.ptr()]);
        let mut methods = vec![ll::const_null(method_slot_type); method_count];
        let mut occupied = vec![false; method_count];
        for (&id, &target) in &info.methods {
            let function = self.get_function_value(m, target);
            assert!(!function.is_null(), "missing dispatch target {}", m.var_name(target));
            assert!(id != 0, "zero is reserved for empty method slots");
            let mut pos = (method_hash(id as u64) as usize) & (method_count - 1);
            while occupied[pos] {
                pos = (pos + 1) & (method_count - 1);
            }
            occupied[pos] = true;
            methods[pos] = ll::const_struct(self.c, &[self.b.c_i64(id as i64), function]);
        }
        let lt = self.get_llvm_type(m, t);
        let value_size = ll::alloc_size(self.m(), lt);
        let inline_value = ll::pointer_size(self.m()) == ANY_INLINE_BYTES
            && value_size <= ANY_INLINE_BYTES
            && (matches!(
                kind,
                TypeKind::Int { .. } | TypeKind::Bool | TypeKind::Float | TypeKind::Float32 | TypeKind::Float16 | TypeKind::BFloat16
            ) || info.base_name == "NoneType");

        let fields = [
            self.b.c_i64(info.id),
            self.b.c_i64(mro.len() as i64),
            self.const_array_global(i64t, &mro, &format!("{name}.mro")),
            self.const_array_global(i64t, &offsets, &format!("{name}.offsets")),
            self.const_array_global(i64t, &sorted_mro, &format!("{name}.sorted")),
            self.b.c_i64(slots.len() as i64),
            self.const_array_global(slot_type, &slots, &format!("{name}.slots")),
            self.b.c_i64(method_count as i64),
            self.const_array_global(method_slot_type, &methods, &format!("{name}.methods")),
            self.b.c_i8(polymorphic as i64),
            self.string_const(&info.raw_name),
            self.string_const(&info.nice_name),
            self.string_const(&info.base_name),
            self.b.c_i64(parameters.len() as i64),
            self.const_array_global(i64t, &parameters, &format!("{name}.params")),
            self.b.c_i64(value_size as i64),
            self.b.c_i8(reference as i64),
            self.b.c_i8(inline_value as i64),
            self.b.c_i64(exception_offset),
            self.b.c_i64(system_exit_offset),
        ];
        let initializer = ll::const_struct(self.c, &fields);
        ll::set_initializer(result, initializer);
        result
    }

    /// `loadRuntimeField`.
    pub(super) fn load_runtime_field(&self, info: V, field: u32) -> V {
        let t = self.get_runtime_type_info_type();
        let ft = ll::struct_elements(t)[field as usize];
        self.b.load(ft, self.b.struct_gep(t, info, field))
    }

    /// `getFieldAddress`.
    pub(super) fn get_field_address(&mut self, m: &Module, object: V, t: TypeId, field: &str, exact: Option<TypeId>) -> V {
        let t = m.actual_type(t);
        let index = m.member_index(t, field);
        assert!(index >= 0, "unknown field {field} of {}", m.type_name(t));
        let index = index as usize;
        if !Self::is_polymorphic(m, t) {
            let layout = self.get_object_layout(m, t);
            return self.b.struct_gep(layout, object, index as u32);
        }
        let metadata = Self::runtime_info_or_panic(m, t).clone();
        let owner = m.actual_type(metadata.field_owners[index]);
        let local = metadata.field_owners[..index].iter().filter(|o| m.actual_type(**o) == owner).count();
        let own = self.get_own_layout(m, owner);
        let local_offset = ll::element_offset(self.m(), own, local as u32);
        let i8t = self.b.i8();
        if let Some(exact) = exact {
            let off = self.get_object_owner_offset(m, exact, owner);
            return self.b.gep(i8t, object, &[self.b.c_i64((off + local_offset) as i64)]);
        }
        let owner_id = Self::runtime_info_or_panic(m, owner).id;
        {
            let key = (metadata.id, owner_id);
            if !self.field_offset_resolvers.contains_key(&key) {
                let mut offsets: Vec<(i64, u64)> = Vec::new();
                for cand in m.types() {
                    if !Self::is_polymorphic(m, cand) {
                        continue;
                    }
                    let Some(ci) = m.runtime_info(cand) else { continue };
                    if !ci.mro.iter().any(|b| m.actual_type(*b) == t) {
                        continue;
                    }
                    let cid = ci.id;
                    let off = self.get_object_owner_offset(m, cand, owner);
                    offsets.push((cid, off));
                }
                let mut resolver = FieldOffsetResolver { constant: false, offset: 0, function: null_mut() };
                if !offsets.is_empty() {
                    resolver.offset = offsets[0].1;
                    resolver.constant = offsets.iter().all(|e| e.1 == resolver.offset);
                    if !resolver.constant {
                        let fty = ll::fn_ty(self.b.i64(), &[self.b.i64()], false);
                        let f = ll::add_function(self.m(), &format!(".typhon.field.offset.{}.{}", key.0, key.1), fty);
                        ll::set_linkage(f, LLVMPrivateLinkage);
                        ll::set_memory_none(f);
                        ll::add_fn_attr(f, "nounwind");
                        let fb = Builder::new(self.c);
                        let entry = ll::append_block(self.c, f, "entry");
                        let missing = ll::append_block(self.c, f, "missing");
                        fb.at_end(entry);
                        let sel = fb.switch(ll::param(f, 0), missing, 0);
                        let mut dests: BTreeMap<u64, ll::BB> = BTreeMap::new();
                        for (id, off) in &offsets {
                            let d = *dests.entry(*off).or_insert_with(|| {
                                let d = ll::append_block(self.c, f, "found");
                                fb.at_end(d);
                                fb.ret(fb.c_i64(*off as i64));
                                d
                            });
                            Builder::add_case(sel, fb.c_i64(*id), d);
                        }
                        fb.at_end(missing);
                        let (tt, tf) = ll::intrinsic(self.m(), "llvm.trap", &[]);
                        fb.call(tt, tf, &[]);
                        fb.unreachable();
                        resolver.function = f;
                    }
                }
                self.field_offset_resolvers.insert(key, resolver);
            }
            let resolver = self.field_offset_resolvers[&key];
            if resolver.constant {
                return self.b.gep(i8t, object, &[self.b.c_i64((resolver.offset + local_offset) as i64)]);
            }
            if !resolver.function.is_null() {
                let header = self.b.load(self.b.ptr(), object);
                ll::set_metadata_empty(header, "invariant.load");
                let id = self.load_runtime_field(header, rtti_field::TYPE_ID);
                ll::set_metadata_empty(id, "invariant.load");
                let fty = ll::function_type(resolver.function);
                let off = self.b.call(fty, resolver.function, &[id]);
                let addr = self.b.add(off, self.b.c_i64(local_offset as i64));
                return self.b.gep(i8t, object, &[addr]);
            }
        }
        // Open-world lookup through one non-inlined resolver.
        let rname = ".typhon.field.offset.dynamic";
        let resolver = match ll::get_function(self.m(), rname) {
            Some(f) => f,
            None => self.make_dynamic_field_resolver(rname),
        };
        let info = self.b.load(self.b.ptr(), object);
        ll::set_metadata_empty(info, "invariant.load");
        let count = self.load_runtime_field(info, rtti_field::MRO_COUNT);
        let offsets = self.load_runtime_field(info, rtti_field::MRO_OFFSETS);
        ll::set_metadata_empty(count, "invariant.load");
        ll::set_metadata_empty(offsets, "invariant.load");
        let fty = ll::function_type(resolver);
        let off = self.b.call(fty, resolver, &[count, offsets, self.b.c_i64(owner_id)]);
        let addr = self.b.add(off, self.b.c_i64(local_offset as i64));
        self.b.gep(i8t, object, &[addr])
    }

    fn make_dynamic_field_resolver(&mut self, name: &str) -> V {
        let i64t = self.b.i64();
        let fty = ll::fn_ty(i64t, &[i64t, self.b.ptr(), i64t], false);
        let r = ll::add_function(self.m(), name, fty);
        ll::set_linkage(r, LLVMPrivateLinkage);
        ll::set_memory_none(r);
        ll::add_fn_attr(r, "noinline");
        ll::add_fn_attr(r, "nounwind");
        ll::add_fn_attr(r, "willreturn");
        ll::add_nocapture(r, 2);
        let c = self.c;
        let entry = ll::append_block(c, r, "entry");
        let loop_ = ll::append_block(c, r, "field.probe");
        let check = ll::append_block(c, r, "field.check");
        let next = ll::append_block(c, r, "field.next");
        let found = ll::append_block(c, r, "field.found");
        let missing = ll::append_block(c, r, "field.missing");
        let b = Builder::new(c);
        b.at_end(entry);
        let mask_addr = b.gep(i64t, ll::param(r, 1), &[ll::param(r, 0)]);
        let mask = b.load(i64t, mask_addr);
        let slots = b.gep(i64t, mask_addr, &[b.c_i64(1)]);
        let mixed = b.mul(ll::param(r, 2), b.c_i64(METHOD_HASH_MULTIPLIER as i64));
        let hashed = b.xor(mixed, b.lshr(mixed, METHOD_HASH_SHIFT));
        let start = b.and(hashed, mask);
        b.br(loop_);
        b.at_end(loop_);
        let position = b.phi(i64t);
        Builder::add_incoming(position, start, entry);
        let slot = b.gep(i64t, slots, &[b.mul(position, b.c_i64(2))]);
        let offset = b.load(i64t, b.gep(i64t, slot, &[b.c_i64(1)]));
        b.cond_br(b.icmp_eq(offset, b.c_i64(0)), missing, check);
        b.at_end(check);
        let id = b.load(i64t, slot);
        b.cond_br(b.icmp_eq(id, ll::param(r, 2)), found, next);
        b.at_end(next);
        let np = b.and(b.add(position, b.c_i64(1)), mask);
        Builder::add_incoming(position, np, next);
        b.cond_br(b.icmp_eq(np, start), missing, loop_);
        b.at_end(missing);
        let (tt, tf) = ll::intrinsic(self.m(), "llvm.trap", &[]);
        b.call(tt, tf, &[]);
        b.unreachable();
        b.at_end(found);
        for f in [mask, id, offset] {
            ll::set_metadata_empty(f, "invariant.load");
        }
        b.ret(offset);
        r
    }

    /// `codegenTypeQuery`.
    pub(super) fn codegen_type_query(&mut self, m: &Module, f: FuncId) -> V {
        let fd = m.func(f).unwrap();
        let t = fd.intrinsic_type().expect("invalid type query signature");
        match fd.intrinsic() {
            Intrinsic::Sizeof => {
                let lt = self.get_llvm_type(m, t);
                self.b.c_i64(ll::alloc_size(self.m(), lt) as i64)
            }
            Intrinsic::IsAtomic => self.b.c_i8(m.type_is_atomic(t) as i64),
            Intrinsic::IsContentAtomic => self.b.c_i8(m.type_is_content_atomic(t) as i64),
            _ => panic!("intrinsic is not a type query"),
        }
    }

    /// `codegenIntrinsic`.
    pub(super) fn codegen_intrinsic(&mut self, m: &Module, f: FuncId, args: &[V]) -> Option<V> {
        let fd = m.func(f).unwrap();
        let FuncKind::Internal { intrinsic, intrinsic_type } = fd.kind else { return None };
        let it = intrinsic_type;
        match intrinsic {
            Intrinsic::Sizeof | Intrinsic::IsAtomic | Intrinsic::IsContentAtomic => Some(self.codegen_type_query(m, f)),
            Intrinsic::RecordNew => {
                let t = it.map(|t| m.actual_type(t)).expect("invalid record constructor");
                assert!(m.type_kind(t).is_record() && m.type_fields(t).len() == args.len(), "invalid record constructor");
                let lt = self.get_llvm_type(m, t);
                let mut r = ll::undef(lt);
                for (i, a) in args.iter().enumerate() {
                    r = self.b.insert_value(r, *a, i as u32);
                }
                Some(r)
            }
            Intrinsic::RepresentationCast => {
                let lt = self.get_llvm_type(m, it.expect("missing type"));
                if args.len() != 1 || ll::type_of(args[0]) != lt {
                    self.fail("invalid representation-preserving cast");
                    return Some(ll::undef(lt));
                }
                Some(args[0])
            }
            Intrinsic::GeneratorPromise => {
                let t = m.actual_type(it.expect("invalid generator promise access"));
                let TypeKind::Generator { base } = *m.type_kind(t) else { panic!("invalid generator promise access") };
                assert_eq!(args.len(), 1, "invalid generator promise access");
                let bt = self.get_llvm_type(m, base);
                if ll::is_void(bt) {
                    return Some(self.b.null_ptr());
                }
                let (pt, pf) = ll::intrinsic(self.m(), "llvm.coro.promise", &[]);
                let align = self.b.c_i32(ll::pref_align(self.m(), bt) as i64);
                Some(self.b.call(pt, pf, &[args[0], align, self.b.c_i1(false)]))
            }
            Intrinsic::AllocationSize => {
                let t = it.expect("allocation size requires a reference type");
                assert!(m.type_kind(m.actual_type(t)).is_ref(), "allocation size requires a reference type");
                let l = self.get_object_layout(m, t);
                Some(self.b.c_i64(ll::alloc_size(self.m(), l) as i64))
            }
            Intrinsic::Allocate => {
                let t = it.expect("reference allocation requires a reference type");
                assert!(m.type_kind(m.actual_type(t)).is_ref(), "reference allocation requires a reference type");
                let l = self.get_object_layout(m, t);
                let size = ll::alloc_size(self.m(), l).max(1);
                let (at, af) = self.make_alloc_func(m.type_is_content_atomic(t), false);
                let object = self.b.call(at, af, &[self.b.c_i64(size as i64)]);
                if Self::is_polymorphic(m, t) {
                    let ti = self.get_runtime_type_info(m, t);
                    self.b.store(ti, object);
                }
                Some(object)
            }
            Intrinsic::Typeinfo => Some(self.get_runtime_type_info(m, it.expect("missing type"))),
            Intrinsic::TypeinfoLookup => {
                let arg = args[0];
                // AOT knows the full ID set.
                let missing = self.new_block_pub("typeinfo.missing");
                let sel = self.b.switch(arg, missing, 0);
                for t in m.types() {
                    let Some(info) = m.runtime_info(t) else { continue };
                    let id = info.id;
                    let found = self.new_block_pub("typeinfo.found");
                    Builder::add_case(sel, self.b.c_i64(id), found);
                    let ti = self.get_runtime_type_info(m, t);
                    self.b.at_end(found);
                    self.block = found;
                    self.b.ret(ti);
                }
                self.b.at_end(missing);
                self.block = missing;
                let (tt, tf) = ll::intrinsic(self.m(), "llvm.trap", &[]);
                self.b.call(tt, tf, &[]);
                self.b.unreachable();
                None
            }
            Intrinsic::VirtualLookup | Intrinsic::TypeinfoMethodLookup => {
                // Probe the keyed table emitted by getRuntimeTypeInfo().
                let info = if intrinsic == Intrinsic::VirtualLookup { self.b.load(self.b.ptr(), args[0]) } else { args[0] };
                let count = self.load_runtime_field(info, rtti_field::METHOD_COUNT);
                let valid = self.new_block_pub("method.valid");
                let probe = self.new_block_pub("method.probe");
                let next = self.new_block_pub("method.next");
                let missing = self.new_block_pub("method.missing");
                let found = self.new_block_pub("method.found");
                let done = self.new_block_pub("method.done");
                self.b.cond_br(self.b.icmp_ne(count, self.b.c_i64(0)), valid, missing);
                self.b.at_end(valid);
                let methods = self.load_runtime_field(info, rtti_field::METHODS);
                let mask = self.b.sub(count, self.b.c_i64(1));
                let mixed = self.b.mul(args[1], self.b.c_i64(METHOD_HASH_MULTIPLIER as i64));
                let hash = self.b.xor(mixed, self.b.lshr(mixed, METHOD_HASH_SHIFT));
                let start = self.b.and(hash, mask);
                let slot_type = ll::struct_ty(self.c, &[self.b.i64(), self.b.ptr()]);
                self.b.br(probe);
                self.b.at_end(probe);
                let position = self.b.phi(self.b.i64());
                Builder::add_incoming(position, start, valid);
                let slot = self.b.gep(slot_type, methods, &[position]);
                let id = self.b.load(self.b.i64(), self.b.struct_gep(slot_type, slot, 0));
                self.b.cond_br(self.b.icmp_eq(id, args[1]), found, next);
                self.b.at_end(next);
                let following = self.b.and(self.b.add(position, self.b.c_i64(1)), mask);
                Builder::add_incoming(position, following, next);
                let cont = self.b.and(self.b.icmp_ne(id, self.b.c_i64(0)), self.b.icmp_ne(following, start));
                self.b.cond_br(cont, probe, missing);
                self.b.at_end(found);
                let target = self.b.load(self.b.ptr(), self.b.struct_gep(slot_type, slot, 1));
                self.b.cond_br(self.b.is_not_null(target), done, missing);
                self.b.at_end(missing);
                let (tt, tf) = ll::intrinsic(self.m(), "llvm.trap", &[]);
                self.b.call(tt, tf, &[]);
                self.b.unreachable();
                self.b.at_end(done);
                self.block = done;
                Some(target)
            }
            Intrinsic::None => None,
        }
    }
}
