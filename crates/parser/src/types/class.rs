//! Port of codon/parser/ast/types/class.{h,cpp}: `ClassType` and `ClassType::Generic`.

use std::collections::BTreeSet;

use typhon_common::Result;
use typhon_common::mangle::stdlib;

use super::*;

/// `ClassType::PartialFlag`.
pub mod partial_flag {
    pub const MISSING: u8 = b'0';
    pub const INCLUDED: u8 = b'1';
    pub const DEFAULT: u8 = b'2';
}

impl TypeArena {
    // ------------------------------------------------------------------
    // Generic helpers
    // ------------------------------------------------------------------

    /// `Generic::debugString`.
    pub fn generic_debug_string(&self, g: &Generic, mode: i32) -> String {
        let t = g.ty();
        if g.static_kind == LiteralKind::Runtime && self.get_static(t).is_some() && mode != 2 {
            if let Some(ns) = self.non_static_type(t) {
                return self.debug_string(ns, mode);
            }
        }
        self.debug_string(t, mode)
    }
    /// `Generic::realizedName`.
    pub fn generic_realized_name(&self, g: &Generic) -> String {
        let t = g.ty();
        if g.static_kind == LiteralKind::Runtime && self.get_static(t).is_some() {
            if let Some(ns) = self.non_static_type(t) {
                return self.realized_name(ns);
            }
        }
        self.realized_name(t)
    }
    /// `Generic::generalize`.
    pub fn generic_generalize(&self, g: &Generic, at_level: i32) -> Generic {
        let t = match g.ty {
            Some(t) if g.static_kind == LiteralKind::Runtime && self.get_static(t).is_some() => {
                let ns = self.non_static_type(t).unwrap_or(t);
                Some(self.generalize(ns, at_level))
            }
            Some(t) => Some(self.generalize(t, at_level)),
            None => None,
        };
        Generic { name: g.name, id: g.id, ty: t, static_kind: g.static_kind }
    }
    /// `Generic::instantiate`.
    pub(crate) fn generic_instantiate(
        &self,
        g: &Generic,
        at_level: i32,
        use_counter: bool,
        cache: &mut InstantiationCache,
        has_cache: bool,
    ) -> Generic {
        let t = match g.ty {
            Some(t) if g.static_kind == LiteralKind::Runtime && self.get_static(t).is_some() => {
                let ns = self.non_static_type(t).unwrap_or(t);
                Some(self.instantiate_impl(ns, at_level, use_counter, cache, has_cache))
            }
            Some(t) => Some(self.instantiate_impl(t, at_level, use_counter, cache, has_cache)),
            None => None,
        };
        Generic { name: g.name, id: g.id, ty: t, static_kind: g.static_kind }
    }

    // ------------------------------------------------------------------
    // ClassType
    // ------------------------------------------------------------------

    /// `ClassType::freeze`: mark fully concrete plain classes immutable.
    pub fn freeze(&self, this: TypeId) -> bool {
        let Some(c) = self.class(this) else { return false };
        if c.immutable {
            return true;
        }
        if !matches!(c.kind, ClassKind::Plain) || c.name == stdlib().unrealized_type.as_str() || c.name == "__NTuple__" {
            return false;
        }
        let stable = |g: &Generic| -> bool {
            let Some(t) = g.ty else { return false };
            if self.get_static(t).is_some() {
                return true;
            }
            // Links that point to frozen types are fixed in realize() during typechecking
            if self.get_link(t).is_some() {
                return false;
            }
            match self.get_class(t) {
                Some(child) => self.freeze(child),
                None => false,
            }
        };
        if !c.generics.iter().all(stable) || !c.hidden_generics.iter().all(stable) {
            return false;
        }
        self.with_class_mut(this, |c| c.immutable = true);
        true
    }

    /// `ClassType::canonicalOptional`: `Optional[Optional[T]]` / `Optional[NoneType]`
    /// collapse to their (realizable) member.
    pub(crate) fn canonical_optional(&self, this: TypeId) -> Option<TypeId> {
        let (name, g0) = self.with_class(this, |c| {
            (c.name, if c.generics.len() == 1 { c.generics[0].ty } else { None })
        })?;
        if name != stdlib().optional.as_str() {
            return None;
        }
        let member = self.get_class(g0?)?;
        let mname = self.with_class(member, |c| c.name)?;
        if self.can_realize(member) && (mname == stdlib().optional.as_str() || mname == stdlib().none_type.as_str()) {
            Some(member)
        } else {
            None
        }
    }

    /// `ClassType::unify` (plain classes; also used as the base step for subclasses).
    pub(crate) fn class_unify(
        &self,
        this: TypeId,
        typ: TypeId,
        mut us: Option<&mut Unification>,
        host: &mut dyn TypeHost,
    ) -> Result<i32> {
        if let Some(canonical) = self.canonical_optional(this) {
            return self.unify(canonical, typ, us, host);
        }
        let st = stdlib();
        let me = self.class(this).unwrap();
        if let Some(tcid) = self.get_class(typ) {
            let tc = self.class(tcid).unwrap();
            // A defaulted None payload also resolves the original optional slot.
            if me.name == st.optional.as_str()
                && tc.name == st.none_type.as_str()
                && me.generics.len() == 1
                && self.get_unbound(me.generics[0].ty()).is_some()
            {
                return self.unify(me.generics[0].ty(), tcid, us, host);
            }
            if me.name == st.none_type.as_str() && tc.name == st.optional.as_str() {
                return self.unify(tcid, this, us, host);
            }
            if me.name == st.unrealized_type.as_str() && tc.name == me.name {
                if me.generics[0].ty == tc.generics[0].ty {
                    return Ok(0);
                }
                // instantiate + unify!
                let mut gc = InstantiationCache::default();
                let l = self.instantiate(me.generics[0].ty(), 0, Some(&mut gc));
                gc.clear();
                let r = self.instantiate(tc.generics[0].ty(), 0, Some(&mut gc));
                return self.unify(l, r, us, host);
            }

            let mut s1 = 3;
            if me.name == "__NTuple__" && tc.name == me.name {
                let n1 = self.get_int_static(me.generics[0].ty());
                let n2 = self.get_int_static(tc.generics[0].ty());
                if let (Some(n1), Some(n2)) = (n1, n2) {
                    let t1 = self.generics(me.generics[1].ty());
                    let t2 = self.generics(tc.generics[1].ty());
                    if n1 * t1.len() as i64 != n2 * t2.len() as i64 {
                        return Ok(-1);
                    }
                    for i in 0..(t1.len() * n1.max(0) as usize) {
                        let s = self.unify(t1[i % t1.len()].ty(), t2[i % t2.len()].ty(), us.as_deref_mut(), host)?;
                        if s == -1 {
                            return Ok(-1);
                        }
                        s1 += s;
                    }
                    return Ok(s1);
                }
            } else if tc.name == "__NTuple__" {
                return self.unify(tcid, this, us, host);
            } else if me.name == "__NTuple__" && tc.name == st.tuple.as_str() {
                let n1 = self.get_int_static(me.generics[0].ty());
                match n1 {
                    None => {
                        let n = tc.generics.len();
                        let tn = self.new_int_static(n as i64);
                        // If we are unifying NT[N, T] and T[X, X, ...], we assume that N is
                        // number of X's
                        if self.unify(me.generics[0].ty(), tn, us.as_deref_mut(), host)? == -1 {
                            return Ok(-1);
                        }
                        let tt = if n > 0 {
                            let tt = host.instantiate_tuple(1, Some(&[tc.generics[0].ty()]))?;
                            for i in 1..tc.generics.len() {
                                let g0 = self.generic_type(tt, 0).unwrap();
                                let s = self.unify(g0, tc.generics[i].ty(), us.as_deref_mut(), host)?;
                                if s == -1 {
                                    return Ok(-1);
                                }
                                s1 += s;
                            }
                            tt
                        } else {
                            host.instantiate_tuple(1, None)?
                        };
                        if self.unify(me.generics[1].ty(), tt, us.as_deref_mut(), host)? == -1 {
                            return Ok(-1);
                        }
                    }
                    Some(n1) => {
                        let t1 = self.generics(me.generics[1].ty());
                        if n1 * t1.len() as i64 != tc.generics.len() as i64 {
                            return Ok(-1);
                        }
                        for i in 0..(t1.len() * n1.max(0) as usize) {
                            let s = self.unify(t1[i % t1.len()].ty(), tc.generics[i].ty(), us.as_deref_mut(), host)?;
                            if s == -1 {
                                return Ok(-1);
                            }
                            s1 += s;
                        }
                    }
                }
                return Ok(s1);
            }

            // Check names.
            if me.name != tc.name {
                return Ok(-1);
            }
            // Check generics.
            if me.generics.len() != tc.generics.len() {
                return Ok(-1);
            }
            for i in 0..me.generics.len() {
                let s = self.unify(me.generics[i].ty(), tc.generics[i].ty(), us.as_deref_mut(), host)?;
                if s == -1 {
                    return Ok(-1);
                }
                s1 += s;
            }
            for i in 0..me.hidden_generics.len() {
                let Some(other) = tc.hidden_generics.get(i) else { return Ok(-1) };
                let s = self.unify(me.hidden_generics[i].ty(), other.ty(), us.as_deref_mut(), host)?;
                if s == -1 {
                    return Ok(-1);
                }
                s1 += s;
            }
            Ok(s1)
        } else if let Some(tl) = self.get_link(typ) {
            self.unify(tl, this, us, host)
        } else {
            Ok(-1)
        }
    }

    fn rebuild_class(&self, this: TypeId, g: Vec<Generic>, hg: Vec<Generic>) -> TypeId {
        let (name, is_tuple, src) = self.with(this, |d| match &d.kind {
            TypeKind::Class(c) => (c.name, c.is_tuple, d.src),
            _ => unreachable!(),
        });
        let mut c = ClassType::new(name, g, hg);
        c.is_tuple = is_tuple;
        self.alloc(TypeKind::Class(c), src)
    }

    /// Shared early-outs of `ClassType::generalize/instantiate`.
    fn class_shared(&self, this: TypeId, plain_only: bool) -> bool {
        let c = self.class(this).unwrap();
        if c.immutable {
            return true;
        }
        if c.generics.is_empty() && c.hidden_generics.is_empty() && (!plain_only || matches!(c.kind, ClassKind::Plain)) {
            if matches!(c.kind, ClassKind::Plain) {
                return true;
            }
        }
        if c.name == stdlib().unrealized_type.as_str() {
            if let Some(g0) = c.generics.first().and_then(|g| g.ty) {
                if self.get_class(g0).is_some() {
                    return true;
                }
            }
        }
        false
    }

    /// `ClassType::generalize` (base step: always produces a plain class).
    pub(crate) fn class_generalize_base(&self, this: TypeId, at_level: i32, check_shared: bool) -> TypeId {
        if check_shared && self.class_shared(this, true) {
            return this;
        }
        let c = self.class(this).unwrap();
        let g = c.generics.iter().map(|t| self.generic_generalize(t, at_level)).collect();
        let hg = c.hidden_generics.iter().map(|t| self.generic_generalize(t, at_level)).collect();
        self.rebuild_class(this, g, hg)
    }
    pub(crate) fn class_generalize(&self, this: TypeId, at_level: i32) -> TypeId {
        self.class_generalize_base(this, at_level, true)
    }

    /// `ClassType::instantiate` (base step: always produces a plain class).
    pub(crate) fn class_instantiate_base(
        &self,
        this: TypeId,
        at_level: i32,
        use_counter: bool,
        cache: &mut InstantiationCache,
        has_cache: bool,
        check_shared: bool,
    ) -> TypeId {
        if check_shared && self.class_shared(this, true) {
            return this;
        }
        let c = self.class(this).unwrap();
        let g = c.generics.iter().map(|t| self.generic_instantiate(t, at_level, use_counter, cache, has_cache)).collect();
        let hg = c
            .hidden_generics
            .iter()
            .map(|t| self.generic_instantiate(t, at_level, use_counter, cache, has_cache))
            .collect();
        self.rebuild_class(this, g, hg)
    }
    pub(crate) fn class_instantiate(
        &self,
        this: TypeId,
        at_level: i32,
        use_counter: bool,
        cache: &mut InstantiationCache,
        has_cache: bool,
    ) -> TypeId {
        self.class_instantiate_base(this, at_level, use_counter, cache, has_cache, true)
    }

    /// `ClassType::hasUnbounds`.
    pub(crate) fn class_has_unbounds(&self, c: &ClassType, include_generics: bool) -> bool {
        if c.name == stdlib().unrealized_type.as_str() {
            return false;
        }
        c.generics
            .iter()
            .chain(c.hidden_generics.iter())
            .any(|g| g.ty.is_some_and(|t| self.has_unbounds(t, include_generics)))
    }

    /// `ClassType::getUnbounds`.
    pub(crate) fn class_get_unbounds(&self, c: &ClassType, include_generics: bool) -> Vec<TypeId> {
        let mut u = vec![];
        if c.name == stdlib().unrealized_type.as_str() {
            return u;
        }
        for g in c.generics.iter().chain(c.hidden_generics.iter()) {
            if let Some(t) = g.ty {
                let mut tu = self.get_unbounds(t, include_generics);
                tu.append(&mut u);
                u = tu;
            }
        }
        u
    }

    /// `ClassType::canRealize`.
    pub(crate) fn class_can_realize(&self, c: &ClassType) -> bool {
        let st = stdlib();
        if c.name == st.type_.as_str() && !self.class_has_unbounds(c, false) {
            return true; // always true!
        }
        if c.name == st.unrealized_type.as_str() {
            return self.get_class(c.generics[0].ty()).is_some();
        }
        c.generics.iter().chain(c.hidden_generics.iter()).all(|g| g.ty.is_none_or(|t| self.can_realize(t)))
    }

    /// `ClassType::isInstantiated`.
    pub(crate) fn class_is_instantiated(&self, c: &ClassType, skip: Option<TypeId>) -> bool {
        if c.name == stdlib().unrealized_type.as_str() {
            return self.get_class(c.generics[0].ty()).is_some();
        }
        c.generics
            .iter()
            .chain(c.hidden_generics.iter())
            .all(|g| g.ty.is_none_or(|t| self.is_instantiated_impl(t, skip)))
    }

    /// `ClassType::debugString` (also used as the base for `UnionType`/`FuncType` callers
    /// that explicitly ask for `ClassType::debugString`).
    pub fn class_debug_string(&self, this: TypeId, mode: i32) -> String {
        if let Some(canonical) = self.canonical_optional(this) {
            return self.debug_string(canonical, mode);
        }
        let c = self.class(this).unwrap();
        let st = stdlib();
        if c.name == st.named_tuple.as_str() {
            if let Some(id) = self.get_int_static(c.generics[0].ty()) {
                let names = self.generated_tuple_names.borrow().get(id as usize).cloned().unwrap_or_default();
                if names.is_empty() {
                    return c.name.to_string();
                }
                let ts = self.generics(c.generics[1].ty());
                let as_: Vec<String> = names
                    .iter()
                    .enumerate()
                    .map(|(i, n)| format!("{}={}", n, ts.get(i).map(|g| self.generic_debug_string(g, mode)).unwrap_or_default()))
                    .collect();
                return format!("{}[{}]", c.name, join(&as_, ","));
            } else {
                return format!("{}[{}]", c.name, self.debug_string(c.generics[0].ty(), mode));
            }
        } else if c.name == "Partial" && c.generics.len() > 3 && self.get_class(c.generics[3].ty()).is_some() && mode != 2 {
            // Name: function[full_args](instantiated_args...)
            let mut as_: Vec<String> = vec![];
            let known = self.partial_mask(this);
            let func = self.partial_func(this);
            let finfo = self.class(func).unwrap();
            let fdata = finfo.func().unwrap().clone();
            let ast = self.func_ast(fdata.ast);

            let mut args: Vec<String> = vec![];
            if self.get_class(c.generics[1].ty()).is_some() {
                for i in self.generics(c.generics[1].ty()) {
                    args.push(self.generic_debug_string(&i, mode));
                }
            }
            let (mut ai, mut gi) = (0usize, 0usize);
            for (i, &k) in known.as_bytes().iter().enumerate() {
                let is_value = ast.params.get(i).is_none_or(|p| p.1 == ParamStatus::Value);
                if is_value {
                    as_.push(if ai < args.len() {
                        if k == partial_flag::INCLUDED { args[ai].clone() } else { "...".into() }
                    } else {
                        "...".into()
                    });
                    if k == partial_flag::INCLUDED {
                        ai += 1;
                    }
                } else {
                    let s = fdata.func_generics.get(gi).map(|g| self.generic_debug_string(g, mode)).unwrap_or_default();
                    as_.push(if k == partial_flag::INCLUDED {
                        s
                    } else {
                        format!("...{}", if mode == 0 { String::new() } else { s })
                    });
                    gi += 1;
                }
            }
            if let Some(last) = args.last() {
                if last != st.tuple.as_str() {
                    // unused *args (by default always 0 in mask)
                    as_.push(last.clone());
                }
            }
            let ks = self.debug_string(c.generics[2].ty(), mode);
            if ks.len() > 10 {
                // if **kwargs is used; chop off NamedTuple[...]
                as_.push(ks[11.min(ks.len())..ks.len() - 1].to_string());
            }
            let fnname = if mode == 0 { self.rev(&ast.name) } else { ast.name.clone() };
            return format!("{}({})", fnname, join(&as_, ","));
        }
        let mut gs: Vec<String> = vec![];
        for a in &c.generics {
            if !a.name.is_empty() {
                gs.push(self.generic_debug_string(a, mode));
            }
        }
        if mode == 2 && !c.hidden_generics.is_empty() {
            for a in &c.hidden_generics {
                if !a.name.is_empty() {
                    gs.push(format!("-{}", self.generic_debug_string(a, mode)));
                }
            }
        }
        // Special formatting for Functions and Tuples
        let n = if mode == 0 { self.rev(&c.name) } else { c.name.to_string() };
        let s = format!("{}{}", n, if gs.is_empty() { String::new() } else { format!("[{}]", join(&gs, ",")) });
        if s == "Int[64]" { "int".into() } else { s }
    }

    /// `ClassType::realizedName` (with the `_rn` cache). Use [`TypeArena::realized_name`]
    /// for virtual dispatch.
    pub fn class_realized_name(&self, this: TypeId) -> String {
        if let Some(canonical) = self.canonical_optional(this) {
            return self.realized_name(canonical);
        }
        let c = self.class(this).unwrap();
        if !c.rn.is_empty() {
            return c.rn;
        }
        let s = if c.name == "Partial" {
            self.class_debug_string(this, 1)
        } else {
            let mut gs: Vec<String> = vec![];
            if c.name == stdlib().union.as_str() && self.get_class(c.generics[0].ty()).is_some() {
                let mut gss = BTreeSet::new();
                for a in self.generics(c.generics[0].ty()) {
                    gss.insert(self.generic_realized_name(&a));
                }
                gs = vec![gss.into_iter().collect::<Vec<_>>().join("|")];
            } else {
                for a in &c.generics {
                    if !a.name.is_empty() {
                        gs.push(self.generic_realized_name(a));
                    }
                }
            }
            let s = join(&gs, ",");
            format!("{}{}", c.name, if s.is_empty() { String::new() } else { format!("[{s}]") })
        };
        if self.class_can_realize(&c) {
            self.with_class_mut(this, |c| c.rn = s.clone());
        }
        s
    }

    /// `getPartialFunc()`.
    pub fn partial_func(&self, this: TypeId) -> TypeId {
        let g3 = self.generic_type(this, 3).expect("not a partial");
        let n = self.generic_type(g3, 0).expect("not a partial func");
        self.get_func(n).expect("not a partial func")
    }
    /// `getPartialMask()`.
    pub fn partial_mask(&self, this: TypeId) -> String {
        let n = self.get_str_static(self.generic_type(this, 0).unwrap()).unwrap_or_default();
        if self.is_partial_virtual(this) { n[1..].to_string() } else { n }
    }
    /// `isPartialVirtual()`.
    pub fn is_partial_virtual(&self, this: TypeId) -> bool {
        self.get_str_static(self.generic_type(this, 0).unwrap()).is_some_and(|s| s.starts_with('v'))
    }
    /// `isPartialEmpty()`.
    pub fn is_partial_empty(&self, this: TypeId) -> bool {
        let a = self.generic_type(this, 1).unwrap();
        let ka = self.generic_type(this, 2).unwrap();
        let ag = self.generics(a);
        ag.len() == 1
            && self.generics(ag[0].ty()).is_empty()
            && self.generic_type(ka, 1).is_some_and(|k| self.generics(k).is_empty())
    }
}
