//! `UnionType`.

use std::collections::{BTreeMap, BTreeSet};

use typhon_common::Result;

use super::*;

impl TypeArena {
    /// `UnionType::unify`.
    pub(crate) fn union_unify(
        &self,
        this: TypeId,
        typ: TypeId,
        mut us: Option<&mut Unification>,
        host: &mut dyn TypeHost,
    ) -> Result<i32> {
        if let Some(tr) = self.get_union(typ) {
            // Do not hard-unify if we have unbounds
            if !self.can_realize(this) || !self.can_realize(tr) {
                return Ok(0);
            }
            let u1 = self.union_realization_types(this);
            let u2 = self.union_realization_types(tr);
            if u1.len() != u2.len() {
                return Ok(-1);
            }
            let mut s1 = 2;
            for (a, b) in u1.iter().zip(u2.iter()) {
                let s = self.unify(*a, *b, us.as_deref_mut(), host)?;
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

    fn union_from_base(&self, this: TypeId, base: TypeId) -> TypeId {
        let g = self.with_class(base, |c| c.generics.clone()).unwrap();
        let t = self.new_union(g);
        self.set_src(t, self.src(this));
        t
    }

    /// `UnionType::generalize`.
    pub(crate) fn union_generalize(&self, this: TypeId, at_level: i32) -> TypeId {
        let r = self.class_generalize_base(this, at_level, false);
        self.union_from_base(this, r)
    }

    /// `UnionType::instantiate`.
    pub(crate) fn union_instantiate(
        &self,
        this: TypeId,
        at_level: i32,
        use_counter: bool,
        cache: &mut InstantiationCache,
        has_cache: bool,
    ) -> TypeId {
        let r = self.class_instantiate_base(this, at_level, use_counter, cache, has_cache, false);
        self.union_from_base(this, r)
    }

    /// `UnionType::debugString`.
    pub(crate) fn union_debug_string(&self, this: TypeId, c: &ClassType, mode: i32) -> String {
        if mode == 2 || c.generics.is_empty() || self.get_class(c.generics[0].ty()).is_none() {
            return self.class_debug_string(this, mode);
        }
        let mut gss = BTreeSet::new();
        for a in self.generics(c.generics[0].ty()) {
            gss.insert(self.generic_debug_string(&a, mode));
        }
        let s = gss.into_iter().collect::<Vec<_>>().join("|");
        format!("{}{}", c.name, if s.is_empty() { String::new() } else { format!("[{s}]") })
    }

    /// `UnionType::getRealizationTypes`: member types, deduplicated and ordered by
    /// realized name.
    pub fn union_realization_types(&self, this: TypeId) -> Vec<TypeId> {
        typhon_common::seqassert!(self.can_realize(this), "cannot realize {}", self.debug_string(this, 2));
        let g0 = self.generic_type(this, 0).unwrap();
        let mut m = BTreeMap::new();
        for u in self.generics(g0) {
            m.insert(self.realized_name(u.ty()), u.ty());
        }
        m.into_values().collect()
    }
}
