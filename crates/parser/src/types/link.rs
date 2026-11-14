//! `LinkType` operations.

use typhon_common::Result;

use super::*;

impl TypeArena {
    /// `LinkType::unify`.
    pub(crate) fn link_unify(
        &self,
        this: TypeId,
        typ: TypeId,
        mut undo: Option<&mut Unification>,
        host: &mut dyn TypeHost,
    ) -> Result<i32> {
        let l = self.link(this).expect("not a link");
        if l.kind == LinkKind::Link {
            // Case 1: Just follow the link
            return self.unify(l.ty.unwrap(), typ, undo, host);
        }
        // Case 3: Unbound unification
        let my_sk = self.static_kind(this);
        let their_sk = self.static_kind(typ);
        if my_sk != their_sk {
            if my_sk == LiteralKind::Runtime {
                // other one is; move this to non-static equivalent
                if let Some(u) = undo.as_deref_mut() {
                    u.statics.push(this);
                    self.with_link_mut(this, |l| l.static_kind = their_sk);
                }
            } else {
                return Ok(-1);
            }
        }
        if let Some(t) = self.link(typ) {
            if t.kind == LinkKind::Link {
                return self.unify(t.ty.unwrap(), this, undo, host);
            }
            if l.kind != t.kind {
                return Ok(-1);
            }
            // Identical unbound types get a score of 1
            if l.id == t.id {
                return Ok(1);
            }
            // Generics must have matching IDs unless we are doing non-destructive unification
            if l.kind == LinkKind::Generic {
                return Ok(if undo.is_some() { -1 } else { 1 });
            }
            // Always merge a newer type into the older type
            if l.id < t.id {
                return self.unify(typ, this, undo, host);
            }
        } else if l.kind == LinkKind::Generic {
            return Ok(-1);
        }
        typhon_common::seqassert!(l.kind == LinkKind::Unbound, "not an unbound");

        // Ensure that we do not have recursive unification! (e.g. unify ?1 with list[?1])
        if self.occurs(this, typ, undo.as_deref_mut()) {
            return Ok(-1);
        }
        // Handle traits
        let my_trait = self.link(this).and_then(|l| l.trait_);
        if let Some(tr) = my_trait {
            if self.unify(tr, typ, undo.as_deref_mut(), host)? == -1 {
                return Ok(-1);
            }
        }
        // Unification: destructive part.
        if let Some(u) = undo {
            typhon_common::log_flag!(TYPECHECK, "[unify] {} := {}", l.id, self.debug_string(typ, 2));
            u.linked.push(this);
            let target = self.follow(typ);
            self.with_link_mut(this, |l| {
                l.kind = LinkKind::Link;
                l.ty = Some(target);
            });
            if let Some(tr) = my_trait {
                if let Some(tl) = self.link(target) {
                    if tl.kind == LinkKind::Unbound && tl.trait_.is_none() {
                        u.traits.push(target);
                        self.with_link_mut(target, |l| l.trait_ = Some(tr));
                    }
                }
            }
        }
        Ok(0)
    }

    /// `LinkType::generalize`.
    pub(crate) fn link_generalize(&self, this: TypeId, at_level: i32) -> TypeId {
        let l = self.link(this).unwrap();
        match l.kind {
            LinkKind::Generic => this,
            LinkKind::Unbound => {
                if l.level >= at_level {
                    let trait_ = l.trait_.map(|t| self.generalize(t, at_level));
                    let default_type = l.default_type.map(|t| self.generalize(t, at_level));
                    self.new_link_type(
                        LinkType {
                            kind: LinkKind::Generic,
                            id: l.id,
                            level: 0,
                            ty: None,
                            static_kind: l.static_kind,
                            trait_,
                            generic_name: l.generic_name,
                            default_type,
                            pass_through: l.pass_through,
                        late_default: l.late_default,
                        },
                        typhon_common::SrcInfo::default(),
                    )
                } else {
                    this
                }
            }
            LinkKind::Link => self.generalize(l.ty.unwrap(), at_level),
        }
    }

    /// `LinkType::instantiate`.
    pub(crate) fn link_instantiate(
        &self,
        this: TypeId,
        at_level: i32,
        use_counter: bool,
        cache: &mut InstantiationCache,
        has_cache: bool,
    ) -> TypeId {
        let l = self.link(this).unwrap();
        match l.kind {
            LinkKind::Generic => {
                if has_cache {
                    if let Some(r) = cache.get(&l.id) {
                        return *r;
                    }
                }
                let id = if use_counter { self.next_unbound_id() } else { l.id };
                let trait_ = l.trait_.map(|t| self.instantiate_impl(t, at_level, use_counter, cache, has_cache));
                let default_type =
                    l.default_type.map(|t| self.instantiate_impl(t, at_level, use_counter, cache, has_cache));
                let t = self.new_link_type(
                    LinkType {
                        kind: LinkKind::Unbound,
                        id,
                        level: at_level,
                        ty: None,
                        static_kind: l.static_kind,
                        trait_,
                        generic_name: l.generic_name,
                        default_type,
                        pass_through: l.pass_through,
                        late_default: l.late_default,
                    },
                    typhon_common::SrcInfo::default(),
                );
                if has_cache {
                    cache.insert(l.id, t);
                }
                t
            }
            LinkKind::Unbound => this,
            LinkKind::Link => self.instantiate_impl(l.ty.unwrap(), at_level, use_counter, cache, has_cache),
        }
    }

    /// `LinkType::debugString`.
    pub(crate) fn link_debug_string(&self, l: &LinkType, mode: i32) -> String {
        if l.kind == LinkKind::Unbound || l.kind == LinkKind::Generic {
            if mode == 2 {
                return format!(
                    "{}{}{}{}{}",
                    if l.generic_name.is_empty() { String::new() } else { format!("{}:", l.generic_name) },
                    if l.kind == LinkKind::Unbound { "?" } else { "#" },
                    l.id,
                    l.trait_.map(|t| format!(":{}", self.debug_string(t, mode))).unwrap_or_default(),
                    if l.static_kind != LiteralKind::Runtime {
                        format!(":S{}", literal_index(l.static_kind))
                    } else {
                        String::new()
                    }
                );
            } else if let Some(t) = l.trait_ {
                return self.debug_string(t, mode);
            }
            return if l.generic_name.is_empty() {
                if mode != 0 { "?".into() } else { "<unknown type>".into() }
            } else {
                l.generic_name.clone()
            };
        }
        self.debug_string(l.ty.unwrap(), mode)
    }

    /// `LinkType::occurs`: checks if `this` (unbound) occurs within `typ`; lowers levels of
    /// free unbounds (recorded in `undo`). Shared sub-graphs are visited once.
    pub(crate) fn occurs(&self, this: TypeId, typ: TypeId, mut undo: Option<&mut Unification>) -> bool {
        let me = self.link(this).unwrap();
        let mut visited = Visited::default();
        let mut stack = vec![typ];
        // Note: iterative DFS instead of recursive std::function (same visit order
        // is irrelevant for the boolean result and level adjustments).
        while let Some(t) = stack.pop() {
            if !visited.insert(t) {
                continue;
            }
            match self.kind(t) {
                TypeKind::Link(tl) => match tl.kind {
                    LinkKind::Unbound => {
                        if tl.id == me.id {
                            return true;
                        }
                        if let Some(tr) = tl.trait_ {
                            stack.push(tr);
                        }
                        if let Some(u) = undo.as_deref_mut() {
                            if tl.level > me.level {
                                u.leveled.push((t, tl.level));
                                self.with_link_mut(t, |l| l.level = me.level);
                            }
                        }
                    }
                    LinkKind::Link => stack.push(tl.ty.unwrap()),
                    LinkKind::Generic => {}
                },
                TypeKind::Class(c) => {
                    if c.is_static() {
                        continue;
                    }
                    let c = self.get_class(t).and_then(|c| self.class(c)).unwrap_or(c);
                    for g in c.generics.iter().rev() {
                        if let Some(gt) = g.ty {
                            stack.push(gt);
                        }
                    }
                }
                TypeKind::Trait(tk) => {
                    // the reference `typ->getClass()` is null for traits -> false; but the trait's
                    // own unbound args are not visited either.
                    let _ = tk;
                }
            }
        }
        false
    }
}
