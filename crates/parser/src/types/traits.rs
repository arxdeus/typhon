//! `CallableTrait`, `TypeTrait`.

use typhon_common::Result;
use typhon_common::mangle::stdlib;

use super::*;

impl TypeArena {
    /// `CallableTrait::unify` / `TypeTrait::unify`.
    pub(crate) fn trait_unify(
        &self,
        this: TypeId,
        typ: TypeId,
        mut us: Option<&mut Unification>,
        host: &mut dyn TypeHost,
    ) -> Result<i32> {
        match self.trait_kind(this).unwrap() {
            TraitKind::Callable { args } => {
                if self.get_class(typ).is_some() {
                    let st = stdlib();
                    if self.is(typ, &st.none_type) {
                        return Ok(1);
                    }
                    if args.is_empty() {
                        return Ok(
                            if self.is(typ, &st.function) || self.get_partial(typ).is_some() || self.is(typ, &st.type_wrap) {
                                1
                            } else {
                                -1
                            },
                        );
                    }
                    let arg_types: Vec<TypeId> = self.generics(args[0].unwrap()).iter().map(|g| g.ty()).collect();
                    let r = host.match_callable(typ, &arg_types, args[1], us)?;
                    return Ok(if r >= 0 { 1 } else { -1 });
                } else if let Some(tl) = self.link(typ) {
                    if tl.kind == LinkKind::Link {
                        return self.trait_unify(this, tl.ty.unwrap(), us, host);
                    }
                    if tl.kind == LinkKind::Unbound {
                        if let Some(tt) = tl.trait_ {
                            let Some(TraitKind::Callable { args: targs }) = self.trait_kind(tt) else {
                                return Ok(-1);
                            };
                            if targs.len() != args.len() {
                                return Ok(-1);
                            }
                            for (a, b) in args.iter().zip(targs.iter()) {
                                let (Some(a), Some(b)) = (a, b) else { continue };
                                if self.unify(*a, *b, us.as_deref_mut(), host)? == -1 {
                                    return Ok(-1);
                                }
                            }
                        }
                        return Ok(1);
                    }
                }
                Ok(-1)
            }
            TraitKind::Type { ty } => {
                if self.get_class(typ).is_some() {
                    // does not make sense otherwise and results in infinite cycles
                    return self.unify(typ, ty, us, host);
                }
                if self.get_unbound(typ).is_some() {
                    return Ok(0);
                }
                Ok(-1)
            }
        }
    }

    /// `CallableTrait::generalize` / `TypeTrait::generalize`.
    pub(crate) fn trait_generalize(&self, this: TypeId, at_level: i32) -> TypeId {
        let src = self.src(this);
        let t = match self.trait_kind(this).unwrap() {
            TraitKind::Callable { args } => {
                let g = args.iter().map(|t| t.map(|t| self.generalize(t, at_level))).collect();
                self.new_callable_trait(g)
            }
            TraitKind::Type { ty } => {
                let g = self.generalize(ty, at_level);
                self.new_type_trait(g)
            }
        };
        self.set_src(t, src);
        t
    }

    /// `CallableTrait::instantiate` / `TypeTrait::instantiate`.
    pub(crate) fn trait_instantiate(
        &self,
        this: TypeId,
        at_level: i32,
        use_counter: bool,
        cache: &mut InstantiationCache,
        has_cache: bool,
    ) -> TypeId {
        let src = self.src(this);
        let t = match self.trait_kind(this).unwrap() {
            TraitKind::Callable { args } => {
                let g = args
                    .iter()
                    .map(|t| t.map(|t| self.instantiate_impl(t, at_level, use_counter, cache, has_cache)))
                    .collect();
                self.new_callable_trait(g)
            }
            TraitKind::Type { ty } => {
                let g = self.instantiate_impl(ty, at_level, use_counter, cache, has_cache);
                self.new_type_trait(g)
            }
        };
        self.set_src(t, src);
        t
    }

    /// `CallableTrait::debugString` / `TypeTrait::debugString`.
    pub(crate) fn trait_debug_string(&self, t: &TraitKind, mode: i32) -> String {
        match t {
            TraitKind::Callable { args } => {
                let s = args.first().copied().flatten().map(|a| self.debug_string(a, mode)).unwrap_or_default();
                let s = if s.starts_with("Tuple") { s[5..].to_string() } else { s };
                let r = args.get(1).copied().flatten().map(|a| self.debug_string(a, mode)).unwrap_or_default();
                format!("CallableTrait[{s},{r}]")
            }
            TraitKind::Type { ty } => {
                format!("Trait[{}]", self.class_name(*ty).map(|n| n.to_string()).unwrap_or_else(|| "-".into()))
            }
        }
    }
}
