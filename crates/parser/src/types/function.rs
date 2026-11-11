//! Port of codon/parser/ast/types/function.{h,cpp}: `FuncType`.

use typhon_common::Result;

use super::*;

impl TypeArena {
    fn func_data(&self, this: TypeId) -> FuncData {
        self.with_class(this, |c| c.func().cloned()).flatten().expect("not a function type")
    }

    /// `FuncType::unify`.
    pub(crate) fn func_unify(
        &self,
        this: TypeId,
        typ: TypeId,
        mut us: Option<&mut Unification>,
        host: &mut dyn TypeHost,
    ) -> Result<i32> {
        if this == typ {
            return Ok(0);
        }
        let mut s1 = 2;
        if let Some(t) = self.get_func(typ) {
            let me = self.func_data(this);
            let other = self.func_data(t);
            // Check if names and parents match.
            if self.func_ast(me.ast).name != self.func_ast(other.ast).name
                || (me.func_parent.is_some() ^ other.func_parent.is_some())
            {
                return Ok(-1);
            }
            let mut s = 0;
            if let (Some(p1), Some(p2)) = (me.func_parent, other.func_parent) {
                s = self.unify(p1, p2, us.as_deref_mut(), host)?;
                if s == -1 {
                    return Ok(-1);
                }
            }
            s1 += s;
            // Check if function generics match.
            typhon_common::seqassert!(
                me.func_generics.len() == other.func_generics.len(),
                "generic size mismatch for {}",
                self.func_ast(me.ast).name
            );
            for (a, b) in me.func_generics.iter().zip(other.func_generics.iter()) {
                let s = self.unify(a.ty(), b.ty(), us.as_deref_mut(), host)?;
                if s == -1 {
                    return Ok(-1);
                }
                s1 += s;
            }
        }
        let s = self.class_unify(this, typ, us, host)?;
        Ok(if s == -1 { s } else { s1 + s })
    }

    fn rebuild_func(&self, base: TypeId, ast: StmtId, fg: Vec<Generic>, parent: Option<TypeId>) -> TypeId {
        // `FuncType(ClassType &&base, ...)`: take the private base's storage.
        self.with_class_mut(base, |c| {
            c.kind = ClassKind::Func(Box::new(FuncData { ast, func_generics: fg, func_parent: parent }));
            c.immutable = false;
            c.rn = String::new();
        });
        base
    }

    /// `FuncType::generalize`.
    pub(crate) fn func_generalize(&self, this: TypeId, at_level: i32) -> TypeId {
        let f = self.func_data(this);
        let fg = f.func_generics.iter().map(|t| self.generic_generalize(t, at_level)).collect();
        let p = f.func_parent.map(|p| self.generalize(p, at_level));
        let r = self.class_generalize_base(this, at_level, false);
        self.rebuild_func(r, f.ast, fg, p)
    }

    /// `FuncType::instantiate`.
    pub(crate) fn func_instantiate(
        &self,
        this: TypeId,
        at_level: i32,
        use_counter: bool,
        cache: &mut InstantiationCache,
        has_cache: bool,
    ) -> TypeId {
        let f = self.func_data(this);
        let mut fg = Vec::with_capacity(f.func_generics.len());
        for t in &f.func_generics {
            let g = self.generic_instantiate(t, at_level, use_counter, cache, has_cache);
            if has_cache {
                if let Some(gt) = g.ty {
                    if let Some(c) = cache.get_mut(&t.id) {
                        *c = gt;
                    }
                }
            }
            fg.push(g);
        }
        let p = f.func_parent.map(|p| self.instantiate_impl(p, at_level, use_counter, cache, has_cache));
        let r = self.class_instantiate_base(this, at_level, use_counter, cache, has_cache, false);
        self.rebuild_func(r, f.ast, fg, p)
    }

    /// Argument types (`for (auto &a : *this)`): generics of `generics[0]`.
    pub fn func_args(&self, this: TypeId) -> Vec<Generic> {
        let g0 = self.generic_type(this, 0).expect("function without argument tuple");
        self.generics(g0)
    }
    /// `(*func)[i]`.
    pub fn func_arg(&self, this: TypeId, i: usize) -> TypeId {
        self.func_args(this)[i].ty()
    }
    /// `size()`.
    pub fn func_size(&self, this: TypeId) -> usize {
        self.func_args(this).len()
    }
    /// `getRetType()`.
    pub fn func_ret(&self, this: TypeId) -> TypeId {
        self.with_class(self.follow(this), |c| c.generics[1].ty()).expect("not a function")
    }
    /// `funcGenerics`.
    pub fn func_generics(&self, this: TypeId) -> Vec<Generic> {
        self.func_data(self.follow(this)).func_generics
    }
    /// `funcParent`.
    pub fn func_parent(&self, this: TypeId) -> Option<TypeId> {
        self.func_data(self.follow(this)).func_parent
    }
    /// `ast` handle.
    pub fn func_ast_id(&self, this: TypeId) -> StmtId {
        self.func_data(self.follow(this)).ast
    }
    /// `getFuncName()` (`ast->getName()`).
    pub fn func_name(&self, this: TypeId) -> String {
        self.func_ast(self.func_ast_id(this)).name.clone()
    }

    /// `FuncType::hasUnbounds`.
    pub(crate) fn func_has_unbounds(&self, this: TypeId, ig: bool) -> bool {
        let f = self.func_data(this);
        if f.func_generics.iter().any(|t| t.ty.is_some_and(|t| self.has_unbounds(t, ig))) {
            return true;
        }
        if f.func_parent.is_some_and(|p| self.has_unbounds(p, ig)) {
            return true;
        }
        if self.func_args(this).iter().any(|a| self.has_unbounds(a.ty(), ig)) {
            return true;
        }
        self.has_unbounds(self.func_ret(this), ig)
    }

    /// `FuncType::getUnbounds`.
    pub(crate) fn func_get_unbounds(&self, this: TypeId, ig: bool) -> Vec<TypeId> {
        let f = self.func_data(this);
        let mut u: Vec<TypeId> = vec![];
        let prepend = |mut tu: Vec<TypeId>, u: &mut Vec<TypeId>| {
            tu.append(u);
            *u = tu;
        };
        for t in &f.func_generics {
            if let Some(t) = t.ty {
                prepend(self.get_unbounds(t, ig), &mut u);
            }
        }
        if let Some(p) = f.func_parent {
            prepend(self.get_unbounds(p, ig), &mut u);
        }
        // Important: return type unbounds are not important, so skip them.
        for a in self.func_args(this) {
            prepend(self.get_unbounds(a.ty(), ig), &mut u);
        }
        u
    }

    /// `FuncType::canRealize`.
    pub(crate) fn func_can_realize(&self, this: TypeId) -> bool {
        let f = self.func_data(this);
        let allow = self.func_ast(f.ast).allow_pass_through;
        let can = |t: TypeId| -> bool {
            if self.can_realize(t) {
                return true;
            }
            if !allow {
                return false;
            }
            for u in self.get_unbounds(t, true) {
                let l = self.link(u).unwrap();
                if l.kind == LinkKind::Generic || !l.pass_through {
                    return false;
                }
            }
            true
        };
        // Important: return type does not have to be realized.
        for a in self.func_args(this) {
            if self.get_func(a.ty()).is_none() && !can(a.ty()) {
                return false;
            }
        }
        let generics = f.func_generics.iter().all(|a| {
            let Some(t) = a.ty else { return true };
            if self.can_realize(t) {
                return true;
            }
            // Self must not defer __init__
            allow && !a.name.is_empty() && self.unmangled(&a.name) == "Self" && can(t)
        });
        generics && f.func_parent.is_none_or(can)
    }

    /// `FuncType::isInstantiated`.
    /// PORT-NOTE: C++ temporarily nulls `retType->funcParent` if it points back to `this`;
    /// here the back edge is skipped through `skip_parent_of`.
    pub(crate) fn func_is_instantiated(&self, this: TypeId, skip_parent_of: Option<TypeId>) -> bool {
        let f = self.func_data(this);
        let parent = if skip_parent_of.is_some_and(|s| f.func_parent == Some(s)) { None } else { f.func_parent };
        let ret = self.func_ret(this);
        let skip_for_ret = Some(this);
        let gens = f.func_generics.iter().all(|a| a.ty.is_none_or(|t| self.is_instantiated_impl(t, skip_parent_of)));
        let par = parent.is_none_or(|p| self.is_instantiated_impl(p, skip_parent_of));
        // ClassType::isInstantiated over generics[0] (args) and generics[1] (ret).
        let c = self.class(this).unwrap();
        let cls = c.generics.iter().chain(c.hidden_generics.iter()).all(|g| match g.ty {
            None => true,
            Some(t) if t == ret || self.follow(t) == self.follow(ret) => self.is_instantiated_impl(t, skip_for_ret),
            Some(t) => self.is_instantiated_impl(t, skip_parent_of),
        });
        gens && par && cls
    }

    /// `FuncType::debugString`.
    pub(crate) fn func_debug_string(&self, this: TypeId, mode: i32) -> String {
        let f = self.func_data(this);
        let ast = self.func_ast(f.ast);
        let mut gs: Vec<String> = vec![];
        for a in &f.func_generics {
            if !a.name.is_empty() {
                gs.push(if mode < 2 {
                    self.debug_string(a.ty(), mode)
                } else {
                    format!("{}={}", self.rev(&a.name), self.debug_string(a.ty(), mode))
                });
            }
        }
        let mut s = join(&gs, ",");
        let mut as_: Vec<String> = vec![];
        // Important: return type does not have to be realized.
        if mode == 2 {
            as_.push(format!("RET={}", self.debug_string(self.func_ret(this), mode)));
        }
        let args = self.func_args(this);
        if mode < 2 || ast.params.is_empty() {
            for a in &args {
                as_.push(self.generic_debug_string(a, mode));
            }
        } else {
            let mut si = 0;
            for (name, status) in &ast.params {
                if *status == ParamStatus::Generic {
                    continue;
                }
                if let Some(a) = args.get(si) {
                    as_.push(format!("{}={}", name, self.debug_string(a.ty(), mode)));
                }
                si += 1;
            }
        }
        let a = join(&as_, ",");
        s = if s.is_empty() { a } else { format!("{s};{a}") };
        let fnname = if mode == 0 { self.rev(&ast.name) } else { ast.name.clone() };
        if mode == 2 {
            if let Some(p) = f.func_parent {
                s += &format!(";{}", self.debug_string(p, mode));
            }
        }
        format!("{}{}", fnname, if s.is_empty() { String::new() } else { format!("[{s}]") })
    }

    /// `FuncType::realizedName`.
    pub(crate) fn func_realized_name(&self, this: TypeId) -> String {
        let f = self.func_data(this);
        let mut gs: Vec<String> = vec![];
        for a in &f.func_generics {
            if !a.name.is_empty() {
                gs.push(self.generic_realized_name(a));
            }
        }
        let s = join(&gs, ",");
        let mut as_: Vec<String> = vec![];
        // Important: return type does not have to be realized.
        for a in self.func_args(this) {
            as_.push(match self.get_func(a.ty()) {
                Some(af) => self.func_realized_name(af),
                None => self.generic_realized_name(&a),
            });
        }
        let a = join(&as_, ",");
        let s = if s.is_empty() { a } else { format!("{a},{s}") };
        format!(
            "{}{}{}",
            f.func_parent.map(|p| format!("{}:", self.realized_name(p))).unwrap_or_default(),
            self.func_ast(f.ast).name,
            if s.is_empty() { String::new() } else { format!("[{s}]") }
        )
    }

    /// `FuncType::replaceArgTypes`: replace non-null argument types, preserving shared
    /// tuples (a fresh argument tuple is allocated) and clearing the name cache.
    pub fn replace_arg_types(&self, this: TypeId, replacements: &[Option<TypeId>]) {
        typhon_common::seqassert!(replacements.len() == self.func_size(this), "function argument count mismatch");
        let g0 = self.generic_type(this, 0).unwrap();
        let args = self.copy_class(g0);
        self.with_class_mut(args, |c| {
            for (i, r) in replacements.iter().enumerate() {
                if let Some(r) = r {
                    c.generics[i].ty = Some(*r);
                }
            }
        });
        self.with_class_mut(this, |c| {
            c.generics[0].ty = Some(args);
            c.rn.clear();
        });
    }
}
