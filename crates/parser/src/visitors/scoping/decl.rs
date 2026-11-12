
use std::collections::BTreeMap;

use typhon_common::ErrorKind;

use super::{Context, Node, ScopeBlock, ScopingVisitor};
use crate::ast::{Attr, AttrValue, Binding, BindingsAttr, CaptureType, ExprKind, IdExpr, ParamKind, is_id};
use crate::ids::StmtId;

macro_rules! stop {
    ($self:ident, $kind:expr, $src:expr $(, $arg:expr)*) => {{
        let s = $src;
        $self.add_error($kind, &s, &[$(&$arg),*]);
        return;
    }};
}
macro_rules! check {
    ($e:expr) => {
        if !$e {
            return;
        }
    };
}

impl ScopingVisitor<'_> {
    fn esrc(&self, e: Option<crate::ids::ExprId>) -> typhon_common::srcinfo::SrcInfo {
        e.map(|e| self.cache.ast[e].src.clone()).unwrap_or_default()
    }

    pub(crate) fn scope_import(&mut self, id: StmtId) {
        let st = self.cache.ast[id].kind.as_import().unwrap().clone();
        let ssrc = self.cache.ast[id].src.clone();
        // Validate
        if let Some(from) = st.from {
            let mut e = from;
            while let Some(d) = self.cache.ast[e].kind.as_dot() {
                match d.expr {
                    Some(x) => e = x,
                    None => break,
                }
            }
            if !is_id(&self.cache.ast, st.from, "C") && !is_id(&self.cache.ast, st.from, "python") {
                if self.cache.ast[e].kind.as_id().is_none() {
                    stop!(self, ErrorKind::IMPORT_IDENTIFIER, self.esrc(Some(e)));
                }
                if let Some(a) = st.args.first() {
                    stop!(self, ErrorKind::IMPORT_FN, a.src.clone());
                }
                if st.ret.is_some() {
                    stop!(self, ErrorKind::IMPORT_FN, self.esrc(st.ret));
                }
                if st.what.is_some_and(|w| self.cache.ast[w].kind.as_id().is_none()) {
                    stop!(self, ErrorKind::IMPORT_IDENTIFIER, self.esrc(st.what));
                }
            }
            if st.is_cvar()
                && let Some(a) = st.args.first()
            {
                stop!(self, ErrorKind::IMPORT_FN, a.src.clone());
            }
        }
        if self.ctx.function_scope.is_some() && st.what.is_some() && is_id(&self.cache.ast, st.what, "*") {
            stop!(self, ErrorKind::IMPORT_STAR, ssrc.clone());
        }

        // dylib C imports
        if st.from.is_some()
            && is_id(&self.cache.ast, st.from, "C")
            && let Some(d) = st.what.and_then(|w| self.cache.ast[w].kind.as_dot().cloned())
        {
            check!(self.transform_e(d.expr));
        }

        if st.as_.is_empty() {
            let target = if st.what.is_some() {
                if is_id(&self.cache.ast, st.what, "*") { None } else { st.what }
            } else {
                st.from
            };
            if target.is_some() {
                let o = self.set_root(Some(Node::Stmt(id)));
                let ok = self.transform_e(target);
                self.ctx.root = o;
                check!(ok);
            }
        } else {
            self.visit_name(&st.as_, true, Some(Node::Stmt(id)), &ssrc);
        }
        for a in &st.args {
            check!(self.transform_e(a.ty));
            check!(self.transform_e(a.default_value));
        }
        check!(self.transform_e(st.ret));
    }

    pub(crate) fn scope_function(&mut self, id: StmtId) {
        let ssrc = self.cache.ast[id].src.clone();
        // Validate
        let decorators = self.cache.ast[id].kind.as_function().unwrap().decorators.clone();
        let mut new_decorators = vec![];
        for (di, &d) in decorators.iter().enumerate() {
            let a = &self.cache.ast;
            let attr = if is_id(a, Some(d), "__attribute__") {
                Some(Attr::Attribute)
            } else if is_id(a, Some(d), "llvm") {
                Some(Attr::LLVM)
            } else if is_id(a, Some(d), "python") {
                if decorators.len() != 1 {
                    let other = decorators[if di == 1 { 0 } else { 1 }];
                    let _ = other;
                    stop!(self, ErrorKind::FN_SINGLE_DECORATOR, self.esrc(Some(decorators[1])), "python");
                }
                Some(Attr::Python)
            } else if is_id(a, Some(d), "__internal__") {
                Some(Attr::Internal)
            } else if is_id(a, Some(d), "__hidden__") {
                Some(Attr::HiddenFromUser)
            } else if is_id(a, Some(d), "atomic") {
                Some(Attr::Atomic)
            } else if is_id(a, Some(d), "property") {
                Some(Attr::Property)
            } else if is_id(a, Some(d), "staticmethod") {
                Some(Attr::StaticMethod)
            } else if is_id(a, Some(d), "__force__") {
                Some(Attr::ForceRealize)
            } else if is_id(a, Some(d), "C") {
                Some(Attr::C)
            } else {
                None
            };
            match attr {
                Some(at) => self.cache.ast[id].attrs.set(at),
                None => new_decorators.push(d),
            }
        }
        let has = |s: &Self, a: Attr| s.cache.ast[id].attrs.has(a);
        let params = self.cache.ast[id].kind.as_function().unwrap().items.clone();
        if has(self, Attr::C) {
            for a in &params {
                let b = a.name.as_bytes();
                if b.len() > 1 && b[0] == b'*' && b[1] != b'*' {
                    self.cache.ast[id].attrs.set(Attr::CVarArg);
                }
            }
        }
        if let Some(f) = params.first()
            && f.ty.is_none()
            && f.name == "self"
        {
            self.cache.ast[id].attrs.set(Attr::HasSelf);
        }
        let (name, ret, suite) = {
            let f = self.cache.ast[id].kind.as_function_mut().unwrap();
            f.decorators = new_decorators.clone();
            (f.name.clone(), f.ret, f.suite)
        };
        if ret.is_none() && (has(self, Attr::LLVM) || has(self, Attr::C)) {
            stop!(self, ErrorKind::FN_LLVM, self.src.clone());
        }
        // Set attributes
        let mut seen_args = std::collections::HashSet::new();
        let (mut defaults_started, mut has_star_arg, mut has_kw_arg) = (false, false, false);
        for (ia, a) in params.iter().enumerate() {
            let (stars, n) = a.name_with_stars();
            if stars == 2 {
                if has_kw_arg {
                    stop!(self, ErrorKind::FN_MULTIPLE_ARGS, a.src.clone());
                }
                if a.default_value.is_some() {
                    stop!(self, ErrorKind::FN_DEFAULT_STARARG, self.esrc(a.default_value));
                }
                if ia != params.len() - 1 {
                    stop!(self, ErrorKind::FN_LAST_KWARG, a.src.clone());
                }
                has_kw_arg = true;
            } else if stars == 1 {
                if has_star_arg {
                    stop!(self, ErrorKind::FN_MULTIPLE_ARGS, a.src.clone());
                }
                if a.default_value.is_some() {
                    stop!(self, ErrorKind::FN_DEFAULT_STARARG, self.esrc(a.default_value));
                }
                has_star_arg = true;
            }
            if seen_args.contains(&n) {
                stop!(self, ErrorKind::FN_ARG_TWICE, a.src.clone(), n);
            }
            seen_args.insert(n.clone());
            if a.default_value.is_none()
                && defaults_started
                && stars == 0
                && a.is_value()
                && a.kind != ParamKind::KeywordOnly
            {
                stop!(self, ErrorKind::FN_DEFAULT, a.src.clone(), n);
            }
            defaults_started |= a.default_value.is_some();
            if has(self, Attr::C) {
                if a.default_value.is_some() {
                    stop!(self, ErrorKind::FN_C_DEFAULT, self.esrc(a.default_value), n);
                }
                if stars != 1 && a.ty.is_none() {
                    stop!(self, ErrorKind::FN_C_TYPE, a.src.clone(), n);
                }
            }
        }

        if self.ctx.in_class && !has(self, Attr::StaticMethod) && params.first().is_some_and(|p| p.name == "self") {
            // Append default generic Self
            let src = self.src.clone();
            if params[0].ty.is_none() {
                let t = self.cache.ast.new_expr(IdExpr::new("Self"), src.clone());
                self.cache.ast[id].kind.as_function_mut().unwrap().items[0].ty = Some(t);
            }
            let f = self.cache.ast[id].kind.as_function().unwrap();
            if is_id(&self.cache.ast, f.items[0].ty, "Self") && !f.items.iter().any(|p| p.name == "Self") {
                let ty = self.cache.ast.new_expr(IdExpr::new(typhon_common::mangle::stdlib().type_.clone()), src.clone());
                let p = self.cache.ast.new_param(src, "Self", Some(ty), None, 0, ParamKind::Normal);
                self.cache.ast[id].kind.as_function_mut().unwrap().add_param(p);
            }
        }
        let params = self.cache.ast[id].kind.as_function().unwrap().items.clone();

        let is_overload = new_decorators.iter().any(|d| is_id(&self.cache.ast, Some(*d), "overload"));
        if !is_overload {
            self.visit_name(&name, true, Some(Node::Stmt(id)), &ssrc);
        }

        let mut c = Context { function_scope: Some(id), renames: self.ctx.renames.clone(), ..Default::default() };
        if self.ctx.in_class && !params.is_empty() {
            c.class_deduce = (params[0].name.clone(), Default::default());
        }
        let mut errs = Default::default();
        c.scope.push(ScopeBlock { id: 0, suite: None, seen_vars: None });
        self.in_ctx(&mut c, &mut errs, |v| v.visit_name(&name, true, Some(Node::Stmt(id)), &ssrc));
        for a in &params {
            let (_, n) = a.name_with_stars();
            self.in_ctx(&mut c, &mut errs, |v| v.visit_name(&n, true, Some(Node::Stmt(id)), &a.src));
            if a.default_value.is_some() {
                check!(self.transform_e(a.default_value));
            }
        }
        c.scope.pop();

        c.scope.push(ScopeBlock { id: 0, suite, seen_vars: None });
        self.in_ctx(&mut c, &mut errs, |v| {
            v.transform_s(suite);
            v.process_child_captures();
        });
        c.scope.pop();
        self.errors.append(&errs);
        if !self.can_continue() {
            return;
        }

        let mut b = BindingsAttr::default();
        b.captures = c.captures.iter().map(|(k, v)| (k.clone(), *v)).collect();
        for (k, v) in &c.captures {
            self.ctx.child_captures.entry(k.clone()).or_insert(*v);
        }
        // Recursive capture if used
        let self_count = c.map.entry(name.clone()).or_default().len();
        if self_count == 1 && c.first_seen.contains_key(&name) {
            b.captures.insert(name.clone(), CaptureType::Read);
        }
        for (u, v) in &c.map {
            let mut bind = Binding { name: u.clone(), count: v.len(), is_nonlocal: false };
            let cp = c.child_captures.get(u).or_else(|| c.captures.get(u));
            if cp == Some(&CaptureType::Nonlocal) {
                bind.is_nonlocal = true;
                self.ctx.child_captures.insert(u.clone(), CaptureType::Nonlocal);
            }
            b.bindings.insert(u.clone(), bind);
        }
        self.cache.ast[id].attrs.set_value(Attr::Bindings, AttrValue::Bindings(b));

        if !c.class_deduce.1.is_empty() {
            // Note: the reference iterates an unordered_set; insertion order is used here.
            let v: Vec<String> = c.class_deduce.1.iter().cloned().collect();
            self.cache.ast[id].attrs.set_value(Attr::ClassDeduce, AttrValue::StrList(v));
        }
    }

    pub(crate) fn scope_class(&mut self, id: StmtId) {
        let cls = self.cache.ast[id].kind.as_class().unwrap().clone();
        let ssrc = self.cache.ast[id].src.clone();
        // @tuple(init=, repr=, eq=, order=, hash=, pickle=, container=, python=, add=,
        // internal=...)
        // @dataclass(...)
        // @extend
        let mut magics: BTreeMap<&str, bool> = [
            ("new", true), ("repr", false), ("hash", false), ("eq", false), ("ne", false),
            ("lt", false), ("le", false), ("gt", false), ("ge", false), ("pickle", true),
            ("unpickle", true), ("to_py", false), ("from_py", false), ("iter", false),
            ("getitem", false), ("len", false), ("to_gpu", false), ("from_gpu", false),
            ("from_gpu_new", false), ("tuplesize", true),
        ]
        .into_iter()
        .collect();
        let has = |s: &Self, a: Attr| s.cache.ast[id].attrs.has(a);
        let set = |s: &mut Self, a: Attr| s.cache.ast[id].attrs.set(a);

        for &d in &cls.decorators {
            let a = &self.cache.ast;
            if is_id(a, Some(d), "__notuple__") {
                set(self, Attr::ClassNoTuple);
            } else if is_id(a, Some(d), "__noextend__") {
                set(self, Attr::NoExtend);
            } else if is_id(a, Some(d), "dataclass") {
                set(self, Attr::Dataclass);
            } else if let ExprKind::Call(c) = &a[d].kind {
                let c = c.clone();
                if is_id(a, c.expr, "tuple") {
                    set(self, Attr::Tuple);
                    magics.values_mut().for_each(|v| *v = true);
                } else if !is_id(a, c.expr, "dataclass") {
                    stop!(self, ErrorKind::CLASS_BAD_DECORATOR, self.esrc(c.expr));
                } else if has(self, Attr::Tuple) {
                    stop!(self, ErrorKind::CLASS_CONFLICT_DECORATOR, self.esrc(Some(d)), "dataclass", "tuple");
                }
                for arg in &c.items {
                    let Some(b) = arg.value.and_then(|v| self.cache.ast[v].kind.as_bool()).map(|b| b.value) else {
                        stop!(self, ErrorKind::CLASS_NONSTATIC_DECORATOR, arg.src.clone());
                    };
                    let keys: &[&str] = match arg.name.as_str() {
                        "init" => &["new"],
                        "repr" => &["repr"],
                        "eq" => &["eq", "ne"],
                        "order" => &["lt", "le", "gt", "ge"],
                        "hash" => &["hash"],
                        "pickle" => &["pickle", "unpickle"],
                        "python" => &["to_py", "from_py"],
                        "gpu" => &["to_gpu", "from_gpu", "from_gpu_new"],
                        "container" => &["iter", "getitem"],
                        _ => stop!(self, ErrorKind::CLASS_BAD_DECORATOR_ARG, arg.src.clone()),
                    };
                    for k in keys {
                        magics.insert(k, b);
                    }
                }
            } else if is_id(a, Some(d), "tuple") {
                if has(self, Attr::Tuple) {
                    stop!(self, ErrorKind::CLASS_MULTIPLE_DECORATORS, self.esrc(Some(d)), "tuple");
                }
                set(self, Attr::Tuple);
                magics.values_mut().for_each(|v| *v = true);
            } else if is_id(a, Some(d), "extend") {
                set(self, Attr::Extend);
                if cls.decorators.len() != 1 {
                    let i = usize::from(cls.decorators[0] == d);
                    stop!(self, ErrorKind::CLASS_SINGLE_DECORATOR, self.esrc(Some(cls.decorators[i])), "extend");
                }
            } else if is_id(a, Some(d), "__internal__") {
                set(self, Attr::Internal);
            } else {
                stop!(self, ErrorKind::CLASS_BAD_DECORATOR, self.esrc(Some(d)));
            }
        }
        if !has(self, Attr::Tuple) && !has(self, Attr::Internal) && !has(self, Attr::Dataclass) && cls.items.is_empty() {
            set(self, Attr::ClassDeduce);
        }
        if !has(self, Attr::Tuple) {
            let n = magics["new"];
            magics.insert("init", n);
            magics.insert("new", true);
            magics.insert("raw", true);
            magics.insert("len", false);
            magics.insert("repr_default", true);
        }
        magics.insert("dict", true);
        // Internal classes do not get any auto-generated members.
        let mut ms: Vec<String> = vec![];
        if !has(self, Attr::Internal) {
            for (k, v) in &magics {
                if *v {
                    if *k == "new" {
                        ms.insert(0, k.to_string());
                    } else {
                        ms.push(k.to_string());
                    }
                }
            }
        }
        self.cache.ast[id].attrs.set_value(Attr::ClassMagic, AttrValue::StrList(ms));
        if has(self, Attr::Extend) && !cls.items.is_empty() {
            stop!(self, ErrorKind::CLASS_EXTENSION, cls.items[0].src.clone());
        }
        if has(self, Attr::Extend) && !cls.base_classes.is_empty() {
            stop!(self, ErrorKind::CLASS_EXTENSION, self.esrc(Some(cls.base_classes[0])));
        }
        let mut seen = std::collections::HashSet::new();
        for a in &cls.items {
            if a.ty.is_none() && a.default_value.is_none() {
                stop!(self, ErrorKind::CLASS_MISSING_TYPE, a.src.clone(), a.name);
            }
            if seen.contains(&a.name) {
                stop!(self, ErrorKind::CLASS_ARG_TWICE, a.src.clone(), a.name);
            }
            seen.insert(a.name.clone());
        }

        if has(self, Attr::Extend) {
            self.visit_name(&cls.name, false, None, &typhon_common::srcinfo::SrcInfo::default());
        } else {
            self.visit_name(&cls.name, true, Some(Node::Stmt(id)), &ssrc);
        }

        let mut c = Context { renames: self.ctx.renames.clone(), in_class: true, ..Default::default() };
        c.scope.push(ScopeBlock { id: 0, suite: None, seen_vars: None });
        let mut errs = Default::default();
        self.in_ctx(&mut c, &mut errs, |v| {
            for a in &cls.items {
                v.transform_e(a.ty);
                v.transform_e(a.default_value);
            }
            v.transform_s(cls.suite);
        });
        c.scope.pop();
        self.errors.append(&errs);
        if !self.can_continue() {
            return;
        }
        for &d in &cls.base_classes {
            check!(self.transform_e(Some(d)));
        }
    }
}
