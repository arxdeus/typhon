//! Port of `DocVisitor::visit(FunctionStmt|ClassStmt|ImportStmt|AssignStmt)` (doc.cpp).

use serde_json::{Map, Value};
use typhon_common::Result;

use super::{DocContextRef, DocVisitor, custom, is_empty_json, is_valid_name, jsonify, obj, parse_file, strs};
use crate::ast::{Attr, ExprKind, Param, ParamStatus, is_id};
use crate::ids::StmtId;

impl DocVisitor<'_> {
    fn jsonify_params(&mut self, params: &[Param]) -> Result<Vec<Value>> {
        let mut args = vec![];
        for a in params {
            let mut jj = Map::new();
            jj.insert("name".into(), Value::String(a.name.clone()));
            if let Some(t) = a.ty {
                let tt = self.transform_expr(Some(t))?;
                if is_empty_json(&tt) {
                    let src = self.shared.cache.ast[t].src.clone();
                    eprintln!("{}: warning: cannot resolve argument {}", src, self.fmt(t));
                    jj.insert("type".into(), Value::String(self.fmt(t)));
                } else {
                    jj.insert("type".into(), tt);
                }
            }
            if let Some(d) = a.default_value {
                jj.insert("default".into(), Value::String(self.fmt(d)));
            }
            args.push(Value::Object(jj));
        }
        Ok(args)
    }

    pub(super) fn visit_function(&mut self, s: StmtId) -> Result<String> {
        let id = self.shared.next_id();
        let f = self.shared.cache.ast[s].kind.as_function().unwrap().clone();
        self.ctx.borrow_mut().add(&f.name, id);
        let mut j = obj(&[("kind", "function"), ("name", &f.name)]);
        j.insert("pos".into(), jsonify(&self.shared.cache.ast[s].src));

        let mut generics = vec![];
        for (i, a) in f.items.iter().enumerate() {
            if !a.is_value() {
                self.ctx.borrow_mut().add(&a.name, 0);
                generics.push(a.name.clone());
                self.shared.cache.ast[s].kind.as_function_mut().unwrap().items[i].status = ParamStatus::Generic;
            }
        }
        let args = self.jsonify_params(&f.items)?;
        j.insert("generics".into(), strs(&generics));
        let mut is_llvm = false;
        let mut attrs: Vec<String> = vec![];
        for &d in &f.decorators {
            if let ExprKind::Id(e) = &self.shared.cache.ast[d].kind {
                attrs.push(e.value.clone());
                is_llvm |= e.value == "llvm";
            }
        }
        let a = &self.shared.cache.ast[s].attrs;
        for (at, n) in [
            (Attr::Property, "property"),
            (Attr::Attribute, "__attribute__"),
            (Attr::Python, "python"),
            (Attr::LLVM, "llvm"),
            (Attr::Internal, "__internal__"),
            (Attr::HiddenFromUser, "__hidden__"),
            (Attr::Atomic, "atomic"),
            (Attr::StaticMethod, "staticmethod"),
            (Attr::C, "C"),
        ] {
            if a.has(at) {
                attrs.push(n.into());
            }
        }
        if !attrs.is_empty() {
            j.insert("attrs".into(), strs(&attrs));
        }
        if f.ret.is_some() {
            j.insert("return".into(), self.transform_expr(f.ret)?);
        }
        j.insert("args".into(), Value::Array(args));
        let mut docstr = String::new();
        Self::flatten(&self.shared.cache, f.suite, Some(&mut docstr), true);
        for g in &generics {
            self.ctx.borrow_mut().remove(g);
        }
        if !docstr.is_empty() && !is_llvm {
            j.insert("doc".into(), Value::String(docstr));
        }
        self.shared.j.insert(id.to_string(), Value::Object(j));
        Ok(id.to_string())
    }

    pub(super) fn visit_class(&mut self, s: StmtId) -> Result<String> {
        let c = self.shared.cache.ast[s].kind.as_class().unwrap().clone();
        let is_record = self.shared.cache.ast.class_is_record(s);
        let mut j = obj(&[("name", &c.name), ("kind", "class"), ("type", if is_record { "type" } else { "class" })]);
        let id = self.shared.next_id();
        let is_extend = c.decorators.iter().any(|d| is_id(&self.shared.cache.ast, Some(*d), "extend"));
        let mut generics: Vec<String> = vec![];
        if is_extend {
            j.insert("type".into(), Value::String("extension".into()));
            let src = self.shared.cache.ast[s].src.clone();
            let i = self
                .shared
                .find(&self.ctx, &c.name)
                .ok_or_else(|| custom(&src, format!("unknown identifier {}", c.name)))?;
            j.insert("parent".into(), Value::String(i.to_string()));
            generics = self.shared.generics.get(&i).cloned().unwrap_or_default();
        } else {
            self.ctx.borrow_mut().add(&c.name, id);
        }
        for a in &c.items {
            if !a.is_value() {
                generics.push(a.name.clone());
            }
        }
        self.shared.generics.insert(id, generics.clone());
        for g in &generics {
            self.ctx.borrow_mut().add(g, 0);
        }
        let args = self.jsonify_params(&c.items)?;
        j.insert("generics".into(), strs(&generics));
        j.insert("args".into(), Value::Array(args));
        j.insert("pos".into(), jsonify(&self.shared.cache.ast[s].src));

        let mut docstr = String::new();
        let mut members = vec![];
        for f in Self::flatten(&self.shared.cache, c.suite, Some(&mut docstr), true) {
            let name = self.shared.cache.ast[f].kind.as_function().map(|x| x.name.clone());
            if let Some(name) = name {
                let i = self.transform_stmt(Some(f))?;
                if !i.is_empty() {
                    members.push(i);
                }
                if is_valid_name(&name) {
                    self.ctx.borrow_mut().remove(&name);
                }
            }
        }
        for g in &generics {
            self.ctx.borrow_mut().remove(g);
        }
        j.insert("members".into(), strs(&members));
        if !docstr.is_empty() {
            j.insert("doc".into(), Value::String(docstr));
        }
        self.shared.j.insert(id.to_string(), Value::Object(j));
        Ok(id.to_string())
    }

    pub(super) fn visit_import(&mut self, s: StmtId) -> Result<String> {
        let st = self.shared.cache.ast[s].kind.as_import().unwrap().clone();
        let src = self.shared.cache.ast[s].src.clone();
        if is_id(&self.shared.cache.ast, st.from, "C") || is_id(&self.shared.cache.ast, st.from, "python") {
            let id = self.shared.next_id();
            let (mut name, mut lib) = (String::new(), String::new());
            match st.what.map(|w| self.shared.cache.ast[w].kind.clone()) {
                Some(ExprKind::Id(i)) => name = i.value,
                Some(ExprKind::Dot(d)) => {
                    name = d.member;
                    lib = d.expr.map(|e| self.fmt(e)).unwrap_or_default();
                }
                _ => typhon_common::seqassert!(false, "invalid C import statement"),
            }
            self.ctx.borrow_mut().add(&name, id);
            let name = if st.as_.is_empty() { name } else { st.as_.clone() };
            let mut j = obj(&[("name", &name), ("kind", "function")]);
            if let Some(ExprKind::Id(i)) = st.from.map(|f| &self.shared.cache.ast[f].kind) {
                j.insert("extern".into(), Value::String(i.value.clone()));
            }
            j.insert("pos".into(), jsonify(&src));
            if st.ret.is_some() {
                j.insert("return".into(), self.transform_expr(st.ret)?);
            }
            let mut args = vec![];
            for a in &st.args {
                let mut ja = Map::new();
                ja.insert("name".into(), Value::String(a.name.clone()));
                ja.insert("type".into(), self.transform_expr(a.ty)?);
                args.push(Value::Object(ja));
            }
            j.insert("dylib".into(), Value::String(lib));
            j.insert("args".into(), Value::Array(args));
            self.shared.j.insert(id.to_string(), Value::Object(j));
            return Ok(id.to_string());
        }

        let ast = &self.shared.cache.ast;
        let mut dirs: Vec<String> = vec![]; // Path components
        let mut e = st.from;
        let what_bad = st.what.is_some_and(|w| ast[w].kind.as_id().is_none());
        let invalid = |src: &typhon_common::srcinfo::SrcInfo| custom(src, "invalid import statement".into());
        while let Some(d) = e.and_then(|x| ast[x].kind.as_dot()) {
            while let Some(d2) = e.and_then(|x| ast[x].kind.as_dot()) {
                dirs.push(d2.member.clone());
                e = d2.expr;
            }
            let _ = d;
            if !e.is_some_and(|x| ast[x].kind.as_id().is_some()) || !st.args.is_empty() || st.ret.is_some() || what_bad {
                return Err(invalid(&src));
            }
        }
        if let Some(ee) = e.and_then(|x| ast[x].kind.as_id()) {
            if !st.args.is_empty() || st.ret.is_some() || what_bad {
                return Err(invalid(&src));
            }
            // We have an empty stmt->from in "from .. import".
            if !ee.value.is_empty() {
                dirs.push(ee.value.clone());
            }
        }
        // Handle dots (e.g. .. in from ..m import x).
        for _ in 1..st.dots.max(1) {
            dirs.push("..".into());
        }
        let path: String = dirs.iter().rev().cloned().collect::<Vec<_>>().join("/");
        // Fetch the import!
        let filename = self.ctx.borrow().filename().to_string();
        let Some(file) = crate::typecheck::helpers::get_import_file(&mut self.shared.cache, &path, &filename, false) else {
            return Err(custom(&src, format!("cannot locate import '{path}'")));
        };

        let ictx: DocContextRef = match self.shared.modules.get(&file.path) {
            Some(c) => c.clone(),
            None => {
                let ictx: DocContextRef = std::rc::Rc::new(std::cell::RefCell::new(crate::ctx::Context::new(file.path.clone())));
                self.shared.modules.insert(file.path.clone(), ictx.clone());
                let tmp = parse_file(&mut self.shared.cache, &file.path)?;
                DocVisitor::new(self.shared, ictx.clone()).transform_module(tmp)?;
                ictx
            }
        };

        if st.what.is_none() {
            // TODO: implement this corner case
            for i in &dirs {
                if self.shared.find(&self.ctx, i).is_none() {
                    let n = self.shared.next_id();
                    self.ctx.borrow_mut().add(i, n);
                }
            }
        } else if is_id(&self.shared.cache.ast, st.what, "*") {
            let items: Vec<(String, i32)> =
                ictx.borrow().map.iter().filter_map(|(k, v)| v.first().map(|x| (k.clone(), *x))).collect();
            for (k, v) in items {
                self.ctx.borrow_mut().add(&k, v);
            }
        } else {
            let name = self.shared.cache.ast[st.what.unwrap()].kind.as_id().unwrap().value.clone();
            match self.shared.find(&ictx, &name) {
                Some(c) => {
                    let n = if st.as_.is_empty() { name } else { st.as_.clone() };
                    self.ctx.borrow_mut().add(&n, c);
                }
                None => return Err(custom(&src, format!("symbol '{}' not found in {}", name, file.path))),
            }
        }
        Ok(String::new())
    }

    pub(super) fn visit_assign(&mut self, s: StmtId) -> Result<String> {
        let st = self.shared.cache.ast[s].kind.as_assign().unwrap().clone();
        let Some(e) = st.lhs.and_then(|l| self.shared.cache.ast[l].kind.as_id().cloned()) else {
            return Ok(String::new());
        };
        if let Some(ei) = st.rhs.and_then(|r| self.shared.cache.ast[r].kind.as_id().cloned())
            && let Some(i) = self.shared.find(&self.ctx, &ei.value)
        {
            let k = self.shared.j.get(&i.to_string()).and_then(|v| v.get("kind")).and_then(|v| v.as_str());
            if k == Some("class") {
                self.ctx.borrow_mut().add(&e.value, i);
                return Ok(String::new());
            }
        }
        let id = self.shared.next_id();
        self.ctx.borrow_mut().add(&e.value, id);
        let mut j = obj(&[("name", &e.value), ("kind", "variable")]);
        if st.type_expr.is_some() {
            j.insert("type".into(), self.transform_expr(st.type_expr)?);
        }
        if let Some(r) = st.rhs {
            j.insert("value".into(), Value::String(self.fmt(r)));
        }
        j.insert("pos".into(), jsonify(&self.shared.cache.ast[s].src));
        self.shared.j.insert(id.to_string(), Value::Object(j));
        Ok(id.to_string())
    }
}
