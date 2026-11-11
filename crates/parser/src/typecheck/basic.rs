//! Port of codon/parser/visitors/typecheck/basic.cpp: literal expressions.

use typhon_common::{ErrorKind, Result};
use typhon_common::mangle::{self, stdlib};

use super::TypecheckVisitor;
use crate::ast::{CallExpr, DotExpr, FloatExpr, IdExpr, IndexExpr, IntExpr, StringExpr, CallArg};
use crate::cache::MAX_INT_WIDTH;
use crate::ids::ExprId;

impl<'a> TypecheckVisitor<'a> {
    fn call_expr(&mut self, f: ExprId, args: Vec<ExprId>) -> ExprId {
        let items = args.into_iter().map(|a| CallArg::new("", Some(a))).collect();
        self.n_expr(CallExpr { expr: Some(f), items, ordered: false, partial: false })
    }
    fn dot(&mut self, e: ExprId, m: &str) -> ExprId {
        self.n_expr(DotExpr { expr: Some(e), member: m.into() })
    }
    fn id(&mut self, s: &str) -> ExprId {
        self.n_expr(IdExpr::new(s))
    }

    /// `visit(NoneExpr*)`.
    pub fn visit_none(&mut self, e: ExprId) -> Result<()> {
        let opt = self.get_stdlib_type(&stdlib().optional)?;
        let t = self.instantiate_type(opt, None)?;
        let et = self.ety(e);
        self.unify(et, t)?;
        if self.realize(et)?.is_some() {
            // Keep the None provisional until inference resolves or defaults it to NoneType.
            let name = if self.types().is(et, &stdlib().optional) { "Optional" } else { "NoneType" };
            let f = self.force_find(&mangle::mangled_method("", name, "__new__", 0, 0)).borrow().ty();
            let c = self.extract_class_type_expr(e);
            let inst = self.instantiate_type(f, c)?;
            self.realize(inst)?;
            self.set_done_expr(e);
        }
        Ok(())
    }

    /// `visit(BoolExpr*)`: set type to static `bool`.
    pub fn visit_bool(&mut self, e: ExprId) -> Result<()> {
        let v = self.ast()[e].kind.as_bool().unwrap().value;
        let t = self.types().new_bool_static(v);
        let et = self.ety(e);
        self.unify(et, t)?;
        self.set_done_expr(e);
        Ok(())
    }

    /// `visit(IntExpr*)`.
    pub fn visit_int(&mut self, e: ExprId) -> Result<()> {
        self.result_expr = self.transform_int(e)?;
        Ok(())
    }

    /// `visit(FloatExpr*)`.
    pub fn visit_float(&mut self, e: ExprId) -> Result<()> {
        self.result_expr = self.transform_float(e)?;
        Ok(())
    }

    /// `visit(StringExpr*)`: concatenate parts and apply transformations.
    pub fn visit_string(&mut self, e: ExprId) -> Result<()> {
        let s = self.ast()[e].kind.as_string().unwrap().clone();
        if s.is_simple() {
            let t = if s.strings[0].prefix == "b" {
                self.get_stdlib_type("bytes")?
            } else {
                self.types().new_str_static(s.value())
            };
            let et = self.ety(e);
            self.unify(et, t)?;
            self.set_done_expr(e);
            return Ok(());
        }
        let mut items = vec![];
        for p in &s.strings {
            if let Some(pe) = p.expr {
                let mut format = "{".to_string();
                if !p.format.conversion.is_empty() {
                    format += &format!("!{}", p.format.conversion);
                }
                if !p.format.spec.is_empty() {
                    format += &format!(":{}", p.format.spec);
                }
                format += "}";
                let fs = self.n_expr(StringExpr::new(format, ""));
                let d = self.dot(fs, "format");
                let mut x = self.call_expr(d, vec![pe]);
                let str_id = self.id("str");
                x = self.call_expr(str_id, vec![x]);
                if !p.value.is_empty() {
                    let sid = self.id(&stdlib().string);
                    let cat = self.dot(sid, "cat");
                    let v = self.n_expr(StringExpr::new(p.value.clone(), ""));
                    x = self.call_expr(cat, vec![v, x]);
                }
                items.push(x);
            } else if p.prefix.to_lowercase() == "b" {
                items.push(self.n_expr(StringExpr::new(p.value.clone(), "b")));
            } else if !p.prefix.is_empty() {
                // Custom prefix strings: `str.__prefix_[prefix]__(str, [static length of str])`
                let sid = self.id(&stdlib().string);
                let d = self.dot(sid, &format!("__prefix_{}__", p.prefix));
                let v = self.n_expr(StringExpr::new(p.value.clone(), ""));
                let l = self.n_expr(IntExpr::new(p.value.len() as i64));
                items.push(self.call_expr(d, vec![v, l]));
            } else {
                items.push(self.n_expr(StringExpr::new(p.value.clone(), "")));
            }
        }
        let r = if items.len() == 1 {
            items[0]
        } else {
            let sid = self.id(&stdlib().string);
            let cat = self.dot(sid, "cat");
            self.call_expr(cat, items)
        };
        self.result_expr = self.transform(Some(r))?;
        Ok(())
    }

    /// `transformInt(expr)`: `123u` -> `UInt[64](123)`, `123i56` -> `Int[56](123)`,
    /// `123pf` -> `int.__suffix_pf__(123)`.
    pub fn transform_int(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let ie = self.ast()[e].kind.as_int().unwrap().clone();
        let (value, suffix) = (ie.value.clone(), ie.suffix.clone());
        let holder = match ie.int_value {
            None => {
                if suffix.is_empty() {
                    let src = self.ast()[e].src;
                    typhon_common::bail_e!(ErrorKind::INT_RANGE, src, value);
                }
                self.n_expr(StringExpr::new(value.clone(), ""))
            }
            Some(v) => self.n_expr(IntExpr::new(v)),
        };
        // Fixed-width integers: `uNNN` or `iNNN`.
        let mut suffix_value: Option<i16> = None;
        if suffix.len() > 1 && (suffix.starts_with('u') || suffix.starts_with('i')) && suffix[1..].bytes().all(|c| c.is_ascii_digit()) {
            suffix_value = suffix[1..].parse::<i16>().ok();
            if suffix_value.is_some_and(|v| v as i32 > MAX_INT_WIDTH) {
                suffix_value = None;
            }
        }
        if suffix.is_empty() {
            // A normal integer (int64_t)
            let t = self.types().new_int_static(ie.value());
            let et = self.ety(e);
            self.unify(et, t)?;
            self.set_done_expr(e);
            return Ok(None);
        }
        let callee = if suffix == "u" {
            let u = self.id(&stdlib().uint);
            let n = self.n_expr(IntExpr::new(64));
            self.n_expr(IndexExpr { expr: Some(u), index: Some(n) })
        } else if let Some(sv) = suffix_value {
            let base = if suffix.starts_with('u') { stdlib().uint.clone() } else { stdlib().int.clone() };
            let u = self.id(&base);
            let n = self.n_expr(IntExpr::new(sv as i64));
            self.n_expr(IndexExpr { expr: Some(u), index: Some(n) })
        } else {
            let i = self.id("Int");
            self.dot(i, &format!("__suffix_{suffix}__"))
        };
        let c = self.call_expr(callee, vec![holder]);
        self.transform(Some(c))
    }

    /// `transformFloat(expr)`: `123.4pf` -> `float.__suffix_pf__(123.4)`.
    pub fn transform_float(&mut self, e: ExprId) -> Result<Option<ExprId>> {
        let fe = self.ast()[e].kind.as_float().unwrap().clone();
        let holder = match fe.float_value {
            None => self.n_expr(StringExpr::new(fe.value.clone(), "")),
            Some(v) => self.n_expr(FloatExpr::new(v)),
        };
        if fe.suffix.is_empty() && fe.float_value.is_some() {
            // A normal float (double)
            let t = self.get_stdlib_type("float")?;
            let et = self.ety(e);
            self.unify(et, t)?;
            self.set_done_expr(e);
            return Ok(None);
        }
        let f = self.id("float");
        let d = if fe.suffix.is_empty() { self.dot(f, "__new__") } else { self.dot(f, &format!("__suffix_{}__", fe.suffix)) };
        let c = self.call_expr(d, vec![holder]);
        self.transform(Some(c))
    }
}

