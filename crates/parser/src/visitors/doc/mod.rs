//! `typhon doc`: walks the given files (plus the stdlib prelude) and produces a JSON
//! documentation database `{ "<id>": item, ... }`.
//!
//! Note: the reference `json` helper (string / list / object over an `unordered_map`) is
//! replaced by `serde_json::Value`; [`json_to_string`] reproduces the reference `toString()`
//! layout (`{ "k": v }`, `[ a, b ]`) with keys in sorted order instead of hash order.
//! The the reference visitor is an `ASTVisitor`; this port dispatches with a `match` since only
//! seven node kinds are handled.

mod items;

use std::cell::RefCell;
use std::rc::Rc;

use rustc_hash::FxHashMap;
use serde_json::{Map, Value};
use typhon_common::srcinfo::SrcInfo;
use typhon_common::{CompilerError, ErrorKind, Result};

use crate::ast::{ExprKind, StmtKind};
use crate::cache::{Cache, STDLIB_INTERNAL_MODULE};
use crate::ctx::Context;
use crate::ids::{ExprId, StmtId};
use crate::visitors::FormatVisitor;
use crate::visitors::scoping::ScopingVisitor;

/// the reference `json_escape`.
pub fn json_escape(s: &str) -> String {
    let mut r = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\u{8}' => r += "\\b",
            '\u{c}' => r += "\\f",
            '\n' => r += "\\n",
            '\r' => r += "\\r",
            '\t' => r += "\\t",
            '\\' => r += "\\\\",
            '"' => r += "\\\"",
            c => r.push(c),
        }
    }
    r
}

/// the reference `json::toString()`.
pub fn json_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => format!("\"{}\"", json_escape(s)),
        Value::Array(a) if a.is_empty() => "{}".into(),
        Value::Array(a) => format!("[ {} ]", a.iter().map(json_to_string).collect::<Vec<_>>().join(", ")),
        Value::Object(o) if o.is_empty() => "{}".into(),
        Value::Object(o) => format!(
            "{{ {} }}",
            o.iter().map(|(k, v)| format!("\"{}\": {}", json_escape(k), json_to_string(v))).collect::<Vec<_>>().join(", ")
        ),
        Value::Null => "{}".into(),
        other => format!("\"{}\"", json_escape(&other.to_string())),
    }
}

fn obj(kv: &[(&str, &str)]) -> Map<String, Value> {
    kv.iter().map(|(k, v)| (k.to_string(), Value::String(v.to_string()))).collect()
}
fn strs(v: &[String]) -> Value {
    Value::Array(v.iter().cloned().map(Value::String).collect())
}
fn is_empty_json(v: &Value) -> bool {
    matches!(v, Value::Object(o) if o.is_empty()) || matches!(v, Value::Array(a) if a.is_empty()) || v.is_null()
}

/// `DocContext`: identifier -> item id (0 = generic).
pub type DocContextRef = Rc<RefCell<Context<i32>>>;

/// `DocShared`.
pub struct DocShared {
    pub item_id: i32,
    pub j: Map<String, Value>,
    pub modules: FxHashMap<String, DocContextRef>,
    pub argv0: String,
    pub cache: Cache,
    pub generics: FxHashMap<i32, Vec<String>>,
}

impl DocShared {
    /// Fresh shared state with an empty prelude module `""`.
    pub fn new(argv0: &str, cache: Cache) -> Self {
        let mut s = Self {
            item_id: 1,
            j: Map::new(),
            modules: FxHashMap::default(),
            argv0: argv0.to_string(),
            cache,
            generics: FxHashMap::default(),
        };
        s.modules.insert(String::new(), Rc::new(RefCell::new(Context::new(""))));
        s
    }
    /// New `DocContext` for a module file.
    pub fn new_context(&self, filename: &str) -> DocContextRef {
        Rc::new(RefCell::new(Context::new(filename)))
    }
    fn next_id(&mut self) -> i32 {
        let i = self.item_id;
        self.item_id += 1;
        i
    }
}

/// `DocVisitor`.
pub struct DocVisitor<'a> {
    pub shared: &'a mut DocShared,
    pub ctx: DocContextRef,
}

fn custom(src: &SrcInfo, msg: String) -> CompilerError {
    CompilerError::parser(ErrorKind::CUSTOM, src.clone(), &[&msg])
}

fn parse_file(cache: &mut Cache, path: &str) -> Result<StmtId> {
    let h = cache.hooks.parse_file.clone().ok_or_else(|| CompilerError::custom("no parser hook installed", SrcInfo::default()))?;
    h(cache, path)
}

impl DocShared {
    /// `DocContext::find`: fall back to the prelude module (`modules[""]`).
    fn find(&self, ctx: &DocContextRef, s: &str) -> Option<i32> {
        let i = ctx.borrow().find(s);
        if i.is_none()
            && let Some(root) = self.modules.get("")
            && !Rc::ptr_eq(root, ctx)
        {
            return root.borrow().find(s);
        }
        i
    }
}

/// `getDocstr`.
fn get_docstr(cache: &Cache, s: StmtId) -> String {
    if let StmtKind::Expr(se) = &cache.ast[s].kind
        && let Some(e) = se.expr
        && let ExprKind::String(st) = &cache.ast[e].kind
    {
        return st.strings.iter().map(|p| p.value.as_str()).collect();
    }
    String::new()
}

/// `DocVisitor::jsonify`.
pub fn jsonify(s: &SrcInfo) -> Value {
    strs(&[s.line.to_string(), s.len.to_string()])
}

/// `isValidName`.
fn is_valid_name(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if s.len() > 4 && s.starts_with("__") && s.ends_with("__") {
        return true;
    }
    !s.starts_with('_')
}

impl<'a> DocVisitor<'a> {
    pub fn new(shared: &'a mut DocShared, ctx: DocContextRef) -> Self {
        Self { shared, ctx }
    }

    /// `DocVisitor::apply(argv0, files)`. The cache must have `hooks.parse_file` /
    /// `hooks.parse_code` installed and a file system.
    pub fn apply(argv0: &str, files: &[String], cache: Cache) -> Result<Value> {
        let mut shared = DocShared::new(argv0, cache);
        let root = shared.modules[""].clone();

        let stdlib = crate::typecheck::helpers::get_import_file(&mut shared.cache, STDLIB_INTERNAL_MODULE, "", true)
            .ok_or_else(|| CompilerError::custom("cannot find stdlib", SrcInfo::default()))?;
        let ast = parse_file(&mut shared.cache, &stdlib.path)?;
        let pc = shared
            .cache
            .hooks
            .parse_code
            .clone()
            .ok_or_else(|| CompilerError::custom("no parser hook installed", SrcInfo::default()))?;
        let core = pc(&mut shared.cache, &stdlib.path, "from internal.core import *", 0)?;
        root.borrow_mut().set_filename(stdlib.path.clone());
        for name in [
            "int", "__py_numerics__", "__debug__", "__apple__", "__windows__", "__linux__", "__android__", "__ios__", "__macos__",
            "__wasm__", "__wasi__", "__aarch64__", "__x86_64__", "__dict_unordered__", "__typhon_version_major__", "__typhon_version_minor__", "__typhon_version_micro__",
        ] {
            let id = shared.next_id();
            root.borrow_mut().add(name, id);
        }
        for name in ["type", "Literal"] {
            let mut j = obj(&[("name", name), ("kind", "class"), ("type", "type")]);
            j.insert("generics".into(), strs(&["T".into()]));
            root.borrow_mut().add(name, shared.item_id);
            let id = shared.next_id();
            shared.j.insert(id.to_string(), Value::Object(j));
        }
        DocVisitor::new(&mut shared, root.clone()).transform_module(core)?;
        DocVisitor::new(&mut shared, root.clone()).transform_module(ast)?;

        let ctx: DocContextRef = Rc::new(RefCell::new(Context::new("")));
        for f in files {
            let path = match shared.cache.fs.clone() {
                Some(fs) => fs.borrow().canonical(std::path::Path::new(f)).to_string_lossy().replace('\\', "/"),
                None => f.clone(),
            };
            ctx.borrow_mut().set_filename(path.clone());
            let a = parse_file(&mut shared.cache, &path)?;
            DocVisitor::new(&mut shared, ctx.clone()).transform_module(a)?;
        }
        Ok(Value::Object(shared.j))
    }

    /// `DocVisitor::flatten`.
    pub fn flatten(cache: &Cache, stmt: Option<StmtId>, docstr: Option<&mut String>, deep: bool) -> Vec<StmtId> {
        let mut out = vec![];
        let Some(stmt) = stmt else { return out };
        let mut docstr = docstr;
        if let StmtKind::Suite(s) = &cache.ast[stmt].kind {
            let n = if deep { s.items.len() } else { s.items.len().min(1) };
            for (i, &x) in s.items.iter().take(n).enumerate() {
                let d = if i == 0 { docstr.as_deref_mut() } else { None };
                out.extend(Self::flatten(cache, Some(x), d, deep));
            }
        } else {
            if let Some(d) = docstr {
                *d = get_docstr(cache, stmt);
            }
            out.push(stmt);
        }
        out
    }

    fn fmt(&self, e: ExprId) -> String {
        FormatVisitor::apply_expr(&self.shared.cache.ast, e, None, false)
    }

    /// `transform(Expr*)`: an empty object for `None` / unhandled nodes.
    pub fn transform_expr(&mut self, e: Option<ExprId>) -> Result<Value> {
        let Some(e) = e else { return Ok(Value::Object(Map::new())) };
        let src = self.shared.cache.ast[e].src.clone();
        match self.shared.cache.ast[e].kind.clone() {
            ExprKind::Int(i) => Ok(Value::String(i.raw_data().0)),
            ExprKind::Id(i) => {
                let Some(id) = self.shared.find(&self.ctx, &i.value) else {
                    return Err(custom(&src, format!("unknown identifier {}", i.value)));
                };
                Ok(Value::String(if id != 0 { id.to_string() } else { i.value }))
            }
            ExprKind::Index(ix) => {
                let mut v = vec![self.transform_expr(ix.expr)?];
                let tuple = ix.index.and_then(|x| self.shared.cache.ast[x].kind.as_tuple().cloned());
                if let Some(tp) = tuple {
                    let first_list = tp.items.first().and_then(|x| self.shared.cache.ast[*x].kind.as_list().cloned());
                    if let Some(l) = first_list {
                        for e in l.items {
                            v.push(self.tr_index(e)?);
                        }
                        // Note: the reference indexes `(*tp)[1]` unconditionally.
                        if let Some(&x) = tp.items.get(1) {
                            v.push(self.tr_index(x)?);
                        }
                    } else {
                        for e in tp.items {
                            v.push(self.tr_index(e)?);
                        }
                    }
                } else if let Some(x) = ix.index {
                    v.push(self.tr_index(x)?);
                }
                Ok(Value::Array(v))
            }
            _ => Ok(Value::Object(Map::new())),
        }
    }

    fn tr_index(&mut self, e: ExprId) -> Result<Value> {
        if matches!(self.shared.cache.ast[e].kind, ExprKind::Int(_) | ExprKind::Bool(_) | ExprKind::String(_)) {
            Ok(Value::String(self.fmt(e)))
        } else {
            self.transform_expr(Some(e))
        }
    }

    /// `transform(Stmt*)`: the id of the documented item or "".
    pub fn transform_stmt(&mut self, s: Option<StmtId>) -> Result<String> {
        let Some(s) = s else { return Ok(String::new()) };
        match self.shared.cache.ast[s].kind.tag() {
            crate::ast::StmtTag::Function => self.visit_function(s),
            crate::ast::StmtTag::Class => self.visit_class(s),
            crate::ast::StmtTag::Assign => self.visit_assign(s),
            crate::ast::StmtTag::Import => self.visit_import(s),
            _ => {
                // the reference DocVisitor is a CallbackASTVisitor: unhandled statements (if/while/
                // suite/...) still visit their child statements, so names bound inside them
                // (e.g. `if __dict_unordered__: dict = ...`) are registered. The child
                // result is discarded (`resultStmt` of the outer visitor stays "").
                for c in crate::visitors::visitor::stmt_children(&self.shared.cache.ast, s) {
                    if let crate::ast::children::Child::Stmt(x) | crate::ast::children::Child::Suite(x) = c {
                        self.transform_stmt(Some(x))?;
                    }
                }
                Ok(String::new())
            }
        }
    }

    /// `transformModule`.
    pub fn transform_module(&mut self, stmt: StmtId) -> Result<()> {
        ScopingVisitor::apply(&mut self.shared.cache, stmt, None, false)?;
        let mut children = vec![];
        let mut docstr = String::new();
        let flat = Self::flatten(&self.shared.cache, Some(stmt), Some(&mut docstr), true);
        for (i, &s) in flat.iter().enumerate() {
            let id = self.transform_stmt(Some(s))?;
            if id.is_empty() {
                continue;
            }
            if i + 1 < flat.len() && matches!(self.shared.cache.ast[s].kind, StmtKind::Assign(_)) {
                let ds = get_docstr(&self.shared.cache, flat[i + 1]);
                if !ds.is_empty()
                    && let Some(Value::Object(o)) = self.shared.j.get_mut(&id)
                {
                    o.insert("doc".into(), Value::String(ds));
                }
            }
            children.push(id);
        }
        let id = self.shared.next_id().to_string();
        let path = self.ctx.borrow().filename().to_string();
        let mut ja = obj(&[("kind", "module"), ("path", &path)]);
        ja.insert("children".into(), strs(&children));
        if !docstr.is_empty() {
            ja.insert("doc".into(), Value::String(docstr));
        }
        self.shared.j.insert(id, Value::Object(ja));
        Ok(())
    }
}
