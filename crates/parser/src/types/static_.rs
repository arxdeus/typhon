//! `IntStaticType`, `StrStaticType`,
//! `BoolStaticType`.

use typhon_common::Result;
use typhon_common::strings::escape;

use super::*;
use crate::ast::{AstArena, BoolExpr, IntExpr, StringExpr};
use crate::ids::ExprId;

impl TypeArena {
    /// `{Int,Str,Bool}StaticType::unify`.
    pub(crate) fn static_unify(
        &self,
        this: TypeId,
        typ: TypeId,
        us: Option<&mut Unification>,
        host: &mut dyn TypeHost,
    ) -> Result<i32> {
        let me = self.class(this).unwrap();
        match &me.kind {
            ClassKind::IntStatic(v) => {
                if let Some(t) = self.get_int_static(typ) {
                    return Ok(if *v == t { 1 } else { -1 });
                } else if let Some(c) = self.get_class(typ) {
                    return match self.non_static_type(this) {
                        Some(ns) => self.unify(ns, c, us, host),
                        None => Ok(-1),
                    };
                }
            }
            ClassKind::StrStatic(v) => {
                if let Some(t) = self.get_str_static(typ) {
                    return Ok(if **v == *t { 1 } else { -1 });
                } else if let Some(c) = self.get_class(typ) {
                    return self.class_unify(this, c, us, host);
                }
            }
            ClassKind::BoolStatic(v) => {
                if let Some(t) = self.get_bool_static(typ) {
                    return Ok(if *v == t { 1 } else { -1 });
                } else if let Some(c) = self.get_class(typ) {
                    return self.class_unify(this, c, us, host);
                }
            }
            _ => unreachable!(),
        }
        if let Some(tl) = self.get_link(typ) {
            self.unify(tl, this, us, host)
        } else {
            Ok(-1)
        }
    }

    /// `{Int,Str,Bool}StaticType::debugString`.
    pub(crate) fn static_debug_string(&self, c: &ClassType, mode: i32) -> String {
        match &c.kind {
            ClassKind::IntStatic(v) => {
                if mode < 2 {
                    format!("{v}")
                } else {
                    format!("Literal[{v}]")
                }
            }
            ClassKind::StrStatic(v) => {
                if mode < 2 {
                    format!("'{}'", escape(v))
                } else {
                    format!("Literal['{}']", escape(v))
                }
            }
            ClassKind::BoolStatic(v) => {
                if mode < 2 {
                    if *v { "1".into() } else { "0".into() }
                } else {
                    format!("Literal[{}]", if *v { "True" } else { "False" })
                }
            }
            _ => unreachable!(),
        }
    }

    /// `StaticType::getStaticExpr()`: a fresh literal expression for the static value.
    pub fn static_expr(&self, this: TypeId, ast: &mut AstArena) -> Option<ExprId> {
        let s = self.get_static(this)?;
        let src = self.src(s);
        let k = self.with_class(s, |c| c.kind.clone())?;
        Some(match k {
            ClassKind::IntStatic(v) => ast.new_expr(IntExpr::new(v), src),
            ClassKind::StrStatic(v) => ast.new_expr(StringExpr::new(v, ""), src),
            ClassKind::BoolStatic(v) => ast.new_expr(BoolExpr { value: v }, src),
            _ => return None,
        })
    }
}
