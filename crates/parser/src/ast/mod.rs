//! Port of codon/parser/ast.h + codon/parser/ast/{node.h,attr.h,expr.{h,cpp},stmt.{h,cpp},
//! codon_string.cpp}.
//!
//! See `../../AST_API.md` for the API contract and the C++ -> Rust name table.

pub mod arena;
pub mod attr;
pub mod children;
pub mod codon_string;
pub mod expr;
pub mod python_string;
pub mod serial;
pub mod sexpr;
pub mod stmt;

pub use arena::{AstArena, GenSlot, LiteralKind, TypeFormatter, is_id};
pub use attr::{Attr, AttrValue, Attrs, Binding, BindingsAttr, CaptureType, INDENT_SIZE};
pub use children::{Child, IdMapper};
pub use expr::*;
pub use stmt::*;

pub use crate::ids::{ExprId, StmtId};

/// Serde adapter for [`typhon_common::SrcInfo`] (which is defined without serde support).
pub(crate) mod src_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use typhon_common::SrcInfo;

    #[derive(Serialize, Deserialize)]
    struct Repr {
        file: String,
        line: i32,
        col: i32,
        len: i32,
        id: i32,
    }

    pub fn serialize<S: Serializer>(s: &SrcInfo, ser: S) -> Result<S::Ok, S::Error> {
        Repr { file: s.file.as_str().to_string(), line: s.line, col: s.col, len: s.len, id: s.id }.serialize(ser)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<SrcInfo, D::Error> {
        let r = Repr::deserialize(d)?;
        Ok(SrcInfo { file: ustr::ustr(&r.file), line: r.line, col: r.col, len: r.len, id: r.id })
    }
}
