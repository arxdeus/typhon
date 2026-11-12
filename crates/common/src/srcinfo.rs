//! Source locations. Port of `InternedString`, `SrcInfo`, `SrcObject`.

use std::cmp::Ordering;
use std::fmt;
use std::sync::atomic::{AtomicI32, Ordering as AtomicOrdering};

/// Interned file name (`InternedString` / `SourceFile`). Backed by the `ustr` crate,
/// which gives process-lifetime, pointer-comparable strings exactly like the reference version.
pub type SourceFile = ustr::Ustr;

static NEXT_ID: AtomicI32 = AtomicI32::new(0);

/// A source location. Equality is by unique `id`, ordering is by
/// `(file, line, col)`.
#[derive(Clone, Copy, Debug)]
pub struct SrcInfo {
    pub file: SourceFile,
    pub line: i32,
    pub col: i32,
    pub len: i32,
    /// Used to differentiate different instances.
    pub id: i32,
}

impl SrcInfo {
    pub fn new(file: &str, line: i32, col: i32, len: i32) -> Self {
        let mut line = line;
        if file.is_empty() && line != 0 {
            line += 1;
        }
        Self {
            file: ustr::ustr(file),
            line,
            col,
            len,
            id: NEXT_ID.fetch_add(1, AtomicOrdering::Relaxed),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.file.is_empty() && self.line == 0 && self.col == 0
    }
}

impl Default for SrcInfo {
    fn default() -> Self {
        Self::new("", 0, 0, 0)
    }
}

impl PartialEq for SrcInfo {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for SrcInfo {}

impl PartialOrd for SrcInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(
            (self.file.as_str(), self.line, self.col).cmp(&(
                other.file.as_str(),
                other.line,
                other.col,
            )),
        )
    }
}

impl fmt::Display for SrcInfo {
    /// `file_basename:line:col` (`operator<<(ostream&, SrcInfo)`).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = std::path::Path::new(self.file.as_str())
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        write!(f, "{}:{}:{}", name, self.line, self.col)
    }
}

/// anything carrying a source location.
pub trait SrcObject {
    fn src_info(&self) -> SrcInfo;
    fn set_src_info(&mut self, info: SrcInfo);
}
