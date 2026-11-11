//! Port of `codon/runtime/re.cpp` (RE2-backed `re` module) on top of `regex-automata`
//! (the engine behind the `regex` crate; same linear-time guarantees as RE2).
//!
//! PORT-NOTE: behavioral differences vs RE2:
//! - Error messages come from `regex-syntax` and differ in wording from RE2's.
//! - `\w`, `\d`, `\s`, `\b` are Unicode-aware for `str` patterns (RE2: ASCII-only),
//!   which is closer to Python. `ASCII` flag switches them to ASCII.
//! - `VERBOSE` is honored (RE2 ignores it).
//! - `ANCHOR_BOTH` is implemented by compiling a second program `(?:pat)\z`.
//! - Bytes patterns (RE2 Latin-1 encoding) are compiled with Unicode and UTF-8 mode
//!   off; non-ASCII pattern bytes are translated to `\xHH` byte escapes.
//! - `endpos` truncates the haystack (Python semantics: `$` matches at `endpos`),
//!   `pos` keeps left context (so `^` only matches at the real start, as in RE2/Python).

use std::cell::RefCell;
use std::collections::HashMap;

use regex_automata::meta::Regex;
use regex_automata::util::captures::Captures;
use regex_automata::util::syntax;
use regex_automata::{Anchored, Input, PatternID};

use crate::lib_rt::seq_alloc_atomic;
use crate::types::{SeqBytes, SeqInt};

// Flags: must match Codon's `re` module.
pub const ASCII: SeqInt = 1 << 0;
pub const DEBUG: SeqInt = 1 << 1;
pub const IGNORECASE: SeqInt = 1 << 2;
pub const LOCALE: SeqInt = 1 << 3;
pub const MULTILINE: SeqInt = 1 << 4;
pub const DOTALL: SeqInt = 1 << 5;
pub const VERBOSE: SeqInt = 1 << 6;

// RE2::Anchor
pub const UNANCHORED: SeqInt = 0;
pub const ANCHOR_START: SeqInt = 1;
pub const ANCHOR_BOTH: SeqInt = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    pub start: SeqInt,
    pub end: SeqInt,
}

const NO_SPAN: Span = Span { start: -1, end: -1 };

/// Compiled pattern handle (the opaque `cobj` handed to Codon code).
pub struct Pattern {
    re: Option<Regex>,
    full: Option<Regex>,
    error: String,
    groups: usize,
}

impl Pattern {
    fn compile(pattern: &[u8], flags: SeqInt, is_bytes: bool) -> Pattern {
        let src = if is_bytes {
            let mut s = String::with_capacity(pattern.len());
            for &b in pattern {
                if b < 0x80 {
                    s.push(b as char);
                } else {
                    s.push_str(&format!("\\x{b:02X}"));
                }
            }
            s
        } else {
            match std::str::from_utf8(pattern) {
                Ok(s) => s.to_owned(),
                Err(e) => return Pattern::failed(format!("invalid UTF-8: {e}")),
            }
        };
        let unicode = !is_bytes && flags & ASCII == 0;
        let cfg = syntax::Config::new()
            .case_insensitive(flags & IGNORECASE != 0)
            .multi_line(flags & MULTILINE != 0)
            .dot_matches_new_line(flags & DOTALL != 0)
            .ignore_whitespace(flags & VERBOSE != 0)
            .unicode(unicode)
            .utf8(!is_bytes);
        let build = |p: &str| Regex::builder().syntax(cfg).build(p);
        match build(&src) {
            Ok(re) => {
                let full = build(&format!("(?:{src})\\z")).ok();
                let groups = re.group_info().group_len(PatternID::ZERO) - 1;
                Pattern { re: Some(re), full, error: String::new(), groups }
            }
            Err(e) => Pattern::failed(e.to_string()),
        }
    }

    fn failed(error: String) -> Pattern {
        Pattern { re: None, full: None, error, groups: 0 }
    }

    pub fn ok(&self) -> bool {
        self.re.is_some()
    }

    pub fn error(&self) -> &str {
        &self.error
    }

    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Match returning all group spans (index 0 = whole match), or None.
    pub fn match_at(&self, anchor: SeqInt, s: &[u8], pos: SeqInt, endpos: SeqInt) -> Option<Vec<Option<(usize, usize)>>> {
        let re = if anchor == ANCHOR_BOTH { self.full.as_ref()? } else { self.re.as_ref()? };
        let end = (endpos.max(0) as usize).min(s.len());
        let start = pos.max(0) as usize;
        if start > end {
            return None;
        }
        let input = Input::new(&s[..end]).span(start..end).anchored(if anchor == UNANCHORED {
            Anchored::No
        } else {
            Anchored::Yes
        });
        let mut caps: Captures = re.create_captures();
        re.search_captures(&input, &mut caps);
        if !caps.is_match() {
            return None;
        }
        Some((0..=self.groups).map(|i| caps.get_group(i).map(|sp| (sp.start, sp.end))).collect())
    }

    pub fn name_to_index(&self, name: &str) -> Option<usize> {
        let re = self.re.as_ref()?;
        re.group_info().to_index(PatternID::ZERO, name)
    }

    pub fn index_to_name(&self, index: usize) -> Option<&str> {
        let re = self.re.as_ref()?;
        re.group_info().to_name(PatternID::ZERO, index)
    }
}

/// RE2::QuoteMeta.
pub fn quote_meta(p: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(p.len() * 2);
    for &c in p {
        if !c.is_ascii_alphanumeric() && c != b'_' && c < 0x80 {
            if c == 0 {
                out.extend_from_slice(b"\\x00");
                continue;
            }
            out.push(b'\\');
        }
        out.push(c);
    }
    out
}

/// RE2::CheckRewriteString.
pub fn check_rewrite_string(groups: usize, rewrite: &[u8]) -> Result<(), String> {
    let mut max_token: i64 = -1;
    let mut i = 0;
    while i < rewrite.len() {
        let c = rewrite[i];
        i += 1;
        if c != b'\\' {
            continue;
        }
        if i == rewrite.len() {
            return Err("Rewrite schema error: '\\' not allowed at end.".into());
        }
        let n = rewrite[i];
        i += 1;
        if n == b'\\' {
            continue;
        }
        if !n.is_ascii_digit() {
            return Err("Rewrite schema error: '\\' must be followed by a digit or '\\'.".into());
        }
        max_token = max_token.max((n - b'0') as i64);
    }
    if max_token > groups as i64 {
        return Err(format!(
            "Rewrite schema requests {max_token} matches, but the regexp only has {groups} parenthesized subexpressions."
        ));
    }
    Ok(())
}

type Key = (Vec<u8>, SeqInt);

thread_local! {
    static CACHE: RefCell<HashMap<Key, Box<Pattern>>> = RefCell::new(HashMap::new());
}

fn get(p: &[u8], flags: SeqInt, is_bytes: bool) -> *mut Pattern {
    let key_flags = flags | if is_bytes { 1i64 << 32 } else { 0 };
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        let e = c.entry((p.to_vec(), key_flags)).or_insert_with(|| Box::new(Pattern::compile(p, flags, is_bytes)));
        &mut **e as *mut Pattern
    })
}

/*
 * Matching
 */

/// # Safety
/// `re` from `seq_re_compile`; `s` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_re_match(re: *mut Pattern, anchor: SeqInt, s: SeqBytes, pos: SeqInt, endpos: SeqInt) -> *mut Span {
    let re = unsafe { &*re };
    let n = re.groups() + 1;
    let spans = seq_alloc_atomic(n * core::mem::size_of::<Span>()) as *mut Span;
    let res = re.match_at(anchor, unsafe { s.as_slice() }, pos, endpos);
    for i in 0..n {
        let sp = match &res {
            Some(g) => g[i].map_or(NO_SPAN, |(a, b)| Span { start: a as SeqInt, end: b as SeqInt }),
            None => NO_SPAN,
        };
        unsafe { spans.add(i).write(sp) };
    }
    spans
}

/// # Safety
/// `re` from `seq_re_compile`; `s` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_re_match_one(re: *mut Pattern, anchor: SeqInt, s: SeqBytes, pos: SeqInt, endpos: SeqInt) -> Span {
    let re = unsafe { &*re };
    match re.match_at(anchor, unsafe { s.as_slice() }, pos, endpos) {
        Some(g) => g[0].map_or(NO_SPAN, |(a, b)| Span { start: a as SeqInt, end: b as SeqInt }),
        None => NO_SPAN,
    }
}

/*
 * General functions
 */

/// # Safety
/// `p` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_re_escape(p: SeqBytes) -> SeqBytes {
    SeqBytes::from_slice_gc(&quote_meta(unsafe { p.as_slice() }))
}

/// # Safety
/// `p` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_re_compile(p: SeqBytes, flags: SeqInt, is_bytes: bool) -> *mut Pattern {
    get(unsafe { p.as_slice() }, flags, is_bytes)
}

#[unsafe(no_mangle)]
pub extern "C" fn seq_re_purge() {
    CACHE.with(|c| c.borrow_mut().clear());
}

/*
 * Pattern methods
 */

/// # Safety
/// `pattern` from `seq_re_compile`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_re_pattern_groups(pattern: *mut Pattern) -> SeqInt {
    unsafe { (*pattern).groups() as SeqInt }
}

/// # Safety
/// `pattern` from `seq_re_compile`; `name` valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_re_group_name_to_index(pattern: *mut Pattern, name: SeqBytes) -> SeqInt {
    let name = String::from_utf8_lossy(unsafe { name.as_slice() });
    unsafe { (*pattern).name_to_index(&name).map_or(-1, |i| i as SeqInt) }
}

/// # Safety
/// `pattern` from `seq_re_compile`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_re_group_index_to_name(pattern: *mut Pattern, index: SeqInt) -> SeqBytes {
    if index < 0 {
        return SeqBytes::EMPTY;
    }
    match unsafe { (*pattern).index_to_name(index as usize) } {
        Some(n) => SeqBytes::from_slice_gc(n.as_bytes()),
        None => SeqBytes::EMPTY,
    }
}

/// # Safety
/// Pointers valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_re_check_rewrite_string(pattern: *mut Pattern, rewrite: SeqBytes, error: *mut SeqBytes) -> bool {
    match check_rewrite_string(unsafe { (*pattern).groups() }, unsafe { rewrite.as_slice() }) {
        Ok(()) => true,
        Err(e) => {
            unsafe { *error = SeqBytes::from_slice_gc(e.as_bytes()) };
            false
        }
    }
}

/// # Safety
/// `pattern` from `seq_re_compile`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn seq_re_pattern_error(pattern: *mut Pattern) -> SeqBytes {
    let p = unsafe { &*pattern };
    if p.ok() { SeqBytes::EMPTY } else { SeqBytes::from_slice_gc(p.error().as_bytes()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(p: &str, f: SeqInt, a: SeqInt, s: &str) -> Option<(usize, usize)> {
        Pattern::compile(p.as_bytes(), f, false).match_at(a, s.as_bytes(), 0, s.len() as SeqInt).and_then(|g| g[0])
    }

    #[test]
    fn basic_matching() {
        assert_eq!(m("b+", 0, UNANCHORED, "abbbc"), Some((1, 4)));
        assert_eq!(m("b+", 0, ANCHOR_START, "abbbc"), None);
        assert_eq!(m("a|ab", 0, ANCHOR_BOTH, "ab"), Some((0, 2)));
        assert_eq!(m("ABC", IGNORECASE, UNANCHORED, "xabc"), Some((1, 4)));
        assert_eq!(m("^b", MULTILINE, UNANCHORED, "a\nb"), Some((2, 3)));
        assert_eq!(m("^b", 0, UNANCHORED, "a\nb"), None);
        assert_eq!(m("a.b", DOTALL, UNANCHORED, "a\nb"), Some((0, 3)));
        assert_eq!(m("a b # c", VERBOSE, UNANCHORED, "ab"), Some((0, 2)));
    }

    #[test]
    fn pos_endpos_groups_names() {
        let p = Pattern::compile(br"(?P<x>\d+)-(\d+)?", 0, false);
        assert_eq!(p.groups(), 2);
        assert_eq!(p.name_to_index("x"), Some(1));
        assert_eq!(p.index_to_name(1), Some("x"));
        assert_eq!(p.index_to_name(2), None);
        let g = p.match_at(UNANCHORED, b"12-", 0, 3).unwrap();
        assert_eq!(g, vec![Some((0, 3)), Some((0, 2)), None]);
        assert!(p.match_at(UNANCHORED, b"12-34", 1, 2).is_none());
        let d = Pattern::compile(b"a$", 0, false);
        assert!(d.match_at(UNANCHORED, b"aab", 0, 2).is_some());
    }

    #[test]
    fn errors_bytes_escape_rewrite() {
        let e = Pattern::compile(b"(abc", 0, false);
        assert!(!e.ok() && !e.error().is_empty());
        let b = Pattern::compile(b"\xff+", 0, true);
        assert_eq!(b.match_at(UNANCHORED, b"a\xff\xffb", 0, 4).unwrap()[0], Some((1, 3)));
        assert_eq!(quote_meta(b"a.b\0_"), b"a\\.b\\x00_".to_vec());
        assert!(check_rewrite_string(1, br"\1\\").is_ok());
        assert!(check_rewrite_string(1, br"\2").is_err());
        assert!(check_rewrite_string(1, br"\q").is_err());
    }

    #[test]
    fn abi_cache() {
        crate::lib_rt::gc_init();
        let pat = b"x(y)";
        let pb = SeqBytes { ptr: pat.as_ptr() as *mut u8, len: 4 };
        unsafe {
            let a = seq_re_compile(pb, 0, false);
            assert_eq!(a, seq_re_compile(pb, 0, false));
            assert_eq!(seq_re_pattern_groups(a), 1);
            let s = b"zxy";
            let sb = SeqBytes { ptr: s.as_ptr() as *mut u8, len: 3 };
            let sp = seq_re_match(a, 0, sb, 0, 3);
            assert_eq!(*sp, Span { start: 1, end: 3 });
            assert_eq!(*sp.add(1), Span { start: 2, end: 3 });
            assert_eq!(seq_re_match_one(a, 1, sb, 0, 3), NO_SPAN);
            assert_eq!(seq_re_pattern_error(a).len, 0);
            seq_re_purge();
        }
    }
}
