//! String helpers. Port of the free functions.
//!
//! Many the reference helpers (`split`, `join`, `in`, `startswith`, `ltrim`, `tolower`, ...) map
//! directly onto `std` methods; only the non-trivial ones are reimplemented here.

use std::fmt::Write as _;

/// Error from [`unescape`] (throws `std::invalid_argument`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct UnescapeError(pub String);

/// Python-style escape of a byte string.
pub fn escape(s: &str) -> String {
    let mut r = String::with_capacity(s.len());
    for &c in s.as_bytes() {
        match c {
            0x07 => r.push_str("\\a"),
            0x08 => r.push_str("\\b"),
            0x0c => r.push_str("\\f"),
            b'\n' => r.push_str("\\n"),
            b'\r' => r.push_str("\\r"),
            b'\t' => r.push_str("\\t"),
            0x0b => r.push_str("\\v"),
            b'\'' => r.push_str("\\'"),
            b'\\' => r.push_str("\\\\"),
            c if !(32..127).contains(&c) => {
                let _ = write!(r, "\\x{c:x}");
            }
            c => r.push(c as char),
        }
    }
    r
}

fn push_cp(out: &mut Vec<u8>, cp: u32) {
    let mut buf = [0u8; 4];
    match char::from_u32(cp) {
        Some(ch) => out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes()),
        // Surrogates: encode like the reference `utf8_append` (3-byte form) for fidelity.
        None => {
            out.push((0xE0 | (cp >> 12)) as u8);
            out.push((0x80 | ((cp >> 6) & 0x3F)) as u8);
            out.push((0x80 | (cp & 0x3F)) as u8);
        }
    }
}

/// Python-style unescape. With `utf8 == false`
/// (bytes literals) non-ASCII input is rejected and `\x` yields a raw byte.
/// Returns raw bytes because bytes literals may be non-UTF-8.
pub fn unescape(s: &str, utf8: bool) -> Result<Vec<u8>, UnescapeError> {
    let b = s.as_bytes();
    let mut r: Vec<u8> = Vec::with_capacity(b.len());
    let mut i = 0usize;
    while i < b.len() {
        if !utf8 && b[i] >= 0x80 {
            return Err(UnescapeError("bytes can only contain ASCII literal characters".into()));
        }
        if b[i] == b'\\' && i + 1 < b.len() {
            let n = b[i + 1];
            match n {
                b'a' => { r.push(0x07); i += 1; }
                b'b' => { r.push(0x08); i += 1; }
                b'f' => { r.push(0x0c); i += 1; }
                b'n' => { r.push(b'\n'); i += 1; }
                b'r' => { r.push(b'\r'); i += 1; }
                b't' => { r.push(b'\t'); i += 1; }
                b'v' => { r.push(0x0b); i += 1; }
                b'"' => { r.push(b'"'); i += 1; }
                b'\'' => { r.push(b'\''); i += 1; }
                b'\\' => { r.push(b'\\'); i += 1; }
                b'x' => {
                    if i + 3 > b.len() {
                        return Err(UnescapeError("invalid \\x code".into()));
                    }
                    let end = (i + 4).min(b.len());
                    let digits: String = s[i + 2..end]
                        .chars()
                        .take_while(|c| c.is_ascii_hexdigit())
                        .collect();
                    if digits.is_empty() {
                        return Err(UnescapeError("invalid \\x code".into()));
                    }
                    let code = u32::from_str_radix(&digits, 16).unwrap();
                    if utf8 { push_cp(&mut r, code) } else { r.push(code as u8) }
                    i += digits.len() + 1;
                }
                b'u' | b'U' if utf8 => {
                    let digits = if n == b'u' { 4 } else { 8 };
                    if i + digits + 2 > b.len() {
                        return Err(UnescapeError("truncated Unicode escape".into()));
                    }
                    let hex = &s[i + 2..i + 2 + digits];
                    let code = u32::from_str_radix(hex, 16)
                        .map_err(|_| UnescapeError("invalid Unicode escape".into()))?;
                    if code > 0x10FFFF {
                        return Err(UnescapeError("Unicode escape out of range".into()));
                    }
                    push_cp(&mut r, code);
                    i += digits + 1;
                }
                b'0'..=b'7' => {
                    let end = (i + 4).min(b.len());
                    let digits: String = s[i + 1..end]
                        .chars()
                        .take_while(|c| ('0'..='7').contains(c))
                        .collect();
                    let code = u32::from_str_radix(&digits, 8).unwrap();
                    if utf8 { push_cp(&mut r, code) } else { r.push(code as u8) }
                    i += digits.len();
                }
                _ => r.push(b[i]),
            }
        } else {
            r.push(b[i]);
        }
        i += 1;
    }
    Ok(r)
}

/// Double `{` and `}` in `s[start..start+len]`.
pub fn escape_fstring_braces(s: &str, start: usize, len: usize) -> String {
    let mut t = String::with_capacity(len);
    for c in s[start..start + len].chars() {
        match c {
            '{' => t.push_str("{{"),
            '}' => t.push_str("}}"),
            c => t.push(c),
        }
    }
    t
}

/// the reference `findStar`: index after an opening paren or of the first non-space char.
pub fn find_star(s: &str) -> usize {
    for (i, c) in s.bytes().enumerate() {
        if c == b'(' {
            return i + 1;
        }
        if !c.is_ascii_whitespace() {
            return i;
        }
    }
    s.len()
}

/// Number of Unicode scalar values (`utf8_strlen`).
pub fn utf8_strlen(s: &str) -> usize {
    s.chars().count()
}

/// Code-point slice with Python step semantics (`utf8_substr`).
pub fn utf8_substr(s: &str, start: usize, stop: usize, step: isize) -> String {
    assert!(step != 0, "step cannot be zero");
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::new();
    if step > 0 {
        let mut i = start;
        while i < stop.min(chars.len()) {
            out.push(chars[i]);
            i += step as usize;
        }
    } else {
        let mut picked = Vec::new();
        let mut i = stop as isize + 1;
        while i >= 0 && (i as usize) <= start && (i as usize) < chars.len() {
            picked.push(chars[i as usize]);
            i += -step;
        }
        for c in picked {
            out.insert(0, c);
        }
    }
    out
}

/// Replace the first occurrence.
pub fn replace_first(text: &str, pat: &str, with: &str) -> String {
    text.replacen(pat, with, 1)
}

/// `isdigit`: all ASCII digits (true for empty).
pub fn is_all_digits(s: &str) -> bool {
    s.bytes().all(|c| c.is_ascii_digit())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_escape() {
        assert_eq!(escape("a\nb'\\"), "a\\nb\\'\\\\");
        assert_eq!(unescape("a\\nb\\x41\\u00e9\\101", true).unwrap(), "a\nbAéA".as_bytes());
        assert!(unescape("é", false).is_err());
        assert_eq!(unescape("\\xff", false).unwrap(), vec![0xff]);
    }

    #[test]
    fn utf8_ops() {
        assert_eq!(utf8_strlen("héllo"), 5);
        assert_eq!(utf8_substr("héllo", 1, 4, 1), "éll");
        assert_eq!(utf8_substr("abcdef", 0, 6, 2), "ace");
    }
}
