//! Core ABI types and string helpers shared by the runtime (port of `lib.h` and the
//! string-conversion section of `lib.cpp`).

use crate::lib_rt::{seq_alloc_atomic, seq_free};

/// Compiled/running in debug mode.
pub const SEQ_FLAG_DEBUG: i32 = 1 << 0;
// Bit 1 was SEQ_FLAG_CAPTURE_OUTPUT (JIT/Jupyter output capture); unused.
/// Compiled as a standalone object/binary.
pub const SEQ_FLAG_STANDALONE: i32 = 1 << 2;

pub const SEQ_EXCEPTION_CLASS: u64 = 0x6f62_6a00_7365_7100;

pub const SEQ_STR_KIND_ASCII: u32 = 0;
pub const SEQ_STR_KIND_LATIN1: u32 = 1;
pub const SEQ_STR_KIND_UCS2: u32 = 2;
pub const SEQ_STR_KIND_UCS4: u32 = 3;
pub const SEQ_STR_KIND_SHIFT: u32 = 56;
pub const SEQ_STR_LEN_MASK: u64 = (1u64 << SEQ_STR_KIND_SHIFT) - 1;
pub const SEQ_STR_KIND_MASK: u64 = 3u64 << SEQ_STR_KIND_SHIFT;

pub type SeqInt = i64;

/// `seq_str_t`: pointer plus packed (kind, length) metadata.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SeqStr {
    pub ptr: *mut u8,
    pub meta: SeqInt,
}

/// `seq_bytes_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SeqBytes {
    pub ptr: *mut u8,
    pub len: SeqInt,
}

impl SeqBytes {
    pub const EMPTY: SeqBytes = SeqBytes { ptr: core::ptr::null_mut(), len: 0 };

    /// # Safety
    /// `ptr` must be valid for `len` bytes (or `len <= 0`).
    pub unsafe fn as_slice<'a>(&self) -> &'a [u8] {
        if self.ptr.is_null() || self.len <= 0 {
            &[]
        } else {
            unsafe { core::slice::from_raw_parts(self.ptr, self.len as usize) }
        }
    }

    /// Copy `data` into fresh atomic GC memory (port of re.cpp `convert`).
    pub fn from_slice_gc(data: &[u8]) -> SeqBytes {
        let n = data.len();
        let p = seq_alloc_atomic(n) as *mut u8;
        if n > 0 {
            unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), p, n) };
        }
        SeqBytes { ptr: p, len: n as SeqInt }
    }
}

/// `seq_time_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SeqTime {
    pub year: i16,
    pub yday: i16,
    pub sec: i8,
    pub min: i8,
    pub hour: i8,
    pub mday: i8,
    pub mon: i8,
    pub wday: i8,
    pub isdst: i8,
}

impl SeqStr {
    pub const EMPTY: SeqStr = SeqStr { ptr: core::ptr::null_mut(), meta: 0 };

    #[inline]
    pub fn len(&self) -> usize {
        ((self.meta as u64) & SEQ_STR_LEN_MASK) as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn kind(&self) -> u32 {
        (((self.meta as u64) & SEQ_STR_KIND_MASK) >> SEQ_STR_KIND_SHIFT) as u32
    }

    /// Code point at `i`.
    ///
    /// # Safety
    /// `ptr` must be valid for `len()` elements of the string's kind.
    pub unsafe fn char_at(&self, i: usize) -> u32 {
        unsafe {
            match self.kind() {
                SEQ_STR_KIND_ASCII | SEQ_STR_KIND_LATIN1 => *self.ptr.add(i) as u32,
                SEQ_STR_KIND_UCS2 => (self.ptr as *const u16).add(i).read_unaligned() as u32,
                _ => (self.ptr as *const u32).add(i).read_unaligned(),
            }
        }
    }

    /// UTF-8 encoding (port of `seq_str_t::encode`).
    pub fn encode(&self) -> Vec<u8> {
        let n = self.len();
        let mut out = Vec::with_capacity(n);
        if n == 0 || self.ptr.is_null() {
            return out;
        }
        for i in 0..n {
            let c = unsafe { self.char_at(i) };
            utf8_push(&mut out, c);
        }
        out
    }

    /// Lossy `String` view, handy for diagnostics.
    pub fn to_string_lossy(&self) -> String {
        String::from_utf8_lossy(&self.encode()).into_owned()
    }

    /// Port of `seq_str_t::operator==(const char *)`.
    pub fn eq_utf8(&self, other: &[u8]) -> bool {
        if self.kind() == SEQ_STR_KIND_ASCII && other.len() != self.len() {
            return false;
        }
        let mut pos = 0;
        let mut index = 0;
        while let Some(cp) = utf8_read(other, &mut pos) {
            if index == self.len() {
                return false;
            }
            if unsafe { self.char_at(index) } != cp {
                return false;
            }
            index += 1;
        }
        index == self.len()
    }
}

#[inline]
pub fn utf8_size(c: u32) -> usize {
    if c <= 0x7f {
        1
    } else if c <= 0x7ff {
        2
    } else if c <= 0xffff {
        3
    } else {
        4
    }
}

#[inline]
pub fn utf8_push(out: &mut Vec<u8>, c: u32) {
    if c <= 0x7f {
        out.push(c as u8);
    } else if c <= 0x7ff {
        out.push((0xc0 | (c >> 6)) as u8);
        out.push((0x80 | (c & 0x3f)) as u8);
    } else if c <= 0xffff {
        out.push((0xe0 | (c >> 12)) as u8);
        out.push((0x80 | ((c >> 6) & 0x3f)) as u8);
        out.push((0x80 | (c & 0x3f)) as u8);
    } else {
        out.push((0xf0 | (c >> 18)) as u8);
        out.push((0x80 | ((c >> 12) & 0x3f)) as u8);
        out.push((0x80 | ((c >> 6) & 0x3f)) as u8);
        out.push((0x80 | (c & 0x3f)) as u8);
    }
}

/// Strict UTF-8 decoder (port of `utf8_read`). Aborts on malformed input exactly like
/// the C++ runtime.
pub fn utf8_read(data: &[u8], pos: &mut usize) -> Option<u32> {
    match utf8_read_checked(data, pos) {
        Ok(v) => v,
        Err(()) => std::process::abort(),
    }
}

/// Non-aborting variant used by tests and callers that can recover.
pub fn utf8_read_checked(data: &[u8], pos: &mut usize) -> Result<Option<u32>, ()> {
    if *pos == data.len() {
        return Ok(None);
    }
    let byte = data[*pos];
    *pos += 1;
    if byte < 0x80 {
        return Ok(Some(byte as u32));
    }
    let (width, mut value) = match byte {
        0xC2..=0xDF => (1usize, (byte & 0x1F) as u32),
        0xE0..=0xEF => (2, (byte & 0x0F) as u32),
        0xF0..=0xF4 => (3, (byte & 0x07) as u32),
        _ => return Err(()),
    };
    if *pos + width > data.len() {
        return Err(());
    }
    for _ in 0..width {
        let c = data[*pos];
        *pos += 1;
        if c & 0xC0 != 0x80 {
            return Err(());
        }
        value = (value << 6) | (c & 0x3F) as u32;
    }
    if (width == 1 && value < 0x80)
        || (width == 2 && value < 0x800)
        || (width == 3 && value < 0x10000)
        || value > 0x10FFFF
        || (0xD800..=0xDFFF).contains(&value)
    {
        return Err(());
    }
    Ok(Some(value))
}

/// Build a Codon `str` from UTF-8 bytes, choosing the narrowest storage kind
/// (port of `string_conv`).
pub fn string_conv(s: &[u8]) -> SeqStr {
    if s.is_empty() {
        return SeqStr::EMPTY;
    }
    let mut length = 0usize;
    let mut maxchar = 0u32;
    let mut pos = 0;
    while let Some(cp) = utf8_read(s, &mut pos) {
        length += 1;
        maxchar = maxchar.max(cp);
    }
    let (kind, width) = if maxchar > 0xFFFF {
        (SEQ_STR_KIND_UCS4, 4usize)
    } else if maxchar > 0xFF {
        (SEQ_STR_KIND_UCS2, 2)
    } else if maxchar > 0x7F {
        (SEQ_STR_KIND_LATIN1, 1)
    } else {
        (SEQ_STR_KIND_ASCII, 1)
    };
    let out = seq_alloc_atomic(length * width) as *mut u8;
    pos = 0;
    let mut i = 0;
    while let Some(cp) = utf8_read(s, &mut pos) {
        unsafe {
            match kind {
                SEQ_STR_KIND_UCS4 => (out as *mut u32).add(i).write_unaligned(cp),
                SEQ_STR_KIND_UCS2 => (out as *mut u16).add(i).write_unaligned(cp as u16),
                _ => *out.add(i) = cp as u8,
            }
        }
        i += 1;
    }
    let meta = ((length as u64) & SEQ_STR_LEN_MASK) | ((kind as u64) << SEQ_STR_KIND_SHIFT);
    SeqStr { ptr: out, meta: meta as SeqInt }
}

/// Free a temporary produced by GC allocation, mirroring the C++ `seq_free` of scratch
/// buffers. Kept for API symmetry; Rust-side encodings use `Vec`.
pub fn free_gc(p: *mut u8) {
    seq_free(p.cast());
}
