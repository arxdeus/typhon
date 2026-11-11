//! Port of codon/cir/transform/pythonic/format.{h,cpp}

use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::util::irtools;
use typhon_ir::util::operator::{Operator, OperatorState};
use typhon_ir::{FuncId, Module, TypeId, TypeKind, ValueId, magic};

use crate::lowering::{operator_pass, operator_state};

pub(crate) const FORMAT_FLAG_ALT: i64 = 1;
pub(crate) const FORMAT_FLAG_ZERO: i64 = 2;
pub(crate) const FORMAT_FLAG_SPACE: i64 = 8;
pub(crate) const FORMAT_FLAG_PLUS: i64 = 16;

/// Parsed format spec (`[[fill]align][sign][z][#][0][width][grouping][.precision][type]`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Directive {
    pub flags: i64,
    pub width: i64,
    pub precision: i64,
    pub conversion: i64,
    pub alignment: i64,
    pub fill: i64,
    pub grouping: i64,
    pub locale: bool,
    pub coerce_negative_zero: bool,
    pub sign_minus: bool,
}

impl Default for Directive {
    fn default() -> Self {
        Self {
            flags: 0,
            width: 0,
            precision: -1,
            conversion: 0,
            alignment: 0,
            fill: 0x20,
            grouping: 0,
            locale: false,
            coerce_negative_zero: false,
            sign_minus: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FieldAccess {
    Member(String),
    IntegerElement(i64),
    StringElement(String),
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Part {
    pub literal: String,
    pub index: i64,
    pub accesses: Vec<FieldAccess>,
    pub field_conversion: i64,
    pub directive: Directive,
}

impl Part {
    fn literal(s: String) -> Self {
        Self { literal: s, index: -1, ..Default::default() }
    }
    pub fn is_field(&self) -> bool {
        self.index >= 0
    }
}

fn is_alignment(c: u32) -> bool {
    matches!(char::from_u32(c), Some('<' | '>' | '^' | '='))
}

fn decode_codepoint(s: &[u8], pos: &mut usize) -> Option<u32> {
    let first = *s.get(*pos)?;
    *pos += 1;
    if first < 0x80 {
        return Some(first as u32);
    }
    let (n, mut value, min) = if first & 0xe0 == 0xc0 {
        (1, (first & 0x1f) as u32, 0x80)
    } else if first & 0xf0 == 0xe0 {
        (2, (first & 0x0f) as u32, 0x800)
    } else if first & 0xf8 == 0xf0 {
        (3, (first & 0x07) as u32, 0x10000)
    } else {
        return None;
    };
    for _ in 0..n {
        let next = *s.get(*pos)?;
        *pos += 1;
        if next & 0xc0 != 0x80 {
            return None;
        }
        value = (value << 6) | (next & 0x3f) as u32;
    }
    if value < min || value > 0x10ffff || (0xd800..=0xdfff).contains(&value) {
        return None;
    }
    Some(value)
}

fn append_digit(value: &mut i64, digit: i64) -> bool {
    if *value > (i64::MAX - digit) / 10 {
        return false;
    }
    *value = *value * 10 + digit;
    true
}

/// `parseDirective(spec, directive)`.
pub fn parse_directive(spec: &str) -> Option<Directive> {
    let s = spec.as_bytes();
    let mut d = Directive::default();
    let mut pos = 0usize;
    if !s.is_empty() {
        let mut first_end = 0usize;
        let first = decode_codepoint(s, &mut first_end)?;
        if first_end < s.len() && is_alignment(s[first_end] as u32) {
            d.fill = first as i64;
            d.alignment = s[first_end] as i64;
            pos = first_end + 1;
        } else if is_alignment(first) {
            d.alignment = first as i64;
            pos = first_end;
        }
    }
    let take = |pos: &mut usize, c: u8| {
        if *pos < s.len() && s[*pos] == c {
            *pos += 1;
            true
        } else {
            false
        }
    };
    if take(&mut pos, b'-') {
        d.sign_minus = true;
    } else if take(&mut pos, b' ') {
        d.flags |= FORMAT_FLAG_SPACE;
    } else if take(&mut pos, b'+') {
        d.flags |= FORMAT_FLAG_PLUS;
    }
    d.coerce_negative_zero = take(&mut pos, b'z');
    if take(&mut pos, b'#') {
        d.flags |= FORMAT_FLAG_ALT;
    }
    if take(&mut pos, b'0') {
        d.flags |= FORMAT_FLAG_ZERO;
    }
    while pos < s.len() && s[pos].is_ascii_digit() {
        if !append_digit(&mut d.width, (s[pos] - b'0') as i64) {
            return None;
        }
        pos += 1;
    }
    if pos < s.len() && (s[pos] == b',' || s[pos] == b'_') {
        d.grouping = s[pos] as i64;
        pos += 1;
    }
    if take(&mut pos, b'.') {
        d.precision = 0;
        while pos < s.len() && s[pos].is_ascii_digit() {
            if !append_digit(&mut d.precision, (s[pos] - b'0') as i64) {
                return None;
            }
            pos += 1;
        }
    }
    if pos == s.len() {
        return Some(d);
    }
    let conv = s[pos];
    pos += 1;
    if pos != s.len() {
        return None;
    }
    if conv == b'n' {
        d.locale = true;
        return Some(d);
    }
    if !b"bsc%diuoxXeEfFgG".contains(&conv) {
        return None;
    }
    d.conversion = conv as i64;
    Some(d)
}

fn parse_index(text: &[u8]) -> Option<i64> {
    if text.is_empty() {
        return None;
    }
    let mut idx = 0i64;
    for &c in text {
        if !c.is_ascii_digit() || !append_digit(&mut idx, (c - b'0') as i64) {
            return None;
        }
    }
    Some(idx)
}

fn sub(t: &[u8], a: usize, b: usize) -> String {
    String::from_utf8_lossy(&t[a..b]).into_owned()
}

fn parse_field(text: &[u8], automatic: &mut i64, mode: &mut i32, part: &mut Part) -> bool {
    let mut pos = 0usize;
    while pos < text.len() && !matches!(text[pos], b'.' | b'[' | b'!' | b':') {
        pos += 1;
    }
    let index_text = &text[..pos];
    if index_text.is_empty() {
        if *mode == 2 {
            return false;
        }
        *mode = 1;
        part.index = *automatic;
        *automatic += 1;
    } else {
        if *mode == 1 {
            return false;
        }
        let Some(i) = parse_index(index_text) else { return false };
        part.index = i;
        *mode = 2;
    }
    while pos < text.len() && (text[pos] == b'.' || text[pos] == b'[') {
        if text[pos] == b'.' {
            pos += 1;
            let start = pos;
            while pos < text.len() && !matches!(text[pos], b'.' | b'[' | b'!' | b':' | b']') {
                pos += 1;
            }
            if pos == start {
                return false;
            }
            part.accesses.push(FieldAccess::Member(sub(text, start, pos)));
            continue;
        }
        pos += 1;
        let start = pos;
        let Some(off) = text[start..].iter().position(|&c| c == b']') else { return false };
        let end = start + off;
        if end == start {
            return false;
        }
        let key = &text[start..end];
        match parse_index(key) {
            Some(i) => part.accesses.push(FieldAccess::IntegerElement(i)),
            None => part.accesses.push(FieldAccess::StringElement(sub(text, start, end))),
        }
        pos = end + 1;
    }
    if pos < text.len() && text[pos] == b'!' {
        pos += 1;
        if pos >= text.len() {
            return false;
        }
        let c = text[pos];
        pos += 1;
        if !matches!(c, b's' | b'r' | b'a') {
            return false;
        }
        part.field_conversion = c as i64;
    }
    if pos == text.len() {
        return true;
    }
    if text[pos] != b':' {
        return false;
    }
    pos += 1;
    let spec = &text[pos..];
    if spec.contains(&b'{') || spec.contains(&b'}') {
        return false;
    }
    let Ok(spec) = std::str::from_utf8(spec) else { return false };
    match parse_directive(spec) {
        Some(d) => {
            part.directive = d;
            true
        }
        None => false,
    }
}

/// `parseFormat(format)`.
pub fn parse_format(format: &str) -> Option<Vec<Part>> {
    let f = format.as_bytes();
    let mut parts = Vec::new();
    let mut literal: Vec<u8> = Vec::new();
    let mut automatic = 0i64;
    let mut mode = 0i32;
    let mut pos = 0usize;
    let flush = |literal: &mut Vec<u8>, parts: &mut Vec<Part>| {
        if !literal.is_empty() {
            parts.push(Part::literal(String::from_utf8_lossy(literal).into_owned()));
            literal.clear();
        }
    };
    while pos < f.len() {
        let c = f[pos];
        if c == b'{' {
            if pos + 1 < f.len() && f[pos + 1] == b'{' {
                literal.push(b'{');
                pos += 2;
                continue;
            }
            flush(&mut literal, &mut parts);
            let end = pos + 1 + f[pos + 1..].iter().position(|&c| c == b'}')?;
            let field = &f[pos + 1..end];
            if field.contains(&b'{') {
                return None;
            }
            let mut part = Part { index: -1, ..Default::default() };
            if !parse_field(field, &mut automatic, &mut mode, &mut part) {
                return None;
            }
            parts.push(part);
            pos = end + 1;
            continue;
        }
        if c == b'}' {
            if pos + 1 >= f.len() || f[pos + 1] != b'}' {
                return None;
            }
            literal.push(b'}');
            pos += 2;
            continue;
        }
        literal.push(c);
        pos += 1;
    }
    flush(&mut literal, &mut parts);
    Some(parts)
}

fn supports_parsed_formatting(m: &mut Module, t: TypeId) -> bool {
    let f = m.get_float_type();
    let s = m.get_string_type();
    matches!(m.type_kind(t), TypeKind::Int { .. }) || m.type_is(t, f) || m.type_is(t, s)
}

fn parsed_helper_args(m: &mut Module, value: ValueId, part: &Part) -> Vec<ValueId> {
    let d = &part.directive;
    vec![
        value,
        m.get_int(part.field_conversion),
        m.get_int(d.conversion),
        m.get_int(d.flags),
        m.get_int(d.width),
        m.get_int(d.precision),
        m.get_int(d.alignment),
        m.get_int(d.fill),
        m.get_int(d.grouping),
        m.get_bool(d.locale),
        m.get_bool(d.coerce_negative_zero),
        m.get_bool(d.sign_minus),
    ]
}

fn get_parsed_helper(m: &mut Module, value_type: TypeId) -> Option<FuncId> {
    let mut types = vec![value_type];
    let it = m.get_int_type();
    let bt = m.get_bool_type();
    types.extend(std::iter::repeat_n(it, 8));
    types.extend(std::iter::repeat_n(bt, 3));
    m.get_or_realize_func("_format_parsed", &types, &[], "std.internal.format")
}

#[derive(Default, Clone)]
struct ResolvedPart {
    ty: Option<TypeId>,
    element_accessors: Vec<Option<FuncId>>,
}

fn resolve_part(m: &mut Module, part: &Part, element_types: &[TypeId]) -> Option<ResolvedPart> {
    if part.index < 0 || part.index as usize >= element_types.len() {
        return None;
    }
    let mut ty = element_types[part.index as usize];
    let mut accessors = Vec::with_capacity(part.accesses.len());
    for access in &part.accesses {
        if let FieldAccess::Member(name) = access {
            if !m.type_kind(ty).is_membered() {
                return None;
            }
            ty = m.member_type(ty, name)?;
            accessors.push(None);
            continue;
        }
        if let FieldAccess::IntegerElement(i) = access {
            if m.type_kind(ty).is_record() && m.type_name(ty) == "Tuple" {
                ty = m.member_type(ty, &format!("item{}", i + 1))?;
                accessors.push(None);
                continue;
            }
        }
        let key_type = match access {
            FieldAccess::IntegerElement(_) => m.get_int_type(),
            _ => m.get_string_type(),
        };
        let getitem = m.get_or_realize_method(ty, magic::GETITEM, &[ty, key_type], &[])?;
        ty = irtools::return_type(m, getitem)?;
        accessors.push(Some(getitem));
    }
    Some(ResolvedPart { ty: Some(ty), element_accessors: accessors })
}

fn apply_accesses(m: &mut Module, mut value: ValueId, part: &Part, resolved: &ResolvedPart) -> ValueId {
    for (i, access) in part.accesses.iter().enumerate() {
        value = match (access, resolved.element_accessors[i]) {
            (FieldAccess::Member(name), _) => m.nr_extract(value, name.clone()),
            (FieldAccess::IntegerElement(idx), None) => irtools::tuple_get(m, value, *idx as usize),
            (FieldAccess::IntegerElement(idx), Some(f)) => {
                let k = m.get_int(*idx);
                irtools::call(m, f, vec![value, k])
            }
            (FieldAccess::StringElement(s), Some(f)) => {
                let k = m.get_string(s.clone());
                irtools::call(m, f, vec![value, k])
            }
            (FieldAccess::StringElement(_), None) => unreachable!("string element without accessor"),
        };
    }
    value
}

/// Pass to lower literal `str.format` / `__format__` calls to pre-parsed helpers.
#[derive(Default)]
pub struct FormattingOptimization {
    state: OperatorState,
}

impl FormattingOptimization {
    pub const KEY: &'static str = "core-pythonic-formatting-opt";
    pub fn new() -> Self {
        Self::default()
    }
}

impl Operator for FormattingOptimization {
    operator_state!();
    fn handle_call_instr(&mut self, m: &mut Module, v: ValueId) {
        let c = m.as_call_instr(v).unwrap().clone();
        let Some(func) = irtools::get_func(m, c.callee) else { return };
        let fd = m.func(func).unwrap();
        let Some(parent_type) = fd.parent_type else { return };
        let name = fd.unmangled_name.clone();
        let st = m.get_string_type();
        let src = m.src_info(v.into());

        if name == "format" && m.type_is(parent_type, st) && c.args.len() == 2 {
            let Some(fmt) = m.as_string_const(c.args[0]).map(|s| s.val.clone()) else { return };
            let tt = m.value_type_mut(c.args[1]);
            let TypeKind::Record { fields } = m.type_kind(tt).clone() else { return };
            let Some(parsed) = parse_format(&fmt) else { return };
            let element_types: Vec<TypeId> = fields.iter().map(|f| f.ty).collect();
            let mut resolved = vec![ResolvedPart::default(); parsed.len()];
            let mut helpers = Vec::with_capacity(parsed.len());
            for (i, part) in parsed.iter().enumerate() {
                if !part.is_field() {
                    helpers.push(None);
                    continue;
                }
                let Some(r) = resolve_part(m, part, &element_types) else { return };
                let rt = r.ty.unwrap();
                if !supports_parsed_formatting(m, rt) {
                    return;
                }
                resolved[i] = r;
                let Some(h) = get_parsed_helper(m, rt) else { return };
                helpers.push(Some(h));
            }
            let mut cat = None;
            if parsed.len() > 1 {
                let pt = m.get_tuple_type(vec![st; parsed.len()]);
                cat = m.get_or_realize_method(st, "cat", &[pt], &[]);
                if cat.is_none() {
                    return;
                }
            }
            let series = m.nr_series_flow();
            let mut cv = CloneVisitor::new(false);
            let cl = cv.clone_value(m, c.args[1]);
            let parent = self.parent_func(m);
            let tuple_var = irtools::make_var(m, cl, series, parent, false);
            let mut pieces = Vec::with_capacity(parsed.len());
            for (i, part) in parsed.iter().enumerate() {
                if !part.is_field() {
                    pieces.push(m.get_string(part.literal.clone()));
                    continue;
                }
                let tv = m.nr_var_value(tuple_var);
                let val = irtools::tuple_get(m, tv, part.index as usize);
                let val = apply_accesses(m, val, part, &resolved[i]);
                let args = parsed_helper_args(m, val, part);
                pieces.push(irtools::call(m, helpers[i].unwrap(), args));
            }
            let result = match pieces.len() {
                0 => m.get_string(""),
                1 => pieces[0],
                _ => {
                    let Some(t) = irtools::make_tuple(m, &pieces) else { return };
                    irtools::call(m, cat.unwrap(), vec![t])
                }
            };
            let rep = m.nr_flow_instr(series, result);
            m.set_src_info(rep.into(), src);
            m.replace_all_value(v, rep);
            return;
        }

        if name == "__format__" && c.args.len() == 2 {
            let ft = m.value_type_mut(c.args[0]);
            if !supports_parsed_formatting(m, ft) {
                return;
            }
            let Some(spec) = m.as_string_const(c.args[1]).map(|s| s.val.clone()) else { return };
            let Some(directive) = parse_directive(&spec) else { return };
            let part = Part { index: 0, directive, ..Default::default() };
            let Some(helper) = get_parsed_helper(m, ft) else { return };
            let mut cv = CloneVisitor::new(false);
            let cl = cv.clone_value(m, c.args[0]);
            let args = parsed_helper_args(m, cl, &part);
            let rep = irtools::call(m, helper, args);
            m.set_src_info(rep.into(), src);
            m.replace_all_value(v, rep);
        }
    }
}

operator_pass!(FormattingOptimization);
