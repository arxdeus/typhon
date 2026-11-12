
use once_cell::sync::Lazy;
use typhon_common::mangle::func;

/// Ordered from strongest to weakest guarantee.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SideEffectStatus {
    Pure = 0,
    NoSideEffect,
    NoCapture,
    #[default]
    Unknown,
}

pub static NON_PURE_ATTR: Lazy<String> = Lazy::new(|| func("std.internal.attributes", "nonpure"));
pub static PURE_ATTR: Lazy<String> = Lazy::new(|| func("", "pure"));
pub static NO_SIDE_EFFECT_ATTR: Lazy<String> = Lazy::new(|| func("std.internal.attributes", "no_side_effect"));
pub static NO_CAPTURE_ATTR: Lazy<String> = Lazy::new(|| func("std.internal.attributes", "nocapture"));
pub static DERIVES_ATTR: Lazy<String> = Lazy::new(|| func("", "derives"));
pub static SELF_CAPTURES_ATTR: Lazy<String> = Lazy::new(|| func("std.internal.attributes", "self_captures"));
