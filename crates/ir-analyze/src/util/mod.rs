//! Port of `codon/cir/util/{inlining,outlining,side_effect}` plus helpers.
pub mod bitset;
pub mod inlining;
pub mod outlining;
pub mod walk;

/// Port of `codon/cir/util/side_effect.{h,cpp}` (lives in typhon-ir, re-exported).
pub mod side_effect {
    pub use typhon_ir::util::side_effect::*;
}
