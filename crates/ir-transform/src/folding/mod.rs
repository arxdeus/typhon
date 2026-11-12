
pub mod const_fold;
pub mod const_prop;
pub mod folding;
pub mod rule;

pub use const_fold::FoldingPass;
pub use const_prop::ConstPropPass;
pub use folding::FoldingPassGroup;
