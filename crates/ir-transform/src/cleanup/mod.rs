//! Port of codon/cir/transform/cleanup/* (canonicalization, DCE, devirtualization,
//! global demotion, physical replacement).

pub mod canonical;
pub mod dead_code;
pub mod devirtualize;
pub mod global_demote;
pub mod replacer;

pub use canonical::CanonicalizationPass;
pub use dead_code::DeadCodeCleanupPass;
pub use devirtualize::DevirtualizationPass;
pub use global_demote::GlobalDemotionPass;
pub use replacer::ReplaceCleanupPass;
