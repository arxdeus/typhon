//! Port of `test/cir/test.h` (`CIRCoreTest` fixture).
#![allow(dead_code)]

use typhon_ir::util::cloning::CloneVisitor;
use typhon_ir::{Module, reset_id};

/// `SetUp()`: resets ids and creates an empty module without a parser cache.
pub fn setup() -> (Module, CloneVisitor) {
    reset_id();
    (Module::new("test"), CloneVisitor::new(true))
}

pub fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}
