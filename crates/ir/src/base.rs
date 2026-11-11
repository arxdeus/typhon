//! Port of `codon/cir/base.{h,cpp}` (node data common to every IR node).

use crate::attribute::AttributeMap;

/// Common node state (`Node`): name, attribute map and `ReplaceableNodeBase::replaceable`.
#[derive(Clone, Debug)]
pub struct NodeBase {
    pub name: String,
    pub attributes: AttributeMap,
    pub replaceable: bool,
}

impl NodeBase {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), attributes: AttributeMap::default(), replaceable: true }
    }
}

impl Default for NodeBase {
    fn default() -> Self {
        Self::new("")
    }
}
