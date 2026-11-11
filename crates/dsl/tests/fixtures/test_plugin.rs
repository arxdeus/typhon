//! Test plugin (Rust counterpart of the C++ `TEST_PLUGIN_COMPILER` fixture).

use std::rc::Rc;
use typhon_dsl::{BlockKeyword, Dsl, ExprKeyword};
use typhon_ir_transform::manager::PassManager;
use typhon_llvm::PassStage;

#[derive(Default)]
pub struct TestDsl;

impl Dsl for TestDsl {
    fn add_ir_passes(&self, _pm: &mut PassManager, _debug: bool) {}
    fn pipeline_text(&self, stage: PassStage, debug: bool) -> Option<String> {
        (matches!(stage, PassStage::OptimizerLast) && !debug).then(|| "instcombine".to_string())
    }
    fn expr_keywords(&self) -> Vec<ExprKeyword> {
        vec![ExprKeyword { keyword: "test_expr".into(), callback: Rc::new(|_, s| Ok(Some(s))) }]
    }
    fn block_keywords(&self) -> Vec<BlockKeyword> {
        vec![BlockKeyword { keyword: "test_block".into(), callback: Rc::new(|_, _| Ok(None)), has_expr: true }]
    }
}

typhon_dsl::export_plugin!(TestDsl);
