//! Port of codon/cir/transform/lowering/*

pub mod async_for;
pub mod await_;
pub mod imperative;
pub mod pipeline;
pub mod return_;
pub mod throw;

pub use async_for::AsyncForLowering;
pub use await_::AwaitLowering;
pub use imperative::ImperativeForFlowLowering;
pub use pipeline::PipelineLowering;
pub use return_::MissingReturnLowering;
pub use throw::ThrowLowering;

/// Implements `OperatorState` plumbing and `Pass` (C++ `OperatorPass`) for a struct
/// with a `state: OperatorState` field and a `KEY` constant.
macro_rules! operator_pass {
    ($t:ty) => {
        impl typhon_ir::pass::Pass for $t {
            fn key(&self) -> String {
                <$t>::KEY.to_string()
            }
            fn run(&mut self, m: &mut typhon_ir::Module, _ctx: &dyn typhon_ir::pass::PassContext) {
                typhon_ir::pass::run_operator_pass(self, m);
            }
        }
    };
}
pub(crate) use operator_pass;

/// Common operator state accessors.
macro_rules! operator_state {
    () => {
        fn state(&self) -> &typhon_ir::util::operator::OperatorState {
            &self.state
        }
        fn state_mut(&mut self) -> &mut typhon_ir::util::operator::OperatorState {
            &mut self.state
        }
    };
}
pub(crate) use operator_state;
