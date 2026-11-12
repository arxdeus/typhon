//! .
//!
//! `PassManager` lives in `typhon-ir-transform`; passes access it through the
//! [`PassContext`] trait object passed to `run` (replaces `Pass::manager`).

use std::any::Any;

use typhon_common::Options;

use crate::module::Module;
use crate::util::operator::{Operator, run_operator};

pub trait AnalysisResult: Any {
    fn as_any(&self) -> &dyn Any;
}

/// What a pass may ask its manager (`getAnalysisResult`, `getOptions`).
pub trait PassContext {
    /// `manager->getAnalysisResult(key)`.
    fn analysis_result(&self, key: &str) -> Option<&dyn AnalysisResult>;
    /// `manager->getOptions()`.
    fn options(&self) -> Option<&Options>;
}

impl dyn PassContext + '_ {
    /// `getAnalysisResult<T>(key)`.
    pub fn get_analysis<T: AnalysisResult>(&self, key: &str) -> Option<&T> {
        self.analysis_result(key).and_then(|r| r.as_any().downcast_ref::<T>())
    }
}

/// A context with no analyses and no options (pass run without a manager).
#[derive(Clone, Copy, Debug, Default)]
pub struct NullPassContext;

impl PassContext for NullPassContext {
    fn analysis_result(&self, _key: &str) -> Option<&dyn AnalysisResult> {
        None
    }
    fn options(&self) -> Option<&Options> {
        None
    }
}

pub trait Pass {
    /// `getKey()`.
    fn key(&self) -> String;
    /// `shouldRun(module)`.
    fn should_run(&self, _m: &Module) -> bool {
        true
    }
    /// `run(module)`.
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext);
    /// `shouldRepeat(num)`.
    fn should_repeat(&self, _num: i32) -> bool {
        false
    }
    /// Sub-passes (`PassGroup`) so that managers can register nested passes
    /// (`setManager` recursion).
    fn sub_passes(&mut self) -> Option<&mut Vec<Box<dyn Pass>>> {
        None
    }
}

pub struct PassGroup {
    pub key: String,
    pub repeat: i32,
    pub passes: Vec<Box<dyn Pass>>,
}

impl PassGroup {
    pub fn new(key: impl Into<String>, repeat: i32, passes: Vec<Box<dyn Pass>>) -> Self {
        Self { key: key.into(), repeat, passes }
    }
    pub fn push_back(&mut self, p: Box<dyn Pass>) {
        self.passes.push(p);
    }
    pub fn repeat(&self) -> i32 {
        self.repeat
    }
    pub fn set_repeat(&mut self, r: i32) {
        self.repeat = r;
    }
}

impl Pass for PassGroup {
    fn key(&self) -> String {
        self.key.clone()
    }
    fn run(&mut self, m: &mut Module, ctx: &dyn PassContext) {
        for p in self.passes.iter_mut() {
            p.run(m, ctx);
        }
    }
    fn should_repeat(&self, num: i32) -> bool {
        num < self.repeat
    }
    fn sub_passes(&mut self) -> Option<&mut Vec<Box<dyn Pass>>> {
        Some(&mut self.passes)
    }
}

/// resets the operator and processes the module.
pub fn run_operator_pass<O: Operator + ?Sized>(op: &mut O, m: &mut Module) {
    run_operator(op, m);
}

pub trait Analysis {
    fn key(&self) -> String;
    fn run(&mut self, m: &Module, ctx: &dyn PassContext) -> Box<dyn AnalysisResult>;
}
