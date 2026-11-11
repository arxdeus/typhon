//! Port of `codon/cir/transform/parallel/schedule.{h,cpp}`.

use crate::ids::{Id, ValueId};
use crate::module::Module;

/// Port of `transform::parallel::OMPSched`.
#[derive(Clone, Debug, PartialEq)]
pub struct OMPSched {
    pub code: i32,
    pub dynamic: bool,
    pub threads: Option<ValueId>,
    pub chunk: Option<ValueId>,
    pub ordered: bool,
    pub collapse: i64,
    pub gpu: bool,
}

impl Default for OMPSched {
    fn default() -> Self {
        Self {
            code: schedule_code("static", false, false, false),
            dynamic: false,
            threads: None,
            chunk: None,
            ordered: false,
            collapse: 0,
            gpu: false,
        }
    }
}

/// Port of anonymous `getScheduleCode`.
pub fn schedule_code(schedule: &str, chunked: bool, ordered: bool, monotonic: bool) -> i32 {
    let modifier: i32 = if monotonic { 1 << 29 } else { 1 << 30 };
    match schedule {
        "static" => {
            if chunked {
                if ordered { 65 } else { 33 }
            } else if ordered {
                66
            } else {
                34
            }
        }
        "dynamic" => (if ordered { 67 } else { 35 }) | modifier,
        "guided" => (if ordered { 68 } else { 36 }) | modifier,
        "runtime" => (if ordered { 69 } else { 37 }) | modifier,
        "auto" => (if ordered { 70 } else { 38 }) | modifier,
        _ => schedule_code("static", false, false, false),
    }
}

fn null_if_neg(m: &Module, v: Option<ValueId>) -> Option<ValueId> {
    let v = v?;
    if let Some(c) = m.as_int_const(v) {
        if c.val <= 0 {
            return None;
        }
    }
    Some(v)
}

impl OMPSched {
    /// `OMPSched(int code, bool dynamic, Value *threads, Value *chunk, bool ordered, int64_t collapse, bool gpu)`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: &Module,
        code: i32,
        dynamic: bool,
        threads: Option<ValueId>,
        chunk: Option<ValueId>,
        ordered: bool,
        collapse: i64,
        gpu: bool,
    ) -> Self {
        let code = if code < 0 { schedule_code("static", false, false, false) } else { code };
        Self {
            code,
            dynamic,
            threads: null_if_neg(m, threads),
            chunk: null_if_neg(m, chunk),
            ordered,
            collapse,
            gpu,
        }
    }

    /// `OMPSched(const std::string &schedule, ...)`.
    pub fn from_schedule(
        m: &Module,
        schedule: &str,
        threads: Option<ValueId>,
        chunk: Option<ValueId>,
        ordered: bool,
        collapse: i64,
        gpu: bool,
    ) -> Self {
        let chunked = null_if_neg(m, chunk).is_some();
        Self::new(
            m,
            schedule_code(schedule, chunked, ordered, false),
            schedule != "static" || ordered,
            threads,
            chunk,
            ordered,
            collapse,
            gpu,
        )
    }

    pub fn used_values(&self) -> Vec<ValueId> {
        self.threads.iter().chain(self.chunk.iter()).copied().collect()
    }

    pub fn replace_used_value(&mut self, m: &Module, id: Id, new: Option<ValueId>) -> i32 {
        let mut count = 0;
        if let Some(t) = self.threads {
            if m.value_uid(t) == id {
                self.threads = new;
                count += 1;
            }
        }
        if let Some(c) = self.chunk {
            if m.value_uid(c) == id {
                self.chunk = new;
                count += 1;
            }
        }
        count
    }
}
