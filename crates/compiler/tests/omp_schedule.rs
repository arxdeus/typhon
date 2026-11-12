//! Regression: a `@par(schedule=..., ordered=...)` loop whose ForStmt is re-visited by the
//! typechecker (e.g. the body calls a not-yet-realized `@omp.ordered` function) must keep
//! its schedule. Previously the second `transform_for_decorator` rebuilt the
//! `for_par(...)` call from the already-transformed one, losing every Literal argument,
//! so the loop silently became `schedule(static)` without `ordered`.

use std::collections::HashMap;

use typhon_common::Options;
use typhon_compiler::Compiler;
use typhon_ir::ValueKind;

#[test]
fn par_schedule_survives_typecheck_revisit() {
    let src = r#"
import openmp as omp
@omp.ordered
def f(A, i):
    A.append(i)
def t1(N: int = 1000):
    A = []
    @par(schedule='dynamic', chunk_size=1, num_threads=2, ordered=True)
    for i in range(N):
        f(A, i)
    print(len(A), A == list(range(N)))
t1()
"#;
    let mut c = Compiler::new(Options::with_argv0("build/typhon_test"), None);
    c.parse_code("omp_sched.py", src, 0, 0, &HashMap::new()).expect("compile");
    let m = c.module().borrow();
    let scheds: Vec<_> = m
        .values()
        .into_iter()
        .filter_map(|v| match m.kind(v) {
            ValueKind::ForFlow(f) => f.schedule.clone(),
            ValueKind::ImperativeForFlow(f) => f.schedule.clone(),
            _ => None,
        })
        .collect();
    assert_eq!(scheds.len(), 1, "expected one parallel loop");
    assert!(scheds[0].dynamic, "schedule lost: {:?}", scheds[0]);
    assert!(scheds[0].ordered, "ordered lost: {:?}", scheds[0]);
}
