//! Integration tests for typhon-llvm. Ports of the IR-level tests in
//! `test/cir/llvm/optimize.cpp` plus target/engine smoke tests.

use inkwell::context::Context;
use inkwell::memory_buffer::MemoryBuffer;
use inkwell::module::Module;
use inkwell::values::AsValueRef;
use typhon_common::Options;
use typhon_llvm::optimize::passes;
use typhon_llvm::test_support::*;

mod common;
use typhon_llvm::optimize;

fn parse<'c>(ctx: &'c Context, ir: &str) -> Module<'c> {
    let buf = MemoryBuffer::create_from_memory_range_copy(ir.as_bytes(), "test");
    ctx.create_module_from_ir(buf).unwrap_or_else(|e| panic!("parse error: {e}\n{ir}"))
}

fn opts() -> Options {
    let mut o = Options::default();
    o.debug = false;
    o.native = false;
    o.standalone = true;
    o
}

/// Port of `compileAndOptimizeIR`.
fn compile_and_optimize_ir<'c>(ctx: &'c Context, ir: &str) -> Module<'c> {
    let m = parse(ctx, ir);
    optimize(&m, &opts()).expect("optimize");
    m.verify().expect("verify");
    m
}

fn make_allocation_loop_ir(declaration: &str, allocation_use: &str) -> String {
    format!(
        "declare noalias ptr @seq_alloc_atomic(i64)\n{declaration}
define i64 @test(i64 %count) {{
entry:
  br label %header

header:
  %index = phi i64 [ 0, %entry ], [ %next, %body ]
  %total = phi i64 [ 0, %entry ], [ %updated, %body ]
  %done = icmp eq i64 %index, %count
  br i1 %done, label %exit, label %body

body:
  %allocation = call ptr @seq_alloc_atomic(i64 65536)
{allocation_use}
  %extended = zext i8 %value to i64
  %updated = add i64 %total, %extended
  %next = add i64 %index, 1
  br label %header

exit:
  ret i64 %total
}}
"
    )
}

// ------------------------------------------------------------------ targets

#[test]
fn host_target_machine() {
    let tm = typhon_llvm::host_target_machine(&Options::default(), false).expect("tm");
    assert!(!typhon_llvm::data_layout_string(&tm).is_empty());
    assert!(!typhon_llvm::host_triple().is_empty());
    assert_eq!(typhon_llvm::features_str(&["avx2".into(), "-sse4.1".into()]), "+avx2,-sse4.1");
    let (cpu, _features) = typhon_llvm::native::native_cpu_and_features(&typhon_llvm::host_triple()).expect("native");
    assert!(!cpu.is_empty());
    assert!(typhon_llvm::target_machine_for("nvptx64-nvidia-cuda", "sm_30", "+ptx42", "nvptx64", false).unwrap().is_some());
}

#[test]
fn native_attributes_and_x86_cpu_defaults() {
    use typhon_llvm::native::*;
    let t = Triple::parse("x86_64-apple-macosx10.15.0");
    assert!(t.is_x86_64() && t.is_os_darwin());
    let ctx = Context::create();
    let m = parse(&ctx, "target triple = \"aarch64-unknown-linux-gnu\"\ndefine void @f() { ret void }");
    apply_native_attributes(&m, "cortex-a72", "+neon");
    let ir = m.print_to_string().to_string();
    assert!(ir.contains("\"target-cpu\"=\"cortex-a72\""), "{ir}");
    assert!(ir.contains("\"frame-pointer\"=\"non-leaf\""), "{ir}");
}

// ------------------------------------------------------------------ individual passes

#[test]
fn allocation_remover_removes_unused_and_demotes_small() {
    let ctx = Context::create();
    let m = parse(
        &ctx,
        r#"
declare noalias ptr @seq_alloc(i64)
declare void @use(ptr nocapture) nofree
define void @unused() {
  %a = call ptr @seq_alloc(i64 64)
  store i64 1, ptr %a
  ret void
}
define void @small() {
  %a = call ptr @seq_alloc(i64 64)
  store i64 1, ptr %a
  call void @use(ptr %a)
  ret void
}
"#,
    );
    for name in ["unused", "small"] {
        let f = m.get_function(name).unwrap();
        assert!(passes::allocation_remover(f.as_value_ref()));
    }
    m.verify().unwrap();
    assert!(calls_in(&m, "unused").iter().all(|(n, _)| n != "seq_alloc"));
    assert!(calls_in(&m, "small").iter().all(|(n, _)| n != "seq_alloc"));
    // Passed to a nocapture/nofree call: aligned like the heap allocation (2 * ptr size).
    assert_eq!(alloca_alignments(&m, "small"), vec![16]);
}

#[test]
fn allocation_hoister_hoists_simple_case() {
    let ctx = Context::create();
    let m = parse(
        &ctx,
        &make_allocation_loop_ir(
            "declare i8 @read(ptr nocapture) nofree memory(read)\n",
            "  store i8 42, ptr %allocation\n  %value = call i8 @read(ptr %allocation)\n",
        ),
    );
    let f = m.get_function("test").unwrap();
    assert_eq!(count_fixed_allocations(&m, 65536, true), 1);
    assert!(passes::allocation_hoister(f.as_value_ref()));
    m.verify().unwrap();
    // The loop may not execute the allocation (exit edge), so it is cached lazily.
    assert_eq!(count_lazy_fixed_allocation_caches(&m, 65536), 0, "cache phi is created by mem2reg later");
    let ir = m.print_to_string().to_string();
    assert!(ir.contains("alloc_hoist.cache"), "{ir}");
}

#[test]
fn allocation_auto_free_inserts_free() {
    let ctx = Context::create();
    let m = parse(
        &ctx,
        r#"
declare noalias ptr @seq_alloc_atomic(i64)
define i8 @f() {
  %a = call ptr @seq_alloc_atomic(i64 4096)
  store i8 1, ptr %a
  %v = load i8, ptr %a
  ret i8 %v
}
"#,
    );
    let f = m.get_function("f").unwrap();
    assert!(passes::allocation_auto_free(f.as_value_ref()));
    m.verify().unwrap();
    assert!(calls_in(&m, "f").iter().any(|(n, _)| n == "seq_free"));
}

#[test]
fn openmp_thread_id_optimizer_reuses_ids() {
    let ctx = Context::create();
    let m = parse(
        &ctx,
        r#"
declare void @__kmpc_fork_call(ptr, i32, ptr, ...)
declare i32 @__kmpc_global_thread_num(ptr)
declare i32 @omp_get_thread_num()
define private void @outlined(ptr %gtid, ptr %btid, ptr %result) noinline {
  %global = call i32 @__kmpc_global_thread_num(ptr null)
  %team = call i32 @omp_get_thread_num()
  %total = add i32 %global, %team
  store i32 %total, ptr %result
  ret void
}
define void @launch(ptr %result) {
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr null, i32 1, ptr @outlined, ptr %result)
  ret void
}
"#,
    );
    let f = m.get_function("outlined").unwrap();
    assert!(passes::openmp_thread_id_optimizer(f.as_value_ref()));
    m.verify().unwrap();
    assert!(calls_in(&m, "outlined").is_empty());
}

#[test]
fn fast_math_and_debug_transformations() {
    let ctx = Context::create();
    let m = parse(
        &ctx,
        r#"
define internal double @f(double %a, double %b) {
  %c = fadd double %a, %b
  %d = tail call double @g(double %c)
  ret double %d
}
declare double @g(double)
"#,
    );
    passes::apply_fast_math_transformations(m.as_mut_ptr());
    passes::apply_debug_transformations(m.as_mut_ptr(), true);
    m.verify().unwrap();
    let ir = m.print_to_string().to_string();
    assert!(ir.contains("fadd fast double"), "{ir}");
    assert!(!ir.contains("tail call"), "{ir}");
    assert!(ir.contains("double @f(") && !ir.contains("internal double @f"), "linkage made external: {ir}");
    assert!(ir.contains("noinline"), "{ir}");
}

#[test]
fn coro_unroll_control_marks_and_cleans_suspending_loops() {
    let ctx = Context::create();
    let m = parse(
        &ctx,
        r#"
declare i8 @llvm.coro.suspend(token, i1)
define void @gen(i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %s = call i8 @llvm.coro.suspend(token none, i1 false)
  %next = add i64 %i, 1
  %c = icmp ult i64 %next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret void
}
"#,
    );
    let f = m.get_function("gen").unwrap();
    assert!(passes::coro_unroll_control(f.as_value_ref()));
    m.verify().unwrap();
    let hints = first_loop_hints(&m, "gen").expect("loop id");
    assert!(hints.contains(&"typhon.coro.unroll".to_string()));
    assert!(hints.contains(&"llvm.loop.unroll.disable".to_string()));
    // Second run is a no-op.
    assert!(!passes::coro_unroll_control(f.as_value_ref()));
    // After removing the suspend, cleanup removes our markers.
    for i in f.get_first_basic_block().unwrap().get_next_basic_block().unwrap().get_instructions() {
        if i.get_opcode() == inkwell::values::InstructionOpcode::Call {
            i.erase_from_basic_block();
            break;
        }
    }
    passes::coro_unroll_cleanup(m.as_mut_ptr());
    m.verify().unwrap();
    assert!(first_loop_hints(&m, "gen").is_none_or(|h| h.is_empty()));
}

// ------------------------------------------------------------------ ports of optimize.cpp tests

#[test]
fn preserves_user_unroll_metadata() {
    let ctx = Context::create();
    let m = compile_and_optimize_ir(
        &ctx,
        r#"
define i64 @user_unroll_guard(ptr %data, i64 %count) {
entry:
  br label %loop

loop:
  %index = phi i64 [ 0, %entry ], [ %next, %loop ]
  %total = phi i64 [ 0, %entry ], [ %updated, %loop ]
  %address = getelementptr i64, ptr %data, i64 %index
  %value = load volatile i64, ptr %address
  %updated = add i64 %total, %value
  %next = add i64 %index, 1
  %again = icmp ult i64 %next, %count
  br i1 %again, label %loop, label %exit, !llvm.loop !0

exit:
  ret i64 %updated
}

!0 = distinct !{!0, !1, !2}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = !{!"codon.test.preserve"}
"#,
    );
    let hints = first_loop_hints(&m, "user_unroll_guard").expect("loop id");
    assert!(hints.contains(&"llvm.loop.unroll.disable".to_string()), "{hints:?}");
    assert!(hints.contains(&"codon.test.preserve".to_string()), "{hints:?}");
}

#[test]
fn merges_equivalent_guarded_pointer_states() {
    for same_initial in [true, false] {
        for null_edge in [true, false] {
            let code = r#"
declare ptr @advance(ptr)
define i1 @test(ptr %initial, ptr %other, i64 %count) {
entry:
  br label %header
header:
  %left = phi ptr [ %initial, %entry ], [ %left.next, %latch ]
  %right = phi ptr [ INITIAL, %entry ], [ %right.next, %latch ]
  %index = phi i64 [ 0, %entry ], [ %next, %latch ]
  %finished = icmp eq i64 %index, %count
  br i1 %finished, label %exit, label %pull
pull:
  %done = icmp PREDICATE ptr %right, null
  br i1 %done, label %latch, label %resume
resume:
  %value = call ptr @advance(ptr %left)
  br label %latch
latch:
  %left.next = phi ptr [ %left, %pull ], [ %value, %resume ]
  %right.next = phi ptr [ null, %pull ], [ %value, %resume ]
  %next = add i64 %index, 1
  br label %header
exit:
  %equal = icmp eq ptr %left, %right
  ret i1 %equal
}
"#
            .replace("INITIAL", if same_initial { "%initial" } else { "%other" })
            .replace("PREDICATE", if null_edge { "eq" } else { "ne" });
            let ctx = Context::create();
            let m = compile_and_optimize_ir(&ctx, &code);
            let always_equal = constant_returns(&m, "test").is_some_and(|r| !r.is_empty() && r.iter().all(|&v| v == 1));
            assert_eq!(same_initial && null_edge, always_equal, "same_initial={same_initial} null_edge={null_edge}");
        }
    }
}

#[test]
fn respects_resume_noinline_attributes() {
    for callee_noinline in [false, true] {
        for call_noinline in [false, true] {
            let code = format!(
                "define internal void @values.resume(ptr nocapture %frame) \"typhon.generator\" {} {{
entry:
  %value = load i64, ptr %frame
  %updated = add i64 %value, 1
  store i64 %updated, ptr %frame
  ret void
}}

define i64 @test(i64 %initial) {{
entry:
  %frame = alloca i64
  store i64 %initial, ptr %frame
  call void @values.resume(ptr %frame){}
  %result = load i64, ptr %frame
  ret i64 %result
}}
",
                if callee_noinline { "noinline" } else { "" },
                if call_noinline { " noinline" } else { "" }
            );
            let ctx = Context::create();
            let m = compile_and_optimize_ir(&ctx, &code);
            let has_resume = calls_in(&m, "test").iter().any(|(n, _)| n.starts_with("values.resume"));
            assert_eq!(callee_noinline || call_noinline, has_resume, "callee={callee_noinline} call={call_noinline}");
        }
    }
}

#[test]
fn keeps_openmp_queries_without_stable_context() {
    let ctx = Context::create();
    let m = compile_and_optimize_ir(
        &ctx,
        r#"
declare void @__kmpc_fork_call(ptr, i32, ptr, ...)
declare ptr @__kmpc_omp_task_alloc(ptr, i32, i32, i64, i64, ptr, ptr)
declare i32 @__kmpc_global_thread_num(ptr)
declare i32 @omp_get_thread_num()
declare void @escape(ptr)

define private void @escaped(ptr %gtid, ptr %btid, ptr %result) noinline {
  %global = call i32 @__kmpc_global_thread_num(ptr null)
  %team = call i32 @omp_get_thread_num()
  %total = add i32 %global, %team
  store i32 %total, ptr %result
  ret void
}

define private i32 @untied(i32 %gtid, ptr %task) noinline {
  %global = call i32 @__kmpc_global_thread_num(ptr null)
  ret i32 %global
}

define void @launch(ptr %result) {
  call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr null, i32 1,
                                                ptr @escaped, ptr %result)
  call void @escape(ptr @escaped)
  %task = call ptr @__kmpc_omp_task_alloc(ptr null, i32 0, i32 0, i64 64,
                                        i64 0, ptr @untied, ptr null)
  call void @escape(ptr %task)
  ret void
}
"#,
    );
    let mut queries = 0;
    for f in m.get_functions() {
        let name = f.get_name().to_string_lossy().into_owned();
        queries += calls_in(&m, &name)
            .iter()
            .filter(|(n, _)| n == "__kmpc_global_thread_num" || n == "omp_get_thread_num")
            .count();
    }
    assert_eq!(3, queries);
}

#[test]
fn preserves_required_heap_allocation_alignment() {
    for alignment in [8u32, 32] {
        let code = format!(
            r#"
declare noalias ptr @seq_alloc_atomic(i64)
declare i64 @collect()
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)

define void @test(ptr %source, ptr %output) {{
entry:
  %allocation = call ptr @seq_alloc_atomic(i64 32)
  call void @llvm.memcpy.p0.p0.i64(ptr align {alignment} %allocation, ptr align 1 %source, i64 32, i1 false)
  %count = call i64 @collect()
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %output, ptr align 1 %allocation, i64 %count, i1 false)
  ret void
}}
"#
        );
        let ctx = Context::create();
        let m = compile_and_optimize_ir(&ctx, &code);
        assert_eq!(count_fixed_allocations(&m, 32, false), if alignment > 8 { 1 } else { 0 }, "align {alignment}");
    }
}

#[test]
fn uses_allocation_alignment_contracts() {
    for alignment in [1u64, 2, 4, 8, 16, 32, 64] {
        let ret_align = if alignment > 16 { format!("align {alignment} ") } else { String::new() };
        let code = format!(
            r#"declare noalias {ret_align}ptr @seq_alloc_atomic(i64)
declare i64 @collect()
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)

define void @test(ptr %source, ptr %output) {{
entry:
  %allocation = call ptr @seq_alloc_atomic(i64 33)
  call void @llvm.memcpy.p0.p0.i64(ptr align {alignment} %allocation, ptr align 1 %source, i64 33, i1 false)
  %count = call i64 @collect()
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %output, ptr align 1 %allocation, i64 %count, i1 false)
  ret void
}}
"#
        );
        let ctx = Context::create();
        let m = compile_and_optimize_ir(&ctx, &code);
        assert_eq!(0, count_fixed_allocations(&m, 33, false), "align {alignment}");
        assert_eq!(alloca_alignments(&m, "test"), vec![alignment], "align {alignment}");
    }
}

#[test]
fn aligns_demoted_allocations_for_garbage_collection() {
    for allocator in ["seq_alloc", "seq_alloc_atomic", "seq_alloc_uncollectable", "seq_alloc_atomic_uncollectable"] {
        let code = format!(
            r#"declare noalias ptr @{allocator}(i64)
declare i64 @collect()
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1 immarg)

define i8 @test(ptr %source) {{
entry:
  %allocation = call ptr @{allocator}(i64 17)
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %allocation, ptr align 1 %source, i64 17, i1 false)
  %index = call i64 @collect()
  %selected = getelementptr i8, ptr %allocation, i64 %index
  %result = load i8, ptr %selected, align 1
  ret i8 %result
}}
"#
        );
        let ctx = Context::create();
        let m = compile_and_optimize_ir(&ctx, &code);
        let atomic = allocator.contains("atomic");
        assert_eq!(alloca_alignments(&m, "test"), vec![if atomic { 1 } else { 8 }], "{allocator}");
        assert!(calls_in(&m, "test").iter().all(|(n, _)| n != allocator), "{allocator}");
    }
}

#[test]
fn hoists_and_releases_freed_loop_allocation() {
    let ctx = Context::create();
    let m = compile_and_optimize_ir(
        &ctx,
        &make_allocation_loop_ir(
            "declare {} @seq_free(ptr) nounwind\ndeclare i8 @read(ptr nocapture) nofree memory(read)\n",
            "  store i8 42, ptr %allocation\n  %value = call i8 @read(ptr %allocation)\n  call {} @seq_free(ptr %allocation)\n",
        ),
    );
    let calls = calls_in(&m, "test");
    let releases: Vec<_> = calls.iter().filter(|(n, _)| n == "seq_free").collect();
    assert_eq!(releases.len(), 1, "{calls:?}");
    assert!(releases.iter().all(|(_, in_loop)| !in_loop), "{calls:?}");
    assert_eq!(1, count_fixed_allocations(&m, 65536, false));
    assert!(count_fixed_allocations(&m, 65536, true) == 0 || count_lazy_fixed_allocation_caches(&m, 65536) == 1);
}

#[test]
fn does_not_hoist_mixed_owner_free() {
    let ctx = Context::create();
    let m = compile_and_optimize_ir(
        &ctx,
        &make_allocation_loop_ir(
            "declare void @seq_free(ptr) nounwind\ndeclare ptr @other()\ndeclare i1 @choose()\n\
             declare i8 @read(ptr nocapture) nofree memory(read)\n",
            "  store i8 42, ptr %allocation\n  %value = call i8 @read(ptr %allocation)\n  %other = call ptr @other()\n  \
             %choose = call i1 @choose()\n  %owner = select i1 %choose, ptr %allocation, ptr %other\n  \
             call void @seq_free(ptr %owner)\n",
        ),
    );
    assert!(count_fixed_allocations(&m, 65536, true) > 0);
    assert_eq!(0, count_lazy_fixed_allocation_caches(&m, 65536));
}

#[test]
fn does_not_hoist_escaping_freed_allocation() {
    let ctx = Context::create();
    let m = compile_and_optimize_ir(
        &ctx,
        &make_allocation_loop_ir(
            "@escaped = global ptr null\ndeclare void @seq_free(ptr) nounwind\n\
             declare i8 @read(ptr nocapture) nofree memory(read)\n",
            "  store ptr %allocation, ptr @escaped\n  %value = call i8 @read(ptr %allocation)\n  \
             call void @seq_free(ptr %allocation)\n",
        ),
    );
    assert!(count_fixed_allocations(&m, 65536, true) > 0);
    assert_eq!(0, count_lazy_fixed_allocation_caches(&m, 65536));
}

fn assert_not_hoisted(decl: &str, body: &str) {
    let ctx = Context::create();
    let m = compile_and_optimize_ir(&ctx, &make_allocation_loop_ir(decl, body));
    assert!(count_fixed_allocations(&m, 65536, true) > 0, "{}", m.print_to_string());
    assert_eq!(0, count_lazy_fixed_allocation_caches(&m, 65536));
}

#[test]
fn does_not_hoist_readonly_call_without_nocapture() {
    assert_not_hoisted(
        "declare i8 @read_and_capture(ptr) nofree memory(read)\n",
        "  %value = call i8 @read_and_capture(ptr %allocation)\n",
    );
}

#[test]
fn does_not_hoist_call_without_nofree() {
    assert_not_hoisted(
        "declare i8 @read_and_maybe_free(ptr nocapture)\n",
        "  %value = call i8 @read_and_maybe_free(ptr %allocation)\n",
    );
}

#[test]
fn does_not_hoist_pointer_returned_through_aggregate() {
    assert_not_hoisted(
        "@escaped = global ptr null\ndeclare { ptr, i8 } @return_and_read(ptr) nofree memory(read)\n",
        "  %result = call { ptr, i8 } @return_and_read(ptr %allocation)\n  %returned = extractvalue { ptr, i8 } %result, 0\n  \
         store ptr %returned, ptr @escaped\n  %value = extractvalue { ptr, i8 } %result, 1\n",
    );
}

#[test]
fn does_not_hoist_reallocated_pointer() {
    assert_not_hoisted(
        "declare ptr @seq_realloc(ptr, i64, i64)\ndeclare i8 @read(ptr nocapture) nofree memory(read)\n",
        "  %resized = call ptr @seq_realloc(ptr %allocation, i64 131072, i64 65536)\n  %value = call i8 @read(ptr %resized)\n",
    );
}

// ------------------------------------------------------------------ full optimize / gpu

#[test]
fn optimize_debug_and_release() {
    for debug in [true, false] {
        let ctx = Context::create();
        let m = parse(
            &ctx,
            r#"
define i64 @sum(i64 %n) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc2, %loop ]
  %acc2 = add i64 %acc, %i
  %next = add i64 %i, 1
  %c = icmp ult i64 %next, %n
  br i1 %c, label %loop, label %exit
exit:
  ret i64 %acc2
}
"#,
        );
        let mut o = opts();
        o.debug = debug;
        o.native = true;
        m.set_triple(&inkwell::targets::TargetTriple::create(&typhon_llvm::host_triple()));
        optimize(&m, &o).expect("optimize");
        m.verify().unwrap();
        if !debug {
            assert_eq!(loop_count(&m, "sum"), 0, "closed form expected:\n{}", m.print_to_string());
        }
    }
}

#[test]
fn gpu_patches_ptx_without_kernels() {
    let ctx = Context::create();
    let m = parse(
        &ctx,
        r#"
declare ptr @__typhon_ptx__()
define ptr @f() {
  %p = call ptr @__typhon_ptx__()
  ret ptr %p
}
"#,
    );
    let g = typhon_llvm::gpu::prepare_gpu_module(&m, &opts()).unwrap();
    assert!(g.is_none());
    assert!(m.get_function("__typhon_ptx__").is_none());
    m.verify().unwrap();
}

#[test]
fn gpu_generates_ptx_for_kernel() {
    let dir = std::env::temp_dir().join(format!("typhon-libdevice-{}.ll", std::process::id()));
    std::fs::write(&dir, "; empty libdevice\n").unwrap();
    let ctx = Context::create();
    let m = parse(
        &ctx,
        r#"
declare ptr @__typhon_ptx__()
declare double @llvm.sqrt.f64(double)
define {} @kern(ptr %p) "kernel" {
  %v = load double, ptr %p
  %s = call double @llvm.sqrt.f64(double %v)
  store double %s, ptr %p
  ret {} zeroinitializer
}
define ptr @host() {
  %p = call ptr @__typhon_ptx__()
  ret ptr %p
}
"#,
    );
    let mut o = opts();
    o.libdevice = dir.to_string_lossy().into_owned();
    optimize(&m, &o).expect("optimize with gpu");
    let ptx = m.get_global(".ptx").expect("embedded ptx");
    let init = ptx.get_initializer().unwrap().into_array_value();
    let s = String::from_utf8_lossy(init.as_const_string().unwrap()).into_owned();
    assert!(s.contains(".entry kern"), "{s}");
    assert!(s.contains("__nv_sqrt"), "{s}");
    let _ = std::fs::remove_file(dir);
}

// ------------------------------------------------------------------ AOT

#[test]
fn aot_object_links_with_runtime_and_libc() {
    let ctx = Context::create();
    let m = parse(
        &ctx,
        "declare i64 @host_seven()\ndeclare i64 @labs(i64)\ndeclare ptr @seq_alloc(i64)\n\
         define i64 @plus() {\n  %a = call i64 @host_seven()\n  %b = call i64 @labs(i64 -35)\n  \
         %p = call ptr @seq_alloc(i64 16)\n  store i64 %b, ptr %p\n  %c = load i64, ptr %p\n  \
         %s = add i64 %a, %c\n  ret i64 %s\n}\n",
    );
    optimize(&m, &opts()).expect("optimize");
    let driver = "#include <stdio.h>\n#include <stdint.h>\nvoid seq_init(int); int64_t plus(void);\n\
                  int64_t host_seven(void) { return 7; }\n\
                  int main(void) { seq_init(0); printf(\"%lld\\n\", (long long)plus()); return 0; }\n";
    let out = common::build_and_run(&m, &opts(), driver, "aot_plus", &[]);
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "42");
}
