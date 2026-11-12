//! AOT execution helper for LLVM-level tests: write an LLVM module to an object file,
//! link it with a small C driver and the static runtime (`libtyphonrt.a`, found like
//! `typhon build` finds it: `TYPHON_LIB_DIR`, else `target/<profile>/`), run the binary.
#![allow(dead_code)]

use std::path::PathBuf;
use std::process::{Command, Output};

use inkwell::module::Module;
use inkwell::targets::FileType;
use typhon_common::Options;

/// `target/<profile>/typhon`: the argv0 used to locate `libtyphonrt.a`.
fn typhon_argv0() -> PathBuf {
    // tests run from target/<profile>/deps/<test-binary>
    std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("typhon")
}

/// Compile `module` + `driver_c` into an executable and run it with `args`.
/// The module's own `main` (if any) is renamed to `typhon_main` so the driver can call
/// it to initialize the runtime.
pub fn build_and_run(module: &Module, options: &Options, driver_c: &str, tag: &str, args: &[&str]) -> Output {
    let dir = tempfile::tempdir().expect("tempdir");
    if let Some(f) = module.get_function("main") {
        f.as_global_value().set_name("typhon_main");
    }
    let tm = typhon_llvm::host_target_machine(options, false).expect("target machine");
    module.set_triple(&tm.get_triple());
    module.set_data_layout(&tm.get_target_data().get_data_layout());
    if let Err(e) = module.verify() {
        panic!("invalid module: {}\n{}", e.to_string(), module.print_to_string().to_string());
    }
    let obj = dir.path().join(format!("{tag}.o"));
    tm.write_to_file(module, FileType::Object, &obj).expect("write object");
    let drv = dir.path().join(format!("{tag}_driver.c"));
    std::fs::write(&drv, driver_c).unwrap();
    let exe = dir.path().join(tag);
    let rt = typhon_llvm::llvisitor::runtime_link_args(&typhon_argv0().to_string_lossy()).unwrap_or_else(|e| {
        panic!("{e}: build the runtime first (`cargo build [--release] -p typhon-runtime`)")
    });
    let st = Command::new("cc")
        .arg("-Wno-unused-command-line-argument")
        .arg(&drv)
        .arg(&obj)
        .args(&rt)
        .arg("-lpthread")
        .arg("-o")
        .arg(&exe)
        .status()
        .expect("cc");
    assert!(st.success(), "link failed for {tag}");
    let out = Command::new(&exe).args(args).output().expect("run test exe");
    drop(dir);
    out
}

/// C declarations matching the runtime ABI used by the drivers: a minimal Codon
/// exception object `{ TypeInfo*, ExceptionHeader }` (see `typhonrt::rtti::TypeInfo`
/// and `typhonrt::exc::ExceptionHeader`).
pub const C_PRELUDE: &str = r#"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { char *ptr; int64_t len; } seq_str;
typedef struct {
  int64_t id, n_mro; const int64_t *mro, *offsets, *sorted_mro;
  int64_t n_slots; const void *slots; int64_t n_methods; const void *methods;
  _Bool rtti; seq_str raw_name, nice_name, base_name;
  int64_t n_params; const int64_t *params; int64_t size;
  _Bool reference, inline_value; int64_t exception_offset, system_exit_offset;
} type_info;
typedef struct { seq_str msg, func, file; int64_t line, col; void *python_type, *cause; } exc_header;
typedef struct { const type_info *info; exc_header hdr; } exc_obj;
static seq_str S(const char *s) { seq_str r = {(char *)s, (int64_t)strlen(s)}; return r; }
static void *fake_exception(const char *ty, const char *msg) {
  type_info *t = calloc(1, sizeof(type_info));
  t->id = 1; t->raw_name = t->nice_name = t->base_name = S(ty);
  t->reference = 1; t->exception_offset = 8; t->system_exit_offset = -1;
  exc_obj *o = calloc(1, sizeof(exc_obj));
  o->info = t; o->hdr.msg = S(msg); o->hdr.func = S("boom"); o->hdr.file = S("test.py");
  o->hdr.line = 3; o->hdr.col = 5;
  return o;
}
"#;
