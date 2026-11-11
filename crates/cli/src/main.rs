//! `typhon` CLI: port of `codon/app/main.cpp` (modes `run`, `build`, `doc`, `version`).
//! `jit` and `jupyter` modes are out of scope for typhon.

mod args;

use std::collections::{HashMap, HashSet};
use std::path::Path;

use args::{BuildKind, Cli};
use typhon_common::diag::{MessageGroupPos, compilation_error, compilation_warning};
use typhon_common::{CompilerError, ParserErrors, logger};
use typhon_compiler::Compiler;
use typhon_compiler::typhon_llvm::inkwell::context::Context;
use typhon_compiler::typhon_llvm::{EngineConfig, LlvmVisitor};

const SUPPORTED_EXTENSIONS: [&str; 3] = [".codon", ".py", ".seq"];

fn make_output_filename(filename: &str, extension: &str) -> String {
    for ext in SUPPORTED_EXTENSIONS {
        if let Some(stem) = filename.strip_suffix(ext) {
            return format!("{stem}{extension}");
        }
    }
    format!("{filename}{extension}")
}

fn fatal(msg: &str) -> ! {
    compilation_error(msg, "", 0, 0, 0, -1, true, MessageGroupPos::None);
    std::process::exit(1)
}

/// `display(ParserErrorInfo)`: print each message once, grouped.
fn display(errs: &ParserErrors) {
    let mut seen = HashSet::new();
    for group in &errs.errors {
        let n = group.trace.len();
        let mut i = 0;
        for msg in &group.trace {
            let key = format!("{}:{}:{}:{}", msg.file(), msg.loc.line, msg.loc.col, msg.msg);
            if !seen.insert(key) {
                continue;
            }
            let pos = if i == 0 {
                MessageGroupPos::Head
            } else if i == n - 1 {
                MessageGroupPos::Last
            } else {
                MessageGroupPos::Mid
            };
            i += 1;
            compilation_error(&msg.msg, msg.file(), msg.loc.line, msg.loc.col, msg.loc.len, msg.error_code, false, pos);
        }
    }
}

fn report(err: &CompilerError) {
    match err {
        CompilerError::Parser(p) => display(p),
        other => compilation_error(&other.to_string(), "", 0, 0, 0, -1, false, MessageGroupPos::None),
    }
}

fn init_log_flags(log: &str) {
    logger().parse(log);
    if let Ok(d) = std::env::var("CODON_DEBUG") {
        logger().parse(&d);
    }
}

fn flush_c_stdio() {
    // SAFETY: fflush(NULL) flushes all C streams written by the runtime.
    unsafe {
        libc_fflush();
    }
}

unsafe fn libc_fflush() {
    unsafe extern "C" {
        fn fflush(f: *mut std::ffi::c_void) -> i32;
    }
    unsafe {
        fflush(std::ptr::null_mut());
    }
}

/// `processSource`: parse + typecheck + translate + IR passes. Returns the compiler or
/// `None` after printing errors.
fn process_source(cli: &mut Cli, standalone: bool) -> Option<Compiler> {
    let test_mode = std::env::var("TYPHON_TEST_MODE").is_ok_and(|v| v == "1");
    cli.options.standalone = standalone || test_mode;
    cli.options.pyext = cli.build_kind == BuildKind::PyExtension;
    if test_mode {
        cli.options.test = true;
    }
    init_log_flags(&cli.options.log);

    let mut defmap: HashMap<String, String> = HashMap::new();
    for define in &cli.options.defines {
        match define.find('=') {
            Some(eq) if eq > 0 => {
                let (name, value) = (&define[..eq], &define[eq + 1..]);
                if defmap.contains_key(name) {
                    compilation_warning(&format!("ignoring duplicate definition: {define}"), "", 0, 0, 0, -1, false, MessageGroupPos::None);
                    continue;
                }
                defmap.insert(name.to_string(), value.to_string());
            }
            _ => {
                compilation_warning(&format!("ignoring malformed definition: {define}"), "", 0, 0, 0, -1, false, MessageGroupPos::None);
            }
        }
    }

    let mut compiler = Compiler::new(cli.options.clone(), None);
    for plugin in cli.options.plugins.clone() {
        if let Err(e) = compiler.load(&plugin) {
            compilation_error(&e.to_string(), "", 0, 0, 0, -1, false, MessageGroupPos::None);
            return None;
        }
    }

    let test_flags: i32 = std::env::var("CODON_TEST_FLAGS").ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    if let Err(e) = compiler.parse_file(&cli.input, test_flags, &defmap) {
        report(&e);
        return None;
    }
    if test_mode {
        typhon_compiler::test_passes::register_test_passes(compiler.pass_manager());
    }
    let mut t = typhon_common::Timer::new("compile");
    compiler.run_passes();
    t.log();
    Some(compiler)
}

fn codegen<'ctx>(compiler: &Compiler, ctx: &'ctx Context) -> LlvmVisitor<'ctx> {
    match compiler.codegen(ctx) {
        Ok(v) => v,
        Err(e) => {
            report(&e);
            std::process::exit(1);
        }
    }
}

fn run_mode(argv0: &str, args: &[String]) -> i32 {
    let mut cli = args::parse(argv0, args, false);
    let Some(compiler) = process_source(&mut cli, false) else { return 1 };
    let ctx = Context::create();
    let mut v = codegen(&compiler, &ctx);
    let rt = match typhon_compiler::runtime::load_runtime() {
        Ok(rt) => rt,
        Err(e) => fatal(&e.to_string()),
    };
    // PORT-NOTE: C++ loads `-l` libraries via DynamicLibrary::LoadLibraryPermanently; same here.
    let mut prog_args = vec![compiler.input().to_string()];
    prog_args.extend(cli.prog_args.iter().cloned());
    let config = EngineConfig { gc_roots: rt.gc_roots, ..Default::default() };
    let r = v.run(&prog_args, &cli.libs, config);
    flush_c_stdio();
    match r {
        Ok(_) => 0,
        Err(e) => {
            report(&e);
            1
        }
    }
}

fn build_mode(argv0: &str, exe_argv0: &str, args: &[String]) -> i32 {
    let mut cli = args::parse(argv0, args, true);
    let Some(compiler) = process_source(&mut cli, true) else { return 1 };
    let mut kind = cli.build_kind;
    if cli.pie {
        if kind != BuildKind::Detect && kind != BuildKind::Executable {
            fatal("--pie requires executable output");
        }
        kind = BuildKind::Executable;
    }
    if cli.output.is_empty() && compiler.input() == "-" {
        fatal("output file must be specified when reading from stdin");
    }
    let extension = match kind {
        BuildKind::Llvm => ".ll",
        BuildKind::Bitcode => ".bc",
        BuildKind::Object | BuildKind::PyExtension => ".o",
        BuildKind::Assembly => ".s",
        BuildKind::Library => {
            if cfg!(windows) {
                ".dll"
            } else if cfg!(target_os = "macos") {
                ".dylib"
            } else {
                ".so"
            }
        }
        BuildKind::Executable | BuildKind::Detect => "",
        BuildKind::Cir => ".cir",
    };
    let filename = if cli.output.is_empty() { make_output_filename(compiler.input(), extension) } else { cli.output.clone() };
    if kind == BuildKind::Cir {
        let s = typhon_ir::util::format::format(&compiler.module().borrow(), typhon_ir::NodeRef::Module);
        if let Err(e) = std::fs::write(&filename, s) {
            fatal(&e.to_string());
        }
        return 0;
    }
    let ctx = Context::create();
    let mut v = codegen(&compiler, &ctx);
    let r = match kind {
        BuildKind::Llvm => v.write_to_ll_file(&filename, true),
        BuildKind::Bitcode => v.write_to_bitcode_file(&filename),
        BuildKind::Object => v.write_to_object_file(&filename, false, false),
        BuildKind::Assembly => v.write_to_object_file(&filename, false, true),
        BuildKind::Executable => {
            let lf = if cli.pie { format!("{} -pie", cli.lflags) } else { cli.lflags.clone() };
            v.write_to_executable(&filename, exe_argv0, false, &cli.libs, &lf)
        }
        BuildKind::Library => v.write_to_executable(&filename, exe_argv0, true, &cli.libs, &cli.lflags),
        BuildKind::PyExtension => {
            let cache = compiler.cache().borrow();
            match cache.py_module.clone() {
                Some(pm) => {
                    let name = if cli.py_module.is_empty() {
                        Path::new(compiler.input()).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
                    } else {
                        cli.py_module.clone()
                    };
                    pm.borrow_mut().name = name;
                    let pm = pm.borrow();
                    v.write_to_python_extension(&compiler.module().borrow(), &pm, &filename)
                }
                None => Err(CompilerError::Io("no Python extension module was generated".into())),
            }
        }
        BuildKind::Detect => v.compile(&filename, exe_argv0, &cli.libs, &cli.lflags),
        BuildKind::Cir => Ok(()),
    };
    match r {
        Ok(()) => 0,
        Err(e) => {
            report(&e);
            1
        }
    }
}

fn doc_mode(argv0: &str, args: &[String]) -> i32 {
    let input = args.iter().find(|a| !a.starts_with('-') || *a == "-").cloned().unwrap_or_else(|| "-".into());
    let mut files = Vec::new();
    if !args.is_empty() {
        let p = Path::new(&input);
        if !p.exists() {
            compilation_error(&format!("'{input}' does not exist"), "", 0, 0, 0, -1, false, MessageGroupPos::None);
        }
        if p.is_file() {
            files.push(input.clone());
        } else if p.is_dir() {
            collect_files(p, &mut files);
        }
    }
    files.sort();
    let options = typhon_common::Options::with_argv0(argv0);
    let compiler = Compiler::new(options, None);
    match compiler.docgen(&files) {
        Ok(s) => {
            println!("{s}");
            0
        }
        Err(e) => {
            report(&e);
            1
        }
    }
}

fn collect_files(dir: &Path, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect_files(&p, out);
        } else if p.is_file() && !p.to_string_lossy().ends_with("__init_test__.codon") {
            out.push(p.to_string_lossy().into_owned());
        }
    }
}

fn show_commands_and_exit() -> ! {
    fatal("Available commands: typhon <run|build|doc>")
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() < 2 {
        show_commands_and_exit();
    }
    let exe = argv[0].clone();
    let mode = argv[1].as_str();
    let rest = &argv[2..];
    let argv0 = format!("{exe} {mode}");
    let code = match mode {
        "run" => run_mode(&argv0, rest),
        "build" => build_mode(&argv0, &exe, rest),
        "doc" => doc_mode(&exe, rest),
        "version" | "--version" | "-version" => {
            println!("{}", typhon_common::config::CODON_VERSION);
            0
        }
        "jit" | "jupyter" => fatal(&format!("'{mode}' mode is not supported in typhon (only run, build and doc)")),
        "--help" | "-help" | "-h" => {
            println!("USAGE: typhon <mode> [options]\n\nMODES:\n\n  run   - run a program interactively\n  build - build a program\n  doc   - generate program documentation\n");
            0
        }
        _ => show_commands_and_exit(),
    };
    flush_c_stdio();
    std::process::exit(code);
}
