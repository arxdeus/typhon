//! Command-line option parsing compatible with the LLVM `cl::opt` style flags of
//! `codon/app/main.cpp` and `codon/compiler/options.cpp` (single-dash long options,
//! `-o file`, `-D<name>=<val>`, `-numerics=py`, positional input, program args after it).

use typhon_common::Options;
use typhon_common::options::GlobalCtorMode;

/// Build kinds of `build` mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildKind {
    Llvm,
    Bitcode,
    Object,
    Assembly,
    Executable,
    Library,
    PyExtension,
    Detect,
    Cir,
}

/// Parsed command line for `run` / `build`.
#[derive(Debug)]
pub struct Cli {
    pub options: Options,
    pub input: String,
    pub prog_args: Vec<String>,
    pub libs: Vec<String>,
    pub lflags: String,
    pub pie: bool,
    pub build_kind: BuildKind,
    pub output: String,
    pub py_module: String,
    pub relocation_model: String,
}

pub fn usage_error(msg: &str) -> ! {
    eprintln!("typhon: {msg}");
    std::process::exit(1);
}

/// Split `-name=value` / `--name=value` / `-name`.
fn split(arg: &str) -> (String, Option<String>) {
    let a = arg.trim_start_matches('-');
    match a.split_once('=') {
        Some((n, v)) => (n.to_string(), Some(v.to_string())),
        None => (a.to_string(), None),
    }
}

fn parse_bool(v: Option<String>) -> bool {
    !matches!(v.as_deref(), Some("false" | "0" | "False" | "FALSE"))
}

/// Parse args (excluding argv[0] and the mode). `build` enables build-only flags.
pub fn parse(argv0: &str, args: &[String], build: bool) -> Cli {
    let mut o = Options::with_argv0(argv0);
    let mut cli = Cli {
        options: Options::default(),
        input: String::new(),
        prog_args: Vec::new(),
        libs: Vec::new(),
        lflags: String::new(),
        pie: false,
        build_kind: BuildKind::Detect,
        output: String::new(),
        py_module: String::new(),
        relocation_model: String::new(),
    };
    let mut march = String::new();
    let mut i = 0;
    let mut input_seen = false;
    while i < args.len() {
        let arg = &args[i];
        i += 1;
        if input_seen {
            cli.prog_args.push(arg.clone());
            continue;
        }
        if arg == "--" {
            cli.prog_args.extend(args[i..].iter().cloned());
            break;
        }
        if !arg.starts_with('-') || arg == "-" {
            cli.input = arg.clone();
            input_seen = true;
            continue;
        }
        // `-Dname=value` / `-D name=value` (cl::Prefix)
        if let Some(rest) = arg.strip_prefix("-D") {
            if rest.is_empty() {
                let Some(v) = args.get(i) else { usage_error("-D requires a value") };
                i += 1;
                o.defines.push(v.clone());
            } else {
                o.defines.push(rest.to_string());
            }
            continue;
        }
        // `-lfoo` (prefix form) or `-l foo`
        if let Some(rest) = arg.strip_prefix("-l")
            && !arg.starts_with("-log")
            && !arg.starts_with("-lib")
            && !arg.starts_with("-linker")
            && !arg.starts_with("-llvm")
        {
            if rest.is_empty() {
                let Some(v) = args.get(i) else { usage_error("-l requires a value") };
                i += 1;
                cli.libs.push(v.clone());
            } else {
                cli.libs.push(rest.trim_start_matches('=').to_string());
            }
            continue;
        }
        let (name, val) = split(arg);
        let mut take = |val: Option<String>| -> String {
            if let Some(v) = val {
                return v;
            }
            let Some(v) = args.get(i) else { usage_error(&format!("option '-{name}' requires a value")) };
            i += 1;
            v.clone()
        };
        match name.as_str() {
            "debug" => o.debug = true,
            "release" => o.debug = false,
            "disable-exceptions" => o.noexc = parse_bool(val),
            "auto-free" => o.autofree = parse_bool(val),
            "fast-math" => o.fastmath = parse_bool(val),
            "disable-native" => o.native = !parse_bool(val),
            "auto-python" => o.autopy = parse_bool(val),
            "disable-opt" => {
                let v = take(val);
                o.disabled.push(v)
            }
            "plugin" => {
                let v = take(val);
                o.plugins.push(v)
            }
            "log" => o.log = take(val),
            "numerics" => match take(val).as_str() {
                "c" => o.pynum = false,
                "py" => o.pynum = true,
                other => usage_error(&format!("Cannot find option named '{other}' for -numerics")),
            },
            "libdevice" => o.libdevice = take(val),
            "gpu-name" => o.gpu_name = take(val),
            "gpu-features" => o.gpu_feat = take(val),
            "ptx" => o.gpu_output = take(val),
            "unordered-dict" => o.unordereddict = parse_bool(val),
            "global-ctor" => {
                o.ctor = match take(val).as_str() {
                    "no" => GlobalCtorMode::No,
                    "yes" => GlobalCtorMode::Yes,
                    "auto" => GlobalCtorMode::Auto,
                    other => usage_error(&format!("Cannot find option named '{other}' for -global-ctor")),
                }
            }
            "march" => march = take(val),
            "mcpu" => o.mcpu = take(val),
            "mattr" => {
                let v = take(val);
                o.mattrs.extend(v.split(',').filter(|s| !s.is_empty()).map(String::from))
            }
            "relocation-model" => cli.relocation_model = take(val),
            "linker-flags" if build => cli.lflags = take(val),
            "pie" if build => cli.pie = parse_bool(val),
            "o" if build => cli.output = take(val),
            "module" if build => cli.py_module = take(val),
            "llvm" if build => cli.build_kind = BuildKind::Llvm,
            "bc" if build => cli.build_kind = BuildKind::Bitcode,
            "obj" if build => cli.build_kind = BuildKind::Object,
            "asm" | "assembly" if build => cli.build_kind = BuildKind::Assembly,
            "exe" if build => cli.build_kind = BuildKind::Executable,
            "lib" if build => cli.build_kind = BuildKind::Library,
            "pyext" if build => cli.build_kind = BuildKind::PyExtension,
            "cir" if build => cli.build_kind = BuildKind::Cir,
            "detect" if build => cli.build_kind = BuildKind::Detect,
            "std" | "std-version" => {
                // PORT-NOTE: accepted for compatibility; no effect (as in Codon).
                let _ = val;
            }
            "help" | "h" => {
                print_help(build);
                std::process::exit(0);
            }
            "version" => {
                println!("{}", typhon_common::config::CODON_VERSION);
                std::process::exit(0);
            }
            _ => usage_error(&format!("Unknown command line argument '{arg}'.  Try: 'typhon --help'")),
        }
    }
    if cli.input.is_empty() {
        cli.input = "-".into();
    }
    // options.cpp: `-march=native` -> host CPU and features.
    if march == "native" {
        march.clear();
        if o.mcpu.is_empty() {
            let host = typhon_llvm::target::host_cpu_name();
            if !host.is_empty() && host != "generic" {
                o.mcpu = host;
            }
        }
        if o.mattrs.is_empty() {
            o.mattrs = typhon_llvm::target::host_cpu_features().split(',').filter(|s| !s.is_empty()).map(String::from).collect();
        }
    }
    o.march = march;
    cli.options = o;
    cli
}

pub fn print_help(build: bool) {
    println!("USAGE: typhon {} [options] <input file>{}", if build { "build" } else { "run" }, if build { "" } else { " <program arguments>..." });
    println!(
        "\nOPTIONS:\n  -debug / -release        optimization mode (default: debug)\n  -numerics=c|py           numerical semantics (default: py)\n  -D<name>=<value>         static definitions\n  -disable-opt=<pass>      disable an IR pass\n  -plugin=<dir>            load a plugin\n  -log=<streams>           enable log streams\n  -fast-math, -disable-exceptions, -disable-native, -unordered-dict, -auto-python\n  -march/-mcpu/-mattr      target selection\n  -libdevice/-gpu-name/-gpu-features/-ptx  GPU options\n  -l<lib>                  load/link library"
    );
    if build {
        println!("  -llvm|-bc|-obj|-asm|-exe|-lib|-pyext|-cir   output kind\n  -o <file>                output file\n  -linker-flags=<flags>    pass flags to the linker\n  -pie                     position-independent executable\n  -module=<name>           Python extension module name");
    }
}
