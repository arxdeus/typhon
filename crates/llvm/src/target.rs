//! Port of the target-machine half of `codon/cir/llvm/optimize.{h,cpp}`
//! (`getTargetMachine`, `getFeaturesStr`).

use std::sync::Once;

use inkwell::OptimizationLevel;
use inkwell::module::Module;
use inkwell::targets::{CodeModel, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple};
use typhon_common::{CompilerError, Options, Result};

use crate::ffi;

static INIT: Once = Once::new();

/// Initialize every LLVM target compiled into the toolchain (the C++ code calls
/// `InitializeAllTargets()` etc. in `LLVMVisitor`'s constructor). Idempotent.
pub fn initialize_targets() {
    INIT.call_once(|| {
        let cfg = InitializationConfig {
            asm_parser: true,
            asm_printer: true,
            base: true,
            disassembler: true,
            info: true,
            machine_code: true,
        };
        Target::initialize_all(&cfg);
        // Native must also be initialized for the JIT.
        let _ = Target::initialize_native(&cfg);
    });
}

/// C++ `getFeaturesStr`: `llvm::SubtargetFeatures::AddFeature` prefixes `+`
/// unless a sign is already present; features are joined with `,`.
pub fn features_str(mattrs: &[String]) -> String {
    mattrs
        .iter()
        .filter(|m| !m.is_empty())
        .map(|m| {
            let lower = m.to_ascii_lowercase();
            if lower.starts_with('+') || lower.starts_with('-') { lower } else { format!("+{lower}") }
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Default host triple (`llvm::sys::getDefaultTargetTriple()`, normalized).
pub fn host_triple() -> String {
    let t = TargetMachine::get_default_triple();
    t.as_str().to_string_lossy().into_owned()
}

pub fn host_cpu_name() -> String {
    TargetMachine::get_host_cpu_name().to_string()
}

pub fn host_cpu_features() -> String {
    TargetMachine::get_host_cpu_features().to_string()
}

fn lookup_target(triple: &str, march: &str) -> std::result::Result<Target, String> {
    if !march.is_empty() {
        // `TargetRegistry::lookupTarget(march, triple, err)`: by arch name.
        return Target::from_name(march).ok_or_else(|| format!("invalid target '{march}'"));
    }
    Target::from_triple(&TargetTriple::create(triple)).map_err(|e| e.to_string())
}

/// Port of `getTargetMachine(triple, cpu, features, targetOptions, march, pic)`.
///
/// Returns `Ok(None)` if the target cannot be found (C++ returns `nullptr`), and
/// an error if creation fails (C++ `compilationError`).
///
/// PORT-NOTE: `llvm::codegen::InitTargetOptionsFromCodeGenFlags` and the explicit
/// reloc/code model command-line flags are not reachable through the C API; LLVM's
/// default TargetOptions, `Default` reloc model (unless `pic`) and `Default` code
/// model are used, which is what those flags resolve to when unset.
pub fn target_machine_for(triple: &str, cpu: &str, features: &str, march: &str, pic: bool) -> Result<Option<TargetMachine>> {
    initialize_targets();
    let target = match lookup_target(triple, march) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let tm = target.create_target_machine(
        &TargetTriple::create(triple),
        cpu,
        features,
        OptimizationLevel::Aggressive,
        if pic { RelocMode::PIC } else { RelocMode::Default },
        CodeModel::Default,
    );
    match tm {
        Some(tm) => Ok(Some(tm)),
        None => Err(CompilerError::Llvm("could not create LLVM target machine".into())),
    }
}

/// Port of `getTargetMachine(module, options, setFunctionAttributes, pic)`.
pub fn target_machine_for_module(
    module: &Module,
    options: &Options,
    set_function_attributes: bool,
    pic: bool,
) -> Result<Option<TargetMachine>> {
    let triple = module.get_triple().as_str().to_string_lossy().into_owned();
    // `moduleTriple.getArch()` is UnknownArch for an empty / unparsable triple.
    if triple.is_empty() || triple.starts_with("unknown") {
        return Ok(None);
    }
    let cpu = options.mcpu.clone();
    let features = features_str(&options.mattrs);
    let machine = target_machine_for(&triple, &cpu, &features, &options.march, pic)?;
    if set_function_attributes {
        set_function_attributes_for(module, &cpu, &features);
    }
    Ok(machine)
}

/// Port of `llvm::codegen::setFunctionAttributes(cpu, features, module)`:
/// adds `target-cpu` / `target-features` to every function definition that does
/// not already carry them (features are appended to existing ones).
pub fn set_function_attributes_for(module: &Module, cpu: &str, features: &str) {
    for f in ffi::functions(module.as_mut_ptr()) {
        if ffi::is_declaration(f) {
            continue;
        }
        if !cpu.is_empty() && !ffi::fn_has_string_attr(f, "target-cpu") {
            ffi::add_fn_string_attr(f, "target-cpu", cpu);
        }
        if !features.is_empty() {
            let existing = string_attr_value(f, "target-features");
            let new = match existing {
                Some(e) if !e.is_empty() => format!("{e},{features}"),
                _ => features.to_string(),
            };
            ffi::add_fn_string_attr(f, "target-features", &new);
        }
    }
}

pub(crate) fn string_attr_value(f: ffi::V, key: &str) -> Option<String> {
    unsafe {
        let a = llvm_sys::core::LLVMGetStringAttributeAtIndex(f, ffi::FN_INDEX, key.as_ptr() as *const _, key.len() as u32);
        if a.is_null() {
            return None;
        }
        let mut len = 0u32;
        let p = llvm_sys::core::LLVMGetStringAttributeValue(a, &mut len);
        if p.is_null() {
            return Some(String::new());
        }
        Some(String::from_utf8_lossy(std::slice::from_raw_parts(p as *const u8, len as usize)).into_owned())
    }
}

/// Host target machine honoring `-march/-mcpu/-mattr` like `EngineBuilder::selectTarget`.
/// An empty `mcpu` selects the host CPU and host features (EngineBuilder behaviour
/// with `setMCPU("")` is the generic CPU; Codon's JIT then overrides per-function
/// attributes through the native pass, so we match the effective result).
pub fn host_target_machine(options: &Options, pic: bool) -> Result<TargetMachine> {
    let triple = host_triple();
    let cpu = options.mcpu.clone();
    let features = features_str(&options.mattrs);
    target_machine_for(&triple, &cpu, &features, &options.march, pic)?
        .ok_or_else(|| CompilerError::Llvm(format!("could not find target for '{triple}'")))
}

/// Data layout string of a target machine.
pub fn data_layout_string(tm: &TargetMachine) -> String {
    tm.get_target_data().get_data_layout().as_str().to_string_lossy().into_owned()
}
