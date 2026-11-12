//! Lives in `typhon-common` (not `typhon-compiler`) because IR passes, the LLVM
//! backend and the parser all read options, and Rust forbids the include cycles
//! the reference code relies on.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GlobalCtorMode {
    No,
    Yes,
    #[default]
    Auto,
}

#[derive(Clone, Debug)]
pub struct Options {
    pub argv0: String,
    pub debug: bool,
    pub test: bool,
    pub jit: bool,
    pub standalone: bool,
    pub pyext: bool,
    pub pmempty: bool,
    pub capture: bool,
    pub native: bool,
    pub pynum: bool,
    pub noexc: bool,
    pub fastmath: bool,
    pub autopy: bool,
    pub autofree: bool,
    pub unordereddict: bool,
    pub ctor: GlobalCtorMode,
    pub defines: Vec<String>,
    pub disabled: Vec<String>,
    pub libdevice: String,
    pub gpu_name: String,
    pub gpu_feat: String,
    pub gpu_output: String,
    pub log: String,
    pub march: String,
    pub mcpu: String,
    pub mattrs: Vec<String>,
    /// Link executables against `libtyphonrt.a` instead of the shared runtime
    /// (`build -exe` default; `-dynamic-runtime` turns it off). Libraries,
    /// Python extensions and the JIT always use the shared runtime.
    pub static_runtime: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            argv0: String::new(),
            debug: true,
            test: false,
            jit: false,
            standalone: false,
            pyext: false,
            pmempty: false,
            capture: false,
            native: true,
            pynum: true,
            noexc: false,
            fastmath: false,
            autopy: false,
            autofree: false,
            unordereddict: false,
            ctor: GlobalCtorMode::Auto,
            defines: Vec::new(),
            disabled: Vec::new(),
            libdevice: "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc".into(),
            gpu_name: "sm_30".into(),
            gpu_feat: "+ptx42".into(),
            gpu_output: String::new(),
            log: String::new(),
            march: String::new(),
            mcpu: String::new(),
            mattrs: Vec::new(),
            static_runtime: false,
        }
    }
}

impl Options {
    pub fn with_argv0(argv0: impl Into<String>) -> Self {
        Self { argv0: argv0.into(), ..Default::default() }
    }
}
