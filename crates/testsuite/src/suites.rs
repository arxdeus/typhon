//! Suite tables (port of the `INSTANTIATE_TEST_SUITE_P` blocks in test/main.cpp)
//! and `#%%` case splitting (`getTypeTests`).

/// One parametrised test case; mirrors the `SeqTest` parameter tuple.
#[derive(Clone, Debug)]
pub struct Case {
    pub suite: &'static str,
    /// Path relative to the test dir.
    pub file: String,
    pub debug: bool,
    /// Case name; empty for whole-file tests.
    pub name: String,
    /// Case source (only for `#%%` cases).
    pub code: String,
    /// 0-based line of the `#%%` marker in the file.
    pub start_line: usize,
    pub barebones: bool,
    pub py_numerics: bool,
    pub run: bool,
}

impl Case {
    pub fn display_name(&self) -> String {
        let mode = if self.debug { "debug" } else { "release" };
        if self.name.is_empty() {
            format!("{}/{}[{}{}]", self.suite, self.file, mode, if self.py_numerics { "" } else { ",cnum" })
        } else {
            format!("{}/{}::{}", self.suite, self.file, self.name)
        }
    }
}

pub const TYPE_TESTS: &[&str] = &[
    "parser/typecheck/test_access.codon", "parser/typecheck/test_assign.codon",
    "parser/typecheck/test_basic.codon", "parser/typecheck/test_call.codon",
    "parser/typecheck/test_class.codon", "parser/typecheck/test_collections.codon",
    "parser/typecheck/test_cond.codon", "parser/typecheck/test_ctx.codon",
    "parser/typecheck/test_error.codon", "parser/typecheck/test_function.codon",
    "parser/typecheck/test_import.codon", "parser/typecheck/test_infer.codon",
    "parser/typecheck/test_loops.codon", "parser/typecheck/test_op.codon",
    "parser/typecheck/test_parser.codon", "parser/typecheck/test_python.codon",
    "parser/typecheck/test_typecheck.codon", "parser/typecheck/test_poly.codon",
];

pub const CORE: &[&str] = &[
    "core/helloworld.codon", "core/arithmetic.codon", "core/numerics.codon", "core/parser.codon",
    "core/generics.codon", "core/generators.codon", "core/gc.codon", "core/exceptions.codon",
    "core/containers.codon", "core/trees.codon", "core/range.codon", "core/bltin.codon",
    "core/arguments.codon", "core/match.codon", "core/serialization.codon",
    "core/pipeline.codon", "core/empty.codon", "core/vec_simd.codon",
];

pub const STDLIB: &[&str] = &[
    "stdlib/llvm_test.codon", "stdlib/str_test.codon", "stdlib/bytearray_test.codon",
    "stdlib/enum_test.codon", "stdlib/re_test.codon", "stdlib/math_test.codon",
    "stdlib/cmath_test.codon", "stdlib/datetime_test.codon", "stdlib/getopt_test.codon",
    "stdlib/argparse_test.codon", "stdlib/itertools_test.codon", "stdlib/json_test.codon",
    "stdlib/bisect_test.codon", "stdlib/random_test.codon", "stdlib/statistics_test.codon",
    "stdlib/sort_test.codon", "stdlib/heapq_test.codon", "stdlib/operator_test.codon",
    "stdlib/asyncio_test.codon", "stdlib/io_test.codon", "stdlib/sys_test.codon",
    "stdlib/os_test.codon", "stdlib/expanduser_test.codon", "stdlib/pathlib_test.codon",
    "stdlib/ntpath_test.codon", "stdlib/win32_os_test.codon", "stdlib/unicode_test.codon",
    "stdlib/functools_test.codon", "stdlib/hashlib_test.codon", "stdlib/future_test.codon",
    "python/pybridge.codon",
];

pub const CNUMERICS: &[&str] = &[
    "core/numerics.codon", "stdlib/math_test.codon", "numpy/test_routines.codon",
    "numpy/fusion/test_producers.codon",
];

pub const OPT: &[&str] = &[
    "transform/canonical.codon", "transform/dict_opt.codon", "transform/enumerate.codon",
    "transform/escapes.codon", "transform/folding.codon", "transform/for_lowering.codon",
    "transform/format.codon", "transform/io_opt.codon", "transform/inlining.codon",
    "transform/list_opt.codon", "transform/omp.codon", "transform/outlining.codon",
    "transform/str_opt.codon",
];

pub const GPU: &[&str] = &["transform/kernels.codon"];

pub const NUMPY: &[&str] = &[
    "numpy/fusion/test_axis_edges.codon", "numpy/fusion/test_axis_reductions.codon",
    "numpy/fusion/test_basic.codon", "numpy/fusion/test_broadcasting.codon",
    "numpy/fusion/test_control_flow.codon", "numpy/fusion/test_destinations.codon",
    "numpy/fusion/test_div_mod.codon", "numpy/fusion/test_forwarded_reductions.codon",
    "numpy/fusion/test_forwarding.codon", "numpy/fusion/test_highway_broadcast.codon",
    "numpy/fusion/test_inlining.codon", "numpy/fusion/test_last_use.codon",
    "numpy/fusion/test_matmul_special.codon", "numpy/fusion/test_matmul_values.codon",
    "numpy/fusion/test_mean_where.codon", "numpy/fusion/test_mixed_types.codon",
    "numpy/fusion/test_ops_complex128.codon", "numpy/fusion/test_ops_complex64.codon",
    "numpy/fusion/test_ops_float16.codon", "numpy/fusion/test_ops_float32.codon",
    "numpy/fusion/test_ops_float64.codon", "numpy/fusion/test_ops_int16.codon",
    "numpy/fusion/test_ops_int32.codon", "numpy/fusion/test_ops_int64.codon",
    "numpy/fusion/test_ops_int8.codon", "numpy/fusion/test_ops_uint16.codon",
    "numpy/fusion/test_ops_uint32.codon", "numpy/fusion/test_ops_uint64.codon",
    "numpy/fusion/test_ops_uint8.codon", "numpy/fusion/test_ownership.codon",
    "numpy/fusion/test_producers.codon", "numpy/fusion/test_reductions.codon",
    "numpy/fusion/test_shape_validation.codon", "numpy/fusion/test_truth_reductions.codon",
    "numpy/random_tests/test_mt19937.codon", "numpy/random_tests/test_pcg64.codon",
    "numpy/random_tests/test_philox.codon", "numpy/random_tests/test_sfc64.codon",
    "numpy/random_tests/test_legacy.codon", "numpy/test_dtype.codon", "numpy/test_elision.codon",
    "numpy/test_gaps.codon", "numpy/test_fft.codon", "numpy/test_functional.codon",
    "numpy/test_indexing.codon", "numpy/test_io.codon", "numpy/test_lib.codon",
    "numpy/test_linalg.codon", "numpy/test_loops.codon", "numpy/test_misc.codon",
    "numpy/test_ndmath.codon", "numpy/test_npdatetime.codon", "numpy/test_pybridge.codon",
    "numpy/test_reductions.codon", "numpy/test_routines.codon", "numpy/test_sorting.codon",
    "numpy/test_statistics.codon", "numpy/test_ufunc.codon", "numpy/test_window.codon",
];

pub const SCIPY: &[&str] = &[
    "scipy/test_fft.codon", "scipy/test_linalg.codon", "scipy/test_signal.codon",
    "scipy/test_special.codon", "scipy/test_stats.codon", "pyriemann/test_geometry.codon",
    "pyriemann/test_pipeline.codon", "mne/test_acceptance.codon", "mne/test_filter_psd.codon",
    "mne/test_csp.codon", "mne/test_iir_resample_mt.codon", "pandas/test_pipeline.codon",
    "pandas/test_reductions.codon", "pandas/test_groupby.codon", "pandas/test_rolling.codon",
    "pandas/test_frame.codon", "pandas/test_io.codon", "pandas/test_repr.codon",
    "pandas/test_fixes.codon",
];

pub const BRAINDECODE: &[&str] = &[
    "torch/test_core.codon", "torch/test_data.codon", "torch/test_nn.codon",
    "torch/test_nn_init.codon", "torch/test_optim.codon", "torch/test_nn_more.codon",
    "torch/test_conv3d.codon", "torch/test_serialization.codon", "einops/test_einops.codon",
    "braindecode/test_data.codon", "braindecode/test_preprocessing.codon",
    "braindecode/test_augmentation.codon", "braindecode/test_training.codon",
    "braindecode/test_models.codon", "braindecode/test_models_attn.codon",
    "braindecode/test_models_conv.codon", "braindecode/test_models_misc.codon",
    "braindecode/test_models_more.codon", "braindecode/test_models_tcn.codon",
    "braindecode/test_models_fb.codon", "braindecode/test_models_atcnet.codon",
    "braindecode/test_models_ifnet_usleep.codon", "braindecode/test_models_msvtnet.codon",
    "braindecode/test_models_tidnet.codon", "braindecode/test_models_attnsleep.codon",
    "braindecode/test_models_deepsleepnet.codon", "braindecode/test_models_eegsym.codon",
    "braindecode/test_models_fb2.codon", "braindecode/test_models_more2.codon",
    "braindecode/test_models_tcformer.codon", "braindecode/test_models_a_attn.codon",
    "braindecode/test_models_a_clip.codon", "braindecode/test_models_a_conv.codon",
    "braindecode/test_models_a_emg.codon", "braindecode/test_models_a_graph.codon",
    "braindecode/test_models_a_guetschel.codon", "braindecode/test_models_a_jepa.codon",
    "braindecode/test_models_b_barista.codon", "braindecode/test_models_b_bendr.codon",
    "braindecode/test_models_b_biot.codon", "braindecode/test_models_b_brainbert.codon",
    "braindecode/test_models_b_brant.codon", "braindecode/test_models_b_cbramod.codon",
    "braindecode/test_models_b_codebrain.codon", "braindecode/test_models_b_csbrain.codon",
    "braindecode/test_models_b_duin.codon", "braindecode/test_models_b_eegdino.codon",
    "braindecode/test_models_b_eegpt.codon", "braindecode/test_models_b_epint.codon",
    "braindecode/test_models_b_labram.codon", "braindecode/test_models_b_luna.codon",
    "braindecode/test_models_b_mvpformer.codon", "braindecode/test_models_b_neurorvq.codon",
    "braindecode/test_models_b_pbt.codon", "braindecode/test_models_b_reve.codon",
    "braindecode/test_models_b_sleepfm.codon", "braindecode/test_models_b_steegformer.codon",
    "braindecode/test_models_b_tfm.codon", "braindecode/test_models_a_meta.codon",
    "braindecode/test_models_a_popt.codon", "braindecode/test_models_b_axon.codon",
    "braindecode/test_models_b_brainmodule.codon", "braindecode/test_models_b_diver1.codon",
    "braindecode/test_models_b_mapa.codon", "braindecode/test_models_b_zuna.codon",
    "braindecode/test_models_b_brainomni.codon", "braindecode/test_models_b_neurorvq_tok.codon",
    "braindecode/test_models_b_dance.codon", "braindecode/test_models_b_seizure.codon",
    "torch/test_autograd_semantics.codon", "torch/test_audit_fixes.codon",
    "sklearn/test_more.codon", "torch/test_ops_a.codon", "torch/test_ops_b.codon",
    "sklearn/test_estimators.codon", "sklearn/test_extended.codon", "joblib/test_joblib.codon",
    "braindecode/test_training_eegclassifier.codon",
];

/// Split a file on `#%%` markers (port of `getTypeTests`). `source` is the file text.
/// Header is `#%% name[,barebones]`; name `__ignore__` skips the case.
pub fn split_cases(file: &str, source: &str) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut barebones = false;
    let mut code = String::new();
    let mut test_name = String::new();
    let mut code_line = 0usize;
    let mut line = 0usize;
    let mk = |line: usize, name: &str, code: &str, code_line: usize, barebones: bool| Case {
        suite: "types",
        file: file.to_string(),
        debug: true,
        name: format!("{}_{}", line, name),
        code: code.to_string(),
        start_line: code_line,
        barebones,
        py_numerics: false,
        run: true,
    };
    for l in source.lines() {
        if l.starts_with("#%%") {
            if line > 0 && test_name != "__ignore__" {
                cases.push(mk(line, &test_name, &code, code_line, barebones));
            }
            // C++: split(l.substr(4), ',')
            let rest = l.get(4..).unwrap_or("");
            let t: Vec<&str> = rest.split(',').collect();
            barebones = t.len() > 1 && t[1] == "barebones";
            test_name = t[0].to_string();
            code = format!("{}\n", l);
            code_line = line;
        } else {
            code.push_str(l);
            code.push('\n');
        }
        line += 1;
    }
    if line > 0 && test_name != "__ignore__" {
        cases.push(mk(line, &test_name, &code, code_line, barebones));
    }
    cases
}

fn combine(suite: &'static str, files: &[&str], py_numerics: bool, run: bool) -> Vec<Case> {
    let mut v = Vec::new();
    for f in files {
        for debug in [true, false] {
            v.push(Case {
                suite, file: f.to_string(), debug, name: String::new(), code: String::new(),
                start_line: 0, barebones: false, py_numerics, run,
            });
        }
    }
    v
}

/// All cases in C++ registration order. `read` loads a test-dir-relative file.
pub fn all_cases(read: impl Fn(&str) -> Option<String>) -> Vec<Case> {
    let mut v = Vec::new();
    for f in TYPE_TESTS {
        if let Some(src) = read(f) {
            v.extend(split_cases(f, &src));
        }
    }
    v.extend(combine("core", CORE, true, true));
    v.extend(combine("stdlib", STDLIB, true, true));
    v.extend(combine("cnumerics", CNUMERICS, false, true));
    v.extend(combine("opt", OPT, true, true));
    v.extend(combine("gpu", GPU, true, false));
    v.extend(combine("numpy", NUMPY, true, true));
    v.extend(combine("scipy", SCIPY, true, true));
    v.extend(combine("braindecode", BRAINDECODE, true, true));
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn splits() {
        let src = "#%% __ignore__\nimport x\n#%% a,barebones\nprint 1 #: 1\n#%% b\nprint 2\n";
        let c = split_cases("f.codon", src);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].name, "4_a");
        assert!(c[0].barebones);
        assert_eq!(c[0].start_line, 2);
        assert_eq!(c[1].name, "6_b");
        assert!(!c[1].barebones);
        assert_eq!(c[1].code, "#%% b\nprint 2\n");
    }
}
