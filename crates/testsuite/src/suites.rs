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
    "parser/typecheck/test_access.py", "parser/typecheck/test_assign.py",
    "parser/typecheck/test_basic.py", "parser/typecheck/test_call.py",
    "parser/typecheck/test_class.py", "parser/typecheck/test_collections.py",
    "parser/typecheck/test_cond.py", "parser/typecheck/test_ctx.py",
    "parser/typecheck/test_error.py", "parser/typecheck/test_function.py",
    "parser/typecheck/test_import.py", "parser/typecheck/test_infer.py",
    "parser/typecheck/test_loops.py", "parser/typecheck/test_op.py",
    "parser/typecheck/test_parser.py",
    "parser/typecheck/test_typecheck.py", "parser/typecheck/test_poly.py",
];

pub const CORE: &[&str] = &[
    "core/helloworld.py", "core/arithmetic.py", "core/numerics.py", "core/parser.py",
    "core/generics.py", "core/generators.py", "core/gc.py", "core/exceptions.py",
    "core/containers.py", "core/trees.py", "core/range.py", "core/bltin.py",
    "core/arguments.py", "core/match.py", "core/serialization.py",
    "core/pipeline.py", "core/empty.py", "core/vec_simd.py",
    "core/implicit.py", "core/bare_generic_fields.py",
];

pub const STDLIB: &[&str] = &[
    "stdlib/llvm_test.py", "stdlib/str_test.py", "stdlib/bytearray_test.py",
    "stdlib/enum_test.py", "stdlib/re_test.py", "stdlib/math_test.py",
    "stdlib/cmath_test.py", "stdlib/datetime_test.py", "stdlib/getopt_test.py",
    "stdlib/argparse_test.py", "stdlib/itertools_test.py", "stdlib/json_test.py",
    "stdlib/bisect_test.py", "stdlib/random_test.py", "stdlib/statistics_test.py",
    "stdlib/sort_test.py", "stdlib/heapq_test.py", "stdlib/operator_test.py",
    "stdlib/asyncio_test.py", "stdlib/io_test.py", "stdlib/sys_test.py",
    "stdlib/os_test.py", "stdlib/expanduser_test.py", "stdlib/pathlib_test.py",
    "stdlib/ntpath_test.py", "stdlib/win32_os_test.py", "stdlib/unicode_test.py",
    "stdlib/functools_test.py", "stdlib/hashlib_test.py", "stdlib/future_test.py",
    "stdlib/contextlib_test.py", "stdlib/dataclasses_test.py", "stdlib/logging_test.py",
    "stdlib/warnings_test.py", "stdlib/abc_test.py", "stdlib/io_exit_flush_test.py",
    "stdlib/textwrap_test.py", "stdlib/fnmatch_test.py", "stdlib/fractions_test.py",
    "stdlib/struct_test.py", "stdlib/csv_test.py", "stdlib/urllib_parse_test.py",
    "stdlib/uuid_test.py", "stdlib/difflib_test.py", "stdlib/pprint_test.py",
    "stdlib/io_mod_test.py", "stdlib/glob_test.py", "stdlib/shutil_test.py",
    "stdlib/tempfile_test.py", "stdlib/types_test.py",
    "stdlib/math_more_test.py", "stdlib/random_parity_test.py", "stdlib/hmac_secrets_test.py",
    "stdlib/queue_graphlib_test.py", "stdlib/test_threading_queue.py", "stdlib/test_robot_decision.py", "stdlib/zlib_timeit_test.py", "stdlib/itertools_more_test.py",
    "stdlib/string_more_test.py",
    "stdlib/numcore_test.py", "stdlib/struct_more_test.py", "stdlib/deque_more_test.py",
    "stdlib/functools_more_test.py", "stdlib/zlib_deflate_test.py", "stdlib/decimal_test.py",
    "stdlib/ipaddress_test.py", "stdlib/calendar_test.py",
    "stdlib/array_test.py", "stdlib/collections_more_test.py", "stdlib/statistics_more_test.py",
    "stdlib/zlib_identical_test.py", "stdlib/datetime_strftime_test.py", "stdlib/string_template_test.py",
    "stdlib/ipaddress_v6_test.py",
    "stdlib/decimal_context_test.py", "stdlib/difflib_more_test.py", "stdlib/functools_cache_test.py",
];

pub const CNUMERICS: &[&str] = &[
    "core/numerics.py", "stdlib/math_test.py", "numpy/test_routines.py",
    "numpy/fusion/test_producers.py",
];

pub const OPT: &[&str] = &[
    "transform/canonical.py", "transform/dict_opt.py", "transform/enumerate.py",
    "transform/escapes.py", "transform/folding.py", "transform/for_lowering.py",
    "transform/format.py", "transform/io_opt.py", "transform/inlining.py",
    "transform/list_opt.py", "transform/omp.py", "transform/outlining.py",
    "transform/str_opt.py",
];

pub const GPU: &[&str] = &["transform/kernels.py"];

pub const NUMPY: &[&str] = &[
    "numpy/fusion/test_axis_edges.py", "numpy/fusion/test_axis_reductions.py",
    "numpy/fusion/test_basic.py", "numpy/fusion/test_broadcasting.py",
    "numpy/fusion/test_control_flow.py", "numpy/fusion/test_destinations.py",
    "numpy/fusion/test_div_mod.py", "numpy/fusion/test_forwarded_reductions.py",
    "numpy/fusion/test_forwarding.py", "numpy/fusion/test_highway_broadcast.py",
    "numpy/fusion/test_inlining.py", "numpy/fusion/test_last_use.py",
    "numpy/fusion/test_matmul_special.py", "numpy/fusion/test_matmul_values.py",
    "numpy/fusion/test_mean_where.py", "numpy/fusion/test_mixed_types.py",
    "numpy/fusion/test_ops_complex128.py", "numpy/fusion/test_ops_complex64.py",
    "numpy/fusion/test_ops_float16.py", "numpy/fusion/test_ops_float32.py",
    "numpy/fusion/test_ops_float64.py", "numpy/fusion/test_ops_int16.py",
    "numpy/fusion/test_ops_int32.py", "numpy/fusion/test_ops_int64.py",
    "numpy/fusion/test_ops_int8.py", "numpy/fusion/test_ops_uint16.py",
    "numpy/fusion/test_ops_uint32.py", "numpy/fusion/test_ops_uint64.py",
    "numpy/fusion/test_ops_uint8.py", "numpy/fusion/test_ownership.py",
    "numpy/fusion/test_producers.py", "numpy/fusion/test_reductions.py",
    "numpy/fusion/test_shape_validation.py", "numpy/fusion/test_truth_reductions.py",
    "numpy/random_tests/test_mt19937.py", "numpy/random_tests/test_pcg64.py",
    "numpy/random_tests/test_philox.py", "numpy/random_tests/test_sfc64.py",
    "numpy/random_tests/test_legacy.py", "numpy/test_dtype.py", "numpy/test_elision.py",
    "numpy/test_gaps.py", "numpy/test_fft.py", "numpy/test_functional.py",
    "numpy/test_indexing.py", "numpy/test_io.py", "numpy/test_lib.py",
    "numpy/test_linalg.py", "numpy/test_loops.py", "numpy/test_misc.py",
    "numpy/test_ndmath.py", "numpy/test_npdatetime.py",
    "numpy/test_reductions.py", "numpy/test_routines.py", "numpy/test_sorting.py",
    "numpy/test_statistics.py", "numpy/test_ufunc.py", "numpy/test_window.py",
    "numpy/test_polynomial.py", "numpy/test_testing.py", "numpy/test_ma.py",
    "numpy/test_empty_errstate.py", "numpy/test_strings.py", "numpy/test_lib_more.py",
    "numpy/test_generator.py", "numpy/test_ndarray_methods.py", "numpy/test_ma2.py",
    "numpy/test_empty_fancy.py", "numpy/test_savez.py",
    "numpy/test_final2.py", "numpy/test_final3.py", "numpy/test_sort_ties.py",
];

pub const SCIPY: &[&str] = &[
    "scipy/test_fft.py", "scipy/test_linalg.py", "scipy/test_signal.py",
    "scipy/test_special.py", "scipy/test_stats.py", "pyriemann/test_geometry.py",
    "pyriemann/test_pipeline.py", "mne/test_acceptance.py", "mne/test_filter_psd.py",
    "mne/test_csp.py", "mne/test_iir_resample_mt.py", "pandas/test_pipeline.py",
    "pandas/test_reductions.py", "pandas/test_groupby.py", "pandas/test_rolling.py",
    "pandas/test_frame.py", "pandas/test_io.py", "pandas/test_repr.py",
    "pandas/test_fixes.py",
    "scipy/test_spatial.py", "scipy/test_interpolate.py", "scipy/test_cluster.py",
    "scipy/test_ndimage.py", "scipy/test_ndimage_interp.py", "scipy/test_sparse.py",
    "scipy/test_sparse_linalg.py", "scipy/test_csgraph.py", "scipy/test_sparse_formats.py",
    "scipy/test_sparse_lu.py", "scipy/test_sparse_eigen.py", "scipy/test_sparse_linalg2.py",
    "scipy/test_csgraph2.py", "scipy/test_optimize.py", "scipy/test_optimize_global.py",
    "scipy/test_linprog.py", "sklearn/test_neighbors.py", "sklearn/test_naive_bayes.py",
    "sklearn/test_cluster.py", "sklearn/test_tree.py", "sklearn/test_ensemble.py",
    "sklearn/test_decomposition.py", "sklearn/test_mixture.py", "sklearn/test_mixture_bayes.py",
    "sklearn/test_feature_selection.py", "sklearn/test_impute.py",
    "sklearn/test_manifold_tsne.py", "sklearn/test_manifold.py",
    "mne/test_ica.py", "mne/test_ica_bads.py", "mne/test_preprocessing.py", "mne/test_ctps.py",
    "mne/test_xdawn.py", "sklearn/test_cluster_more.py", "sklearn/test_cluster_struct.py",
    "sklearn/test_cluster_density.py", "sklearn/test_birch.py",
    "scipy/test_integrate.py", "scipy/test_integrate_quad.py", "scipy/test_integrate_ivp.py",
    "scipy/test_integrate_stiff.py", "scipy/test_integrate_odeint.py",
    "scipy/test_optimize_constrained.py", "scipy/test_optimize_constrained2.py",
    "scipy/test_optimize_local2.py", "scipy/test_optimize_local3.py",
    "sklearn/test_histgb.py", "sklearn/test_ensemble2.py", "sklearn/test_stacking_clf.py",
    "sklearn/test_inspection.py", "mne/test_channels.py",
    "scipy/test_special_misc.py", "scipy/test_special_misc2.py", "scipy/test_special_bessel.py",
    "scipy/test_special_gamma_hyp.py", "scipy/test_special_orth_ellip.py",
    "scipy/test_special_cdfs.py",
    "scipy/test_dists_common.py", "scipy/test_dists_more.py", "scipy/test_dists_ext.py",
    "scipy/test_dists_misc.py",
    "sklearn/test_metrics_cls.py", "sklearn/test_metrics_rank_reg.py",
    "sklearn/test_metrics_pairwise.py", "sklearn/test_utils.py",
    "scipy/test_linalg2.py", "scipy/test_linalg_matfuncs.py", "scipy/test_signal_design.py",
    "scipy/test_signal_filter.py", "scipy/test_signal_spectral.py",
    "sklearn/test_cluster_bisect_bicluster.py", "mne/test_ica_picard.py",
    "scipy/test_stats_desc.py", "scipy/test_stats_tests1.py", "scipy/test_stats_tests2.py",
    "scipy/test_stats_qmc.py", "scipy/test_stats_misc.py",
    "mne/test_io_fif.py", "mne/test_io_edf_bv_eeglab.py", "mne/test_raw_api.py",
    "mne/test_events.py", "mne/test_events_epochs.py", "mne/test_filter_picks.py",
    "sklearn/test_preprocessing_more.py", "sklearn/test_model_selection_more.py",
    "sklearn/test_search_validation.py", "sklearn/test_learning_curves.py",
    "sklearn/test_pipeline_compose.py", "sklearn/test_compose.py",
    "sklearn/test_linear_sgd.py", "sklearn/test_linear_basic.py", "sklearn/test_linear_cd.py",
    "sklearn/test_linear_lars.py", "sklearn/test_logistic.py", "sklearn/test_logistic_cv.py",
    "sklearn/test_linear_robust_glm.py",
    "mne/test_tfr.py", "mne/test_tfr_mt.py", "mne/test_tfr_csd.py", "mne/test_mne_stats.py",
    "mne/test_decoding.py", "mne/test_decoding_lm.py", "mne/test_decoding_sl.py",
    "mne/test_decoding_gl.py", "mne/test_decoding_spoc.py", "mne/test_decoding_rf.py",
    "mne/test_decoding_ssd.py", "mne/test_infomax.py", "mne/test_muscle.py",
    "scipy/test_dists_discrete.py", "scipy/test_dists_discrete2.py", "scipy/test_dists_sample.py",
    "scipy/test_dists_mv.py", "scipy/test_dists_fit.py", "scipy/test_dists_c2a.py",
    "scipy/test_dists_c2b.py", "scipy/test_dists_c2c.py", "scipy/test_dists_c3a.py",
    "scipy/test_dists_c3b.py", "scipy/test_dists_c4.py",
    "scipy/test_optimize_root.py", "scipy/test_optimize_legacy.py", "scipy/test_integrate_bvp.py",
    "scipy/test_signal_more.py", "scipy/test_signal_more2.py", "scipy/test_linalg3.py",
    "sklearn/test_datasets.py", "sklearn/test_misc_estimators.py", "sklearn/test_svm.py",
    "sklearn/test_svm2.py", "sklearn/test_lda_cov.py", "sklearn/test_multiclass.py",
    "sklearn/test_multioutput.py", "sklearn/test_calibration.py", "sklearn/test_pls_semi.py",
    "sklearn/test_feature_extraction.py", "sklearn/test_gp.py", "sklearn/test_gp2.py",
    "sklearn/test_gp_clf.py", "sklearn/test_mlp.py", "sklearn/test_mlp2.py",
    "mne/test_proj_ref.py", "mne/test_csd_annot.py", "mne/test_montage_io.py",
    "mne/test_epochs_more.py",
    "scipy/test_stats_resampling.py", "scipy/test_stats_tests3.py", "scipy/test_mstats.py",
    "scipy/test_fft2.py", "scipy/test_fftpack.py", "scipy/test_fitpack.py",
    "scipy/test_interpolate2.py", "scipy/test_interpolate3.py", "scipy/test_qhull.py",
    "scipy/test_rbf_griddata.py", "scipy/test_rotation.py", "scipy/test_spatial2.py",
    "sklearn/test_balltree_kdtree.py", "sklearn/test_neighbors2.py", "sklearn/test_decomposition2.py",
    "sklearn/test_lda_nmf.py", "sklearn/test_manifold2.py", "sklearn/test_tree_export.py",
    "mne/test_filter_more.py",
    "pyriemann/test_ajd.py", "pyriemann/test_estimation.py", "pyriemann/test_estimators2.py",
    "pyriemann/test_mest.py", "pyriemann/test_covx.py", "pyriemann/test_spectral.py",
    "pyriemann/test_spatial.py", "pyriemann/test_classif_a.py", "pyriemann/test_classif_b.py",
    "pyriemann/test_means2.py", "pyriemann/test_clust2.py", "pyriemann/test_datasets.py",
    "pyriemann/test_potato.py", "pyriemann/test_tl2.py", "pandas/test_str.py",
    "mne/test_cov.py", "mne/test_forward_sphere.py",
    "pyriemann/test_misc.py", "pandas/test_ewm.py", "pandas/test_stats2.py",
    "pandas/test_groupby2.py",
    "scipy/test_special_seq.py", "scipy/test_special_orthopoly.py",
    "scipy/test_stats_exact2x2.py", "scipy/test_stats_fit.py", "scipy/test_stats_multicomp.py",
    "scipy/test_stats_qmc2.py",
    "scipy/test_special_mathieu.py", "scipy/test_special_wright.py", "scipy/test_special_ellip_harm.py",
    "scipy/test_integrate_odesolver.py", "scipy/test_integrate_tanhsinh.py", "scipy/test_linalg_cossin.py",
    "scipy/test_optimize_elementwise.py", "scipy/test_optimize_qap.py", "scipy/test_signal_placepoles.py",
    "scipy/test_signal_stfft.py", "scipy/test_sparse_names.py",
    "sklearn/test_final_chain.py", "sklearn/test_final_metrics.py", "sklearn/test_final_metrics2.py",
    "sklearn/test_final_nb.py", "sklearn/test_final_svm.py", "sklearn/test_final_threshold.py",
    "sklearn/test_final_utils.py",
    "mne/test_decoding_tf.py", "mne/test_decoding_xdawn.py", "mne/test_dig_readers.py",
    "mne/test_epochs_utils.py", "mne/test_io_gdf.py", "mne/test_io_misc.py", "mne/test_io_read_raw.py",
    "mne/test_layout.py", "mne/test_tf_more.py",
    "pandas/test_dt.py", "pandas/test_multi.py", "pandas/test_ops3.py",
    "mne/test_beamformer.py", "mne/test_simulation.py", "mne/test_eog_regression.py",
    "scipy/test_stats_final_dists_a.py", "scipy/test_stats_final_dists_b.py",
    "scipy/test_stats_final_discrete.py", "scipy/test_stats_final_misc.py",
    "scipy/test_special_spheroidal.py",
    "mne/test_beamformer_eeg.py", "mne/test_inverse.py", "mne/test_preproc3.py", "mne/test_preproc_misc.py",
    "mne/test_sim_labels.py", "mne/test_ssp_exg.py",
    "pandas/test_csv2.py", "pandas/test_ops4.py",
    "pyriemann/test_embed.py", "pyriemann/test_nch.py", "pyriemann/test_stats.py", "pyriemann/test_tl3.py",
    "scipy/test_leftovers2.py", "sklearn/test_leftovers2.py",
    "scipy/test_stats_final_cens.py", "scipy/test_stats_final_cov.py", "scipy/test_stats_final_levy.py",
    "scipy/test_stats_final_methods.py", "scipy/test_stats_final_mgc.py", "scipy/test_stats_final_mv.py",
    "scipy/test_stats_final_newdist.py", "scipy/test_stats_final_qmc.py", "scipy/test_stats_final_quantile.py",
    "scipy/test_stats_final_sobol.py", "scipy/test_stats_final_surv.py",
    "mne/test_maxwell.py", "mne/test_preproc_final.py", "mne/test_toplevel_io.py", "mne/test_toplevel_algo.py",
    "mne/test_io_egi_bci_besa.py", "mne/test_io_cnt.py", "mne/test_io_nedf.py",
    "mne/test_io_fmt3a.py", "mne/test_io_fmt3b.py", "mne/test_io_fmt3c.py",
    "bci/test_logpower_lda.py",
    "bci/test_nb_csp_lda.py", "bci/test_nb_ctx_epochs.py", "bci/test_nb_fbcsp.py",
    "bci/test_nb_iir_psd_misc.py", "bci/test_nb_logpower_lda.py", "bci/test_nb_riemann.py",
    "bci/test_src_compat.py", "bci/test_gb.py",
];

pub const BRAINDECODE: &[&str] = &[
    "torch/test_core.py", "torch/test_data.py", "torch/test_nn.py",
    "torch/test_nn_init.py", "torch/test_optim.py", "torch/test_nn_more.py",
    "torch/test_conv3d.py", "torch/test_serialization.py", "einops/test_einops.py",
    "braindecode/test_data.py", "braindecode/test_preprocessing.py",
    "braindecode/test_augmentation.py", "braindecode/test_training.py",
    "braindecode/test_models.py", "braindecode/test_models_attn.py",
    "braindecode/test_models_conv.py", "braindecode/test_models_misc.py",
    "braindecode/test_models_more.py", "braindecode/test_models_tcn.py",
    "braindecode/test_models_fb.py", "braindecode/test_models_atcnet.py",
    "braindecode/test_models_ifnet_usleep.py", "braindecode/test_models_msvtnet.py",
    "braindecode/test_models_tidnet.py", "braindecode/test_models_attnsleep.py",
    "braindecode/test_models_deepsleepnet.py", "braindecode/test_models_eegsym.py",
    "braindecode/test_models_fb2.py", "braindecode/test_models_more2.py",
    "braindecode/test_models_tcformer.py", "braindecode/test_models_a_attn.py",
    "braindecode/test_models_a_clip.py", "braindecode/test_models_a_conv.py",
    "braindecode/test_models_a_emg.py", "braindecode/test_models_a_graph.py",
    "braindecode/test_models_a_guetschel.py", "braindecode/test_models_a_jepa.py",
    "braindecode/test_models_b_barista.py", "braindecode/test_models_b_bendr.py",
    "braindecode/test_models_b_biot.py", "braindecode/test_models_b_brainbert.py",
    "braindecode/test_models_b_brant.py", "braindecode/test_models_b_cbramod.py",
    "braindecode/test_models_b_codebrain.py", "braindecode/test_models_b_csbrain.py",
    "braindecode/test_models_b_duin.py", "braindecode/test_models_b_eegdino.py",
    "braindecode/test_models_b_eegpt.py", "braindecode/test_models_b_epint.py",
    "braindecode/test_models_b_labram.py", "braindecode/test_models_b_luna.py",
    "braindecode/test_models_b_mvpformer.py", "braindecode/test_models_b_neurorvq.py",
    "braindecode/test_models_b_pbt.py", "braindecode/test_models_b_reve.py",
    "braindecode/test_models_b_sleepfm.py", "braindecode/test_models_b_steegformer.py",
    "braindecode/test_models_b_tfm.py", "braindecode/test_models_a_meta.py",
    "braindecode/test_models_a_popt.py", "braindecode/test_models_b_axon.py",
    "braindecode/test_models_b_brainmodule.py", "braindecode/test_models_b_diver1.py",
    "braindecode/test_models_b_mapa.py", "braindecode/test_models_b_zuna.py",
    "braindecode/test_models_b_brainomni.py", "braindecode/test_models_b_neurorvq_tok.py",
    "braindecode/test_models_b_dance.py", "braindecode/test_models_b_seizure.py",
    "torch/test_autograd_semantics.py", "torch/test_audit_fixes.py",
    "sklearn/test_more.py", "torch/test_ops_a.py", "torch/test_ops_b.py",
    "sklearn/test_estimators.py", "sklearn/test_extended.py", "joblib/test_joblib.py",
    "braindecode/test_training_eegclassifier.py",
];
/// Robot-BCI-Movement-GUI (user repo under examples/, not tracked): skipped when absent.
pub const ROBOT: &[&str] = &[
    "robot/test_pth_load.py", "robot/test_finetune_rng.py", "robot/test_finetune_data.py",
    "robot/test_finetune_short.py", "robot/test_inference.py",
];
/// Workspace-relative root of the user's Robot-BCI package (put on TYPHON_PATH for "robot").
pub const ROBOT_REPO: &str = "examples/Robot-BCI-Movement-GUI";

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
    v.extend(combine("robot", ROBOT, true, true));
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn splits() {
        let src = "#%% __ignore__\nimport x\n#%% a,barebones\nprint 1 #: 1\n#%% b\nprint 2\n";
        let c = split_cases("f.py", src);
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].name, "4_a");
        assert!(c[0].barebones);
        assert_eq!(c[0].start_line, 2);
        assert_eq!(c[1].name, "6_b");
        assert!(!c[1].barebones);
        assert_eq!(c[1].code, "#%% b\nprint 2\n");
    }
}
