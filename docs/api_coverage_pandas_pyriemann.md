# pandas / pyriemann API coverage (generated)

## pandas (top-level): 21/120 names defined

Missing: ArrowDtype, BooleanDtype, Categorical, CategoricalDtype, CategoricalIndex, DateOffset, DatetimeIndex, DatetimeTZDtype, ExcelFile, ExcelWriter, Flags, Float32Dtype, Float64Dtype, Grouper, HDFStore, IndexSlice, Int16Dtype, Int32Dtype, Int64Dtype, Int8Dtype, IntervalDtype, IntervalIndex, MultiIndex, NA, NaT, NamedAgg, NamedFunc, Period, PeriodDtype, PeriodIndex, RangeIndex, SparseDtype, StringDtype, Timedelta, TimedeltaIndex, Timestamp, UInt16Dtype, UInt32Dtype, UInt64Dtype, UInt8Dtype, annotations, api, array, arrays, bdate_range, col, compat, core, date_range, describe_option, errors, eval, from_dummies, get_option, infer_freq, interval_range, io, json_normalize, lreshape, merge_asof, merge_ordered, offsets, option_context, options, pandas, period_range, plotting, read_clipboard, read_excel, read_feather, read_fwf, read_hdf, read_html, read_iceberg, read_json, read_orc, read_parquet, read_pickle, read_sas, read_spss, read_sql, read_sql_query, read_sql_table, read_stata, read_table, read_xml, reset_option, set_eng_float_format, set_option, show_versions, test, testing, timedelta_range, to_datetime, to_pickle, to_timedelta, tseries, util, wide_to_long

## DataFrame: 140/204 names defined

Missing: align, asfreq, asof, at_time, attrs, axes, between_time, boxplot, columns, combine, compare, convert_dtypes, dot, droplevel, eval, explode, flags, from_arrow, from_dict, from_records, hist, infer_objects, isetitem, join, memory_usage, plot, query, reindex, reindex_like, rename_axis, reorder_levels, resample, sample, select, set_axis, set_flags, sparse, stack, style, swaplevel, to_clipboard, to_excel, to_feather, to_hdf, to_html, to_iceberg, to_json, to_latex, to_markdown, to_orc, to_parquet, to_period, to_pickle, to_records, to_sql, to_stata, to_timestamp, to_xarray, to_xml, truncate, tz_convert, tz_localize, unstack, xs

## Series: 147/203 names defined

Missing: align, array, asfreq, asof, at_time, attrs, axes, between_time, case_when, cat, combine, compare, convert_dtypes, divmod, dot, droplevel, dt, explode, flags, from_arrow, hist, infer_objects, list, memory_usage, name, nbytes, plot, rdivmod, reindex, reindex_like, rename_axis, reorder_levels, resample, sample, set_axis, set_flags, sparse, str, struct, swaplevel, to_clipboard, to_excel, to_hdf, to_json, to_latex, to_markdown, to_period, to_pickle, to_sql, to_timestamp, to_xarray, truncate, tz_convert, tz_localize, unstack, xs

## Index: 56/92 names defined

Missing: array, asof, asof_locs, delete, difference, droplevel, get_indexer, get_indexer_for, get_indexer_non_unique, get_level_values, get_slice_bound, has_duplicates, identical, infer_objects, inferred_type, intersection, is_, join, memory_usage, name, names, nbytes, nlevels, putmask, ravel, reindex, set_names, slice_indexer, slice_locs, sortlevel, str, symmetric_difference, to_flat_index, to_series, union, view

## DataFrameGroupBy: 56/64 names defined

Missing: boxplot, hist, indices, level, ohlc, plot, resample, sample

## Rolling: 23/24 names defined

Missing: exclusions

## pyriemann._version: 0/0

Missing: -

## pyriemann.artifact_detection: 0/2

Missing: Potato, PotatoField

## pyriemann.channelselection: 2/2

Missing: -

## pyriemann.classification: 6/8

Missing: NearestConvexHull, class_distinctiveness

## pyriemann.clustering: 1/5

Missing: Gaussian, GaussianMixture, KmeansPerClassTransform, MeanShift

## pyriemann.datasets: 0/0

Missing: -

## pyriemann.datasets.sampling: 0/2

Missing: RandomOverSampler, sample_gaussian

## pyriemann.datasets.simulated: 0/5

Missing: make_classification_transfer, make_gaussian_blobs, make_masks, make_matrices, make_outliers

## pyriemann.embedding: 0/5

Missing: LocallyLinearEmbedding, SpectralEmbedding, TSNE, barycenter_weights, locally_linear_embedding

## pyriemann.estimation: 10/10

Missing: -

## pyriemann.geometry: 0/0

Missing: -

## pyriemann.geometry._backend: 2/11

Missing: apply_xp_cov, as_numpy, diag_indices, fast_mcd, from_numpy, hann_window, nanmean, tril_indices, triu_indices

## pyriemann.geometry._check: 1/5

Missing: check_function, check_init, check_like, check_matrix_pair

## pyriemann.geometry._docs: 0/1

Missing: deprecated

## pyriemann.geometry.ajd: 4/4

Missing: -

## pyriemann.geometry.base: 9/9

Missing: -

## pyriemann.geometry.covariance: 10/11

Missing: eegtocov

## pyriemann.geometry.distance: 16/16

Missing: -

## pyriemann.geometry.geodesic: 8/8

Missing: -

## pyriemann.geometry.kernel: 4/4

Missing: -

## pyriemann.geometry.mean: 17/19

Missing: maskedmean_riemann, nanmean_riemann

## pyriemann.geometry.median: 2/2

Missing: -

## pyriemann.geometry.tangentspace: 23/27

Missing: exp_map_logchol, innerproduct_logchol, log_map_logchol, transport_logchol

## pyriemann.geometry.test: 13/13

Missing: -

## pyriemann.optimization: 0/0

Missing: -

## pyriemann.optimization.grassmann: 0/0

Missing: -

## pyriemann.optimization.positive_definite: 0/0

Missing: -

## pyriemann.preprocessing: 1/1

Missing: -

## pyriemann.regression: 0/2

Missing: KNearestNeighborRegressor, SVR

## pyriemann.spatialfilters: 4/5

Missing: AJDC

## pyriemann.stats: 0/5

Missing: BasePermutation, PermutationDistance, PermutationModel, multiset_perm_number, unique_permutations

## pyriemann.tangentspace: 2/2

Missing: -

## pyriemann.transfer: 0/0

Missing: -

## pyriemann.transfer._estimators: 3/8

Missing: MDWM, TLClassifier, TLEstimator, TLRegressor, TLRotate

## pyriemann.transfer._tools: 2/3

Missing: TLSplitter

## pyriemann.utils: 0/0

Missing: -

## pyriemann.utils._base: 0/3

Missing: SpdClassifMixin, SpdClustMixin, SpdTransfMixin

## pyriemann.utils._check: 0/2

Missing: check_metric, check_param_in_func

## pyriemann.utils._data: 0/1

Missing: get_data_path

## pyriemann.utils._logging: 0/0

Missing: -

## pyriemann.utils.ajd: 0/0

Missing: -

## pyriemann.utils.base: 0/0

Missing: -

## pyriemann.utils.covariance: 0/0

Missing: -

## pyriemann.utils.distance: 0/0

Missing: -

## pyriemann.utils.geodesic: 0/0

Missing: -

## pyriemann.utils.kernel: 0/0

Missing: -

## pyriemann.utils.mean: 0/0

Missing: -

## pyriemann.utils.median: 0/0

Missing: -

## pyriemann.utils.tangentspace: 0/0

Missing: -

## pyriemann.utils.test: 0/0

Missing: -

## pyriemann.utils.utils: 0/0

Missing: -

## pyriemann.utils.viz: 0/6

Missing: plot_bihist, plot_biscatter, plot_cospectra, plot_cov_ellipse, plot_embedding, plot_waveforms
