# pandas / pyriemann API coverage (generated)

## pandas (top-level): 12/120 names defined

Missing: ArrowDtype, BooleanDtype, Categorical, CategoricalDtype, CategoricalIndex, DateOffset, DatetimeIndex, DatetimeTZDtype, ExcelFile, ExcelWriter, Flags, Float32Dtype, Float64Dtype, Grouper, HDFStore, IndexSlice, Int16Dtype, Int32Dtype, Int64Dtype, Int8Dtype, Interval, IntervalDtype, IntervalIndex, MultiIndex, NA, NaT, NamedAgg, NamedFunc, Period, PeriodDtype, PeriodIndex, RangeIndex, SparseDtype, StringDtype, Timedelta, TimedeltaIndex, Timestamp, UInt16Dtype, UInt32Dtype, UInt64Dtype, UInt8Dtype, annotations, api, array, arrays, bdate_range, col, compat, core, crosstab, cut, date_range, describe_option, errors, eval, factorize, from_dummies, get_dummies, get_option, infer_freq, interval_range, io, json_normalize, lreshape, melt, merge_asof, merge_ordered, offsets, option_context, options, pandas, period_range, pivot, pivot_table, plotting, qcut, read_clipboard, read_excel, read_feather, read_fwf, read_hdf, read_html, read_iceberg, read_json, read_orc, read_parquet, read_pickle, read_sas, read_spss, read_sql, read_sql_query, read_sql_table, read_stata, read_table, read_xml, reset_option, set_eng_float_format, set_option, show_versions, test, testing, timedelta_range, to_datetime, to_pickle, to_timedelta, tseries, util, wide_to_long

## DataFrame: 100/204 names defined

Missing: add_prefix, add_suffix, align, asfreq, asof, assign, at_time, attrs, axes, between_time, boxplot, columns, combine, combine_first, compare, convert_dtypes, divide, dot, droplevel, eq, eval, ewm, explode, filter, first_valid_index, flags, floordiv, from_arrow, from_dict, from_records, ge, gt, hist, infer_objects, interpolate, isetitem, join, kurt, kurtosis, last_valid_index, le, lt, melt, memory_usage, mod, mode, multiply, ne, pipe, pivot, pivot_table, plot, pow, query, radd, rdiv, reindex, reindex_like, rename_axis, reorder_levels, resample, rfloordiv, rmod, rmul, rpow, rsub, rtruediv, sample, select, set_axis, set_flags, skew, sparse, squeeze, stack, style, subtract, swaplevel, to_clipboard, to_excel, to_feather, to_hdf, to_html, to_iceberg, to_json, to_latex, to_markdown, to_orc, to_parquet, to_period, to_pickle, to_records, to_sql, to_stata, to_timestamp, to_xarray, to_xml, truediv, truncate, tz_convert, tz_localize, unstack, update, xs

## Series: 102/203 names defined

Missing: add_prefix, add_suffix, align, argsort, array, asfreq, asof, at_time, attrs, autocorr, axes, between_time, case_when, cat, combine, combine_first, compare, convert_dtypes, divide, divmod, dot, droplevel, dt, eq, ewm, explode, factorize, filter, first_valid_index, flags, floordiv, from_arrow, ge, gt, hasnans, hist, infer_objects, interpolate, is_monotonic_decreasing, is_monotonic_increasing, item, kurt, kurtosis, last_valid_index, le, list, lt, memory_usage, mod, mode, multiply, name, nbytes, ne, pipe, plot, pow, radd, rdiv, rdivmod, reindex, reindex_like, rename_axis, reorder_levels, repeat, resample, rfloordiv, rmod, rmul, rpow, rsub, rtruediv, sample, searchsorted, set_axis, set_flags, skew, sparse, squeeze, str, struct, subtract, swaplevel, to_clipboard, to_excel, to_hdf, to_json, to_latex, to_markdown, to_period, to_pickle, to_sql, to_timestamp, to_xarray, truediv, truncate, tz_convert, tz_localize, unstack, update, xs

## Index: 48/92 names defined

Missing: argsort, array, asof, asof_locs, delete, difference, droplevel, factorize, get_indexer, get_indexer_for, get_indexer_non_unique, get_level_values, get_slice_bound, has_duplicates, hasnans, identical, infer_objects, inferred_type, intersection, is_, is_monotonic_decreasing, is_monotonic_increasing, item, join, memory_usage, name, names, nbytes, nlevels, putmask, ravel, reindex, repeat, searchsorted, set_names, slice_indexer, slice_locs, sortlevel, str, symmetric_difference, to_flat_index, to_series, union, view

## DataFrameGroupBy: 48/64 names defined

Missing: boxplot, cumcount, ewm, filter, hist, indices, kurt, level, ngroup, nth, ohlc, pipe, plot, resample, sample, skew

## Rolling: 20/24 names defined

Missing: exclusions, kurt, pipe, skew

## pyriemann._version: 0/0

Missing: -

## pyriemann.artifact_detection: 0/2

Missing: Potato, PotatoField

## pyriemann.channelselection: 0/2

Missing: ElectrodeSelection, FlatChannelRemover

## pyriemann.classification: 1/8

Missing: FgMDM, KNearestNeighbor, MeanField, NearestConvexHull, SVC, TSClassifier, class_distinctiveness

## pyriemann.clustering: 0/5

Missing: Gaussian, GaussianMixture, Kmeans, KmeansPerClassTransform, MeanShift

## pyriemann.datasets: 0/0

Missing: -

## pyriemann.datasets.sampling: 0/2

Missing: RandomOverSampler, sample_gaussian

## pyriemann.datasets.simulated: 0/5

Missing: make_classification_transfer, make_gaussian_blobs, make_masks, make_matrices, make_outliers

## pyriemann.embedding: 0/5

Missing: LocallyLinearEmbedding, SpectralEmbedding, TSNE, barycenter_weights, locally_linear_embedding

## pyriemann.estimation: 1/10

Missing: BlockCovariances, CoSpectra, Coherences, CrossSpectra, ERPCovariances, Kernels, Shrinkage, TimeDelayCovariances, XdawnCovariances

## pyriemann.geometry: 0/0

Missing: -

## pyriemann.geometry._backend: 2/11

Missing: apply_xp_cov, as_numpy, diag_indices, fast_mcd, from_numpy, hann_window, nanmean, tril_indices, triu_indices

## pyriemann.geometry._check: 1/5

Missing: check_function, check_init, check_like, check_matrix_pair

## pyriemann.geometry._docs: 0/1

Missing: deprecated

## pyriemann.geometry.ajd: 0/4

Missing: ajd, ajd_pham, rjd, uwedge

## pyriemann.geometry.base: 9/9

Missing: -

## pyriemann.geometry.covariance: 5/11

Missing: block_covariances, coherence, covariance_mest, covariances_EP, cross_spectrum, eegtocov

## pyriemann.geometry.distance: 15/16

Missing: distance_logchol

## pyriemann.geometry.geodesic: 7/8

Missing: geodesic_logchol

## pyriemann.geometry.kernel: 0/4

Missing: kernel, kernel_euclid, kernel_logeuclid, kernel_riemann

## pyriemann.geometry.mean: 11/19

Missing: maskedmean_riemann, mean_ale, mean_alm, mean_bmp, mean_cheap, mean_logchol, mean_thompson, nanmean_riemann

## pyriemann.geometry.median: 0/2

Missing: median_euclid, median_riemann

## pyriemann.geometry.tangentspace: 23/27

Missing: exp_map_logchol, innerproduct_logchol, log_map_logchol, transport_logchol

## pyriemann.geometry.test: 0/13

Missing: is_hankel, is_herm_pos_def, is_herm_pos_semi_def, is_hermitian, is_pos_def, is_pos_semi_def, is_real, is_real_type, is_skew_sym, is_square, is_sym, is_sym_pos_def, is_sym_pos_semi_def

## pyriemann.optimization: 0/0

Missing: -

## pyriemann.optimization.grassmann: 0/0

Missing: -

## pyriemann.optimization.positive_definite: 0/0

Missing: -

## pyriemann.preprocessing: 0/1

Missing: Whitening

## pyriemann.regression: 0/2

Missing: KNearestNeighborRegressor, SVR

## pyriemann.spatialfilters: 0/5

Missing: AJDC, BilinearFilter, CSP, SPoC, Xdawn

## pyriemann.stats: 0/5

Missing: BasePermutation, PermutationDistance, PermutationModel, multiset_perm_number, unique_permutations

## pyriemann.tangentspace: 1/2

Missing: FGDA

## pyriemann.transfer: 0/0

Missing: -

## pyriemann.transfer._estimators: 0/8

Missing: MDWM, TLCenter, TLClassifier, TLDummy, TLEstimator, TLRegressor, TLRotate, TLScale

## pyriemann.transfer._tools: 0/3

Missing: TLSplitter, decode_domains, encode_domains

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
