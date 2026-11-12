# Ported from scipy 1.18.1 (BSD-3-Clause)

from ._stats import ttest_ind, ttest_rel, ttest_1samp, pearsonr, spearmanr, \
                    linregress, zscore, zmap, sem, describe, skew, kurtosis, \
                    moment, rankdata, tiecorrect, f_oneway, kruskal, \
                    mannwhitneyu, wilcoxon, chi2_contingency, TtestResult, \
                    LinregressResult, DescribeResult

# --- distributions (lane scipy-stats-dists) ---
from ._distn_infrastructure import rv_continuous, rv_frozen, check_random_state
from ._continuous_distns import norm, t, f, chi2, uniform, expon, gamma, beta, \
    lognorm, laplace, logistic, cauchy, weibull_min, weibull_max, gumbel_r, \
    gumbel_l, pareto, lomax, chi, rayleigh, maxwell, halfnorm, halfcauchy, \
    invgamma, invgauss, wald, rice, truncexpon, triang

# --- tests / descriptive (lane scipy-stats-tests) ---
from ._descriptive import gmean, hmean, pmean, mode, tmean, tvar, tstd, tsem, \
    tmin, tmax, trimboth, trim1, trim_mean, variation, iqr, \
    median_abs_deviation, percentileofscore, scoreatpercentile, entropy, \
    differential_entropy, cumfreq, relfreq, find_repeats, \
    wasserstein_distance, energy_distance, circmean, circvar, circstd, \
    false_discovery_control, boxcox, boxcox_llf, boxcox_normmax, yeojohnson, \
    yeojohnson_llf, yeojohnson_normmax, ModeResult
