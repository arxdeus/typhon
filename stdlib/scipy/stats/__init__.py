# Ported from scipy 1.18.1 (BSD-3-Clause)

from ._stats import ttest_ind, ttest_rel, ttest_1samp, pearsonr, spearmanr, \
                    linregress, zscore, zmap, sem, describe, skew, kurtosis, \
                    moment, rankdata, tiecorrect, f_oneway, kruskal, \
                    mannwhitneyu, wilcoxon, chi2_contingency, TtestResult, \
                    LinregressResult, DescribeResult

# --- distributions (lane scipy-stats-dists) ---
from ._distn_infrastructure import rv_continuous, rv_frozen, check_random_state
from ._continuous_distns import norm, t, f, chi2, uniform, expon, gamma, beta, \
    lognorm, laplace, logistic, cauchy
