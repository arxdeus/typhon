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
    invgamma, invgauss, wald, rice, truncexpon, triang, genextreme, genpareto, \
    genlogistic, gompertz, hypsecant, loglaplace, powerlaw, burr, fisk, burr12, \
    dweibull, exponweib, levy, levy_l, moyal, arcsine, anglit, bradford, rdist, \
    semicircular, gibrat, mielke, kappa3
from ._continuous_distns2 import erlang, reciprocal, loguniform, johnsonsb, \
    johnsonsu, skewnorm, exponnorm, gennorm, foldnorm, foldcauchy, nakagami, \
    loggamma, alpha, dgamma, betaprime, recipinvgauss, fatiguelife, wrapcauchy, \
    cosine, vonmises, truncnorm, ncx2, ncf, nct, pearson3
from ._continuous_distns3 import halflogistic, genhalflogistic, halfgennorm, \
    invweibull, exponpow, powernorm, powerlognorm, laplace_asymmetric, trapezoid, \
    truncweibull_min, truncpareto, gengamma, genexpon, kappa4, crystalball, argus, \
    tukeylambda, kstwobign, rel_breitwigner
from ._continuous_distns4 import skewcauchy, irwinhall, gausshyper, ksone, \
    norminvgauss
from ._distn_infrastructure import rv_discrete, rv_discrete_frozen, rv_histogram
from ._discrete_distns import binom, bernoulli, betabinom, nbinom, betanbinom, \
    geom, hypergeom, nhypergeom, logser, poisson, planck, boltzmann, randint, \
    zipf, zipfian, dlaplace, skellam, yulesimon
from ._multivariate import multivariate_normal, multivariate_t, dirichlet, \
    multinomial, wishart, invwishart, matrix_normal, ortho_group, \
    special_ortho_group, random_correlation

# --- tests / descriptive (lane scipy-stats-tests) ---
from ._descriptive import gmean, hmean, pmean, mode, tmean, tvar, tstd, tsem, \
    tmin, tmax, trimboth, trim1, trim_mean, variation, iqr, \
    median_abs_deviation, percentileofscore, scoreatpercentile, entropy, \
    differential_entropy, cumfreq, relfreq, find_repeats, \
    wasserstein_distance, energy_distance, circmean, circvar, circstd, \
    false_discovery_control, boxcox, boxcox_llf, boxcox_normmax, yeojohnson, \
    yeojohnson_llf, yeojohnson_normmax, ModeResult
from ._stats_py2 import skewtest, kurtosistest, normaltest, jarque_bera, \
    power_divergence, chisquare, ttest_ind_from_stats, pointbiserialr, \
    ranksums, brunnermunzel, friedmanchisquare, alexandergovern, kendalltau, \
    theilslopes, siegelslopes, combine_pvalues
from ._morestats import shapiro, anderson, anderson_ksamp, ansari, mood, \
    levene, bartlett, fligner, median_test, binomtest, fisher_exact, \
    ks_1samp, ks_2samp, kstest, cramervonmises, cramervonmises_2samp, \
    poisson_means_test, epps_singleton_2samp
from . import qmc
from . import contingency
from .contingency import somersd, page_trend_test, relative_risk, odds_ratio, \
    crosstab
from ._kde_binned import gaussian_kde, binned_statistic, binned_statistic_2d, \
    binned_statistic_dd, weightedtau
