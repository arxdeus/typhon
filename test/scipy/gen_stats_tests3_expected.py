# Generates expected values for test/scipy/test_stats_tests3.py and test_mstats.py (scipy 1.18.1).
# Usage: python gen_stats_tests3_expected.py <tests3|mstats>
import sys
import numpy as np
import scipy.stats as st
from scipy.stats import mstats

def emit(name, v):
    a = np.asarray(v, dtype=float)
    print(f"{name} = {a.tolist()!r}")

nan = float('nan')
x = np.array([2.1, 3.4, 1.9, 5.6, 4.2, 3.3, 2.8, 6.1, 4.4, 3.9, 2.2, 5.0])
y = np.array([1.8, 2.9, 3.1, 4.7, 5.2, 2.6, 3.0, 5.9, 3.5, 4.1, 2.5, 4.4])
xn = x.copy(); xn[[2, 7]] = nan
yn = y.copy(); yn[[4]] = nan
X2 = np.cos(np.arange(24.)).reshape(4, 6) * 3 + 4
Z = np.array([0.5, -1.2, 2.2, -0.3, 1.1, -2.5, 0.7, 0.0, 1.9, -0.8])

def tests3():
    out = []
    r = st.ttest_ind(xn, yn, nan_policy='omit'); out += [r.statistic, r.pvalue, r.df]
    r = st.ttest_1samp(xn, 3.0, nan_policy='omit'); out += [r.statistic, r.pvalue]
    r = st.ttest_rel(xn, yn, nan_policy='omit'); out += [r.statistic, r.pvalue]
    emit('tt_omit', out)
    out = []
    r = st.mannwhitneyu(xn, yn, nan_policy='omit'); out += [r.statistic, r.pvalue]
    r = st.kendalltau(xn, yn, nan_policy='omit'); out += [r.statistic, r.pvalue]
    r = st.ks_2samp(xn, yn, nan_policy='omit'); out += [r.statistic, r.pvalue]
    r = st.ks_1samp(xn, st.norm.cdf, nan_policy='omit'); out += [r.statistic, r.pvalue]
    emit('np_omit', out)
    emit('zs_omit', st.zscore(xn, nan_policy='omit'))
    emit('rk_omit', st.rankdata(xn, nan_policy='omit'))
    emit('rk_prop', st.rankdata(xn))
    emit('gz', st.gzscore(x))
    r = st.sigmaclip(np.concatenate([x, [40.0, -30.0]]), 2.0, 2.0); emit('sc', list(r.clipped) + [r.lower, r.upper])
    ot = st.obrientransform(x, y); emit('ob', list(ot[0]) + list(ot[1]))
    ds = st.directional_stats(np.array([[3., 4], [6, -8], [1, 2]])); emit('ds', list(ds.mean_direction) + [ds.mean_resultant_length])
    emit('ex', [st.expectile(x), st.expectile(x, 0.2), st.expectile(x, 0.8, weights=y)])
    emit('lm', st.lmoment(x)); emit('lm2', st.lmoment(x, order=[1, 3, 5])); emit('lm3', st.lmoment(X2, axis=1, standardize=False))
    r = st.chatterjeexi(x, y); emit('xi', [r.statistic, r.pvalue])
    r = st.chatterjeexi(np.round(x), np.round(y)); emit('xi2', [r.statistic, r.pvalue])
    r = st.chatterjeexi(x, y, y_continuous=True); emit('xi3', [r.statistic, r.pvalue])
    out = []
    for alt in ['two-sided', 'less', 'greater']:
        r = st.quantile_test(x, q=3.5, p=0.4, alternative=alt); out += [r.statistic, r.statistic_type, r.pvalue]
        ci = r.confidence_interval(0.9); out += [ci.low, ci.high]
    emit('qt', out)
    emit('sl', [st.siegelslopes(y, x, method='separate').slope, st.siegelslopes(y, x, method='separate').intercept])

def mstats_():
    emit('gm', [mstats.gmean(x), mstats.hmean(x)])
    emit('mq', mstats.mquantiles(x))
    emit('mq2', mstats.mquantiles(X2, prob=[0.1, 0.5, 0.9], axis=1))
    emit('mq3', mstats.mquantiles(x, prob=[0.3, 0.7], alphap=0.0, betap=0.0))
    emit('wz', np.asarray(mstats.winsorize(x, limits=[0.1, 0.2])))
    emit('wz2', np.asarray(mstats.winsorize(X2, limits=0.25, axis=1)))
    emit('tm', [mstats.trimmed_mean(x, (0.1, 0.1)), mstats.trimmed_mean(x, (2.0, 5.0), relative=False),
                mstats.trimmed_var(x, (0.1, 0.2)), mstats.trimmed_std(x, (0.1, 0.2), ddof=1)])

{'tests3': tests3, 'mstats': mstats_}[sys.argv[1]]()
