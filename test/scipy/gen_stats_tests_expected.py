# Generates expected values for test/scipy/test_stats_desc.py, test_stats_tests1.py,
# test_stats_tests2.py and test_stats_qmc.py (scipy 1.18.1).
# Usage: python gen_stats_tests_expected.py <desc|tests1|tests2|qmc>
# Prints `name = value` lines to paste into the test file header.
import sys
import numpy as np
import scipy.stats as st

def fl(v):
    a = np.asarray(v, dtype=float)
    return repr(a.tolist()) if a.ndim else repr(float(a))

def emit(name, v):
    print(f"{name} = {fl(v)}")

x = np.array([2.1, 3.4, 1.9, 5.6, 4.2, 3.3, 2.8, 6.1, 4.4, 3.9, 2.2, 5.0])
y = np.array([1.8, 2.9, 3.1, 4.7, 5.2, 2.6, 3.0, 5.9, 3.5, 4.1, 2.5, 4.4])
X2 = np.cos(np.arange(24.)).reshape(4, 6) * 3 + 4
Z = np.array([0.5, -1.2, 2.2, -0.3, 1.1, -2.5, 0.7, 0.0, 1.9, -0.8])

def desc():
    emit('gm', st.gmean(x)); emit('gm_ax1', st.gmean(X2, axis=1))
    emit('gm_w', st.gmean(x, weights=y))
    emit('hm', st.hmean(x)); emit('hm_w', st.hmean(x, weights=y))
    emit('pm', [st.pmean(x, 2.5), st.pmean(x, -1.5), st.pmean(x, 1e-7), st.pmean(x, 3, weights=y)])
    m = st.mode(np.array([1., 2., 2., 3., 3., 3., 4.]))
    emit('mo', [m.mode, m.count])
    m2 = st.mode(np.array([[1, 2, 2], [3, 3, 1], [1, 2, 3]]).astype(float), axis=0)
    emit('mo2', [m2.mode, m2.count])
    emit('tmn', [st.tmean(x), st.tmean(x, (2.0, 5.0)), st.tmean(x, (2.2, 5.0), (False, True))])
    emit('tvr', [st.tvar(x), st.tvar(x, (2.0, 5.0)), st.tstd(x, (2.0, 5.0)), st.tsem(x, (2.0, 5.0)), st.tsem(x)])
    emit('tmm', [st.tmin(x, 2.5), st.tmax(x, 5.0), st.tmin(x), st.tmax(x, 5.0, inclusive=False)])
    emit('tmean2', st.tmean(X2, (2.0, 6.0)))
    emit('tvar_ax', st.tvar(X2, (2.0, 6.5), axis=1))
    emit('trb', np.sort(st.trimboth(x, 0.2)))
    emit('tr1', np.sort(st.trim1(x, 0.25)))
    emit('tr1l', np.sort(st.trim1(x, 0.25, tail='left')))
    emit('trm', [st.trim_mean(x, 0.1), st.trim_mean(x, 0.25)])
    emit('trm_ax', st.trim_mean(X2, 0.2, axis=1))
    emit('var', [st.variation(x), st.variation(x, ddof=1)])
    emit('var_ax', st.variation(X2, axis=0))
    emit('iq', [st.iqr(x), st.iqr(x, rng=(10, 80)), st.iqr(x, scale='normal'),
                st.iqr(x, interpolation='lower'), st.iqr(x, interpolation='midpoint'),
                st.iqr(x, interpolation='nearest'), st.iqr(x, interpolation='higher')])
    emit('iq_ax', st.iqr(X2, axis=1))
    emit('mad', [st.median_abs_deviation(x), st.median_abs_deviation(x, scale='normal')])
    emit('mad_ax', st.median_abs_deviation(X2, axis=1))
    emit('pos', [st.percentileofscore(x, 3.4), st.percentileofscore(x, 3.4, kind='strict'),
                 st.percentileofscore(x, 3.4, kind='weak'), st.percentileofscore(x, 3.4, kind='mean'),
                 st.percentileofscore(x, 3.5)])
    emit('sap', [st.scoreatpercentile(x, 30), st.scoreatpercentile(x, 50, interpolation_method='lower'),
                 st.scoreatpercentile(x, 37, interpolation_method='higher'),
                 st.scoreatpercentile(x, 40, limit=(2.0, 5.0))])
    emit('sap_v', st.scoreatpercentile(x, [10, 90]))
    pk = np.array([0.1, 0.2, 0.3, 0.4]); qk = np.array([0.25, 0.25, 0.3, 0.2])
    emit('ent', [st.entropy(pk), st.entropy(pk, base=2), st.entropy(pk, qk), st.entropy([3, 1, 0, 2.])])
    emit('ent_ax', st.entropy(X2, axis=1))
    rng = np.random.default_rng(0)
    big = np.cos(np.arange(40.) * 1.3) * 2 + np.arange(40.) * 0.1
    emit('big', big)
    emit('dent', [st.differential_entropy(x), st.differential_entropy(big),
                  st.differential_entropy(big, method='vasicek'), st.differential_entropy(big, method='van es'),
                  st.differential_entropy(big, method='correa'), st.differential_entropy(big, window_length=3, base=2)])
    c = st.cumfreq(x, numbins=4)
    emit('cf', list(c.cumcount) + [c.lowerlimit, c.binsize, c.extrapoints])
    r = st.relfreq(x, numbins=5, defaultreallimits=(2.0, 5.0))
    emit('rf', list(r.frequency) + [r.lowerlimit, r.binsize, r.extrapoints])
    r = st.relfreq(x, numbins=3, weights=y)
    emit('rfw', list(r.frequency) + [r.lowerlimit, r.binsize, r.extrapoints])
    from scipy.stats import mstats
    fr = mstats.find_repeats([2., 1., 2., 3., 2., 2., 5., 3.])
    emit('frv', fr[0]); emit('frc', fr[1])
    emit('wd', [st.wasserstein_distance(x, y), st.wasserstein_distance(x, y[:7], y, y[:7]),
                st.energy_distance(x, y), st.energy_distance(x, y[:7], y, y[:7])])
    ang = np.array([0.1, 6.2, 0.4, 5.9, 1.0, 0.3])
    emit('circ', [st.circmean(ang), st.circvar(ang), st.circstd(ang), st.circstd(ang, normalize=True),
                  st.circmean(np.array([355., 5., 2., 359.]), high=360), st.circstd(np.array([355., 5., 2., 359.]), high=360)])
    emit('circ_ax', st.circmean(X2, axis=1))
    ps = np.array([0.01, 0.04, 0.03, 0.2, 0.5, 0.008, 0.9])
    emit('fdr', st.false_discovery_control(ps)); emit('fdr_by', st.false_discovery_control(ps, method='by'))
    emit('bcllf', [st.boxcox_llf(0.5, x), st.boxcox_llf(0.0, x), st.boxcox_llf(-1.3, x)])
    yb, lm = st.boxcox(x)
    emit('bc_l', lm); emit('bc_y', yb)
    emit('bc_fix', st.boxcox(x, 0.3))
    emit('bcn', [st.boxcox_normmax(x), st.boxcox_normmax(x, method='mle')])
    _, _, ci = st.boxcox(x, alpha=0.05)
    emit('bc_ci', ci)
    emit('yjllf', [st.yeojohnson_llf(0.5, Z), st.yeojohnson_llf(0.0, x), st.yeojohnson_llf(1.7, -x)])
    yj, lj = st.yeojohnson(Z)
    emit('yj_l', lj); emit('yj_y', yj)
    emit('yj_fix', st.yeojohnson(Z, 0.7))
    emit('yjn', [st.yeojohnson_normmax(x), st.yeojohnson_normmax(-x)])

def tests1():
    big = np.cos(np.arange(40.) * 1.3) * 2 + np.arange(40.) * 0.1
    emit('big', big)
    for nm, f in [('skt', st.skewtest), ('kut', st.kurtosistest), ('nt', st.normaltest)]:
        r = f(big); emit(nm, [r.statistic, r.pvalue])
    r = st.skewtest(big, alternative='less'); emit('skt_l', [r.statistic, r.pvalue])
    r = st.kurtosistest(x, alternative='greater'); emit('kut_g', [r.statistic, r.pvalue])
    r = st.normaltest(big.reshape(8, 5), axis=0); emit('nt_ax', [r.statistic, r.pvalue])
    r = st.jarque_bera(big); emit('jb', [r.statistic, r.pvalue])
    obs = np.array([16., 18., 16., 14., 12., 12.]); ex = np.array([16., 16., 16., 16., 16., 8.])
    out = []
    for args in [(obs,), (obs, ex)]:
        r = st.chisquare(*args); out += [r.statistic, r.pvalue]
    r = st.chisquare(obs, ddof=1); out += [r.statistic, r.pvalue]
    for lam in ['log-likelihood', 'freeman-tukey', 'mod-log-likelihood', 'neyman', 'cressie-read', 1.5]:
        r = st.power_divergence(obs, ex, lambda_=lam); out += [r.statistic, r.pvalue]
    emit('chi', out)
    r = st.chisquare(np.array([[16., 18.], [16., 14.], [12., 12.]])); emit('chi_ax', [r.statistic, r.pvalue])
    out = []
    for ev in [True, False]:
        for alt in ['two-sided', 'less', 'greater']:
            r = st.ttest_ind_from_stats(15.0, 2.0, 13, 13.5, 3.1, 11, equal_var=ev, alternative=alt); out += [r.statistic, r.pvalue]
    emit('tfs', out)
    b = (x > 3.5).astype(float)
    r = st.pointbiserialr(b, y); emit('pbs', [r.statistic, r.pvalue])
    out = []
    for alt in ['two-sided', 'less', 'greater']:
        r = st.ranksums(x, y[:9], alternative=alt); out += [r.statistic, r.pvalue]
        r = st.brunnermunzel(x, y[:9], alternative=alt); out += [r.statistic, r.pvalue]
    r = st.brunnermunzel(x, y[:9], distribution='normal'); out += [r.statistic, r.pvalue]
    emit('rk', out)
    r = st.friedmanchisquare(x[:6], y[:6], x[6:]); emit('fr', [r.statistic, r.pvalue])
    r = st.friedmanchisquare(np.array([1., 2, 3, 4]), np.array([1., 3, 3, 5]), np.array([2., 2, 4, 4])); emit('fr_t', [r.statistic, r.pvalue])
    r = st.alexandergovern(x, y[:9], big[:7]); emit('ag', [r.statistic, r.pvalue])
    out = []
    for alt in ['two-sided', 'less', 'greater']:
        r = st.kendalltau(x, y, alternative=alt); out += [r.statistic, r.pvalue]
    xt = np.round(x); yt = np.round(y)
    r = st.kendalltau(xt, yt); out += [r.statistic, r.pvalue]
    r = st.kendalltau(xt, yt, variant='c'); out += [r.statistic, r.pvalue]
    r = st.kendalltau(x, y, method='asymptotic'); out += [r.statistic, r.pvalue]
    r = st.kendalltau(big, big[::-1] ** 2); out += [r.statistic, r.pvalue]
    emit('kt', out)
    out = []
    r = st.theilslopes(y, x); out += list(r)
    r = st.theilslopes(y, x, method='joint', alpha=0.9); out += list(r)
    r = st.theilslopes(big); out += list(r)
    r = st.siegelslopes(y, x); out += list(r)
    r = st.siegelslopes(y, x, method='separate'); out += list(r)
    emit('slopes', out)
    ps = np.array([0.01, 0.2, 0.3, 0.04, 0.5])
    out = []
    for m in ['fisher', 'pearson', 'mudholkar_george', 'tippett', 'stouffer']:
        r = st.combine_pvalues(ps, method=m); out += [r.statistic, r.pvalue]
    r = st.combine_pvalues(ps, method='stouffer', weights=np.array([1., 2, 3, 4, 5])); out += [r.statistic, r.pvalue]
    emit('cp', out)

{'desc': desc, 'tests1': tests1}[sys.argv[1]]()
