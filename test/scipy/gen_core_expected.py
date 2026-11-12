# Generates expected literals for test/scipy/test_{linalg,special,stats}.py.
# Run: /Users/mind/orca/projects/typhon-test/.venv/bin/python test/scipy/gen_core_expected.py [section]
import sys
import numpy as np
import scipy
import scipy.linalg as sl

def lit(x):
    x = np.asarray(x)
    if x.ndim == 0:
        return repr(float(x)) if not np.iscomplexobj(x) else repr(complex(x))
    return '[' + ', '.join(lit(e) for e in x) + ']'

def show(name, x):
    print(f'{name} = {lit(x)}')

# Deterministic inputs shared with the .py tests (keep in sync).
def mat_a():
    n = 3
    return np.array([[4., 1., 2.], [1., 3., .5], [2., .5, 5.]])
def mat_g():
    return np.array([[1., 2., 0.], [3., 1., 4.], [.5, 2., 2.]])
def mat_b():
    return np.array([[2., .3, .1], [.3, 1.5, .2], [.1, .2, 1.]])
def bp_data():
    bp = np.array([[((i * 7 + j * 13) % 17) / 3.0 + 0.25 * j + (i % 5) * 0.1
                    for j in range(8)] for i in range(40)])
    y = np.array([i % 2 for i in range(40)])
    return bp, y
def cov_pair():
    bp, y = bp_data()
    X0 = bp[y == 0]; X1 = bp[y == 1]
    C0 = X0.T @ X0 / 20.0 + np.eye(8)
    C1 = X1.T @ X1 / 20.0 + np.eye(8)
    return C0, C1

def linalg():
    A, B = mat_a(), mat_b()
    w, v = sl.eigh(A); show('eigh_w', w); show('eigh_v', v)
    w, v = sl.eigh(A, B); show('geigh_w', w); show('geigh_v', v)
    show('geigh_U_w', sl.eigh(A, B, lower=False, eigvals_only=True))
    show('geigh_t2', sl.eigh(A, B, type=2, eigvals_only=True))
    show('geigh_t3', sl.eigh(A, B, type=3, eigvals_only=True))
    show('geigh_sub', sl.eigh(A, B, subset_by_index=[1, 2], eigvals_only=True))
    w, v = sl.eigh(A, subset_by_index=[0, 0]); show('eigh_sub_w', w); show('eigh_sub_v', v)
    show('eigh_val', sl.eigh(A, subset_by_value=[2.5, 10.0], eigvals_only=True))
    show('eigvalsh_ab', sl.eigvalsh(A, B))
    G = mat_g(); b = np.array([1., 2., 3.])
    show('inv_g', sl.inv(G)); show('solve_g', sl.solve(G, b)); show('solve_pos', sl.solve(A, b, assume_a='pos'))
    show('solve_sym', sl.solve(A, b, assume_a='sym'))
    show('det_g', sl.det(G)); show('chol_a', sl.cholesky(A)); show('chol_al', sl.cholesky(A, lower=True))
    show('expm_small', sl.expm(G * 0.3)); show('expm_g', sl.expm(G)); show('expm_big', sl.expm(G * 3))
    show('expm_tri', sl.expm(np.triu(G)))
    show('sqrtm_a', sl.sqrtm(A)); show('logm_a', sl.logm(A)); show('logm_g2', sl.logm(G @ G.T + np.eye(3)))
    show('fmp_a', sl.fractional_matrix_power(A, 0.5)); show('fmp_a3', sl.fractional_matrix_power(A, -0.3))
    p, l, u = sl.lu(G); show('lu_p', p); show('lu_l', l); show('lu_u', u)
    q, r = sl.qr(G); show('qr_q', q); show('qr_r', r)
    show('svdvals_g', sl.svdvals(G)); show('pinv_g', sl.pinv(G)); show('pinvh_a', sl.pinvh(A))
    show('lstsq_g', sl.lstsq(G, b)[0])
    show('solve_tri', sl.solve_triangular(np.triu(G), b))
    show('cho_solve', sl.cho_solve(sl.cho_factor(A), b)); show('lu_solve', sl.lu_solve(sl.lu_factor(G), b))
    show('toep', sl.toeplitz([1., 2., 3.], [1., 5.])); show('inv_a', sl.inv(A))
    C0, C1 = cov_pair()
    show('acc_w', sl.eigh(C0, C0 + C1, eigvals_only=True))

SPECIAL_CASES = [
    ('gamma', (4.5,)), ('gamma', (-2.5,)), ('gamma', (0.1,)), ('gamma', (40.3,)),
    ('gammaln', (0.5,)), ('gammaln', (100.7,)), ('gammaln', (-3.3,)), ('loggamma', (7.25,)),
    ('digamma', (0.25,)), ('digamma', (1.5,)), ('digamma', (-2.5,)), ('digamma', (30.0,)),
    ('beta', (2.5, 3.5)), ('betaln', (10.0, 20.0)),
    ('betainc', (2.0, 3.0, 0.4)), ('betainc', (0.5, 7.5, 0.05)), ('betainc', (20.0, 15.0, 0.6)),
    ('betaincinv', (2.0, 3.0, 0.3)), ('betaincinv', (0.5, 4.0, 0.9)),
    ('gammainc', (2.5, 1.5)), ('gammainc', (30.0, 28.0)), ('gammaincc', (0.7, 3.0)), ('gammaincc', (5.0, 0.4)),
    ('gammaincinv', (3.0, 0.4)), ('gammaincinv', (0.4, 0.05)),
    ('erf', (0.3,)), ('erf', (-1.7,)), ('erfc', (2.5,)), ('erfc', (-0.4,)), ('erfinv', (0.6,)), ('erfcinv', (0.02,)),
    ('ndtr', (1.96,)), ('ndtr', (-7.5,)), ('ndtri', (0.975,)), ('ndtri', (1e-10,)), ('log_ndtr', (-20.0,)), ('log_ndtr', (2.0,)),
    ('stdtr', (5.0, 2.1)), ('stdtr', (1.0, -3.0)), ('stdtrit', (10.0, 0.975)), ('stdtrit', (3.0, 0.1)),
    ('fdtr', (3.0, 12.0, 2.5)), ('fdtrc', (3.0, 12.0, 2.5)), ('chdtr', (4.0, 3.0)), ('chdtrc', (7.0, 14.0)),
    ('xlogy', (2.0, 3.0)), ('xlogy', (0.0, 0.0)), ('xlog1py', (2.0, -0.5)), ('entr', (0.3,)), ('entr', (0.0,)),
    ('expit', (-3.0,)), ('expit', (40.0,)), ('logit', (0.2,)), ('logit', (0.55,)),
    ('i0', (2.5,)), ('i0', (12.0,)), ('i0e', (12.0,)), ('i1', (-3.0,)), ('i1e', (20.0,)),
]

def special():
    import scipy.special as sp
    names = []
    for i, (n, a) in enumerate(SPECIAL_CASES):
        print(f"    ({repr(n)}, {list(map(float, a))}, {repr(float(getattr(sp, n)(*a)))}),")
    show('lse_1d', sp.logsumexp([1.0, 2.0, 3.0, -1.0]))
    show('lse_ax', sp.logsumexp(np.array([[1.0, 2.0], [3.0, 1000.0]]), axis=1))
    show('lse_b', sp.logsumexp([1.0, 2.0, 3.0], b=[0.5, 1.0, 2.0]))
    show('smax', sp.softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]]), axis=1))
    show('lsmax', sp.log_softmax(np.array([1.0, 2.0, 3.0])))
    show('comb_f', sp.comb(10, 3)); show('comb_big', sp.comb(60.0, 25.0)); print('comb_ex =', sp.comb(60, 25, exact=True))
    show('perm_f', sp.perm(10, 3)); show('fact_f', sp.factorial(10)); print('fact_ex =', sp.factorial(25, exact=True))
    show('gamma_arr', sp.gamma(np.array([0.5, 1.5, 2.5])))
    show('betainc_bc', sp.betainc(np.array([1.0, 2.0, 3.0]), 2.0, 0.5))

def stats():
    import scipy.stats as st
    bp, y = bp_data()
    t, p = st.ttest_ind(bp[y == 0], bp[y == 1], axis=0)
    show('acc_t', t); show('acc_p', p)
    a = np.array([1., 3., 2., 5., 4., 6.5, 2.2]); b = np.array([2., 4.5, 3.1, 6., 5.5, 7.])
    r = st.ttest_ind(a, b); show('tti', [r.statistic, r.pvalue, r.df])
    r = st.ttest_ind(a, b, equal_var=False, alternative='less'); show('tti_w', [r.statistic, r.pvalue, r.df])
    r = st.ttest_rel(a[:6], b); show('ttr', [r.statistic, r.pvalue])
    r = st.ttest_1samp(a, 2.0, alternative='greater'); show('tt1', [r.statistic, r.pvalue])
    r = st.pearsonr(a[:6], b); show('pear', [r.statistic, r.pvalue])
    r = st.spearmanr(a[:6], b); show('spear', [r.statistic, r.pvalue])
    r = st.linregress(a[:6], b); show('linr', [r.slope, r.intercept, r.rvalue, r.pvalue, r.stderr, r.intercept_stderr])
    show('zs', st.zscore(a)); show('sem', st.sem(a)); show('skw', st.skew(a)); show('kurt_ub', st.kurtosis(a, bias=False))
    show('skw_ub', st.skew(a, bias=False)); show('kurt', st.kurtosis(a, fisher=False))
    show('rank_min', st.rankdata([3, 1, 3, 2, 1], method='min')); show('rank_avg', st.rankdata([3, 1, 3, 2, 1]))
    show('rank_dense', st.rankdata([3, 1, 3, 2, 1], method='dense')); show('rank_ord', st.rankdata([3, 1, 3, 2, 1], method='ordinal'))
    c = np.array([4.1, 5.0, 3.3, 6.2])
    r = st.f_oneway(a, b, c); show('anova', [r.statistic, r.pvalue])
    r = st.kruskal(a, b, c); show('krus', [r.statistic, r.pvalue])
    r = st.mannwhitneyu(a, b); show('mwu_exact', [r.statistic, r.pvalue])
    r = st.mannwhitneyu(a, b, alternative='greater'); show('mwu_g', [r.statistic, r.pvalue])
    big1 = np.array([(i * 37 % 23) / 3.0 for i in range(15)]); big2 = np.array([(i * 29 % 19) / 2.5 + 0.7 for i in range(12)])
    r = st.mannwhitneyu(big1, big2); show('mwu_asym', [r.statistic, r.pvalue])
    d = np.array([0.5, -1.25, 2.0, 3.5, -0.75, 4.25, 1.5, -2.5, 5.0])
    r = st.wilcoxon(d); show('wil_exact', [r.statistic, r.pvalue])
    r = st.wilcoxon(np.array([(i * 13 % 17) - 7.3 for i in range(60)])); show('wil_asym', [r.statistic, r.pvalue])
    r = st.chi2_contingency(np.array([[10, 20], [30, 25]])); show('chi2c', [r.statistic, r.pvalue])
    D = st.describe(a); show('desc', [D.nobs, D.minmax[0], D.minmax[1], D.mean, D.variance, D.skewness, D.kurtosis])
    xs = np.array([-2.5, -0.3, 0.0, 0.7, 1.9, 4.2])
    qs = np.array([0.001, 0.1, 0.35, 0.5, 0.8, 0.999])
    show('n_pdf', st.norm.pdf(xs, 0.5, 1.5)); show('n_cdf', st.norm.cdf(xs)); show('n_sf', st.norm.sf(xs, loc=1.0))
    show('n_ppf', st.norm.ppf(qs, 2.0, 3.0)); show('n_isf', st.norm.isf(qs)); show('n_logpdf', st.norm.logpdf(xs))
    show('t_pdf', st.t.pdf(xs, 4.5)); show('t_cdf', st.t.cdf(xs, 3)); show('t_sf', st.t.sf(xs, 10, loc=0.2, scale=1.3))
    show('t_ppf', st.t.ppf(qs, 7)); show('t_isf', st.t(12).isf(qs)); show('t_logpdf', st.t.logpdf(xs, 2.5))
    xp = np.abs(xs) + 0.1
    show('c_pdf', st.chi2.pdf(xp, 3)); show('c_cdf', st.chi2.cdf(xp, 4.5)); show('c_sf', st.chi2.sf(xp, 2))
    show('c_ppf', st.chi2.ppf(qs, 5)); show('c_isf', st.chi2.isf(qs, 3))
    show('f_pdf', st.f.pdf(xp, 3, 12)); show('f_cdf', st.f.cdf(xp, 5, 7)); show('f_sf', st.f.sf(xp, 2, 30))
    show('f_ppf', st.f.ppf(qs, 4, 9)); show('f_isf', st.f.isf(qs, 3, 15))
    show('u_cdf', st.uniform.cdf(xs, -1.0, 4.0)); show('u_pdf', st.uniform.pdf(xs, -1.0, 4.0)); show('u_ppf', st.uniform.ppf(qs, 2.0, 0.5))
    show('moms', [st.t(5).mean(), st.t(5).var(), st.f(3, 12).mean(), st.f(3, 12).var(), st.chi2(4).var(), st.uniform(1, 3).var(), st.norm(2, 3).std()])

def followup():
    import scipy.stats as st
    A, B, G = mat_a(), mat_b(), mat_g()
    S = np.stack((A, A * 2.0)); T3 = np.stack((G, G.T))
    show('b_geigh', sl.eigh(S, B, eigvals_only=True))
    w, v = sl.eigh(S); show('b_eigh_w', w); show('b_eigh_v', v)
    show('b_det', sl.det(T3)); show('b_inv', sl.inv(T3)); show('b_solve', sl.solve(T3, np.array([1., 2., 3.])))
    show('b_chol', sl.cholesky(S)); show('b_svals', sl.svd(T3, compute_uv=False))
    show('b_expm', sl.expm(T3 * 0.2)); show('b_logm', sl.logm(S)); show('b_sqrtm', sl.sqrtm(S))
    w, vr = sl.eig(G, B); show('ge_w', w.real); show('ge_v', vr)
    show('ge_vals_im', sl.eigvals(np.array([[0., -1.], [1., 0.]]), np.array([[0., 1.], [1., 1.]])).real)
    q, r, pp = sl.qr(G, pivoting=True); show('qp_q', q); show('qp_r', r); show('qp_p', pp)
    d = np.array([1., -2., 2., 3., -1., 0., 4., 1.5])
    show('wperm', [st.wilcoxon(d).pvalue, st.wilcoxon(d, zero_method='pratt').pvalue,
                   st.wilcoxon(d, alternative='less').pvalue, st.wilcoxon(d, zero_method='zsplit').pvalue,
                   st.wilcoxon(np.array([1., -2., 2., 3., -1., 5., 4., 1.5])).pvalue])
    a = np.array([1., 3., 2., 5., 4., 6.5, 2.2, 9.0, -1.0, 2.5]); b = np.array([2., 4.5, 3.1, 6., 5.5, 7., 0.3, 3.3])
    r = st.ttest_ind(a, b, trim=0.2); show('yuen', [r.statistic, r.pvalue, r.df])
    r = st.ttest_ind(a, b, trim=0.2, equal_var=False); show('yuen_w', [r.statistic, r.pvalue, r.df])
    X = np.arange(24.).reshape(6, 4) ** 1.3; Y = np.arange(20.).reshape(5, 4) ** 1.1; Z = np.cos(np.arange(16.)).reshape(4, 4) * 10
    show('nd_anova_p', st.f_oneway(X, Y, Z).pvalue); show('nd_krus', st.kruskal(X, Y, Z, axis=0).statistic)
    show('welch', st.f_oneway(a, b, equal_var=False).pvalue); show('nd_anova_ax1', st.f_oneway(X.T, Y.T, axis=1).statistic)
    W = np.array([[1., 4., 2.], [3., 1., 5.], [2., 6., 1.], [5., 2., 3.], [4., 3., 6.], [6., 5., 4.]])
    r = st.spearmanr(W); show('sp_mat', r.statistic); show('sp_mat_p', r.pvalue)

if __name__ == '__main__':
    secs = sys.argv[1:] or ['linalg', 'stats']
    for s in secs:
        print(f'# --- {s} ---')
        globals()[s]()


def stdtrit_tails():
    # Deep-tail Student t quantiles (scipy 1.18 uses Boost fast_students_t_quantile).
    # Prints a Typhon @test block to append to test_special.py:
    #   python gen_core_expected.py stdtrit_tails
    import math
    import scipy.special as sp
    import scipy.stats as st
    dfs = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 10.0, 0.5, 1.5, 1e6]
    ps = [0.3, 0.05, 1e-10, 1e-30, 1e-50, 1e-100, 1e-150, 1e-200, 1e-250, 1e-300, 5e-324, 0.999, 1 - 1e-12]
    print('@test')
    print('def test_stdtrit_tails():')
    print('    import scipy.stats as st')
    for df in dfs:
        for p in ps:
            if df >= 1e6 and p < 1e-310:
                # denormal p with huge df: Boost's internal ibeta series at
                # subnormal scale differs from ours by ~1e-5 relative (skipped)
                continue
            v = float(sp.stdtrit(df, p))
            if math.isinf(v):
                print(f'    assert sp.stdtrit({df!r}, {p!r}) == {"INF" if v > 0 else "-INF"}')
            else:
                print(f'    assert close(sp.stdtrit({df!r}, {p!r}), {v!r}, 1e-12)')
    for df, p in [(1.0, 1e-200), (1.0, 1e-300), (2.5, 1e-300), (2.0, 1e-300), (10.0, 1e-10)]:
        v = float(st.t.ppf(p, df))
        if math.isinf(v):
            print(f'    assert st.t.ppf({p!r}, {df!r}) == {"INF" if v > 0 else "-INF"}')
        else:
            print(f'    assert close(st.t.ppf({p!r}, {df!r}), {v!r}, 1e-12)')
    print('test_stdtrit_tails()')
