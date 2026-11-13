# Copyright (C) 2026 the Typhon authors
# Multivariate distributions vs scipy 1.18.1 (reference values from scipy).
from dists_check import close
import numpy as np
import scipy.stats as st

ok = True
cov = np.array([[2.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 0.5]])
m = np.array([1.0, -1.0, 0.5])
x = np.array([[0.5, -0.5, 0.2], [1.5, -1.2, 1.0]])
mv = st.multivariate_normal(m, cov)
ok &= close('mvn_logpdf', '', mv.logpdf(x).tolist(), [-3.08936705494074, -3.097177773298779], 1e-13)
ok &= close('mvn_pdf1', '', [float(mv.pdf(x[0]))], [0.04553076375825586], 1e-13)
ok &= close('mvn_ent', '', [float(mv.entropy())], [4.191191456309042], 1e-13)
ok &= close('mvn_rvs', '', mv.rvs(size=2, random_state=np.random.RandomState(3)).ravel().tolist(), [-1.3488025425004202, -2.130485022370163, 0.1770412464993414, 3.498004027975604, 0.07560182481401978, 0.623503525198208], 1e-13)
ok &= close('mvn_cdf2', '', [float(st.multivariate_normal([0.5, -0.2], [[1.0, 0.6], [0.6, 2.0]]).cdf([1.0, 0.3]))], [0.5007896943118612], 1e-13)
ok &= close('mvn_cdf2n', '', [float(st.multivariate_normal([0.0, 0.0], [[1.0, -0.95], [-0.95, 1.0]]).cdf([0.4, 0.3]))], [0.2739315755924505], 1e-13)
ok &= close('mvn_cdf3', '', [float(mv.cdf(np.array([1.2, -0.4, 0.9]), rng=np.random.default_rng(5)))], [0.33490550517181866], 1e-13)
mt = st.multivariate_t(m, cov, df=4.5)
ok &= close('mvt_logpdf', '', mt.logpdf(x).tolist(), [-3.1566706669666784, -3.1677149056155027], 1e-13)
ok &= close('mvt_ent', '', [float(mt.entropy())], [4.8331088743877455], 1e-13)
ok &= close('mvt_rvs', '', mt.rvs(size=2, random_state=np.random.RandomState(3)).ravel().tolist(), [1.1778001987058295, -0.7181137276361826, 0.5623471195935199, 1.7965148923815415, -0.6307267330860356, 0.3178822163824866], 1e-13)
al = np.array([1.5, 2.0, 0.7])
xd = np.array([0.2, 0.5, 0.3])
ok &= close('dir_logpdf', '', [float(st.dirichlet.logpdf(xd, al))], [0.7717963325849542], 1e-13)
ok &= close('dir_ent', '', [float(st.dirichlet.entropy(al))], [-1.1419003467173847], 1e-13)
ok &= close('dir_cov', '', st.dirichlet.cov(al).ravel().tolist(), [0.044152276295133436, -0.03270538984824699, -0.011446886446886446, -0.03270538984824699, 0.04796790511076225, -0.01526251526251526, -0.011446886446886446, -0.01526251526251526, 0.026709401709401708], 1e-13)
ok &= close('dir_rvs', '', st.dirichlet.rvs(al, size=2, random_state=np.random.RandomState(3)).ravel().tolist(), [0.5047926645787709, 0.2657639388641543, 0.22944339655707488, 0.4152653521861161, 0.5816452819273601, 0.0030893658865239976], 1e-13)
ok &= close('mn_pmf', '', [float(st.multinomial.pmf([3, 2, 5], 10, [0.2, 0.3, 0.5]))], [0.05670000000000023], 1e-13)
ok &= close('mn_ent', '', [float(st.multinomial.entropy(10, [0.2, 0.3, 0.5]))], [3.341218516394507], 1e-13)
ok &= close('mn_cov', '', st.multinomial.cov(10, [0.2, 0.3, 0.5]).ravel().tolist(), [1.5999999999999999, -0.6, -1.0, -0.6, 2.1, -1.5, -1.0, -1.5, 2.5], 1e-13)
ok &= close('mn_rvs', '', st.multinomial.rvs(10, [0.2, 0.3, 0.5], size=2, random_state=np.random.RandomState(3)).ravel().tolist(), [2.0, 4.0, 4.0, 1.0, 3.0, 6.0], 1e-13)
S = np.array([[2.0, 0.3], [0.3, 1.0]])
W = np.array([[5.0, 1.0], [1.0, 3.0]])
ok &= close('w_logpdf', '', [float(st.wishart.logpdf(W, 4.5, S))], [-5.931200235741868], 1e-13)
ok &= close('w_ent', '', [float(st.wishart.entropy(4.5, S))], [7.547917751436308], 1e-13)
ok &= close('w_rvs', '', st.wishart.rvs(4.5, S, size=2, random_state=np.random.RandomState(3)).ravel().tolist(), [8.213557283862135, 6.241453889039242, 6.241453889039242, 6.865810156729879, 1.2847733690639838, 0.676229942966077, 0.676229942966077, 2.3327418292790516], 1e-13)
ok &= close('iw_logpdf', '', [float(st.invwishart.logpdf(W, 4.5, S))], [-12.543909112170445], 1e-13)
ok &= close('iw_ent', '', [float(st.invwishart.entropy(4.5, S))], [1.9686986784047669], 1e-13)
ok &= close('iw_rvs', '', st.invwishart.rvs(4.5, S, size=2, random_state=np.random.RandomState(3)).ravel().tolist(), [0.6516128850106603, -0.9071011017757216, -0.9071011017757216, 2.749406159981095, 0.8996911091023807, -0.0235175603441681, -0.0235175603441681, 0.32626833713207254], 1e-13)
M = np.array([[1.0, 2.0, 0.0], [0.5, -1.0, 1.0]])
U = np.array([[2.0, 0.3], [0.3, 1.0]])
V = cov
X = np.array([[1.5, 1.0, 0.2], [0.0, -0.5, 1.3]])
ok &= close('matn_logpdf', '', [float(st.matrix_normal.logpdf(X, M, U, V))], [-7.178298809573132], 1e-13)
ok &= close('matn_ent', '', [float(st.matrix_normal.entropy(U, V))], [9.353037775705891], 1e-13)
ok &= close('matn_rvs', '', st.matrix_normal.rvs(M, U, V, size=2, random_state=np.random.RandomState(3)).ravel().tolist(), [4.577256946860637, 2.153230419058076, 0.0626091306670434, 1.639857157758887, -1.2253199699837023, 0.6579137948308057, 1.1929949361440173, 1.9145982379440887, -1.2715688201454367, -2.046448914425732, -1.9979056339022607, 1.1502906797532213], 1e-13)
ok &= close('ortho', '', st.ortho_group.rvs(3, random_state=np.random.RandomState(3)).ravel().tolist(), [0.6921114349153448, 0.12055379885720953, -0.7116519818285605, -0.721080217629734, 0.15921749661593554, -0.6743093566860912, -0.03201689245504324, -0.9798553823664236, -0.19712520956253188], 1e-13)
ok &= close('sortho', '', st.special_ortho_group.rvs(3, random_state=np.random.RandomState(3)).ravel().tolist(), [-0.6921114349153448, -0.12055379885720953, 0.7116519818285605, -0.721080217629734, 0.15921749661593554, -0.6743093566860912, -0.03201689245504324, -0.9798553823664236, -0.19712520956253188], 1e-13)
rc = st.random_correlation.rvs(np.array([0.5, 1.0, 1.5]), random_state=np.random.RandomState(3)).ravel()
rc_ref = np.array([0.9999999999999999, -1.0741130368276374e-16, 0.4933011667935992, -8.767128735066347e-17, 0.9999999999999999, 0.081571801745907, 0.49330116679359937, 0.08157180174590696, 0.9999999999999998])
if float(np.max(np.abs(rc - rc_ref))) > 1e-14:
    print('rcorr', rc, rc_ref)
    ok = False

print('ALL OK' if ok else 'FAILED')
assert ok
