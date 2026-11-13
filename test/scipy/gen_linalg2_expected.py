# Generates expected values for test/scipy/test_linalg2.py and
# test/scipy/test_linalg_matfuncs.py from the real scipy 1.18.1:
#   /Users/mind/.jcode/scratch/.venv/bin/python test/scipy/gen_linalg2_expected.py
import numpy as np
import scipy.linalg as sl

def lit(a):
    a = np.asarray(a)
    if np.iscomplexobj(a):
        return "[" + ", ".join(f"complex({float(v.real)!r}, {float(v.imag)!r})" for v in a.ravel()) + "]"
    return "[" + ", ".join(repr(float(v)) for v in a.ravel()) + "]"

A = np.array([[4., 1., 2., .5], [1., 3., .5, .2], [2., .5, 5., 1.], [.3, .2, 1., 2.]])
G = np.array([[1., 2., 0., 1.], [-1., 1., 3., 0.], [2., 0., 1., -2.], [0., 1., 1., 1.]])
S = np.array([[4., 1., 2.], [1., 3., .5], [2., .5, 5.]])
B = np.array([[2., .3, .1, 0.], [.3, 1.5, .2, .1], [.1, .2, 1., .3], [0., .1, .3, 1.2]])
R = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 10.], [1., 0., 1.]])

out = {}
def put(name, v):
    out[name] = v

# linalg2: special matrices
put('had4', sl.hadamard(4)); put('les', sl.leslie([0.1, 2.0, 1.0, 0.1], [0.2, 0.8, 0.7]))
put('comp', sl.companion([1, -10, 31, -30])); put('helm', sl.helmert(5)); put('helmf', sl.helmert(4, full=True))
put('hilb', sl.hilbert(4)); put('ihilb', sl.invhilbert(5)); put('ihilbx', sl.invhilbert(5, exact=True))
put('pasc', sl.pascal(5)); put('pascl', sl.pascal(4, kind='lower')); put('pascf', sl.pascal(4, exact=False))
put('ipasc', sl.invpascal(5)); put('ipascu', sl.invpascal(4, kind='upper'))
put('dft4', sl.dft(5)); put('dftn', sl.dft(4, scale='sqrtn'))
put('fied', sl.fiedler([1., 4., 12., 45., 77.])); put('fcomp', sl.fiedler_companion([1., -16., 86., -176., 105.]))
put('convf', sl.convolution_matrix([1., 2., 3.], 5)); put('convs', sl.convolution_matrix([1., 2., 3., 4.], 3, 'same'))
put('convv', sl.convolution_matrix([1., 2., 3.], 5, 'valid'))
put('kr', sl.khatri_rao(np.array([[1., 2., 3.], [4., 5., 6.]]), np.array([[3., 4., 5.], [6., 7., 8.], [2., 3., 9.]])))
put('dsvd', sl.diagsvd([1., 2.], 3, 2))
# decompositions
T, Z = sl.schur(G); put('schurT', T); put('schurZ', Z)
T, Z = sl.schur(G, output='complex'); put('schurTc', T); put('schurZc', Z)
T, Z, sd = sl.schur(G, sort='lhp'); put('schurTs', T); put('schurZs', Z); put('schursd', [sd])
T, Z = sl.schur(G); T2, Z2 = sl.rsf2csf(T, Z); put('rsfT', T2); put('rsfZ', Z2)
H, Q = sl.hessenberg(G, calc_q=True); put('hessH', H); put('hessQ', Q)
Bb, Tb = sl.matrix_balance(np.array([[1., 100., 0.], [0.01, 1., 1e4], [1e-3, 1., 1.]])); put('balB', Bb); put('balT', Tb)
Bb, (sc, pm) = sl.matrix_balance(np.array([[1., 100., 0.], [0.01, 1., 1e4], [1e-3, 1., 1.]]), separate=True); put('balS', sc); put('balP', pm)
AA, BB, Qz, Zz = sl.qz(G, B); put('qzAA', AA); put('qzBB', BB); put('qzQ', Qz); put('qzZ', Zz)
AA, BB, al, be, Qz, Zz = sl.ordqz(G, B, sort='lhp'); put('oqAA', AA); put('oqBB', BB); put('oqal', al); put('oqbe', be); put('oqQ', Qz); put('oqZ', Zz)
Asym = np.array([[2., -1., 3.], [-1., 2., 0.], [3., 0., 1.]])
lu, d, p = sl.ldl(Asym); put('ldlL', lu); put('ldlD', d); put('ldlP', p)
lu, d, p = sl.ldl(Asym, lower=False); put('ldlLu', lu); put('ldlDu', d); put('ldlPu', p)
r, q = sl.rq(R.T); put('rqR', r); put('rqQ', q)
r, q = sl.rq(R); put('rqR2', r); put('rqQ2', q)
r, q = sl.rq(R.T, mode='economic'); put('rqRe', r); put('rqQe', q)
u, p = sl.polar(R); put('polU', u); put('polP', p)
u, p = sl.polar(R, side='left'); put('polUl', u); put('polPl', p)
rr, scl = sl.orthogonal_procrustes(R, R @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])); put('procR', rr); put('procS', [scl])
put('sang', sl.subspace_angles(R, np.array([[1., 0.], [0., 1.], [1., 1.], [2., 0.]])))
put('bw', list(sl.bandwidth(np.array([[1., 2., 0., 0.], [3., 1., 0., 4.], [0., 5., 1., 0.], [0., 0., 0., 1.]]))))
# banded / tridiagonal
ab = np.array([[0., 1., 2., 3., 1.], [5., 6., 7., 8., 9.], [1., 2., 3., 4., 0.]])
bb = np.array([1., 2., 3., 4., 5.])
put('sbtri', sl.solve_banded((1, 1), ab, bb))
ab2 = np.array([[0., 0., 1., 2., 3.], [0., 1., 2., 3., 1.], [5., 6., 7., 8., 9.], [1., 2., 3., 4., 0.]])
put('sbgen', sl.solve_banded((1, 2), ab2[[0, 1, 2, 3]][::-1][::-1], np.column_stack([bb, bb[::-1]])))
hb = np.array([[0., 1., 1., 1., 1.], [4., 4., 4., 4., 4.]])
put('shb', sl.solveh_banded(hb, bb))
hb3 = np.array([[0., 0., .5, .5, .5], [0., 1., 1., 1., 1.], [6., 6., 6., 6., 6.]])
put('shb3', sl.solveh_banded(hb3, bb))
put('shbl', sl.solveh_banded(np.array([[4., 4., 4., 4., 4.], [1., 1., 1., 1., 0.]]), bb, lower=True))
cb = sl.cholesky_banded(hb3); put('cbf', cb); put('cbs', sl.cho_solve_banded((cb, False), bb))
w, v = sl.eig_banded(hb3); put('ebw', w); put('ebv', v)
put('ebsel', sl.eig_banded(hb3, eigvals_only=True, select='i', select_range=(1, 3)))
w, v = sl.eig_banded(hb3, select='v', select_range=(5.0, 7.0)); put('ebvw', w); put('ebvv', v)
d = np.array([3., 1., 4., 1., 5.]); e = np.array([1., 2., -1., 0.5])
w, v = sl.eigh_tridiagonal(d, e); put('etw', w); put('etv', v)
put('etsterf', sl.eigvalsh_tridiagonal(d, e, lapack_driver='sterf'))
w, v = sl.eigh_tridiagonal(d, e, select='i', select_range=(1, 2)); put('etiw', w); put('etiv', v)
put('etvv', sl.eigvalsh_tridiagonal(d, e, select='v', select_range=(0.0, 4.0)))
w, v = sl.eigh_tridiagonal(d, e, lapack_driver='stev'); put('etsw', w); put('etsv', v)
put('stoep', sl.solve_toeplitz(([1., 3., 6., 10.], [1., -1., -2., -3.]), [1., 2., 2., 5.]))
put('stoep2', sl.solve_toeplitz([4., 1., .5], np.array([[1., 2.], [3., 4.], [5., 6.]])))
put('mtoep', sl.matmul_toeplitz(([1., 2., 3.], [1., 5., 6., 7.]), np.array([1., 2., 3., 4.])))
put('scirc', sl.solve_circulant([2., -1., 0., 1.], [1., 2., 3., 4.]))
# solvers
a = np.array([[-3., -2., 0.], [-1., -1., 0.], [0., -5., -1.]]); b2 = np.array([[1.]]); q = np.array([[1., 2., 3.]]).T
put('sylv', sl.solve_sylvester(a, b2, q))
put('sylv2', sl.solve_sylvester(G, S[:2, :2] + 2, R[:, :2]))
put('lyap', sl.solve_continuous_lyapunov(a, np.eye(3)))
put('dlyap', sl.solve_discrete_lyapunov(0.4 * G, np.eye(4)))
put('dlyapb', sl.solve_discrete_lyapunov(0.4 * G, np.eye(4), method='bilinear'))
ca = np.array([[4., 3.], [-4.5, -3.5]]); cb_ = np.array([[1.], [-1.]]); cq = np.array([[9., 6.], [6., 4.]]); cr = np.array([[1.]])
put('care', sl.solve_continuous_are(ca, cb_, cq, cr))
put('dare', sl.solve_discrete_are(ca * 0.3, cb_, cq + np.eye(2), cr))
put('care2', sl.solve_continuous_are(G, R, np.eye(4), np.eye(3)))
put('dare2', sl.solve_discrete_are(0.5 * G, R, np.eye(4), np.eye(3)))
put('caree', sl.solve_continuous_are(G, R, np.eye(4), np.eye(3), e=np.eye(4) + 0.1 * np.triu(np.ones((4, 4)))))

with open('linalg2_expected.py', 'w') as f:
    f.write('# Generated by gen_linalg2_expected.py (scipy 1.18.1). Do not edit.\n')
    for k, v in out.items():
        f.write(f'{k} = {lit(v)}\n')

# matfuncs
m = {}
M = np.array([[1., .5, .2], [.1, 2., .3], [.4, .2, 1.5]])
N = np.array([[0., 1.], [-2., -3.]])
Mc = np.array([[1 + 1j, .5, 0], [.1j, 2., .3], [.4, .2 - .5j, 1.5]])
m['cosm'] = sl.cosm(M); m['sinm'] = sl.sinm(M); m['tanm'] = sl.tanm(M)
m['coshm'] = sl.coshm(M); m['sinhm'] = sl.sinhm(M); m['tanhm'] = sl.tanhm(M)
m['cosmc'] = sl.cosm(Mc); m['sinhmc'] = sl.sinhm(Mc)
m['funm'] = sl.funm(M, lambda x: x * x + 1)
m['funmexp'] = sl.funm(N, np.exp)
m['signm'] = sl.signm(np.array([[2., 1., 0.], [0., -3., 1.], [0., 0., 1.]]))
m['signm2'] = sl.signm(np.array([[1., 2.], [3., -4.]]))
R_, L_ = sl.expm_frechet(M, N.repeat(2, 0)[:3, :].repeat(2, 1)[:, :3] * 0.1); m['frR'] = R_; m['frL'] = L_
R_, L_ = sl.expm_frechet(4 * M, np.eye(3)); m['frR2'] = R_; m['frL2'] = L_
m['frBE'] = sl.expm_frechet(M, np.ones((3, 3)), method='blockEnlarge', compute_expm=False)
m['econd'] = [sl.expm_cond(M)]
m['expm'] = sl.expm(10 * M); m['expmN'] = sl.expm(N)
m['sqrtm'] = sl.sqrtm(M); m['logm'] = sl.logm(M)
m['fmp'] = sl.fractional_matrix_power(M, 0.3); m['fmp2'] = sl.fractional_matrix_power(M, -1.5)
m['sqrtmc'] = sl.sqrtm(Mc); m['logmc'] = sl.logm(Mc)
m['expmup'] = sl.expm(np.array([[1., 2., 3.], [0., -1., 4.], [0., 0., 3.]]) * 3)
with open('linalg_matfuncs_expected.py', 'w') as f:
    f.write('# Generated by gen_linalg2_expected.py (scipy 1.18.1). Do not edit.\n')
    for k, v in m.items():
        f.write(f'{k} = {lit(v)}\n')
