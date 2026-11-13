# Ported from scipy 1.18.1 (BSD-3-Clause)
# Generates test/scipy/test_optimize_qap.py (quadratic_assignment faq / 2opt).
import numpy as np
from scipy.optimize import quadratic_assignment as qa
from _gen_common import Gen
ns = {'np': np}
g = Gen(ns, tol=1e-9)
g.test('test_qap')
rng = np.random.default_rng(5)
def arr(M):
    return 'np.array(' + repr(M.tolist()) + ')'
for n in [6, 10]:
    A = rng.random((n, n)); B = rng.random((n, n))
    g.raw(f'    A = {arr(A)}\n    B = {arr(B)}')
    ns['A'] = A; ns['B'] = B
    def chk(label, call_t, opts, **kw):
        r = qa(A, B, method=kw.get('method', 'faq'), options=opts)
        g.raw(f'    r = {call_t}')
        g.raw(f'    assert closec(r.col_ind.astype(float), {[float(v) for v in r.col_ind]!r}, 1e-12)')
        g.raw(f'    assert closec(np.array([r.fun, float(r.nit)]), {[float(r.fun), float(r.nit)]!r}, 1e-9)')
    chk('faq', "quadratic_assignment(A, B, 'faq', rng=1)", {'rng': 1})
    chk('faqmax', "quadratic_assignment(A, B, 'faq', maximize=True, rng=2)", {'rng': 2, 'maximize': True})
    chk('faqr', "quadratic_assignment(A, B, 'faq', rng=3, P0='randomized')", {'rng': 3, 'P0': 'randomized'})
    pm = np.array([[0, 1], [2, 0]])
    chk('faqseed', "quadratic_assignment(A, B, 'faq', rng=4, partial_match=np.array([[0, 1], [2, 0]]))", {'rng': 4, 'partial_match': pm})
    chk('2opt', "quadratic_assignment(A, B, '2opt', rng=1)", {'rng': 1}, method='2opt')
    chk('2optmax', "quadratic_assignment(A, B, '2opt', maximize=True, rng=7)", {'rng': 7, 'maximize': True}, method='2opt')
    chk('2optpm', "quadratic_assignment(A, B, '2opt', rng=7, partial_match=np.array([[0, 1], [2, 0]]), partial_guess=np.array([[3, 4], [4, 3]]))",
        {'rng': 7, 'partial_match': pm, 'partial_guess': np.array([[3, 4], [4, 3]])}, method='2opt')
g.write('test/scipy/test_optimize_qap.py', 'gen_optimize_qap.py', ['import numpy as np', 'from scipy.optimize import quadratic_assignment'], [])
