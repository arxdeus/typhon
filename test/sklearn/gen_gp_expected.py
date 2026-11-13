# Generates the expected literals for test/sklearn/test_gp.py (scikit-learn 1.9.1).
#   <venv>/bin/python test/sklearn/gen_gp_expected.py
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared, DotProduct, WhiteKernel, ConstantKernel as C
warnings.filterwarnings("ignore")

def L(name, a):
    print(f"{name} = {[float(v) for v in np.asarray(a, dtype=float).ravel()]!r}")

rs = np.random.RandomState(0)
X = rs.uniform(0, 5, size=(25, 2))
y = np.sin(X[:, 0]) + 0.5 * np.cos(2 * X[:, 1]) + 0.05 * rs.standard_normal(25)
Q = rs.uniform(0, 5, size=(6, 2))
ks = [("rbf", C(1.0) * RBF(1.0)), ("rbfa", C(1.0) * RBF([1.0, 2.0]) + WhiteKernel(0.1)),
      ("m15", Matern(1.0, nu=1.5)), ("m25", C(2.0) * Matern(0.5, nu=2.5)), ("m05", Matern(1.0, nu=0.5) + WhiteKernel()),
      ("rq", RationalQuadratic(1.0, 0.5)),
      ("dot", DotProduct(1.0) ** 2 + WhiteKernel(0.5))]
for name, k in ks:
    g = GaussianProcessRegressor(k, alpha=1e-6, normalize_y=(name == "m25")).fit(X, y)
    L(f"{name}_theta", g.kernel_.theta); print(f"{name}_lml =", g.log_marginal_likelihood_value_)
    m, s = g.predict(Q, return_std=True); L(f"{name}_mean", m); L(f"{name}_std", s)
g = GaussianProcessRegressor(ExpSineSquared(1.0, 6.0, periodicity_bounds=(2.0, 10.0)) + WhiteKernel(0.1), alpha=1e-6).fit(X[:, :1], y)
L("ess_theta", g.kernel_.theta); L("ess_mean", g.predict(Q[:, :1]))
g = GaussianProcessRegressor(C(1.0) * RBF(1.0), n_restarts_optimizer=3, random_state=2).fit(X, y)
L("rst_theta", g.kernel_.theta); L("rst_mean", g.predict(Q))
g = GaussianProcessRegressor(RBF(1.0), optimizer=None).fit(X, np.column_stack([y, 2 * y])); L("mo_mean", g.predict(Q))
L("lml_grad", g.log_marginal_likelihood(np.array([0.3]), eval_gradient=True)[1])
Xc = rs.uniform(-2, 2, size=(30, 2)); yc = (Xc[:, 0] ** 2 + Xc[:, 1] > 1).astype(int)
c = GaussianProcessClassifier(1.0 * RBF(1.0)).fit(Xc, yc)
L("gpc_theta", c.kernel_.theta); print("gpc_lml =", c.log_marginal_likelihood_value_); L("gpc_proba", c.predict_proba(Xc[:8])); L("gpc_pred", c.predict(Xc))
y3 = (Xc[:, 0] > 0).astype(int) + (Xc[:, 1] > 0.5).astype(int)
c = GaussianProcessClassifier(1.0 * RBF(1.0)).fit(Xc, y3); L("gpc3_proba", c.predict_proba(Xc[:8])); L("gpc3_pred", c.predict(Xc))
c = GaussianProcessClassifier(RBF(1.0), multi_class="one_vs_one").fit(Xc, y3); L("gpc3o_pred", c.predict(Xc))
