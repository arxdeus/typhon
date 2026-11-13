# Generates expected values for test/scipy/test_stats_qmc2.py (scipy 1.18.1).
import numpy as np
from scipy.stats import qmc

def e(n, v):
    print(f"{n} = {np.asarray(v, dtype=float).tolist()!r}")

p = qmc.PoissonDisk(d=2, radius=0.2, rng=11); e('pd1', p.random(8))
p = qmc.PoissonDisk(d=2, radius=0.3, rng=11, hypersphere='surface'); e('pd2', p.fill_space())
p = qmc.PoissonDisk(d=3, radius=0.25, rng=4, ncandidates=20); e('pd3', p.random(5)); e('pd3b', p.random(3))
m = qmc.MultivariateNormalQMC(mean=[0, 5.], cov=[[1, .5], [.5, 2]], rng=3); e('mv1', m.random(4))
m = qmc.MultivariateNormalQMC(mean=[0, 5., 1], inv_transform=False, rng=3); e('mv2', m.random(4))
import warnings; warnings.simplefilter('ignore')
m = qmc.MultinomialQMC(pvals=[.2, .3, .5], n_trials=10, rng=5); e('mn', m.random(3))
e('lh2', qmc.LatinHypercube(d=3, strength=2, rng=5).random(9))
e('lh2b', qmc.LatinHypercube(d=4, strength=2, scramble=False, rng=5).random(25)[:6])
e('sob_hi', qmc.Sobol(d=5000, scramble=False).random(4)[:, -3:])
e('sob_hi2', qmc.Sobol(d=21201, scramble=False).random(4)[:, -2:])
e('sob_hi3', qmc.Sobol(d=3000, rng=2).random(2)[:, -2:])
