# Copyright (C) 2026 the Typhon authors
# rv_continuous.fit (MLE with scipy's kernel-specific starting points) vs scipy 1.18.1.
import numpy as np
import scipy.stats as st
from dists_check import close

ok = True
x = np.array([2.1, 3.4, 1.9, 5.6, 4.2, 3.3, 2.8, 6.1, 4.4, 3.9, 2.2, 5.0, 3.1, 2.6, 4.8])
ok &= close('weibull_min', 'fit', [float(v) for v in st.weibull_min.fit(x)],
            [1.381663078605878, 1.8096154274393754, 2.0476143142922725], 1e-9)
ok &= close('genextreme', 'fit', [float(v) for v in st.genextreme.fit(x)],
            [0.13864441880367917, 3.1678296284834784, 1.1179869457550922], 1e-9)
ok &= close('skewnorm', 'fit', [float(v) for v in st.skewnorm.fit(x)][1:],
            [1.8999978137300229, 2.1909188566359683], 1e-9)
ok &= close('nakagami', 'fit', [float(v) for v in st.nakagami.fit(x, floc=0)],
            [2.3236416927507784, 0.0, 3.901617093140063], 1e-9)
ok &= close('gamma', 'fit', [float(v) for v in st.gamma.fit(x, floc=0)],
            [8.453385005204442, 0.0, 0.4369058467181469], 1e-9)
ok &= close('norm', 'fit', [float(v) for v in st.norm.fit(x)],
            [3.6933333333333334, 1.2577581466862202], 1e-12)
print('ALL OK' if ok else 'FAILED')
assert ok
