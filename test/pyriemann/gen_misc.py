# Generates test_misc.py (transfer, channelselection, whitening, median, kmeans, test helpers) from pyriemann 0.12.
import numpy as np, warnings, os
warnings.simplefilter("ignore")
from pyriemann.geometry import covariance as C, median as Md, test as Tt
from pyriemann import channelselection as CS, preprocessing as PP, clustering as CL
from pyriemann.transfer import TLCenter, TLScale, encode_domains
D = os.path.dirname(__file__)
def signals(n_trials, n_ch, n_t):
    X = np.zeros((n_trials, n_ch, n_t))
    for i in range(n_trials):
        for c in range(n_ch):
            for t in range(n_t):
                v = (i * 1009 + c * 7919 + t * 104729 + i * c * t * 13
                     + t * t * (c + 1) * 3 + c * c * t * 7) % 101
                X[i, c, t] = (v - 50.0) / 25.0 + (0.5 * (i % 2)) * ((c + t) % 3 == 0)
    return X
def lit(a): return repr(np.asarray(a, dtype=float).ravel().tolist())
head = open(os.path.join(D, "test_pipeline.py")).read().split("P = spd(5, 4)")[0]
head = head.replace("import numpy as np\n", """import numpy as np
from pyriemann.geometry.covariance import covariances
from pyriemann.geometry.median import median_euclid, median_riemann
from pyriemann.geometry.test import is_sym, is_pos_def, is_hankel, is_sym_pos_def, is_pos_semi_def, is_skew_sym
from pyriemann.channelselection import FlatChannelRemover, ElectrodeSelection
from pyriemann.preprocessing import Whitening
from pyriemann.clustering import Kmeans
from pyriemann.transfer import TLCenter, TLScale, encode_domains
""", 1)
X = signals(18, 5, 40)
X[:, 2, :] = 0.0
cov = C.covariances(X[:, [0, 1, 3, 4], :], "scm")
y = np.array([i % 3 for i in range(18)])
dom = ["src" if i < 9 else "tgt" for i in range(18)]
Xe, ye = encode_domains(cov.copy(), y, dom)
L = []; T = []
def add(name, arr, call, atol=1e-8):
    L.append(f"E_{name} = {lit(arr)}")
    T.append(f"    assert close({call}, E_{name}, atol={atol})")
add("flat", CS.FlatChannelRemover().fit_transform(X), "FlatChannelRemover().fit_transform(X)")
es = CS.ElectrodeSelection(nelec=2).fit(cov, y)
add("es_sub", np.array(es.subelec_, dtype=float), "ElectrodeSelection(2).fit(cov, y).subelec_")
add("es_t", es.transform(cov), "ElectrodeSelection(2).fit_transform(cov, y)")
add("wh", PP.Whitening(metric="riemann").fit_transform(cov), 'Whitening("riemann").fit_transform(cov)')
add("wh_nc", PP.Whitening(dim_red={"n_components": 2}).fit_transform(cov), 'Whitening("euclid", "n_components", 2.0).fit_transform(cov)')
add("wh_ev", PP.Whitening(dim_red={"expl_var": 0.9}).fit_transform(cov), 'Whitening("euclid", "expl_var", 0.9).fit_transform(cov)')
add("wh_inv", (lambda w: w.inverse_transform(w.fit_transform(cov)))(PP.Whitening(metric="logeuclid")), "inv_roundtrip(cov)")
add("med_e", Md.median_euclid(cov), "median_euclid(cov)", 1e-7)
add("med_r", Md.median_riemann(cov), "median_riemann(cov)", 1e-7)
add("tlc", TLCenter("tgt").fit_transform(Xe, ye), "TLCenter(\"tgt\").fit_transform(Xe, ye)")
tlc = TLCenter("tgt").fit(Xe, ye)
add("tlc_t", tlc.transform(cov), "TLCenter(\"tgt\").fit(Xe, ye).transform(cov)")
add("tls", TLScale("tgt", final_dispersion=2.0).fit_transform(Xe.copy(), ye), "TLScale(\"tgt\", 2.0).fit_transform(Xe, ye)", 1e-7)
add("tls_t", TLScale("tgt").fit(Xe, ye).transform(cov), "TLScale(\"tgt\").fit(Xe, ye).transform(cov)", 1e-7)
km = CL.Kmeans(n_clusters=3, random_state=3).fit(cov)
add("km_lab", km.labels_, "Kmeans(3, 100, \"riemann\", 3).fit(cov).labels_")
add("km_cent", km.mdm_.covmeans_, "Kmeans(3, 100, \"riemann\", 3).fit(cov).centroids()", 1e-7)
P = np.array([[2.0, 1], [1, 3]])
chk = [Tt.is_sym(P), Tt.is_pos_def(P), Tt.is_hankel(np.array([[1., 2], [2, 3]])), Tt.is_skew_sym(P), Tt.is_pos_semi_def(np.array([[1., 1], [1, 1]])), Tt.is_sym_pos_def(np.array([[1., 2], [2, 1]]))]
setup = '''X = signals(18, 5, 40)
for i in range(18):
    X[i, 2, :] = 0.0
cov = covariances(X[:, [0, 1, 3, 4], :], "scm")
y = np.array([i % 3 for i in range(18)])
dom = ["src" if i < 9 else "tgt" for i in range(18)]
Xe, ye = encode_domains(cov, y, dom)
def inv_roundtrip(c):
    w = Whitening("logeuclid")
    t = w.fit_transform(c)
    return w.inverse_transform(t)

'''
tests = "\n".join(T) + f'''
    Pm = np.array([[2.0, 1.0], [1.0, 3.0]])
    assert [is_sym(Pm), is_pos_def(Pm), is_hankel(np.array([[1.0, 2.0], [2.0, 3.0]])), is_skew_sym(Pm), is_pos_semi_def(np.array([[1.0, 1.0], [1.0, 1.0]])), is_sym_pos_def(np.array([[1.0, 2.0], [2.0, 1.0]]))] == {[bool(b) for b in chk]!r}
'''
open(os.path.join(D, "test_misc.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_misc():\n" + tests + "test_misc()\n")
