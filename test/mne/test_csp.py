import numpy as np

def make_X(n_ep: int = 40, n_ch: int = 8, n_t: int = 256):
    # mirrors make_X in gen_expected.py (deterministic, no RNG)
    X = np.zeros((n_ep, n_ch, n_t))
    y = np.arange(n_ep) % 2
    for e in range(n_ep):
        for c in range(n_ch):
            for t in range(n_t):
                v = (np.sin(0.37 * t * (c + 1) + 0.11 * e) + 0.5 * np.cos(0.071 * t * (e % 7 + 1) + c)
                     + 0.3 * np.sin(1.3 * t + 0.7 * e * c))
                if c % 3 == 0:
                    v += (y[e] * 0.8 + 0.2) * np.sin(2 * np.pi * 12.0 / 128.0 * t)
                v += 0.05 * c
                X[e, c, t] = v
    return X, y

def close(a, b, rtol: float = 1e-9, atol: float = 0.0) -> bool:
    x = np.asarray(a).ravel()
    z = np.asarray(b).ravel()
    if x.shape[0] != z.shape[0]:
        print("length mismatch", x.shape[0], z.shape[0])
        return False
    scale = float(np.max(np.abs(z)))
    for i in range(x.shape[0]):
        if abs(x[i] - z[i]) > atol + rtol * max(abs(z[i]), 1e-3 * scale):
            print("mismatch at", i, x[i], z[i])
            return False
    return True

def closef(a: float, b: float, rtol: float = 1e-9) -> bool:
    ok = abs(a - b) <= rtol * abs(b)
    if not ok:
        print("mismatch", a, b)
    return ok

import mne
from mne.decoding import CSP

def filtered():
    X, y = make_X()
    info = mne.create_info([f"E{i}" for i in range(8)], 128.0, "eeg")
    ep = mne.EpochsArray(X * 1e-6, info, tmin=0.0)
    ep.filter(8.0, 30.0)
    return ep.get_data(), y

def norms(c):
    return np.sort(np.linalg.norm(c.filters_, axis=1))

@test
def test_csp_variants():
    Xd, y = filtered()
    c = CSP(n_components=3, reg=0.1)
    F = c.fit_transform(Xd, y)
    assert close(F[0], [-2.289038430333651, -2.186100423006206, -0.9991218340541134], 1e-9)
    assert close(norms(c), [973107.6504780058, 987350.6843418002, 999360.1195235347, 1011045.9713884343, 1554996.0674655684, 1796586.0449698002, 2185739.5265645664, 2394533.43394571], 1e-9)
    c = CSP(n_components=3, reg="ledoit_wolf")
    F = c.fit_transform(Xd, y)
    assert close(F[0], [-2.602214927120802, 0.1406136751543415, -1.4527327839380464], 1e-9)
    assert close(norms(c), [1127917.9129569188, 1140440.3525376483, 1263982.0375582916, 1718083.3975423996, 1783544.4938377081, 1954411.2243868983, 2638077.421695091, 2674633.6499341503], 1e-9)
    c = CSP(n_components=3, reg="oas")
    F = c.fit_transform(Xd, y)
    assert close(F[0], [-2.600092177964322, 0.13852376959517348, -1.4550520001982747], 1e-9)
    assert close(norms(c), [1126852.5220271377, 1139788.101694113, 1263542.0739819075, 1716113.5581431543, 1780979.6886046303, 1954376.1825877202, 2634092.940106474, 2672494.7790113073], 1e-9)
    c = CSP(n_components=3, cov_est="epoch")
    F = c.fit_transform(Xd, y)
    assert close(F[0], [-2.6184009498523078, 0.14386981880758543, -1.4854520857857656], 1e-9)
    assert close(norms(c), [1128102.2717618858, 1132253.6636356579, 1256521.9128848128, 1698790.7295299051, 1739086.0743584917, 1980946.532458106, 2632491.3805017634, 2734763.6051928317], 1e-9)
    c = CSP(n_components=3, norm_trace=True, log=False)
    F = c.fit_transform(Xd, y)
    assert close(F[0], [-1.0950694597688382, 2.282160791482908, -1.075074311467634], 1e-9)
    assert close(norms(c), [1.999445824405655, 2.1987859697318775, 2.2253772831321434, 3.0058518984310365, 3.0845353423758435, 3.517523280432535, 4.648260446485127, 4.864970695503937], 1e-9)
    c = CSP(n_components=3, component_order="alternate")
    F = c.fit_transform(Xd, y)
    assert close(F[0], [0.14758838655276865, -2.614682382107143, -1.3440322927961916], 1e-9)
    c = CSP(n_components=3, restr_type=None)
    F = c.fit_transform(Xd, y)
    assert close(F[0], [0.877010229142544, -0.6345035863744745, -0.17724849271479803], 1e-9)
    assert close(norms(c), [1205029.7021779148, 1598817.7381994466, 1787079.9549460406, 2426719.388793876, 2456706.2567110723, 2783830.0202815235, 3798099.8261766187, 3811660.993958937], 1e-9)
    c = CSP(n_components=3, rank="full")
    F = c.fit_transform(Xd, y)
    assert close(F[0], [-2.614682382107143, 0.14758838655276865, -1.4817335180404991], 1e-9)
    assert c.transform_csp_space(Xd).shape == (40, 3, 256)
    assert c.transform(Xd).shape == (40, 3)

test_csp_variants()

@test
def test_csp_multiclass_and_rank_deficient():
    Xd, y = filtered()
    y3 = np.arange(40) % 3
    c = CSP(n_components=3)
    F = c.fit_transform(Xd, y3)
    assert close(F[0], [0.5951261875783659, -1.464793703570126, -0.5167065005335659], 1e-9)
    assert close(norms(c), [1066261.0380769242, 1358904.4797310522, 1879121.4701344445, 2261733.6160620116, 2434007.550200189, 3011840.3006290207, 3494364.332598885, 4296356.992530832], 1e-9)
    assert c.evals_ is None
    Xr = Xd.copy()
    Xr[:, 7] = Xr[:, 0] + Xr[:, 1]
    c = CSP(n_components=3)
    F = c.fit_transform(Xr, y)
    assert c.filters_.shape == (7, 8)
    assert close(F[0], [-2.206506444764829, -2.1432970955663384, -0.4382243483561124], 1e-9)
    assert close(norms(c), [1053596.1335672608, 1086300.0451524537, 1318965.0963301852, 1353391.7986647133, 1362466.0475065594, 1989959.7238256817, 2776159.811276621], 1e-9)

test_csp_multiclass_and_rank_deficient()
