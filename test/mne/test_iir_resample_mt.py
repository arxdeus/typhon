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
from mne.filter import filter_data, resample, construct_iir_filter, IIRParams
from mne.time_frequency import psd_array_multitaper

@test
def test_iir():
    X, y = make_X()
    SF = 128.0
    x2 = X[0] * 1.0
    assert close(filter_data(x2, SF, 8.0, 30.0, method="iir")[1, 100:104], [-1.2413020704030373, -0.7777016551074198, 0.24903731226980208, 0.9878785037905848], 1e-9)
    assert close(filter_data(x2, SF, None, 20.0, method="iir", iir_params=IIRParams(order=2, ftype="butter", output="ba"))[0, 50:53], [-0.9149714501275942, -0.6201070869934374, -0.23814105264343077], 1e-9)
    assert close(filter_data(x2, SF, 2.0, None, method="iir", phase="forward")[2, -3:], [-0.7018691463614242, -0.6240263169724136, 0.28834284521632425], 1e-9)
    assert close(filter_data(x2, SF, 8.0, 30.0, method="iir", iir_params=IIRParams(order=3, ftype="cheby1", rp=1.0))[4, 20:23], [0.19599355430435345, 0.2439701104209661, -0.08525375160552778], 1e-9)
    assert construct_iir_filter({"order": 4}, [8.0, 30.0], None, SF, "bandpass")["padlen"] == 63
    info = mne.create_info([f"E{i}" for i in range(8)], SF, "eeg")
    ep = mne.EpochsArray(X * 1e-6, info)
    ep.filter(8.0, 30.0, method="iir")
    assert closef(float((ep.get_data() ** 2).sum()), 2.5230427069506212e-08, 1e-9)

test_iir()

@test
def test_resample():
    X, y = make_X()
    SF = 128.0
    x2 = X[0] * 1.0
    r = resample(x2, 2.0, 3.0, npad="auto")
    assert r.shape == (8, 171)
    assert close(r[3, 10:13], [0.3516336260641264, -0.824497694247254, 0.3459686768570305], 1e-9)
    r = resample(x2, 3.0, 1.0)
    assert r.shape == (8, 768)
    assert close(r[1, 100:103], [-1.0699050923290236, -0.7182686109289133, -0.3395114191365017], 1e-9)
    info = mne.create_info([f"E{i}" for i in range(8)], SF, "eeg")
    ep = mne.EpochsArray(X[:3] * 1e-6, info, tmin=-0.5)
    ep.resample(64.0)
    d = ep.get_data()
    assert d.shape == (3, 8, 128)
    assert close(d[2, 5, 30:33], [4.409614114757527e-07, 5.394947297651408e-07, 9.778711079825003e-07], 1e-9)
    assert abs(ep.times[-1] - 1.484375) < 1e-12
    assert ep.info["lowpass"] == 32.0 and ep.info["sfreq"] == 64.0
    raw = mne.io.RawArray(X[:2, :3, :].reshape((3, -1))[:, :400] * 1.0, mne.create_info(3, SF, ["eeg", "eeg", "stim"]))
    raw.resample(100.0)
    rd = raw.get_data()
    assert rd.shape == (3, 312)
    assert close(rd[0, 200:203], [0.6312459589051111, 1.4761886888423745, 0.895475462894343], 1e-9)
    assert close(rd[2, 50:53], [-0.4440226233981727, -1.2917362004596242, -1.5078640741655749], 1e-12)

test_resample()

@test
def test_multitaper():
    X, y = make_X()
    SF = 128.0
    p, f = psd_array_multitaper(X[:2, :2, :], SF, fmax=30.0)
    assert p.shape == (2, 2, 61)
    assert close(p[1, 0, :4], [0.03068570829706689, 0.08354255314698696, 1.4892598363909293, 4.2793266393976035], 1e-9)
    p1, f1 = psd_array_multitaper(X[0, 3], SF, bandwidth=3.0, adaptive=True, normalization="full")
    assert p1.shape == (129,)
    assert close(p1[5:8], [0.04703924892094006, 1.0598844326199254e-09, 1.0879998894726503e-09], 1e-9)
    info = mne.create_info([f"E{i}" for i in range(8)], SF, "eeg")
    ep = mne.EpochsArray(X * 1e-6, info)
    ep.filter(8.0, 30.0)
    sp = ep.compute_psd(method="multitaper", fmin=2.0, fmax=40.0)
    assert sp.get_data().shape == (40, 8, 77)
    assert close(sp.get_data()[7, 2, :3], [2.908530864311808e-15, 2.2730406475643855e-15, 2.344249403578243e-15], 1e-9)
    sp2 = ep.compute_psd(method="welch", n_fft=64, average=None)
    seg = sp2.get_data_segments()
    assert seg.shape == (40, 8, 33, 4)
    assert close(seg[1, 1, 10, :], [8.250866842137885e-18, 4.6484376947418046e-18, 5.253571258566813e-18, 1.3677460461063534e-17], 1e-9)
    c = CSP(n_components=3, transform_into="csp_space")
    Z = c.fit_transform(ep.get_data(), y)
    assert Z.shape == (40, 3, 256)
    assert closef(float((Z ** 2).sum()), 15357.000000000002, 1e-9)

test_multitaper()
