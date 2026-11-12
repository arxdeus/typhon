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

@test
def test_acceptance_pipeline():
    X, y = make_X()
    SF = 128.0
    mne.set_log_level("ERROR")
    info = mne.create_info([f"E{i}" for i in range(X.shape[1])], SF, "eeg")
    assert info["sfreq"] == 128.0
    assert info["ch_names"][3] == "E3"
    assert info.nchan == 8
    ev = np.c_[np.arange(len(y)) * X.shape[2], np.zeros(len(y), int), y.astype(int)]
    ep = mne.EpochsArray(X * 1e-6, info, ev, tmin=0.0, verbose=False)
    r = ep.filter(8.0, 30.0, method="fir", fir_design="firwin", verbose=False)
    assert r is ep
    fd = ep.get_data(copy=True)
    assert closef(float((fd ** 2).sum()), 3.36161866577061e-08)
    assert close(fd[0, 0, 100:110], [-7.108595702252508e-07, -4.2957184873969654e-07, 1.4225599352595037e-07, 4.6793660661561615e-07, 3.9380437205368026e-07, 3.509862925089404e-07, 6.878443815025033e-07, 1.1131693712002773e-06, 1.0752778390262964e-06, 5.341701146255255e-07])
    assert close(fd[39, 7, :5], [3.3505662183907503e-07, 4.6326503462142665e-07, 3.396261492889182e-07, 2.852127728603828e-07, 3.008468573338508e-07])
    assert ep.info["highpass"] == 8.0 and ep.info["lowpass"] == 30.0

    psd = ep.compute_psd(method="welch", fmin=4, fmax=40, n_fft=128, verbose=False)
    pd_ = psd.get_data()
    assert pd_.shape == (40, 8, 37)
    assert len(psd.freqs) == 37
    assert closef(float(pd_.sum()), 1.3208315821493515e-10)
    assert close(pd_[0, 0], [1.449469893768098e-17, 3.614066342744705e-18, 3.668800639462693e-15, 1.6743102040632407e-13, 1.9760142654794654e-13, 5.450997298724621e-15, 1.7653295292155073e-17, 2.9171588709958612e-15, 1.4034464541260397e-14, 2.8933016998475396e-15, 7.559851907304498e-18, 6.001713203278818e-18, 4.883648786601509e-18, 4.144355887134005e-18, 3.660334131499778e-18, 3.3469564777178557e-18, 3.1909119931846936e-18, 3.1956571868996656e-18, 3.259472403631161e-18, 2.8962743066630337e-18, 2.19898351023316e-18, 4.870036836942415e-16, 2.2599369958531036e-14, 2.1465085377227962e-14, 3.7210359904452385e-16, 1.3196772303606679e-18, 2.7967410804582933e-18, 3.1151402711021352e-18, 2.689138955199939e-18, 2.0338715244480275e-18, 1.3977835414254827e-18, 9.384321090363321e-19, 6.939179823421574e-19, 5.98834297775526e-19, 5.597602029631424e-19, 5.211721793071802e-19, 4.756433961626289e-19], 1e-9)
    assert closef(float(pd_[5, 3, 10]), 6.009715684796411e-18, 1e-9)
    assert close(psd.freqs, np.arange(4.0, 41.0))

    csp = CSP(n_components=4, log=True, transform_into="average_power")
    F = csp.fit_transform(ep.get_data(copy=True), y)
    assert F.shape == (40, 4)
    assert close(F[0], [-2.614682382107143, 0.14758838655276865, -1.4817335180404991, -1.6279110008438828], 1e-9)
    assert close(F[-1], [-0.05702346628666142, -0.561620191813751, 0.0670159113018898, -0.7946791754021464], 1e-9)
    assert closef(float(F.sum()), -148.96927503320597, 1e-9)
    ev_ = np.sort(np.linalg.norm(csp.filters_, axis=1))
    assert close(ev_, [1130201.6852203887, 1134360.8028901257, 1258860.3170178793, 1701952.2018627462, 1742322.5368685583, 1984633.10052468, 2637390.4822694017, 2739853.036942209], 1e-9)
    # generalized eigenvectors are only defined up to sign (LAPACK dependent):
    # compare filters_ and patterns_ rows up to a global sign per component
    f0 = csp.filters_[0]
    s = 1.0 if f0[6] < 0 else -1.0
    assert close(f0 * s, [-124626.43089922282, 13385.6483763692, -20973.272826774602, -154848.19683879596, -147425.83274450625, 372118.22846637305, -995549.9761020859, 308738.6580926923], 1e-9)
    p0 = csp.patterns_[0]
    s = 1.0 if p0[0] < 0 else -1.0
    assert close(p0 * s, [-7.917180401422176e-07, 6.988087490245849e-09, 6.905873350582561e-09, -7.772458467453144e-07, 5.042296377748092e-09, 4.916351202307735e-09, -7.832945703780986e-07, 4.2818484745022647e-10], 1e-9)
    assert close(csp.mean_, [0.49990234375000037, 0.49990234375, 0.4999023437499998, 0.49990234375000026], 1e-9)
    assert close(csp.std_, [0.3896659779501183, 0.28881986851436, 0.3215206106970177, 0.23811755690168904], 1e-9)
    assert close(csp.evals_.__val__(), [0.11385065836392232, 0.5178123100977063, 0.4833148248438476, 0.49175275510482264, 0.5081240391759759, 0.5037881630330916, 0.49702008320636876, 0.5002946660529083], 1e-9)
    assert len(ep) == 40

    ave = ep.average()
    assert ave.data.shape == (8, 256)
    assert closef(float(ave.data.sum()), -5.865453679928e-07, 1e-9)
    assert ave.times[-1] == 1.9921875

test_acceptance_pipeline()
