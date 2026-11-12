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
from mne.filter import filter_data, create_filter
from mne.time_frequency import psd_array_welch

@test
def test_create_filter():
    X, y = make_X()
    SF = 128.0
    h = create_filter(X[0], SF, 8.0, 30.0)
    assert len(h) == 213
    assert abs(float(h.sum())) < 1e-12
    m = len(h) // 2
    assert close(h[m - 2:m + 3], [-0.12776176092217567, 0.2086147539574088, 0.41732688374314736, 0.2086147539574088, -0.12776176092217567], 1e-10)
    h_lp = create_filter(None, SF, None, 20.0)
    assert len(h_lp) == 85
    assert close(h_lp[:4], [0.0004067227066253439, 0.0006075513136945694, 0.00013202976108879375, -0.0005883076997435931], 1e-9)
    h_hp = create_filter(None, SF, 4.0, None, fir_window="hann")
    assert len(h_hp) == 199
    assert close(h_hp[:4], [0.0, -7.826141433319079e-07, -3.268448305981923e-06, -7.5087665590861716e-06], 1e-9)
    h_bs = create_filter(None, SF, 30.0, 20.0, l_trans_bandwidth=2.0, h_trans_bandwidth=2.0)
    assert len(h_bs) == 213
    m = len(h_bs) // 2
    assert close(h_bs[m - 1:m + 2], [-0.04197441216708048, 0.8748069974305046, -0.04197441216708048], 1e-10)
    h_w2 = create_filter(None, SF, 8.0, 30.0, fir_design="firwin2")
    assert len(h_w2) == 423
    m = len(h_w2) // 2
    assert close(h_w2[m - 1:m + 2], [0.20814591001099506, 0.41796875, 0.20814591001099397], 1e-9)

test_create_filter()

@test
def test_filter_data():
    X, y = make_X()
    SF = 128.0
    x2 = X[0] * 1.0
    fd = filter_data(x2, SF, 8.0, 30.0)
    assert close(fd[2, :6], [-3.0531133177191805e-16, 1.183369109377077, 0.9492055402996562, -0.3954870096636289, -1.2296289062741568, -0.6056907507186284], 1e-9)
    assert closef(float((fd ** 2).sum()), 585.9466311210513)
    assert (x2 == X[0]).all()  # copy=True leaves input untouched
    fd = filter_data(x2, SF, 1.0, None, pad="edge", fir_window="blackman")
    assert close(fd[1, -4:], [-0.3616039216122595, -0.4048049691977424, -0.3119265932024904, 0.18897442457551977], 1e-9)
    assert closef(float((fd ** 2).sum()), 1387.5465682729287)
    fd = filter_data(x2, SF, None, 40.0, phase="zero-double")
    assert close(fd[3, 50:54], [-0.2930724878048031, 0.41051295794401527, 1.169367663065021, 0.5895704479629579], 1e-9)

test_filter_data()

@test
def test_raw():
    X, y = make_X()
    SF = 128.0
    d = X[:, :3, :].transpose(1, 0, 2).reshape((3, -1)) * 1e-6
    raw = mne.io.RawArray(d, mne.create_info(["a", "b", "c"], SF, ["eeg", "eeg", "misc"]))
    raw.filter(1.0, 40.0)
    rd = raw.get_data()
    assert closef(float((rd ** 2).sum()), 2.3440353853634828e-08)
    assert close(rd[0, 1000:1004], [-2.462093870400047e-06, -2.0131590949518157e-06, -1.5340615646681927e-06, -9.955808037999493e-07], 1e-9)
    # misc channel untouched by filter
    assert close(rd[2, 1000:1002], [2.431247425707317e-07, 1.1591119422400313e-06], 1e-12)
    sp = raw.compute_psd(fmax=50.0)
    sd = sp.get_data()
    assert sd.shape == (2, 801)
    assert len(sp.freqs) == 801
    assert closef(float(sd.sum()), 2.5766456280263608e-11, 1e-9)
    assert closef(float(sd[1, 7]), 1.9994502879085642e-16, 1e-9)
    raw.crop(10.0, 20.0)
    assert raw.get_data().shape == (3, 1281)
    assert raw.times[-1] == 10.0

test_raw()

@test
def test_psd_array_welch():
    X, y = make_X()
    SF = 128.0
    p, f = psd_array_welch(X[:2, :2, :], SF, fmin=2.0, fmax=50.0, n_fft=64, n_overlap=32, average="median")
    assert p.shape == (2, 2, 25)
    assert close(p[1, 1, :5], [0.04813294434114963, 0.03701410118730671, 0.00047713062967425743, 2.490056196228108e-05, 1.5173576250958885e-05], 1e-9)
    p2, f2 = psd_array_welch(X[0, 0], SF, n_fft=100, n_per_seg=50, window="hann", average=None)
    assert p2.shape == (51, 5)
    assert close(p2[3, :], [0.02935651525764544, 0.005426919284372018, 0.009982407803833325, 0.01593846033893004, 0.0007880635236440202], 1e-9)

test_psd_array_welch()

@test
def test_epochs_misc():
    X, y = make_X(4, 3, 50)
    info = mne.create_info(3, 100.0, ["eeg", "misc", "eeg"])
    assert info.get_channel_types() == ["eeg", "misc", "eeg"]
    ep = mne.EpochsArray(X, info, tmin=-0.1)
    assert ep.events[1, 0] == 1 and ep.events[1, 2] == 1
    assert ep.event_id == {"1": 1}
    assert abs(ep.times[0] + 0.1) < 1e-15
    ep2 = ep.copy().crop(0.0, 0.2)
    assert ep2.get_data().shape == (4, 3, 21)
    assert ep.get_data().shape == (4, 3, 50)
    assert ep.get_data(picks="eeg").shape == (4, 2, 50)
    assert ep.get_data(tmin=0.0, tmax=0.1).shape == (4, 3, 11)
    assert ep.average().data.shape == (2, 50)
    assert mne.__version__ == "1.13.2"

test_epochs_misc()
