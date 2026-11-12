# Generates the expected literals for test/mne/*.py from real mne 1.13.2.
# Run: /path/to/venv/bin/python test/mne/gen_expected.py
import numpy as np
import mne
from mne.decoding import CSP
from mne.filter import filter_data, create_filter
from mne.time_frequency import psd_array_welch

mne.set_log_level("ERROR")


def make_X(n_ep=40, n_ch=8, n_t=256):
    # deterministic pseudo-random data (no RNG so Typhon reproduces it exactly)
    e = np.arange(n_ep)[:, None, None]
    c = np.arange(n_ch)[None, :, None]
    t = np.arange(n_t)[None, None, :]
    y = np.arange(n_ep) % 2
    X = (np.sin(0.37 * t * (c + 1) + 0.11 * e) + 0.5 * np.cos(0.071 * t * (e % 7 + 1) + c)
         + 0.3 * np.sin(1.3 * t + 0.7 * e * c) + (y[:, None, None] * 0.8 + 0.2) * np.sin(2 * np.pi * 12.0 / 128.0 * t) * (c % 3 == 0)
         + 0.05 * c)
    return X, y


def r(a):
    return repr(float(a))


def lst(a):
    return "[" + ", ".join(r(v) for v in np.ravel(a)) + "]"


X, y = make_X()
SF = 128.0
info = mne.create_info([f"E{i}" for i in range(X.shape[1])], SF, "eeg")
ev = np.c_[np.arange(len(y)) * X.shape[2], np.zeros(len(y), int), y.astype(int)]
ep = mne.EpochsArray(X * 1e-6, info, ev, tmin=0.0, verbose=False)
ep.filter(8.0, 30.0, method="fir", fir_design="firwin", verbose=False)
fd = ep.get_data(copy=True)
print("# acceptance")
print("FILT_SUMSQ =", r((fd ** 2).sum()))
print("FILT_0_0_SLICE =", lst(fd[0, 0, 100:110]))
print("FILT_39_7_SLICE =", lst(fd[39, 7, :5]))
psd = ep.compute_psd(method="welch", fmin=4, fmax=40, n_fft=128, verbose=False)
pd_ = psd.get_data()
print("PSD_SHAPE =", pd_.shape, "NFREQ =", len(psd.freqs))
print("PSD_SUM =", r(pd_.sum()))
print("PSD_0_0 =", lst(pd_[0, 0]))
print("PSD_5_3_10 =", r(pd_[5, 3, 10]))
print("FREQS =", lst(psd.freqs))
csp = CSP(n_components=4, log=True, transform_into="average_power")
F = csp.fit_transform(ep.get_data(copy=True), y)
print("F_FIRST =", lst(F[0]))
print("F_LAST =", lst(F[-1]))
print("F_SUM =", r(F.sum()))
print("NORMS =", lst(np.sort(np.linalg.norm(csp.filters_, axis=1))))
print("FILT0 =", lst(csp.filters_[0]))
print("PAT0 =", lst(csp.patterns_[0]))
print("MEAN =", lst(csp.mean_))
print("STD =", lst(csp.std_))
print("EVALS =", lst(csp.evals_))
print("LEN =", len(ep))
ave = ep.average()
print("AVE_SUM =", r(ave.data.sum()), "AVE_SHAPE", ave.data.shape, "TIMES_LAST", r(ave.times[-1]))

print("# filter_data / create_filter")
x2 = X[0] * 1.0
h = create_filter(x2, SF, 8.0, 30.0)
print("H_LEN =", len(h), "H_SUM =", r(h.sum()), "H_MID =", lst(h[len(h) // 2 - 2: len(h) // 2 + 3]))
h_lp = create_filter(None, SF, None, 20.0)
print("HLP_LEN =", len(h_lp), "HLP =", lst(h_lp[:4]))
h_hp = create_filter(None, SF, 4.0, None, fir_window="hann")
print("HHP_LEN =", len(h_hp), "HHP =", lst(h_hp[:4]))
h_bs = create_filter(None, SF, 30.0, 20.0, l_trans_bandwidth=2.0, h_trans_bandwidth=2.0)
print("HBS_LEN =", len(h_bs), "HBS =", lst(h_bs[len(h_bs) // 2 - 1: len(h_bs) // 2 + 2]))
h_w2 = create_filter(None, SF, 8.0, 30.0, fir_design="firwin2")
print("HW2_LEN =", len(h_w2), "HW2 =", lst(h_w2[len(h_w2) // 2 - 1: len(h_w2) // 2 + 2]))
fdd = filter_data(x2, SF, 8.0, 30.0)
print("FD_RL =", lst(fdd[2, :6]), "FD_RL_SUMSQ =", r((fdd ** 2).sum()))
fdd = filter_data(x2, SF, 1.0, None, pad="edge", fir_window="blackman")
print("FD_HP =", lst(fdd[1, -4:]), "FD_HP_SUMSQ =", r((fdd ** 2).sum()))
fdd = filter_data(x2, SF, None, 40.0, phase="zero-double")
print("FD_ZD =", lst(fdd[3, 50:54]))

print("# raw")
raw = mne.io.RawArray(X[:, :3, :].transpose(1, 0, 2).reshape(3, -1) * 1e-6,
                      mne.create_info(["a", "b", "c"], SF, ["eeg", "eeg", "misc"]))
raw.filter(1.0, 40.0)
rd = raw.get_data()
print("RAW_SUMSQ =", r((rd ** 2).sum()), "RAW_A =", lst(rd[0, 1000:1004]), "RAW_C =", lst(rd[2, 1000:1002]))
sp = raw.compute_psd(fmax=50.0)
sd = sp.get_data()
print("RSP_SHAPE =", sd.shape, "RSP_SUM =", r(sd.sum()), "RSP_1_7 =", r(sd[1, 7]), "RSP_NF", len(sp.freqs))
raw.crop(10.0, 20.0)
print("CROP_N =", raw.get_data().shape, "CROP_T0", r(raw.times[-1]))

print("# psd_array_welch")
p, f = psd_array_welch(X[:2, :2, :], SF, fmin=2.0, fmax=50.0, n_fft=64, n_overlap=32, average="median")
print("PW_MED_SHAPE =", p.shape, "PW_MED =", lst(p[1, 1, :5]))
p, f = psd_array_welch(X[0, 0], SF, n_fft=100, n_per_seg=50, window="hann", average=None)
print("PW_NONE_SHAPE =", p.shape, "PW_NONE =", lst(p[3, :]))

print("# CSP variants")
Xd = fd
for kw in [dict(reg=0.1), dict(reg="ledoit_wolf"), dict(reg="oas"), dict(cov_est="epoch"),
           dict(norm_trace=True, log=False), dict(component_order="alternate"),
           dict(restr_type=None), dict(rank="full")]:
    c = CSP(n_components=3, **kw)
    Fv = c.fit_transform(Xd, y)
    print(kw, "F0 =", lst(Fv[0]), "NORMS =", lst(np.sort(np.linalg.norm(c.filters_, axis=1))))
y3 = np.arange(40) % 3
c = CSP(n_components=3)
Fv = c.fit_transform(Xd, y3)
print("3class F0 =", lst(Fv[0]), "NORMS =", lst(np.sort(np.linalg.norm(c.filters_, axis=1))))
Xr = Xd.copy(); Xr[:, 7] = Xr[:, 0] + Xr[:, 1]
c = CSP(n_components=3)
Fv = c.fit_transform(Xr, y)
print("rankdef F0 =", lst(Fv[0]), "NORMS =", lst(np.sort(np.linalg.norm(c.filters_, axis=1))))

print("# IIR / resample / multitaper / csp_space / average=None")
x2 = X[0] * 1.0
print("IIR_BP =", lst(filter_data(x2, SF, 8.0, 30.0, method="iir")[1, 100:104]))
print("IIR_LP_BA =", lst(filter_data(x2, SF, None, 20.0, method="iir",
                                     iir_params=dict(order=2, ftype="butter", output="ba"))[0, 50:53]))
print("IIR_HP_FWD =", lst(filter_data(x2, SF, 2.0, None, method="iir", phase="forward")[2, -3:]))
print("IIR_CHEBY =", lst(filter_data(x2, SF, 8.0, 30.0, method="iir",
                                     iir_params=dict(order=3, ftype="cheby1", rp=1.0))[4, 20:23]))
from mne.filter import construct_iir_filter
print("IIR_PADLEN =", construct_iir_filter(dict(order=4, ftype="butter", output="sos"), [8.0, 30.0], None, SF, "bandpass")["padlen"])
ep2 = mne.EpochsArray(X * 1e-6, info, ev, tmin=0.0)
ep2.filter(8.0, 30.0, method="iir")
print("EP_IIR_SUMSQ =", r((ep2.get_data() ** 2).sum()))
from mne.filter import resample
y_ = resample(x2, 2.0, 3.0, npad="auto")
print("RS_SHAPE =", y_.shape, "RS =", lst(y_[3, 10:13]))
y_ = resample(x2, 3.0, 1.0)
print("RS_UP_SHAPE =", y_.shape, "RS_UP =", lst(y_[1, 100:103]))
ep3 = mne.EpochsArray(X[:3] * 1e-6, info, tmin=-0.5)
ep3.resample(64.0)
d3 = ep3.get_data()
print("EP_RS_SHAPE =", d3.shape, "EP_RS =", lst(d3[2, 5, 30:33]), "EP_RS_T =", r(ep3.times[-1]), r(ep3.info["lowpass"]))
raw2 = mne.io.RawArray(X[:2, :3, :].reshape(3, -1)[:, :400] * 1.0, mne.create_info(3, SF, ["eeg", "eeg", "stim"]))
raw2.resample(100.0)
rd2 = raw2.get_data()
print("RAW_RS_SHAPE =", rd2.shape, "RAW_RS =", lst(rd2[0, 200:203]), "RAW_RS_STIM =", lst(rd2[2, 50:53]))
from mne.time_frequency import psd_array_multitaper
p, f = psd_array_multitaper(X[:2, :2, :], SF, fmax=30.0)
print("MT_SHAPE =", p.shape, "MT =", lst(p[1, 0, :4]))
p, f = psd_array_multitaper(X[0, 3], SF, bandwidth=3.0, adaptive=True, normalization="full")
print("MT_AD_SHAPE =", p.shape, "MT_AD =", lst(p[5:8]))
sp = ep.compute_psd(method="multitaper", fmin=2.0, fmax=40.0)
print("EP_MT_SHAPE =", sp.get_data().shape, "EP_MT =", lst(sp.get_data()[7, 2, :3]))
sp = ep.compute_psd(method="welch", n_fft=64, average=None)
print("EP_NONE_SHAPE =", sp.get_data().shape, "EP_NONE =", lst(sp.get_data()[1, 1, 10, :]))
c = CSP(n_components=3, transform_into="csp_space")
Z = c.fit_transform(fd, y)
print("CSPSPACE_SHAPE =", Z.shape, "CSPSPACE_SUMSQ =", r((Z ** 2).sum()))
