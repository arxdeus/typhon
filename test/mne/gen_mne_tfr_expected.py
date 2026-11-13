# Generates test/mne/test_tfr.py, test_tfr_mt.py and test_tfr_csd.py from templates/*.tpl using reference
# values from real mne 1.13.2.
# Run: /path/to/venv/bin/python test/mne/gen_mne_tfr_expected.py
import os
import numpy as np
import mne
from mne.time_frequency import (morlet, fwhm, tfr_array_morlet, tfr_array_multitaper,
                                tfr_array_stockwell, stft, istft, stftfreq, tfr_morlet,
                                tfr_multitaper, csd_array_fourier, csd_array_multitaper,
                                csd_array_morlet, psd_array_welch)

mne.set_log_level("ERROR")
HERE = os.path.dirname(os.path.abspath(__file__))


def lst(a):
    a = np.asarray(a).ravel()
    return "[" + ", ".join(repr(float(v)) for v in a) + "]"


def sig(n_ep, n_ch, n_t, sf=100.0):
    e = np.arange(n_ep)[:, None, None]
    c = np.arange(n_ch)[None, :, None]
    t = np.arange(n_t)[None, None, :] / sf
    return (np.sin(2 * np.pi * (8 + 2 * c) * t + 0.3 * e) + 0.5 * np.cos(2 * np.pi * 21.0 * t * (1 + 0.05 * e))
            + 0.2 * np.sin(0.7 * np.arange(n_t)[None, None, :] * (c + 1) + e))


E = {}
X = sig(4, 3, 200)
freqs = np.array([6.0, 10.0, 15.0, 22.0])
W = morlet(100.0, freqs, n_cycles=[3, 4, 5, 6])
E["mw_len"] = "[" + ", ".join(str(len(w)) for w in W) + "]"
E["mw0r"] = lst(W[0].real); E["mw0i"] = lst(W[0].imag)
E["mw2"] = lst(np.abs(morlet(100.0, 12.0, 5.0, sigma=8.0, zero_mean=True)))
E["fwhm"] = lst(fwhm(freqs, 5.0))
p = tfr_array_morlet(X, 100.0, freqs, n_cycles=4.0, output="power")
E["am_p"] = lst(p[:, :, :, ::7])
c = tfr_array_morlet(X, 100.0, freqs, n_cycles=np.array([3.0, 4, 5, 6]), output="complex", decim=3)
E["am_cr"] = lst(c.real[:, :, :, ::5]); E["am_ci"] = lst(c.imag[:, :, :, ::5])
E["am_ph"] = lst(tfr_array_morlet(X, 100.0, freqs, n_cycles=4.0, output="phase", use_fft=False)[:, :, :, ::9])
E["am_ap"] = lst(tfr_array_morlet(X, 100.0, freqs, n_cycles=4.0, output="avg_power", zero_mean=False)[:, :, ::6])
E["am_itc"] = lst(tfr_array_morlet(X, 100.0, freqs, n_cycles=4.0, output="itc")[:, :, ::6])
api = tfr_array_morlet(X, 100.0, freqs, n_cycles=4.0, output="avg_power_itc", decim=2)
E["am_apir"] = lst(api.real[:, :, ::4]); E["am_apii"] = lst(api.imag[:, :, ::4])
E["mt_p"] = lst(tfr_array_multitaper(X, 100.0, freqs, n_cycles=4.0, time_bandwidth=4.0, output="power")[:, :, :, ::7])
E["mt_ap"] = lst(tfr_array_multitaper(X, 100.0, freqs, n_cycles=freqs / 2, time_bandwidth=3.0, output="avg_power", decim=2)[:, :, ::5])
mc, mw = tfr_array_multitaper(X, 100.0, freqs, n_cycles=4.0, output="complex", return_weights=True)
E["mt_c"] = lst(np.abs(mc[:, :, :, :, ::11])); E["mt_w"] = lst(mw)
apim = tfr_array_multitaper(X, 100.0, freqs, n_cycles=4.0, output="avg_power_itc")
E["mt_apir"] = lst(apim.real[:, :, ::9]); E["mt_apii"] = lst(apim.imag[:, :, ::9])
ps, itc, fr = tfr_array_stockwell(X, 100.0, fmin=5.0, fmax=30.0, return_itc=True)
E["st_f"] = lst(fr); E["st_p"] = lst(ps[:, :, ::13]); E["st_i"] = lst(itc[:, :, ::13])
ps, _, fr = tfr_array_stockwell(X, 100.0, n_fft=512, width=1.5, decim=4)
E["st2_f"] = lst(fr[::7]); E["st2_p"] = lst(ps[:, ::7, ::5])
S = stft(X[0], 32, 8)
E["stft_r"] = lst(S.real[:, ::3, ::4]); E["stft_i"] = lst(S.imag[:, ::3, ::4]); E["stft_shape"] = str(list(S.shape))
E["istft"] = lst(istft(S, 8, 200)[:, ::9])
E["stftf"] = lst(stftfreq(32, 100.0))
# containers
info = mne.create_info(["a", "b", "c"], 100.0, "eeg")
ep = mne.EpochsArray(X * 1e-6, info, tmin=-0.5)
pw, it = tfr_morlet(ep, freqs, n_cycles=4.0, return_itc=True, decim=2)
E["tm_p"] = lst(pw.data[:, :, ::5]); E["tm_i"] = lst(it.data[:, :, ::5]); E["tm_t"] = lst(pw.times[::5])
b = pw.copy().apply_baseline((-0.5, -0.1), mode="logratio")
E["bl_lr"] = lst(b.data[:, :, ::5])
for m in ["mean", "ratio", "percent", "zscore", "zlogratio"]:
    E["bl_" + m] = lst(pw.copy().apply_baseline((None, 0.0), mode=m).data[:, :, ::7])
cr = pw.copy().crop(-0.2, 0.6, fmin=8.0, fmax=20.0)
E["cr_d"] = lst(cr.data[:, :, ::3]); E["cr_t"] = lst(cr.times); E["cr_f"] = lst(cr.freqs)
etf = tfr_morlet(ep, freqs, n_cycles=3.0, return_itc=False, average=False, use_fft=True)
E["et_d"] = lst(etf.data[:, :, :, ::17])
E["et_avg"] = lst(etf.average().data[:, :, ::11])
E["et_med"] = lst(etf.average(method="median").data[:, :, ::11])
mt = tfr_multitaper(ep, freqs, n_cycles=4.0, time_bandwidth=4.0, return_itc=False)
E["tmt"] = lst(mt.data[:, :, ::9])
ct = ep.compute_tfr("morlet", freqs, n_cycles=4.0, average=True, return_itc=False, tmin=-0.3, tmax=0.8)
E["ct"] = lst(ct.data[:, :, ::9]); E["ct_t"] = lst(ct.times[::9])
# CSD
cf = csd_array_fourier(X, 100.0, fmin=5, fmax=25, n_fft=128)
E["cf_f"] = lst(cf.frequencies); E["cf_r"] = lst(cf._data.real[:, ::3]); E["cf_i"] = lst(cf._data.imag[:, ::3])
E["cf_m"] = lst(np.abs(cf.mean(fmin=8, fmax=15).get_data()))
cm = csd_array_multitaper(X, 100.0, fmin=5, fmax=25, tmin=0.2, tmax=1.5)
E["cm_r"] = lst(cm._data.real[:, ::2]); E["cm_i"] = lst(cm._data.imag[:, ::2])
cma = csd_array_multitaper(X, 100.0, fmin=5, fmax=25, adaptive=True, bandwidth=6.0)
E["cma"] = lst(np.abs(cma._data[:, ::2]))
cw = csd_array_morlet(X, 100.0, [8.0, 12.0, 20.0], n_cycles=5, tmin=0.3, tmax=1.6)
E["cw_r"] = lst(cw._data.real); E["cw_i"] = lst(cw._data.imag)
E["cw_g"] = lst(np.abs(cw.get_data(frequency=12.0)))
E["cw_tm"] = lst([cw.tmin, cw.tmax])
# psd_array_welch parity
pm, fq = psd_array_welch(X, 100.0, fmin=2, fmax=40, n_fft=64, n_overlap=32, average="median")
E["pw_med"] = lst(pm[:, :, ::3])
pm, fq = psd_array_welch(X, 100.0, n_fft=64, n_per_seg=50, window="hann", average="mean")
E["pw_hann"] = lst(pm[:, :, ::4])
pm, fq = psd_array_welch(X, 100.0, n_fft=64, window=("tukey", 0.25))
E["pw_tuk"] = lst(pm[:, :, ::4])

for name in ["test_tfr", "test_tfr_mt", "test_tfr_csd"]:
    src = open(os.path.join(HERE, "templates", name + ".tpl")).read()
    for k, v in E.items():
        src = src.replace("@@" + k + "@@", v)
    assert "@@" not in src, src[src.index("@@"):src.index("@@") + 40]
    open(os.path.join(HERE, name + ".py"), "w").write(src)
    print("wrote", name)
