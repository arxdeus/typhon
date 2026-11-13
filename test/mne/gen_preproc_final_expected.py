# Generates the expected values embedded in test_maxwell.py / test_preproc_final.py (mne 1.13.2)
import numpy as np, mne
mne.set_log_level('ERROR')
from mne.preprocessing import maxwell_filter, compute_maxwell_basis, find_bad_channels_maxwell


def meg_info(n_mag=40, n_grad=40, sfreq=200.0, seed=1):
    rng = np.random.RandomState(seed)
    n = n_mag + n_grad
    names = [f"MEG{i + 1:04d}" for i in range(n)]
    types = ["mag"] * n_mag + ["grad"] * n_grad
    info = mne.create_info(names, sfreq, types)
    # points on a sphere of radius 0.11 m (upper hemisphere), radial normals
    k = 0
    for i in range(n):
        th = np.arccos(1 - 0.9 * (i + 0.5) / n)
        ph = 2.399963 * i
        pos = 0.11 * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        ez = pos / np.linalg.norm(pos)
        ex = np.cross([0, 0, 1.0], ez)
        if np.linalg.norm(ex) < 1e-8:
            ex = np.array([1.0, 0, 0])
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        info["chs"][i]["loc"][:] = np.concatenate([pos, ex, ey, ez])
        info["chs"][i]["coil_type"] = 3022 if types[i] == "mag" else 3012
        info["chs"][i]["unit"] = 112 if types[i] == "mag" else 201
    info["dev_head_t"] = mne.transforms.Transform("meg", "head", np.eye(4))
    return info


def make_raw(seed=2, n=2000):
    info = meg_info()
    rng = np.random.RandomState(seed)
    d = rng.randn(len(info["ch_names"]), n) * 1e-13
    t = np.arange(n) / info["sfreq"]
    d += 1e-12 * np.sin(2 * np.pi * 7 * t)[None, :] * rng.randn(len(d), 1)
    d[40:] *= 0.01
    return mne.io.RawArray(d, info)


raw = make_raw()
org = (0.0, 0.0, 0.04)
p = dict(origin=org, coord_frame="meg", int_order=6, ext_order=2, mag_scale=100.0, bad_condition="ignore")
r1 = maxwell_filter(raw, **p)
d1 = r1.get_data()
print("SSS", repr(float(np.abs(d1).sum())), repr(float(d1[3, 100])), repr(float(d1[50, 500])), repr(float(d1[:, 0].sum())))
r2 = maxwell_filter(raw, st_duration=4.0, st_correlation=0.9, **p)
d2 = r2.get_data()
print("TSSS", repr(float(np.abs(d2).sum())), repr(float(d2[3, 100])), repr(float(d2[50, 500])))
r3 = maxwell_filter(raw, regularize=None, **p)
print("NOREG", repr(float(np.abs(r3.get_data()).sum())))
S, pS, rm, nin = compute_maxwell_basis(raw.info, origin=org, int_order=6, ext_order=2, coord_frame="meg",
                                       regularize="in", ignore_ref=True, bad_condition="ignore")
print("BASIS", S.shape, pS.shape, len(rm), nin, repr(float(np.abs(S).sum())), repr(float(np.abs(pS).sum())), repr(float(S[5, 3])))
# bads
raw_b = raw.copy()
raw_b.info["bads"] = ["MEG0005"]
r4 = maxwell_filter(raw_b, **p)
print("BADS", repr(float(np.abs(r4.get_data()).sum())), r4.info["bads"])
# HFC
from mne.preprocessing import compute_proj_hfc
pr = compute_proj_hfc(raw.info, order=2)
print("HFC", len(pr), repr(float(np.abs(pr[0]["data"]["data"]).sum())), repr(float(pr[4]["data"]["data"][0, 7])), pr[3]["desc"])

# ICA events
from mne.preprocessing import ica_find_ecg_events, ica_find_eog_events
rng = np.random.RandomState(3)
sf = 100.0
info = mne.create_info(2, sf, "eeg")
t = np.arange(3000) / sf
sig = np.zeros(3000)
for k in range(5, 3000, 80):
    sig[k:k + 3] = np.array([1.0, 3.0, 1.0])
sig += 0.05 * rng.randn(3000)
rr = mne.io.RawArray(np.vstack([sig, rng.randn(3000)]), info)
ev = ica_find_ecg_events(rr, sig, event_id=999)
print("ECG", ev.shape, ev[:5, 0].tolist(), ev[-1, 0])
bl = np.zeros(3000)
for k in range(100, 2900, 250):
    bl += 5 * np.exp(-0.5 * ((np.arange(3000) - k) / 8.0) ** 2)
bl += 0.05 * rng.randn(3000)
ev2 = ica_find_eog_events(rr, bl, event_id=998)
print("EOG", ev2.shape, ev2[:4, 0].tolist())

# score funcs
from mne.preprocessing import get_score_funcs
sfn = get_score_funcs()
x = rng.randn(3, 200)
y = x[1] * 2 + 0.1 * rng.randn(200)
print("SCORE", sorted(sfn.keys()), [repr(float(v)) for v in sfn["pearsonr"](x, y)], [repr(float(v)) for v in sfn["spearmanrho"](x, y)])

# movement
from mne.preprocessing import annotate_movement, compute_average_dev_head_t
info = mne.create_info(3, 100.0, "eeg")
raw_m = mne.io.RawArray(np.zeros((3, 3000)), info)
n = 30
tt = np.arange(n) * 1.0
pos = np.zeros((n, 10))
pos[:, 0] = tt
rng = np.random.RandomState(5)
pos[:, 1:4] = np.cumsum(rng.randn(n, 3) * 0.01, axis=0)
pos[:, 4:7] = np.cumsum(rng.randn(n, 3) * 0.002, axis=0)
pos[10:13, 4] += 0.05
ann, _ = annotate_movement(raw_m, pos, rotation_velocity_limit=2.0, translation_velocity_limit=0.02)
print("MOVE", len(ann), [round(float(v), 6) for v in ann.onset], [round(float(v), 6) for v in ann.duration], list(ann.description))
trans = compute_average_dev_head_t(raw_m, pos)["trans"]
print("AVGT", [repr(float(v)) for v in trans.ravel()])

# make_metadata
events = np.array([[100, 0, 1], [150, 0, 2], [220, 0, 3], [500, 0, 1], [530, 0, 3], [560, 0, 2], [900, 0, 1]])
md, ev3, eid = mne.epochs.make_metadata(events, {"a": 1, "b": 2, "c": 3}, -0.1, 0.5, 100.0)
print("META", md["event_name"].tolist(), md["a"].tolist(), md["b"].tolist(), md["c"].tolist(), ev3.shape, eid)
md2, ev4, eid2 = mne.epochs.make_metadata(events, {"a": 1, "b": 2, "c": 3}, None, None, 100.0, row_events=["a"])
print("META2", md2["a"].tolist(), md2["b"].tolist(), md2["c"].tolist(), ev4[:, 0].tolist(), eid2)

# rERP
from mne.stats import linear_regression_raw
rng = np.random.RandomState(7)
info = mne.create_info(2, 50.0, "eeg")
dat = rng.randn(2, 1500) * 0.1
ev = np.array([[100, 0, 1], [400, 0, 2], [700, 0, 1], [1000, 0, 2], [1300, 0, 1]])
for e in ev:
    dat[:, e[0]:e[0] + 20] += np.hanning(20)[None, :] * (1 + e[2])
rawr = mne.io.RawArray(dat, info)
res = linear_regression_raw(rawr, ev, event_id={"a": 1, "b": 2}, tmin=-0.1, tmax=0.5)
print("RERP", list(res.keys()), [repr(float(res[k].data.sum())) for k in res], res["a"].data.shape, res["b"].nave, repr(float(res["b"].data[1, 10])))

# fmri gradient
from mne.preprocessing import remove_fmri_gradient_artifact
rng = np.random.RandomState(9)
info = mne.create_info(3, 500.0, "eeg")
d = rng.randn(3, 5000) * 0.1
tr_len = 400
tr = np.arange(0, 4800, tr_len)
for s in tr:
    d[:, s:s + tr_len] += 5 * np.sin(2 * np.pi * np.arange(tr_len) / 40)[None, :] + np.linspace(0, 3, tr_len)[None, :]
rf = mne.io.RawArray(d, info)
rc = remove_fmri_gradient_artifact(rf, tr, window=(2, 2))
dc = rc.get_data()
print("FMRI", repr(float(np.abs(dc).sum())), repr(float(dc[1, 1000])), repr(float(dc[0, 4700])))

# realign
from mne.preprocessing import realign_raw
rng = np.random.RandomState(11)
sfa = 100.0
a = mne.io.RawArray(rng.randn(1, 6000), mne.create_info(1, sfa, "eeg"))
b = mne.io.RawArray(rng.randn(1, 6200), mne.create_info(1, sfa, "eeg"))
t_a = np.linspace(2.0, 50.0, 25)
t_b = (t_a - 0.5) / 1.0005 + 0.0
t_b = t_b + np.random.RandomState(1).randn(25) * 0.0005
realign_raw(a, b, t_a, t_b)
print("REALIGN", a.n_times, b.n_times, repr(float(a.get_data().sum())), repr(float(b.get_data().sum())), b.info["sfreq"])

# css
from mne.preprocessing import cortical_signal_suppression
rng = np.random.RandomState(13)
info = mne.create_info(["m1", "m2", "g1", "g2", "e1", "e2"], 100.0, ["mag", "mag", "grad", "grad", "eeg", "eeg"])
ed = rng.randn(6, 200)
ev_ = mne.EvokedArray(ed, info)
out = cortical_signal_suppression(ev_, n_proj=2)
print("CSS", repr(float(np.abs(out.data).sum())), repr(float(out.data[4, 10])), repr(float(out.data[0, 10])))

# interpolate bridged
from mne.preprocessing import interpolate_bridged_electrodes
mont = mne.channels.make_standard_montage("standard_1020")
chs = ["Fz", "Cz", "Pz", "C3", "C4", "F3", "F4", "P3", "P4", "Oz"]
info = mne.create_info(chs, 100.0, "eeg")
info.set_montage(mont)
rng = np.random.RandomState(15)
rb = mne.io.RawArray(rng.randn(10, 300), info)
rb2 = interpolate_bridged_electrodes(rb.copy(), [(3, 5)])
print("BRIDGE", repr(float(np.abs(rb2.get_data()).sum())), repr(float(rb2.get_data()[3, 10])), repr(float(rb2.get_data()[5, 20])))

# pca obs
from mne.preprocessing import apply_pca_obs
rng = np.random.RandomState(17)
n = 4000
sfp = 200.0
qrs = np.arange(100, n - 100, 160)
tmpl = np.exp(-0.5 * ((np.arange(-30, 31)) / 6.0) ** 2) * 5
d = rng.randn(2, n) * 0.3
for q in qrs:
    d[:, q - 30:q + 31] += tmpl[None, :] * (1 + 0.1 * rng.randn(2, 1))
rp = mne.io.RawArray(d, mne.create_info(2, sfp, "eeg"))
rpo = apply_pca_obs(rp, picks=["0"] if False else [0, 1], qrs_times=qrs / sfp, n_components=2)
print("PCAOBS", repr(float(np.abs(rpo.get_data()).sum())), repr(float(rpo.get_data()[0, 500])), repr(float(rpo.get_data()[1, 3000])))

# fine cal io
import tempfile, os
from mne.preprocessing import read_fine_calibration, write_fine_calibration
txt = "# comment\n" + "\n".join(f"{n_} " + " ".join(f"{v:.6f}" for v in np.arange(12) * 0.01 + i) + " 1.012300" for i, n_ in enumerate(["113", "2642", "1"])) + "\n"
open("/tmp/typhon_fc.dat", "w").write(txt)
cal = read_fine_calibration("/tmp/typhon_fc.dat")
print("FINECAL", cal["ch_names"], cal["locs"].shape, repr(float(cal["locs"].sum())), [list(map(float, c)) for c in cal["imb_cals"]])

# ICA save / read + corrmap
from mne.preprocessing import ICA, corrmap, read_ica
rng = np.random.RandomState(0)
info = mne.create_info(5, 100.0, "eeg")
rawi = mne.io.RawArray(rng.randn(5, 3000), info)
ica = ICA(n_components=4, random_state=0, max_iter=200, method="infomax")
ica.fit(rawi)
ica.exclude = [1]
print("ICA", repr(float(np.abs(ica.unmixing_matrix_).sum())), repr(float(np.abs(ica.pca_components_).sum())), ica.n_components_, ica.n_iter_)
ica2 = ICA(n_components=4, random_state=1, max_iter=200, method="infomax")
ica2.fit(rawi)
icas = [ica, ica2]
corrmap(icas, template=(0, 0), threshold=0.2, label="blink", plot=False)
print("CORRMAP", ica.labels_, ica2.labels_)
