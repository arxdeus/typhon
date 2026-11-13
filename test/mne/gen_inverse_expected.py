# Generates test/mne/test_inverse.py from real mne 1.13.2 (synthetic sphere fixtures of
# gen_forward_expected.py). Run: /path/to/venv/bin/python test/mne/gen_inverse_expected.py
import os
import numpy as np
import mne
from mne.minimum_norm import (make_inverse_operator, apply_inverse, apply_inverse_epochs,
                              apply_inverse_raw, apply_inverse_cov, estimate_snr,
                              make_inverse_resolution_matrix, get_point_spread, get_cross_talk,
                              compute_rank_inverse, prepare_inverse_operator)

mne.set_log_level("ERROR")
HERE = os.path.dirname(os.path.abspath(__file__))


def lst(a):
    a = np.asarray(a, dtype=float).ravel()
    return "[" + ", ".join(repr(float(v)) for v in a) + "]"


def sph(n, R, seed, z0=0.04):
    r = np.random.RandomState(seed)
    th = r.uniform(0.2, 1.2, n)
    ph = r.uniform(0, 2 * np.pi, n)
    return np.c_[R * np.sin(th) * np.cos(ph), R * np.sin(th) * np.sin(ph), R * np.cos(th) + z0]


def make_info():
    eeg = sph(8, 0.09, 1)
    meg = sph(6, 0.12, 3)
    names = ["E%d" % i for i in range(8)] + ["M%d" % i for i in range(6)]
    types = ["eeg"] * 8 + ["mag"] * 3 + ["grad"] * 3
    info = mne.create_info(names, 100.0, types)
    for i in range(8):
        info["chs"][i]["loc"][:3] = eeg[i]
        info["chs"][i]["loc"][3:6] = 0.0
    for k in range(6):
        ch = info["chs"][8 + k]
        p = meg[k]
        ez = (p - np.array([0, 0, 0.04])) / np.linalg.norm(p - np.array([0, 0, 0.04]))
        ex = np.cross([0.0, 1.0, 0.0], ez)
        ex /= np.linalg.norm(ex)
        ey = np.cross(ez, ex)
        ch["loc"][:3] = p
        ch["loc"][3:6] = ex
        ch["loc"][6:9] = ey
        ch["loc"][9:12] = ez
    info["dev_head_t"] = mne.transforms.Transform("meg", "head", np.eye(4))
    return info


E = {}
info = make_info()
sphere = mne.make_sphere_model(r0=(0.0, 0.0, 0.04), head_radius=0.09)
sphere["mu"] = np.array([0.9432112879770956, 0.6631031404159992, 0.11394344322830123])
sphere["lambda"] = np.array([0.4271636538254471, 2.086195302728963, -0.05769349392344974])
rr = sph(5, 0.05, 2)
nn = sph(5, 1.0, 4, z0=0.0)
src = mne.setup_volume_source_space(pos=dict(rr=rr, nn=nn))
fwd = mne.make_forward_solution(info, None, src, sphere)
cov = mne.make_ad_hoc_cov(info)
info_ref = info.copy()
data = np.sin(np.arange(10)[None, :] * 0.3 + np.arange(14)[:, None]) * 1e-12
data[:8] *= 1e5
ev = mne.EvokedArray(data, info_ref, tmin=0.0, nave=4)
ev.set_eeg_reference(projection=True)
ev_p = ev.copy()
inv = make_inverse_operator(ev.info, fwd, cov, loose=1.0, depth=0.8)
E["sing"] = lst(inv["sing"])
E["src_cov"] = lst(inv["source_cov"]["data"])
E["depth"] = lst(inv["depth_prior"]["data"])
E["rank"] = lst([compute_rank_inverse(inv)])
for m in ("MNE", "dSPM", "sLORETA"):
    stc = apply_inverse(ev, inv, 1.0 / 9.0, m)
    E["stc_" + m] = lst(stc.data)
    stcv = apply_inverse(ev, inv, 1.0 / 9.0, m, pick_ori="vector")
    E["vec_" + m] = lst(stcv.data)
stc = apply_inverse(ev, inv, 0.05, "eLORETA", method_params=dict(force_equal=False))
E["elo"] = lst(stc.data)
stc = apply_inverse(ev, inv, 0.05, "eLORETA", method_params=dict(force_equal=True))
E["elo_eq"] = lst(stc.data)
inv_d = make_inverse_operator(ev.info, fwd, cov, loose=1.0, depth=None)
E["nodepth_sing"] = lst(inv_d["sing"])
s, r = apply_inverse(ev, inv, 1.0 / 9.0, "MNE", return_residual=True)
E["resid"] = lst(r.data)
snr, snr_est = estimate_snr(ev, inv)
E["snr"] = lst(snr)
E["snr_est"] = lst(snr_est)
E["covsrc"] = lst(apply_inverse_cov(mne.Covariance(np.eye(14) * 1e-24 + 1e-26, ev.info["ch_names"], [], [], 10),
                                    ev.info, inv, nave=4, method="MNE").data) if False else "[]"
epo = mne.EpochsArray(np.stack([data, 0.5 * data + 1e-13]), ev.info.copy(), tmin=0.0)
stcs = apply_inverse_epochs(epo, inv, 1.0 / 9.0, "dSPM", nave=1)
E["epo"] = lst(np.concatenate([s.data.ravel() for s in stcs]))
raw = mne.io.RawArray(data, ev.info.copy())
stcr = apply_inverse_raw(raw, inv, 1.0 / 9.0, "sLORETA", start=2, stop=8, nave=2)
E["raw"] = lst(stcr.data)
fwd_s = mne.convert_forward_solution(fwd, force_fixed=True) if False else None
res = make_inverse_resolution_matrix(fwd, inv, method="MNE", lambda2=1.0 / 9.0)
E["resmat"] = lst(res)
psf = get_point_spread(res, inv["src"], [1], mode=None)
E["psf"] = lst(psf.data)
ctf = get_cross_talk(res, inv["src"], [1, 3], mode="mean")
E["ctf"] = lst(ctf.data)

txt = open(os.path.join(HERE, "templates", "test_inverse.tpl")).read()
for k, v in E.items():
    txt = txt.replace("@@" + k + "@@", v)
assert "@@" not in txt, txt[txt.index("@@"):txt.index("@@") + 40]
open(os.path.join(HERE, "test_inverse.py"), "w").write(txt)
print("wrote test_inverse.py")
