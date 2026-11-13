# Generates test/mne/test_forward_sphere.py from templates/test_forward_sphere.tpl
# using reference values from real mne 1.13.2 (synthetic sensors, sphere model).
# Run: /path/to/venv/bin/python test/mne/gen_forward_expected.py
import os
import numpy as np
import mne

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
E["mu"] = lst(sphere["mu"])
E["lam"] = lst(sphere["lambda"])
sphere2 = mne.make_sphere_model(r0=(0.0, 0.01, 0.03), head_radius=0.1,
                                relative_radii=(0.88, 0.95, 1.0), sigmas=(0.3, 0.006, 0.3))
E["mu2"] = lst(sphere2["mu"])
E["lam2"] = lst(sphere2["lambda"])
rr = sph(5, 0.05, 2)
nn = sph(5, 1.0, 4, z0=0.0)
src = mne.setup_volume_source_space(pos=dict(rr=rr, nn=nn))
fwd = mne.make_forward_solution(info, None, src, sphere)
E["fwd_free"] = lst(fwd["sol"]["data"])
E["fwd_names"] = repr(fwd["sol"]["row_names"])
f2 = mne.convert_forward_solution(fwd, surf_ori=True)
E["fwd_surf"] = lst(f2["sol"]["data"])
E["fwd_surf_nn"] = lst(f2["source_nn"])
f3 = mne.convert_forward_solution(fwd, force_fixed=True)
E["fwd_fixed"] = lst(f3["sol"]["data"])
fe = mne.make_forward_solution(info, None, src, sphere, meg=False)
E["fwd_eeg"] = lst(fe["sol"]["data"])
# MEG only with a no-shell sphere
s0 = mne.make_sphere_model(r0=(0.0, 0.0, 0.04), head_radius=None)
fm = mne.make_forward_solution(info, None, src, s0, eeg=False)
E["fwd_meg0"] = lst(fm["sol"]["data"])
# volume grid
vsrc = mne.setup_volume_source_space(pos=15.0, sphere=sphere, mindist=5.0)
E["vol_nuse"] = lst([vsrc[0]["nuse"]])
E["vol_vertno"] = lst(vsrc[0]["vertno"])
vsrc2 = mne.setup_volume_source_space(pos=20.0, sphere=(0.0, 0.0, 0.04, 0.08), mindist=0.0, exclude=20.0)
E["vol2_vertno"] = lst(vsrc2[0]["vertno"])
fv = mne.make_forward_solution(info, None, vsrc, sphere, mindist=8.0)
E["fv_nsource"] = lst([fv["nsource"]])
E["fv_sum"] = lst(np.abs(fv["sol"]["data"]).sum(axis=1))
# depth / orient priors
dp = mne.forward.compute_depth_prior(fwd, info, exp=0.8, limit=10.0)
E["depth_spec"] = lst(dp)
dp2 = mne.forward.compute_depth_prior(fwd, info, exp=0.5, limit_depth_chs=True, combine_xyz="fro")
E["depth_fro"] = lst(dp2)
op = mne.forward.compute_orient_prior(f2, loose=0.3)
E["orient"] = lst(op)
# apply_forward
data = np.sin(np.arange(10)[None, :] * 0.3 + np.arange(5)[:, None]) * 1e-9
stc = mne.VolSourceEstimate(data, [np.arange(5)], tmin=0.01, tstep=0.01)
ev = mne.apply_forward(fwd, stc, info)
E["apply_fwd"] = lst(ev.data)
E["apply_fwd_times"] = lst(ev.times)
vdata = np.cos(np.arange(5 * 3 * 4).reshape(5, 3, 4) * 0.2) * 1e-9
vstc = mne.VolVectorSourceEstimate(vdata, [np.arange(5)], tmin=0.0, tstep=0.01)
ev2 = mne.apply_forward(fwd, vstc, info)
E["apply_fwd_vec"] = lst(ev2.data)
raw = mne.apply_forward_raw(fwd, stc, info, start=2, stop=8)
E["apply_fwd_raw"] = lst(raw.get_data())
E["apply_fwd_raw_first"] = lst([raw.first_samp])

src_txt = open(os.path.join(HERE, "templates", "test_forward_sphere.tpl")).read()
for k, v in E.items():
    src_txt = src_txt.replace("@@" + k + "@@", v)
assert "@@" not in src_txt, src_txt[src_txt.index("@@"):src_txt.index("@@") + 40]
open(os.path.join(HERE, "test_forward_sphere.py"), "w").write(src_txt)
print("wrote test_forward_sphere.py")
