#DPDEF
def kindunit(name, d):
    out = List[float]()
    for k in ["kind", "unit", "coil_type"]:
        out.append(float(d[k]) if k in d else -99.0)
    check(name, out)

c = mne.io.get_channel_type_constants()
checks("ct_keys", list(c.keys()))
for t in ["grad", "eeg", "eog", "stim", "misc", "csd", "hbo", "fnirs_od", "pupil", "temperature"]:
    kindunit("ct_" + t, c[t])
c = mne.io.get_channel_type_constants(include_defaults=True)
for t in ["grad", "mag", "ref_meg", "eog", "misc", "stim", "bio"]:
    kindunit("ctd_" + t, c[t])

info = mne.create_info(["a", "b", "c"], 250., ["eeg", "eog", "eeg"])
info["bads"] = ["b"]
SETX(info)
mne.io.write_info("build/x-info.fif", info)
i2 = mne.io.read_info("build/x-info.fif")
checks("wi_names", i2["ch_names"])
checks("wi_bads", i2["bads"])
checks("wi_types", i2.get_channel_types())
check("wi_vals", [i2["sfreq"], i2["lowpass"], i2["highpass"]])
checks("wi_text", GETX(i2))

a = mne.create_info(["a"], 100., "eeg")
SETX(a)
b = mne.io.anonymize_info(a)
checks("anon_text", GETX(b))

pts = [DP(1, 1, [-0.08, 0.0, 0.0]),
       DP(1, 2, [0.0, 0.1, 0.0]),
       DP(1, 3, [0.08, 0.0, 0.0]),
       DP(3, 1, [0.01, 0.05, 0.07])]
mne.io.write_fiducials("build/f-fiducials.fif", pts, "head")
r, fr = mne.io.read_fiducials("build/f-fiducials.fif")
check("fid_n", [len(r), fr])
check("fid_kind", [p["kind"] for p in r])
check("fid_ident", [p["ident"] for p in r])
check("fid_r", [p["r"] for p in r], 1e-6, 1e-9)
print("done")
