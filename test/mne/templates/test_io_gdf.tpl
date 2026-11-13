D = "test/mne/data/"
for v in ["1", "2"]:
    r = mne.io.read_raw_gdf(D + "test_gdf" + v + ".gdf", preload=True)
    checks("names" + v, r.ch_names)
    checks("types" + v, r.get_channel_types())
    check("dims" + v, [r.info["sfreq"], r.n_times, r.info["highpass"], r.info["lowpass"]])
    check("data" + v, r.get_data()[:, 40:44], 1e-9, 1e-15)
    check("sum" + v, [float(r.get_data()[k].sum()) for k in range(4)], 1e-9, 1e-12)
    check("annot" + v, [r.annotations.onset[0], r.annotations.onset[1], r.annotations.duration[0], r.annotations.duration[1]])
    checks("desc" + v, r.annotations.description)
    r2 = mne.io.read_raw_gdf(D + "test_gdf" + v + ".gdf", exclude=["Fp2"], eog=["Cz"], preload=True)
    checks("names_ex" + v, r2.ch_names)
    checks("types_ex" + v, r2.get_channel_types())
print("done")
