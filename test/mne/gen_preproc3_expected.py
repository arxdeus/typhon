import numpy as np, mne
mne.set_log_level('ERROR')
h = mne.filter.design_mne_c_filter(1000., 1., 40., 0.5, 5.)
print("H_SUM", repr(float(h.sum())), "H_ABS", repr(float(np.abs(h).sum())), "H_MID", repr(float(h[len(h)//2])), "H_Q", repr(float(h[len(h)//2+7])))
rng=np.random.RandomState(0)
x=rng.randn(6,2000); x[:3]+=rng.randn(1,2000)*3
info=mne.create_info(6,100.,'eeg'); info['bads']=['4']
raw=mne.io.RawArray(x,info)
o=mne.preprocessing.oversampled_temporal_projection(raw,duration=5.)
print("OTP_SUM", repr(float(o._data.sum())), "OTP_ABS", repr(float(np.abs(o._data).sum())), "OTP_00", repr(float(o._data[0,0])), "OTP_MID", repr(float(o._data[2,1000])))
