# Generates EXPECTED for test/mne/test_channels.py (mne 1.13.2 reference).
import warnings
import numpy as np
import mne
warnings.simplefilter("ignore")
mne.set_log_level("ERROR")

def L(v):
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]

out = []
def rec(name, v):
    out.append((name, L(v)))

NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8',
         'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
for kind in ['standard_1020', 'standard_1005', 'biosemi64', 'biosemi128', 'easycap-M1']:
    m = mne.channels.make_standard_montage(kind)
    pos = m.get_positions()
    first = m.ch_names[:3]
    rec(kind + "_n", [len(m.ch_names)])
    rec(kind + "_pos", [pos['ch_pos'][c] for c in first])
    rec(kind + "_fid", [pos['nasion'], pos['lpa'], pos['rpa']])

rng = np.random.RandomState(0)
data = rng.randn(len(NAMES), 50) * 1e-5
info = mne.create_info(NAMES, 100., 'eeg')
raw = mne.io.RawArray(data, info)
raw.set_montage('standard_1020')
locs = np.array([raw.info['chs'][i]['loc'][:3] for i in range(len(NAMES))])
rec("head_locs", locs)
gm = raw.get_montage().get_positions()
rec("get_montage", [gm['ch_pos'][c] for c in ['Cz', 'O2']] + [gm['nasion'], gm['lpa']])
raw.info['bads'] = ['C3', 'Pz']
r1 = raw.copy().interpolate_bads(origin=(0., 0., 0.04))
rec("interp_fixed", r1.get_data(picks=['C3', 'Pz'])[:, :6])
rec("interp_bads_after", [len(r1.info['bads'])])
r2 = raw.copy().interpolate_bads(origin='auto')
rec("interp_auto", r2.get_data(picks=['C3', 'Pz'])[:, :6])
adj, names = mne.channels.find_ch_adjacency(raw.info, 'eeg')
rec("adj", adj.toarray())
dm = mne.channels.make_dig_montage(ch_pos={'A': [0., 0.08, 0.02], 'B': [0.05, 0., 0.06], 'C': [-0.05, 0., 0.06], 'D': [0., -0.08, 0.03]},
                                   nasion=[0., 0.1, 0.], lpa=[-0.08, 0., 0.], rpa=[0.08, 0., 0.], coord_frame='head')
rec("dig", [dm.get_positions()['ch_pos'][c] for c in 'ABCD'])
raw2 = mne.io.RawArray(data[:5], mne.create_info(['Fz', 'Cz', 'X1', 'Fp1', 'Pz'], 100., 'eeg'))
a, b = mne.channels.equalize_channels([raw.copy(), raw2])
rec("eq_n", [len(a.ch_names), len(b.ch_names)])
rec("eq_data", b.get_data()[:, :3])
print("EXPECTED = [")
for n, v in out:
    print(f"    ({n!r}, {v!r}),")
print("]")
