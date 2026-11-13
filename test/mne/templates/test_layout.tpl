import numpy as np
import mne

#EXPECTED#

def check(name: str, got, rtol: float = 1e-9, atol: float = 1e-12):
    want = List[float]()
    for n, v in EXPECTED:
        if n == name:
            want = v
    g = np.asarray(got, dtype=float).ravel()
    ok = g.shape[0] == len(want)
    if ok:
        for i in range(len(want)):
            if abs(g[i] - want[i]) > atol + rtol * abs(want[i]):
                ok = False
    if not ok:
        print("TEST FAILED:", name, g, want)

NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 'C4', 'T8',
         'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'O2']
for k in ['Vectorview-all', 'Vectorview-mag', 'EEG1005', 'CTF-275', 'KIT-157']:
    l = mne.channels.read_layout(k)
    check('rl_' + k, l.pos[:3])
    check('rl_n_' + k, [len(l.names)])
    check('rl_ids_' + k, l.ids[:3])
    check('rlbox_' + k, [l.box[0], l.box[1], l.box[2], l.box[3]])
check('rl_noscale', mne.channels.read_layout('Vectorview-all', scale=False).pos[:3])
info = mne.create_info(NAMES, 100., 'eeg')
info.set_montage('standard_1020')
l = mne.channels.make_eeg_layout(info)
check('eeg_pos', l.pos, 1e-7)
check('eeg_ids', l.ids)
check('eeg_pos2', mne.channels.make_eeg_layout(info, radius=0.4, width=0.1, height=0.08).pos, 1e-7)
inf2 = mne.create_info(['a', 'b', 'c', 'd', 'e', 'f', 'g'], 100., 'misc')
l = mne.channels.make_grid_layout(inf2)
check('grid_pos', l.pos)
check('grid_ids', l.ids)
check('grid_pos3', mne.channels.make_grid_layout(inf2, n_col=3).pos)
rng = np.random.RandomState(1)
xy = rng.rand(8, 2) * 10
l = mne.channels.generate_2d_layout(xy)
check('g2d_pos', l.pos)
check('g2d_box', [l.box[0], l.box[1], l.box[2], l.box[3]])
check('g2d_pos_n', mne.channels.generate_2d_layout(xy, normalize=False, name='x').pos)
check('fl_pos', mne.channels.find_layout(info).pos, 1e-7)
print("done")
