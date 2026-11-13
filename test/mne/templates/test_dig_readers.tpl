import numpy as np
import mne
from mne.channels import Transform

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

def scheck(name: str, got: List[str]):
    want = List[str]()
    for n, v in SEXPECTED:
        if n == name:
            want = v
    if got != want:
        print("TEST FAILED:", name, got, want)

def fid(v: Optional[np.ndarray[float, 1]]) -> List[float]:
    if v is None:
        return List[float]()
    return list(v.__val__())

def mont(n: str, m):
    g = m.get_positions()
    cp = g['ch_pos']
    allp = List[float]()
    for c in m.ch_names:
        for x in cp[c]:
            allp.append(x)
    check(n + '_pos', allp, 1e-6, 1e-12)
    check(n + '_nasion', fid(g['nasion']), 1e-6, 1e-12)
    check(n + '_lpa', fid(g['lpa']), 1e-6, 1e-12)
    check(n + '_rpa', fid(g['rpa']), 1e-6, 1e-12)
    check(n + '_n', [len(m.ch_names)])
    scheck(n + '_names', list(m.ch_names))

d = "/tmp/typhon_dig_fixtures_"
import os
os.makedirs(d, exist_ok=True)
for fn, txt in FILES:
    with open(d + "/" + fn, "w") as f:
        f.write(txt)
P = lambda k: d + "/" + k
mont('dat', mne.channels.read_dig_dat(P('a.dat')))
mont('hpts', mne.channels.read_dig_hpts(P('a.hpts')))
mont('hpts_m', mne.channels.read_dig_hpts(P('a.hpts'), unit='cm'))
mont('loc', mne.channels.read_dig_localite(P('a.csv'), nasion='Nz', lpa='LPA', rpa='RPA'))
mont('elp', mne.channels.read_dig_polhemus_isotrak(P('a.elp')))
mont('hsp', mne.channels.read_dig_polhemus_isotrak(P('b.hsp'), unit='mm'))
mont('hspn', mne.channels.read_dig_polhemus_isotrak(P('b.hsp'), ch_names=['A', 'B', 'C'], unit='mm'))
mont('cap', mne.channels.read_dig_captrak(P('a.bvct')))
mont('egi', mne.channels.read_dig_egi(P('coords.xml')))
check('fast', mne.channels.read_polhemus_fastscan(P('fast.txt'), unit='m'))
m = mne.channels.read_dig_hpts(P('a.hpts'))
t = mne.channels.compute_native_head_t(m)
check('nht', t.trans, 1e-6, 1e-9)
m2 = mne.channels.transform_to_head(m)
mont('toh', m2)
check('toh_frame', [1.0 if m2.coord_frame == 'head' else 0.0])
check('bm_n', [len(mne.channels.get_builtin_montages())])
check('adj_n', [len(mne.channels.get_builtin_ch_adjacencies())])
scheck('bm', mne.channels.get_builtin_montages())
scheck('adj', mne.channels.get_builtin_ch_adjacencies())
info = mne.create_info(['a', 'b', 'c'], 100., 'eeg')
info.bads = ['b']
mne.channels.rename_channels(info, {'a': 'x', 'b': 'y'})
scheck('ren', info.ch_names + info.bads)
mne.channels.rename_channels(info, lambda s: s.upper())
scheck('ren2', info.ch_names + info.bads)
r1 = mne.io.RawArray(np.zeros((3, 5)), mne.create_info(['a', 'b', 'c'], 100., 'eeg'))
r1.info.bads = ['b']
r2 = r1.copy()
r2.info.bads = ['c']
mne.channels.unify_bad_channels([r1, r2])
scheck('unify', r1.info.bads + ['|'] + r2.info.bads)
print("done")
