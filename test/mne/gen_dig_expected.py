# Generates test/mne/test_dig_readers.py (expected values from mne 1.13.2). Fixture files are written by the test.
import os, tempfile, warnings
import numpy as np
import mne
warnings.simplefilter("ignore")
mne.set_log_level("ERROR")
d = tempfile.mkdtemp()
files = {}
files['a.dat'] = "Fp1 1 -3.0 8.0 1.0\nFp2 1 3.0 8.0 1.0\nNas 78 0.0 10.0 0.0\nLPA 76 -8.0 0.0 0.0\nRPA 82 8.0 0.0 0.0\ncen 67 0 0 0\nCz 1 0.0 0.0 9.0\n"
files['a.hpts'] = "# comment\ncardinal 2 0.0 100.0 0.0\ncardinal 1 -80.0 0.0 0.0\ncardinal 3 80.0 0.0 0.0\nhpi 1 -30.4 -11.8 83.3\neeg Fp1 -30.0 80.0 10.0\neeg Cz 0.0 0.0 90.0\nextra 4 1 2 3\n"
files['a.csv'] = "Idx,Name,X,Y,Z\n1,Nz,0,100,0\n2,LPA,-80,0,0\n3,RPA,80,0,0\n4,Cz,0,0,95\n5,Pz,0,-50,80\n"
files['a.elp'] = "%N\tName\n%F\t0.0\t0.1\t0.0\n%F\t-0.08\t0.0\t0.0\n%F\t0.08\t0.0\t0.0\n//\t0.03\t0.05\t0.04\n0.01\t0.02\t0.03\n"
files['a.hsp'] = "% Probe\n%F\t0.0\t0.1\t0.0\n"
files['b.hsp'] = "//Position of fiducials X+, Y+, Z+ (mm) //\n%F\t0.0\t100.0\t0.0\n%F\t-80.0\t0.0\t0.0\n%F\t80.0\t0.0\t0.0\n//No of rows, No of columns; position of digitized points //\n3\t3\n1.0\t2.0\t3.0\n4.0\t5.0\t6.0\n7.0\t8.0\t9.0\n"
files['a.bvct'] = "<?xml version=\"1.0\"?>\n<CapTrakData><CapTrakElectrodeList><CapTrakElectrode><Name>Nasion</Name><X>0</X><Y>100</Y><Z>0</Z></CapTrakElectrode><CapTrakElectrode><Name>LPA</Name><X>-80</X><Y>0</Y><Z>0</Z></CapTrakElectrode><CapTrakElectrode><Name>RPA</Name><X>80</X><Y>0</Y><Z>0</Z></CapTrakElectrode><CapTrakElectrode><Name>Fz</Name><X>0</X><Y>60</Y><Z>70</Z></CapTrakElectrode><CapTrakElectrode><Name>Cz</Name><X>0</X><Y>0</Y><Z>95</Z></CapTrakElectrode></CapTrakElectrodeList></CapTrakData>\n"
files['coords.xml'] = ("<?xml version=\"1.0\"?>\n<coordinates xmlns=\"http://www.egi.com/coordinates_mff\"><sensorLayout><sensors>"
  "<sensor><name></name><number>1</number><type>0</type><x>1.0</x><y>2.0</y><z>3.0</z></sensor>"
  "<sensor><name></name><number>2</number><type>0</type><x>4.0</x><y>5.0</y><z>6.0</z></sensor>"
  "<sensor><name>VREF</name><number>129</number><type>1</type><x>7.0</x><y>8.0</y><z>9.0</z></sensor>"
  "<sensor><name>Nasion</name><number>130</number><type>2</type><x>0.0</x><y>10.0</y><z>0.0</z></sensor>"
  "<sensor><name>Left periauricular point</name><number>131</number><type>2</type><x>-8.0</x><y>0.0</y><z>0.0</z></sensor>"
  "<sensor><name>Right periauricular point</name><number>132</number><type>2</type><x>8.0</x><y>0.0</y><z>0.0</z></sensor>"
  "</sensors></sensorLayout></coordinates>\n")
files['fast.txt'] = "% FastSCAN header\n% more\n1.0 2.0 3.0\n4.0 5.0 6.0\n"
for k, v in files.items():
    open(os.path.join(d, k), 'w').write(v)
P = lambda k: os.path.join(d, k)
out = []
def rec(n, v): out.append((n, [float(x) for x in np.asarray(v, dtype=float).ravel()]))
def mont(n, m):
    g = m.get_positions()
    rec(n + '_pos', [g['ch_pos'][c] for c in m.ch_names])
    for f in ('nasion', 'lpa', 'rpa'):
        rec(n + '_' + f, g[f] if g[f] is not None else [])
    rec(n + '_n', [len(m.ch_names)])
    names.append((n + '_names', list(m.ch_names)))
names = []
mont('dat', mne.channels.read_dig_dat(P('a.dat')))
mont('hpts', mne.channels.read_dig_hpts(P('a.hpts')))
mont('hpts_m', mne.channels.read_dig_hpts(P('a.hpts'), unit='cm'))
mont('loc', mne.channels.read_dig_localite(P('a.csv'), nasion='Nz', lpa='LPA', rpa='RPA'))
mont('elp', mne.channels.read_dig_polhemus_isotrak(P('a.elp')))
mont('hsp', mne.channels.read_dig_polhemus_isotrak(P('b.hsp'), unit='mm'))
mont('hspn', mne.channels.read_dig_polhemus_isotrak(P('b.hsp'), ch_names=['A', 'B', 'C'], unit='mm'))
mont('cap', mne.channels.read_dig_captrak(P('a.bvct')))
mont('egi', mne.channels.read_dig_egi(P('coords.xml')))
rec('fast', mne.channels.read_polhemus_fastscan(P('fast.txt'), unit='m'))
m = mne.channels.read_dig_hpts(P('a.hpts'))
t = mne.channels.compute_native_head_t(m)
rec('nht', t['trans']); 
m2 = mne.channels.transform_to_head(m)
mont('toh', m2)
rec('toh_frame', [float(m2.get_positions()['coord_frame'] == 'head')])
rec('bm_n', [len(mne.channels.get_builtin_montages())]); rec('adj_n', [len(mne.channels.get_builtin_ch_adjacencies())])
snames = [('bm', mne.channels.get_builtin_montages()), ('adj', mne.channels.get_builtin_ch_adjacencies())]
# rename/unify
info = mne.create_info(['a', 'b', 'c'], 100., 'eeg'); info['bads'] = ['b']
mne.channels.rename_channels(info, {'a': 'x', 'b': 'y'})
names.append(('ren', info['ch_names'] + info['bads']))
mne.channels.rename_channels(info, lambda s: s.upper())
names.append(('ren2', info['ch_names'] + info['bads']))
r1 = mne.io.RawArray(np.zeros((3, 5)), mne.create_info(['a', 'b', 'c'], 100., 'eeg')); r1.info['bads'] = ['b']
r2 = r1.copy(); r2.info['bads'] = ['c']
mne.channels.unify_bad_channels([r1, r2])
names.append(('unify', r1.info['bads'] + ['|'] + r2.info['bads']))

src = open('test/mne/templates/test_dig_readers.tpl').read()
files_s = "FILES = [\n" + "".join("    (%r, %r),\n" % kv for kv in files.items()) + "]\n"
exp = files_s + "EXPECTED = [\n" + "".join("    (%r, %r),\n" % (n, v) for n, v in out) + "]\nSEXPECTED = [\n" + "".join("    (%r, %r),\n" % (n, v) for n, v in names + snames) + "]\n"
open('test/mne/test_dig_readers.py', 'w').write("# Generated by gen_dig_expected.py from templates/test_dig_readers.tpl (mne 1.13.2)\n" + src.replace("#EXPECTED#", exp))
