# Generates test/mne/test_io_fif.py, test_io_edf_bv_eeglab.py, test_raw_api.py and
# test_events.py, test_events_epochs.py, test_filter_picks.py from templates/*.tpl using mne 1.13.2 as the reference.
#
# Each template body is plain mne code calling check(name, value, ...) and
# checks(name, strings). Here the body is executed in CPython with recording
# versions of these helpers; the recorded values become FEXP/SEXP tables.
# Run from any cwd with the reference venv:
#   /path/to/.venv/bin/python test/mne/gen_mne_core_expected.py
import os
import sys
import tempfile
import warnings
import numpy as np
import mne

warnings.simplefilter("ignore")
mne.set_log_level("ERROR")
HERE = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(HERE)
NAMES = ["test_io_fif", "test_io_edf_bv_eeglab", "test_raw_api", "test_events",
         "test_events_epochs", "test_filter_picks", "test_proj_ref", "test_csd_annot", "test_montage_io", "test_epochs_more"]


def run(name):
    src = open(os.path.join(HERE, "templates", name + ".tpl")).read()
    head, body = src.split("@@PRELUDE@@\n", 1)
    fexp, sexp = [], []

    def check(n, got, rtol=1e-7, atol=1e-15):
        a = np.asarray(got, dtype=float).ravel()
        fexp.append((n, [float(x) for x in a]))

    def checks(n, got):
        sexp.append((n, [str(v) for v in got]))

    cwd = os.getcwd()
    tmp = tempfile.mkdtemp()
    os.mkdir(os.path.join(tmp, "build"))
    os.symlink(TEST_DIR, os.path.join(tmp, "test"))
    os.chdir(tmp)
    try:
        g = {"np": np, "mne": mne, "check": check, "checks": checks, "V": lambda x: x,
             "List": type("L", (), {"__class_getitem__": classmethod(lambda c, k: list)})}
        exec(compile(body, name + ".tpl", "exec"), g)
    finally:
        os.chdir(cwd)
    names = [n for n, _ in fexp] + [n for n, _ in sexp]
    assert len(names) == len(set(names)), "duplicate check names in " + name
    pre = open(os.path.join(HERE, "templates", "core_prelude.tpl")).read()
    def fl(v):
        return "[" + ", ".join(repr(x) if np.isfinite(x) else f"float('{x}')" for x in v) + "]"

    pre = pre.replace("@@FEXP@@", "\n".join(f"    ({n!r}, {fl(v)})," for n, v in fexp) or
                      "    ('_', [0.0]),")
    pre = pre.replace("@@SEXP@@", "\n".join(f"    ({n!r}, {v!r})," for n, v in sexp) or
                      "    ('_', ['']),")
    out = head + pre + body
    open(os.path.join(HERE, name + ".py"), "w").write(out)
    print("wrote", name + ".py", len(fexp), "numeric +", len(sexp), "string checks")


for nm in (sys.argv[1:] or NAMES):
    run(nm)
