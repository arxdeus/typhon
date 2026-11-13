"""Writes data.bin (X f64 60x6x500, y i64) and ref.bin (labels i64 60, proba f64 60x3, feats f64 60x12).
Run with the reference venv: python gen_ref.py <outdir>"""
import sys, numpy as np
sys.path.insert(0, '/Users/mind/orca/codon/typhon/test/bci')
from gen_logpower_lda import make_data, build
out = sys.argv[1]
X, y = make_data(60); y = y.astype(np.int64)
m = build(); m.fit(X, y)
np.ascontiguousarray(X).tofile(out + '/X.bin'); y.tofile(out + '/y.bin')
m.predict(X).astype(np.int64).tofile(out + '/ref_labels.bin')
m.predict_proba(X).astype(np.float64).tofile(out + '/ref_proba.bin')
m[:-1].transform(X).astype(np.float64).tofile(out + '/ref_feat.bin')
