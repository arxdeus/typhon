# Generates test_ops4.py (merge indicator, value_counts bins, to_csv quoting, concat keys,
# rolling skew/kurt/sem/first/last) from real pandas 3.1.1.
import numpy as np, pandas as pd, os
D = os.path.dirname(__file__)
src = open(os.path.join(D, "test_ops3.py")).read()
head = src.split("nan = float")[0]
L = []; T = []
def fl(a): return repr([float(x) for x in np.asarray(a, dtype=float).ravel()])
def addf(name, arr, call, tol=1e-9):
    L.append(f"E_{name} = {fl(arr)}"); T.append(f"    assert close({call}, E_{name}, {tol})")
def adds(name, arr, call):
    L.append(f"E_{name} = {[str(x) for x in arr]!r}"); T.append(f"    assert {call} == E_{name}")
G = [float((i * 19 + 5) % 17 - 6) * 0.53 + float(i % 4) for i in range(30)]
H = [float((i * 7 + 3) % 13 - 4) * 0.71 for i in range(30)]
G[4] = float("nan")
s = pd.Series(G); h = pd.Series(H)
for w in (4,):
    addf("r_skew", s.rolling(w).skew(), f"fvals(s.rolling({w}).skew())")
    addf("r_kurt", s.rolling(5).kurt(), "fvals(s.rolling(5).kurt())")
    addf("r_sem", s.rolling(w, min_periods=2).sem(), f"fvals(s.rolling({w}, 2).sem())")
    addf("r_first", s.rolling(w, min_periods=2).first(), f"fvals(s.rolling({w}, 2).first())")
    addf("r_last", s.rolling(w, min_periods=2).last(), f"fvals(s.rolling({w}, 2).last())")
vc = s.value_counts(bins=4)
adds("vcb_i", vc.index.astype(str), "[str(x) for x in s.value_counts(False, True, False, True, 4).index.tolist()]")
addf("vcb", vc.values, "vals(s.value_counts(False, True, False, True, 4))")
vc = s.value_counts(bins=3, normalize=True)
addf("vcbn", vc.values, "vals(s.value_counts(True, True, False, True, 3))")
a = pd.DataFrame({"k": [1, 2, 3, 4], "x": [1.5, 2.5, 3.5, 4.5]})
b = pd.DataFrame({"k": [3, 4, 5], "y": [30.0, 40.0, 50.0]})
for how in ("left", "outer", "inner"):
    m = a.merge(b, how=how, on="k", indicator=True)
    adds(f"mg_{how}", m["_merge"].astype(str), f'strs(pd.merge(a, b, "{how}", "k", None, None, ("_x", "_y"), True)["_merge"])')
    addf(f"mgk_{how}", m["k"], f'vals(pd.merge(a, b, "{how}", "k", None, None, ("_x", "_y"), True)["k"])')
c1 = pd.DataFrame({"u": [1.0, 2.0], "v": [3.0, 4.0]}); c2 = pd.DataFrame({"u": [5.0, 6.0]})
cc = pd.concat([c1, c2], axis=1, keys=["A", "B"])
addf("ck_cols", cc.to_numpy(), "fvals(pd.concat([c1, c2], 1, False, ['A', 'B']))")
adds("ck_names", [f"{x}|{y}" for x, y in cc.columns], "[str(x) for x in pd.concat([c1, c2], 1, False, ['A', 'B']).columns.tolist()]") if False else None
cr = pd.concat([c1, c1 * 2], keys=["p", "q"])
addf("ck_rows", cr.to_numpy(), "fvals(pd.concat([c1, c1 * 2], 0, False, ['p', 'q']))")
csv = pd.DataFrame({"a": ["x,y", 'q"r', "z"], "b": [1.5, float("nan"), 3.0]})
L.append(f"E_csv_all = {csv.to_csv(quoting=1)!r}")
T.append('    assert csvf.to_csv(None, ",", "", None, None, True, True, None, "\\n", ".", "w", 1) == E_csv_all')
L.append(f"E_csv_hdr = {csv.to_csv(index=False, header=['A','B'], na_rep='NA')!r}")
T.append('    assert csvf.to_csv(None, ",", "NA", None, None, True, False, None, "\\n", ".", "w", 0, ["A", "B"]) == E_csv_hdr')
setup = '''G = List[float]([float((i * 19 + 5) % 17 - 6) * 0.53 + float(i % 4) for i in range(30)])
G[4] = float("nan")
s = pd.Series(G)
a = pd.DataFrame({"k": pd.Series([1, 2, 3, 4]), "x": pd.Series([1.5, 2.5, 3.5, 4.5])})
b = pd.DataFrame({"k": pd.Series([3, 4, 5]), "y": pd.Series([30.0, 40.0, 50.0])})
c1 = pd.DataFrame({"u": pd.Series([1.0, 2.0]), "v": pd.Series([3.0, 4.0])})
c2 = pd.DataFrame({"u": pd.Series([5.0, 6.0])})
csvf = pd.DataFrame({"a": pd.Series(["x,y", 'q"r', "z"]), "b": pd.Series([1.5, float("nan"), 3.0])})

'''
open(os.path.join(D, "test_ops4.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_ops4():\n" + "\n".join(T) + "\ntest_ops4()\n")
