# Generates test_stats2.py (skew/kurt/mode/quantile interp/interpolate/cut/qcut/melt/pivot/get_dummies) from pandas 3.1.1.
import numpy as np, pandas as pd, os
D = os.path.dirname(__file__)
def lit(a): return repr(a)
src = open(os.path.join(D, "test_rolling.py")).read()
head = src.split("def _r(")[0]
head += '''nan = float("nan")
inf = float("inf")

def close(got, exp: List[float], tol: float = 1e-12) -> bool:
    if len(got) != len(exp):
        print("size mismatch", len(got), len(exp))
        return False
    ok = True
    for i in range(len(exp)):
        a = got[i]
        b = exp[i]
        if a != a and b != b:
            continue
        if (a != a) != (b != b) or abs(a - b) > tol + tol * abs(b):
            print("mismatch at", i, a, b)
            ok = False
    return ok

def vals(s: pd.Series) -> List[float]:
    return [float(x) for x in s.to_numpy()]

def fvals(d: pd.DataFrame) -> List[float]:
    out = List[float]()
    a = d.to_numpy()
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out.append(float(a[i, j]))
    return out

'''
F = np.array([((i * 37 + 11) % 23 - 9) * 0.37 for i in range(40)])
F[[3, 17]] = np.nan
G = np.array([((i * 19 + 5) % 17 - 6) * 0.53 + (i % 4) for i in range(40)])
s = pd.Series(F); d = pd.DataFrame({"a": F, "b": G})
L = []; T = []
def add(name, v, call, kind="s", tol="1e-12"):
    arr = np.asarray(v, dtype=float).ravel().tolist() if not np.isscalar(v) else [float(v)]
    L.append(f"E_{name} = {arr!r}")
    wrap = {"s": f"vals({call})", "f": f"fvals({call})", "x": f"[{call}]"}[kind]
    T.append(f"    assert close({wrap}, E_{name}, {tol})")
add("skew", s.skew(), "s.skew()", "x")
add("kurt", s.kurt(), "s.kurt()", "x")
add("skew_ns", s.skew(skipna=False), "s.skew(skipna=False)", "x")
add("dskew", d.skew(), "d.skew()", "s")
add("dkurt", d.kurt(), "d.kurt()", "s")
for ip in ("linear", "lower", "higher", "midpoint", "nearest"):
    for q in (0.0, 0.33, 0.5, 0.875, 1.0):
        add(f"q{ip}{int(q*1000)}", s.quantile(q, interpolation=ip), f's.quantile({q}, interpolation="{ip}")', "x")
add("dq", d.quantile(0.37, interpolation="midpoint"), 'd.quantile(0.37, "midpoint")', "s")
m = pd.Series([3.0, 1, 3, 2, 2, np.nan, 1])
add("mode", m.mode(), "m.mode()", "s")
add("interp", s.interpolate(), "s.interpolate()", "s")
add("interp_both", s.interpolate(limit_direction="both"), 's.interpolate(limit_direction="both")', "s")
t = pd.Series([np.nan, 1.0, np.nan, np.nan, np.nan, 4.0, np.nan])
add("interp_lim", t.interpolate(limit=2), "t.interpolate(limit=2)", "s")
add("interp_idx", pd.Series([1.0, np.nan, np.nan, 7.0], index=[0, 1, 3, 4]).interpolate(method="index"), 'pd.Series([1.0, nan, nan, 7.0], index=[0, 1, 3, 4]).interpolate(method="index")', "s")
for nm, call in (("cut3", "pd.cut(g, 3)"), ("cutb", "pd.cut(g, [-3.0, 0.0, 2.5, 8.0])"), ("cutl", "pd.cut(g, [-3.0, 0.0, 2.5, 8.0], False)"),
                 ("cutin", "pd.cut(g, [-2.65, 0.0, 2.5, 8.0], True, None, False, 3, True)")):
    pass
g = pd.Series(G)
codes = {
 "cut3": (pd.cut(g, 3), "pd.cut(g, 3)"),
 "cutb": (pd.cut(g, [-3.0, 0.0, 2.5, 8.0]), "pd.cut(g, [-3.0, 0.0, 2.5, 8.0])"),
 "cutl": (pd.cut(g, [-3.0, 0.0, 2.5, 8.0], right=False), "pd.cut(g, [-3.0, 0.0, 2.5, 8.0], right=False)"),
 "cutil": (pd.cut(g, [G.min(), 0.0, 2.5, G.max()], include_lowest=True), f"pd.cut(g, [{G.min()!r}, 0.0, 2.5, {G.max()!r}], include_lowest=True)"),
 "qcut4": (pd.qcut(g, 4), "pd.qcut(g, 4)"),
 "qcutl": (pd.qcut(g, [0, 0.1, 0.5, 0.9, 1.0]), "pd.qcut(g, [0.0, 0.1, 0.5, 0.9, 1.0])"),
}
for k, (r, call) in codes.items():
    labs = [str(x) if x == x else "\x00<NA>" for x in r.astype(object)]
    L.append(f"E_{k} = {labs!r}")
    T.append(f"    assert [str(x) for x in ({call}).tolist()] == [x if x != \"\\x00<NA>\" else \"nan\" for x in E_{k}]")
cc = pd.cut(g, 3, labels=False)
L.append(f"E_cutcodes = {[float(x) for x in cc]!r}")
T.append("    assert close(vals(pd.cut(g, 3, labels=False)), E_cutcodes)")
mlt = d.melt()
L.append(f"E_melt = {mlt['value'].tolist()!r}")
T.append('    assert close([float(x) for x in d.melt()["value"].tolist()], E_melt)')
pv = pd.pivot_table(pd.DataFrame({"r": [i % 3 for i in range(40)], "c": [i % 4 for i in range(40)], "v": G}), values="v", index="r", columns="c", aggfunc="mean")
L.append(f"E_pivot = {pv.to_numpy().ravel().tolist()!r}")
T.append('    assert close(fvals(pv_in.pivot_table("v", "r", "c")), E_pivot)')
gd = pd.get_dummies(pd.Series(["b", "a", "c", "a", None, "b"], name="k"), dummy_na=True)
L.append(f"E_dum = {gd.to_numpy().astype(float).ravel().tolist()!r}\nE_dumc = {[str(c) for c in gd.columns]!r}")
T.append('    dd = pd.get_dummies(pd.Series(["b", "a", "c", "a", None, "b"], name="k"), dummy_na=True)\n    assert [str(c) for c in dd.columns.tolist()] == E_dumc\n    assert close(fvals(dd), E_dum)')
setup = ('F = List[float]([float((i * 37 + 11) % 23 - 9) * 0.37 for i in range(40)])\nG = List[float]([float((i * 19 + 5) % 17 - 6) * 0.53 + float(i % 4) for i in range(40)])\n'
 'F[3] = nan\nF[17] = nan\ns = pd.Series(F)\ng = pd.Series(G)\nd = pd.DataFrame({"a": pd.Series(F), "b": pd.Series(G)})\n'
 'm = pd.Series([3.0, 1.0, 3.0, 2.0, 2.0, nan, 1.0])\nt = pd.Series([nan, 1.0, nan, nan, nan, 4.0, nan])\n'
 'pv_in = pd.DataFrame({"r": pd.Series([i % 3 for i in range(40)]), "c": pd.Series([i % 4 for i in range(40)]), "v": pd.Series(G)})\n')
open(os.path.join(D, "test_stats2.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_stats2():\n" + "\n".join(T) + "\ntest_stats2()\n")
