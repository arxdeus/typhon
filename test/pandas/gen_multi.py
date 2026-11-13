# Generates test_multi.py (MultiIndex, multi-key groupby, stack/unstack, set_index list) from pandas 3.1.1.
import numpy as np, pandas as pd, os
D = os.path.dirname(__file__)
head = open(os.path.join(D, "test_ops3.py")).read().split("G = List[float]")[0]
head += '''def lv(i: pd.Index, j: int) -> List[str]:
    return [str(x) for x in i.get_level_values(j).tolist()]

'''
gg = ["aaa", "b", "aaa", "b", "aaa", "c", "b", "c"]
hh = [1, 1, 22, 2, 1, 1, 1, 2]
kk = ["x", "y", "x", "y", "z", "z", "y", "x"]
vv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5, -8.0]
d = pd.DataFrame({"gg": gg, "h": hh, "k": kk, "v": vv, "w": [x * 2 for x in vv]})
L = []; T = []
def f(a): return np.asarray(a, dtype=float).ravel().tolist()
def add(name, r, call, kind="s"):
    L.append(f"E_{name} = {f(r.to_numpy() if hasattr(r, 'to_numpy') else r)!r}")
    wrap = {"s": f"vals({call})", "f": f"fvals({call})", "x": f"[{call}]"}[kind]
    T.append(f"    assert close({wrap}, E_{name})")
def addl(name, labels, call):
    L.append(f"E_{name} = {[str(x) for x in labels]!r}")
    T.append(f"    assert {call} == E_{name}")
g2 = d.groupby(["gg", "h"])
for m in ["sum", "mean", "min", "max", "count"]:
    r = getattr(g2[["v", "w"]], m)()
    add("g2_" + m, r, f'd.groupby(["gg", "h"])[["v", "w"]].{m}()', "f")
addl("g2_l0", g2[["v", "w"]].sum().index.get_level_values(0), 'lv(d.groupby(["gg", "h"])[["v", "w"]].sum().index, 0)')
addl("g2_l1", g2[["v", "w"]].sum().index.get_level_values(1), 'lv(d.groupby(["gg", "h"])[["v", "w"]].sum().index, 1)')
add("g2_s", g2["v"].sum(), 'd.groupby(["gg", "h"])["v"].sum()')
add("g2_size", g2.size(), 'd.groupby(["gg", "h"]).size()')
add("g2_std", g2["v"].std(), 'd.groupby(["gg", "h"])["v"].std()')
g3 = d.groupby(["gg", "h", "k"])
add("g3_s", g3["w"].sum(), 'd.groupby(["gg", "h", "k"])["w"].sum()')
addl("g3_l2", g3["w"].sum().index.get_level_values(2), 'lv(d.groupby(["gg", "h", "k"])["w"].sum().index, 2)')
u = g2["v"].sum().unstack()
add("unstack", u, 'd.groupby(["gg", "h"])["v"].sum().unstack()', "f")
addl("unstack_c", u.columns, '[str(x) for x in d.groupby(["gg", "h"])["v"].sum().unstack().columns.tolist()]')
addl("unstack_i", u.index, '[str(x) for x in d.groupby(["gg", "h"])["v"].sum().unstack().index.tolist()]')
rs = g2["v"].sum().reset_index()
add("reset_v", rs["v"], 'd.groupby(["gg", "h"])["v"].sum().reset_index()["v"]')
addl("reset_cols", rs.columns, '[str(x) for x in d.groupby(["gg", "h"])["v"].sum().reset_index().columns.tolist()]')
si = d.set_index(["gg", "k"])
add("si_v", si["v"], 'd.set_index(["gg", "k"])["v"]')
addl("si_l1", si.index.get_level_values(1), 'lv(d.set_index(["gg", "k"]).index, 1)')
st = d[["v", "w"]].stack()
add("stack", st, 'd[["v", "w"]].stack()')
addl("stack_l1", st.index.get_level_values(1), 'lv(d[["v", "w"]].stack().index, 1)')
mi = pd.MultiIndex.from_product([[1, 2], ["a", "b", "c"]], names=["n", "l"])
addl("mi_prod0", mi.get_level_values(0), 'lv(pd.MultiIndex.from_product(([1, 2], ["a", "b", "c"]), ["n", "l"]), 0)')
addl("mi_prod1", mi.get_level_values(1), 'lv(pd.MultiIndex.from_product(([1, 2], ["a", "b", "c"]), ["n", "l"]), 1)')
mt = pd.MultiIndex.from_tuples([(1, "a"), (2, "b"), (1, "b")])
addl("mi_tup1", mt.get_level_values(1), 'lv(pd.MultiIndex.from_tuples([(1, "a"), (2, "b"), (1, "b")]), 1)')
L.append(f"E_mi_nlev = {mi.nlevels}\nE_mi_repr = {str(mi)!r}")
T.append('    assert pd.MultiIndex.from_product(([1, 2], ["a", "b", "c"]), ["n", "l"]).nlevels == E_mi_nlev')
T.append('    assert str(pd.MultiIndex.from_product(([1, 2], ["a", "b", "c"]), ["n", "l"])) == E_mi_repr')
L.append(f"E_repr2 = {str(g2[['v', 'w']].sum())!r}\nE_repr3 = {str(g3['w'].sum())!r}\nE_repr1 = {str(g2['v'].sum())!r}")
T.append('    assert str(d.groupby(["gg", "h"])[["v", "w"]].sum()) == E_repr2')
T.append('    assert str(d.groupby(["gg", "h", "k"])["w"].sum()) == E_repr3')
T.append('    assert str(d.groupby(["gg", "h"])["v"].sum()) == E_repr1')
setup = ('d = pd.DataFrame({"gg": pd.Series(' + repr(gg) + '), "h": pd.Series(' + repr(hh) + '), "k": pd.Series(' + repr(kk) + '), "v": pd.Series(' + repr(vv) + '), "w": pd.Series(' + repr([x * 2 for x in vv]) + ')})\n')
open(os.path.join(D, "test_multi.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_multi():\n" + "\n".join(T) + "\ntest_multi()\n")
