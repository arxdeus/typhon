# Generates test_dt.py (Timestamp/Timedelta/date_range/.dt/resample/merge_asof) from pandas 3.1.1.
import numpy as np, pandas as pd, os, warnings
warnings.simplefilter("ignore")
D = os.path.dirname(__file__)
head = open(os.path.join(D, "test_ops3.py")).read().split("G = List[float]")[0]
head += '''def ds(i) -> List[str]:
    return [str(x) for x in i.tolist()]

'''
L = []; T = []
def f(a): return np.asarray(a, dtype=float).ravel().tolist()
def addv(name, r, call):
    L.append(f"E_{name} = {f(r.to_numpy() if hasattr(r, 'to_numpy') else r)!r}")
    T.append(f"    assert close(vals({call}), E_{name})")
def addd(name, r, call):
    L.append(f"E_{name} = {f(r.to_numpy() if hasattr(r, 'to_numpy') else r)!r}")
    T.append(f"    assert close(fvals({call}), E_{name})")
def adds(name, labels, call):
    L.append(f"E_{name} = {[str(x) for x in labels]!r}")
    T.append(f"    assert ds({call}) == E_{name}")
for nm, kw, call in [
    ("d", dict(start="2024-01-30", periods=5, freq="D"), 'pd.date_range("2024-01-30", None, 5, "D")'),
    ("h", dict(start="2024-03-09 22:00", periods=6, freq="h"), 'pd.date_range("2024-03-09 22:00", None, 6, "h")'),
    ("t", dict(start="2024-01-01", periods=4, freq="15min"), 'pd.date_range("2024-01-01", None, 4, "15min")'),
    ("s", dict(start="2024-01-01 00:00:58", periods=4, freq="s"), 'pd.date_range("2024-01-01 00:00:58", None, 4, "s")'),
    ("ms", dict(start="2024-01-01", periods=3, freq="250ms"), 'pd.date_range("2024-01-01", None, 3, "250ms")'),
    ("b", dict(start="2024-01-04", end="2024-01-12", freq="B"), 'pd.date_range("2024-01-04", "2024-01-12", None, "B")'),
    ("w", dict(start="2024-01-03", periods=4, freq="W"), 'pd.date_range("2024-01-03", None, 4, "W")'),
    ("wfri", dict(start="2024-01-03", periods=3, freq="W-FRI"), 'pd.date_range("2024-01-03", None, 3, "W-FRI")'),
    ("me", dict(start="2024-01-31", periods=5, freq="ME"), 'pd.date_range("2024-01-31", None, 5, "ME")'),
    ("me2", dict(start="2024-01-15", periods=3, freq="ME"), 'pd.date_range("2024-01-15", None, 3, "ME")'),
    ("ms_", dict(start="2024-01-15", periods=3, freq="MS"), 'pd.date_range("2024-01-15", None, 3, "MS")'),
    ("qe", dict(start="2024-01-01", periods=4, freq="QE"), 'pd.date_range("2024-01-01", None, 4, "QE")'),
    ("qs", dict(start="2024-02-01", periods=3, freq="QS"), 'pd.date_range("2024-02-01", None, 3, "QS")'),
    ("ye", dict(start="2024-03-01", periods=3, freq="YE"), 'pd.date_range("2024-03-01", None, 3, "YE")'),
    ("ys", dict(start="2024-03-01", periods=3, freq="YS"), 'pd.date_range("2024-03-01", None, 3, "YS")'),
    ("d_end", dict(start="2024-02-27", end="2024-03-02", freq="D"), 'pd.date_range("2024-02-27", "2024-03-02", None, "D")'),
    ("d2", dict(start="2024-01-01", periods=4, freq="2D"), 'pd.date_range("2024-01-01", None, 4, "2D")'),
    ("endp", dict(end="2024-01-10", periods=3, freq="D"), 'pd.date_range(None, "2024-01-10", 3, "D")'),
]:
    r = pd.date_range(**kw)
    adds("dr_" + nm, [x.strftime("%Y-%m-%d %H:%M:%S.%f") for x in r], f"[x.strftime(\"%Y-%m-%d %H:%M:%S.%f\") for x in {call}.tolist()]") if False else None
    L.append(f"E_dr_{nm} = {[x.strftime('%Y-%m-%d %H:%M:%S.%f') for x in r]!r}")
    T.append(f"    assert [pd.Timestamp(x).strftime(\"%Y-%m-%d %H:%M:%S.%f\") for x in {call}.tolist()] == E_dr_{nm}")
ts = pd.Timestamp("2024-03-05 10:20:30.123456")
L.append(f"E_ts = {[ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, ts.microsecond, ts.day_of_week, ts.dayofyear, ts.quarter, int(ts.is_leap_year), ts.days_in_month]!r}")
T.append('    t = pd.Timestamp("2024-03-05 10:20:30.123456")\n    assert [t.year, t.month, t.day, t.hour, t.minute, t.second, t.microsecond, t.day_of_week, t.dayofyear, t.quarter, 1 if t.is_leap_year else 0, t.days_in_month] == E_ts')
L.append(f"E_tsn = {[ts.day_name(), ts.month_name(), str(ts), ts.isoformat(), ts.strftime('%a %d %b %Y %H:%M:%S.%f %j'), str(ts.normalize())]!r}")
T.append('    assert [t.day_name(), t.month_name(), str(t), t.isoformat(), t.strftime("%a %d %b %Y %H:%M:%S.%f %j"), str(t.normalize())] == E_tsn')
td = pd.Timedelta("1 days 02:03:04.5")
L.append(f"E_td = {[td.days, td.seconds, td.microsecond if hasattr(td,'microsecond') else td.microseconds, td.total_seconds(), str(td), str(td + pd.Timedelta(hours=30)), str(pd.Timestamp('2024-01-01') + td), str(pd.Timestamp('2024-03-01') - pd.Timestamp('2024-01-01'))]!r}")
T.append('    u = pd.Timedelta("1 days 02:03:04.5")\n    assert [u.days, u.seconds, u.microseconds, u.total_seconds(), str(u), str(u + pd.Timedelta(None, 0, 30)), str(pd.Timestamp("2024-01-01") + u), str(pd.Timestamp("2024-03-01") - pd.Timestamp("2024-01-01"))] == E_td')
# .dt accessor on a Series
idx = pd.date_range("2023-12-28 05:30", periods=12, freq="19h")
sd = pd.Series(idx)
for a in ["year", "month", "day", "hour", "minute", "dayofweek", "dayofyear", "quarter", "days_in_month"]:
    addv("dt_" + a, getattr(sd.dt, a), f"sd.dt.{a}")
adds("dt_dn", sd.dt.day_name(), "sd.dt.day_name()")
adds("dt_mn", sd.dt.month_name(), "sd.dt.month_name()")
adds("dt_sf", sd.dt.strftime("%Y/%m/%d %H%M"), 'sd.dt.strftime("%Y/%m/%d %H%M")')
adds("dt_fl", sd.dt.floor("h"), 'sd.dt.floor("h")') if False else None
adds("dt_norm", sd.dt.normalize(), "sd.dt.normalize()")
adds("dt_ime", sd.dt.is_month_end, "sd.dt.is_month_end")
dd = sd - pd.Timestamp("2023-12-25")
addv("td_days", dd.dt.days, "(sd - pd.Timestamp(\"2023-12-25\")).dt.days")
addv("td_tsec", dd.dt.total_seconds(), "(sd - pd.Timestamp(\"2023-12-25\")).dt.total_seconds()")
adds("dt_plus", sd + pd.Timedelta(hours=7), "sd + pd.Timedelta(None, 0, 7)")
adds("to_dt", pd.to_datetime(pd.Series(["2024-02-29", "2023-07-04", None])), 'pd.to_datetime(pd.Series(["2024-02-29", "2023-07-04", None]))')
adds("dt_cmp", sd > pd.Timestamp("2023-12-31"), 'sd > pd.Timestamp("2023-12-31")')
# resample
vals = [float(((i * 7 + 3) % 11) - 4) * 0.5 for i in range(45)]
ix = pd.date_range("2024-01-03 06:00", periods=45, freq="19h")
sr = pd.Series(vals, index=ix)
def addr(name, r, call):
    L.append(f"E_{name}_i = {[x.strftime('%Y-%m-%d %H:%M:%S') for x in r.index]!r}")
    L.append(f"E_{name}_v = {f(r.to_numpy())!r}")
    T.append(f"    assert [pd.Timestamp(x).strftime(\"%Y-%m-%d %H:%M:%S\") for x in ({call}).index.tolist()] == E_{name}_i")
    T.append(f"    assert close(fvals(({call}).to_frame()) , E_{name}_v)")
for nm, rule in [("rD", "D"), ("r3D", "3D"), ("rW", "W"), ("r6h", "6h"), ("rME", "ME")]:
    for how in ["sum", "mean"]:
        addr(f"{nm}_{how}", getattr(sr.resample(rule), how)(), f'sr.resample("{rule}").{how}()')
for how in ["min", "max", "count", "first", "last"]:
    addr(f"rD_{how}", getattr(sr.resample("D"), how)(), f'sr.resample("D").{how}()')
oh = sr.resample("2D").ohlc()
L.append(f"E_oh = {f(oh.to_numpy())!r}\nE_oh_i = {[x.strftime('%Y-%m-%d') for x in oh.index]!r}")
T.append('    oh = sr.resample("2D").ohlc()\n    assert close(fvals(oh), E_oh)\n    assert [pd.Timestamp(x).strftime("%Y-%m-%d") for x in oh.index.tolist()] == E_oh_i')
# merge_asof
tr = pd.DataFrame({"time": pd.to_datetime(["2024-01-01 00:00:00.023", "2024-01-01 00:00:00.038", "2024-01-01 00:00:00.048", "2024-01-01 00:00:00.048", "2024-01-01 00:00:00.048"]), "q": [10, 20, 30, 40, 50]})
qt = pd.DataFrame({"time": pd.to_datetime(["2024-01-01 00:00:00.023", "2024-01-01 00:00:00.023", "2024-01-01 00:00:00.030", "2024-01-01 00:00:00.041", "2024-01-01 00:00:00.048", "2024-01-01 00:00:00.049", "2024-01-01 00:00:00.072", "2024-01-01 00:00:00.075"]), "bid": [51.95, 51.95, 720.77, 720.92, 98.0, 720.5, 720.92, 98.0]})
for nm, kw, call in [("b", {}, 'pd.merge_asof(tr, qt, "time")'), ("f", dict(direction="forward"), 'pd.merge_asof(tr, qt, "time", None, "forward")'),
                     ("n", dict(direction="nearest"), 'pd.merge_asof(tr, qt, "time", None, "nearest")'),
                     ("tol", dict(tolerance=pd.Timedelta("10ms")), 'pd.merge_asof(tr, qt, "time", None, "backward", pd.Timedelta(None, 0, 0, 0, 0, 10))'),
                     ("nx", dict(allow_exact_matches=False), 'pd.merge_asof(tr, qt, "time", None, "backward", None, ("_x", "_y"), False)')]:
    r = pd.merge_asof(tr, qt, on="time", **kw)
    addd("asof_" + nm, r[["q", "bid"]], f"{call}[[\"q\", \"bid\"]]")
setup = ('sd = pd.Series(pd.date_range("2023-12-28 05:30", periods=12, freq="19h"))\n'
 'SV = List[float]([float(((i * 7 + 3) % 11) - 4) * 0.5 for i in range(45)])\n'
 'sr = pd.Series(SV, index=pd.date_range("2024-01-03 06:00", None, 45, "19h"))\n'
 'tr = pd.DataFrame({"time": pd.to_datetime(["2024-01-01 00:00:00.023", "2024-01-01 00:00:00.038", "2024-01-01 00:00:00.048", "2024-01-01 00:00:00.048", "2024-01-01 00:00:00.048"]), "q": pd.Series([10, 20, 30, 40, 50])})\n'
 'qt = pd.DataFrame({"time": pd.to_datetime(["2024-01-01 00:00:00.023", "2024-01-01 00:00:00.023", "2024-01-01 00:00:00.030", "2024-01-01 00:00:00.041", "2024-01-01 00:00:00.048", "2024-01-01 00:00:00.049", "2024-01-01 00:00:00.072", "2024-01-01 00:00:00.075"]), "bid": pd.Series([51.95, 51.95, 720.77, 720.92, 98.0, 720.5, 720.92, 98.0])})\n')
open(os.path.join(D, "test_dt.py"), "w").write(head + setup + "\n".join(L) + "\n\n@test\ndef test_dt():\n" + "\n".join(T) + "\ntest_dt()\n")
