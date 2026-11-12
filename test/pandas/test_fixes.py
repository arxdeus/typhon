import pandas as pd
NAN = float("nan")

def _raises_value(f, msg: str) -> bool:
    try:
        f()
    except ValueError as e:
        return str(e) == msg
    return False

def test_fixes():
    RK_AVERAGE = ',0\n0,4.0\n1,1.0\n2,\n3,4.0\n4,2.0\n5,4.0\n'
    RK_AVERAGE_DESC_TOP = ',0\n0,3.0\n1,6.0\n2,1.0\n3,3.0\n4,5.0\n5,3.0\n'
    RK_AVERAGE_PCT_BOT = ',0\n0,0.6666666666666666\n1,0.16666666666666666\n2,1.0\n3,0.6666666666666666\n4,0.3333333333333333\n5,0.6666666666666666\n'
    RK_MIN = ',0\n0,3.0\n1,1.0\n2,\n3,3.0\n4,2.0\n5,3.0\n'
    RK_MIN_DESC_TOP = ',0\n0,2.0\n1,6.0\n2,1.0\n3,2.0\n4,5.0\n5,2.0\n'
    RK_MIN_PCT_BOT = ',0\n0,0.5\n1,0.16666666666666666\n2,1.0\n3,0.5\n4,0.3333333333333333\n5,0.5\n'
    RK_MAX = ',0\n0,5.0\n1,1.0\n2,\n3,5.0\n4,2.0\n5,5.0\n'
    RK_MAX_DESC_TOP = ',0\n0,4.0\n1,6.0\n2,1.0\n3,4.0\n4,5.0\n5,4.0\n'
    RK_MAX_PCT_BOT = ',0\n0,0.8333333333333334\n1,0.16666666666666666\n2,1.0\n3,0.8333333333333334\n4,0.3333333333333333\n5,0.8333333333333334\n'
    RK_FIRST = ',0\n0,3.0\n1,1.0\n2,\n3,4.0\n4,2.0\n5,5.0\n'
    RK_FIRST_DESC_TOP = ',0\n0,2.0\n1,6.0\n2,1.0\n3,3.0\n4,5.0\n5,4.0\n'
    RK_FIRST_PCT_BOT = ',0\n0,0.5\n1,0.16666666666666666\n2,1.0\n3,0.6666666666666666\n4,0.3333333333333333\n5,0.8333333333333334\n'
    RK_DENSE = ',0\n0,3.0\n1,1.0\n2,\n3,3.0\n4,2.0\n5,3.0\n'
    RK_DENSE_DESC_TOP = ',0\n0,2.0\n1,4.0\n2,1.0\n3,2.0\n4,3.0\n5,2.0\n'
    RK_DENSE_PCT_BOT = ',0\n0,0.75\n1,0.25\n2,1.0\n3,0.75\n4,0.5\n5,0.75\n'
    RK_DF_AX1 = ',a,b\n0,1.0,2.0\n1,1.0,2.0\n2,1.0,2.0\n3,1.0,2.0\n'
    RK_DF_DENSE = ',a,b\n0,2.0,3.0\n1,2.0,3.0\n2,1.0,2.0\n3,2.0,1.0\n'
    PCT_INT0 = ',0\n0,\n1,\n2,inf\n'
    PCT_NEG0 = ',0\n0,\n1,-inf\n2,-1.0\n3,inf\n'
    FDIV0 = ',0\n0,inf\n1,-inf\n2,\n3,inf\n'
    MOD0 = ',0\n0,\n1,\n2,\n3,\n'
    FDIVS = ',0\n0,inf\n1,-3.0\n2,\n3,-4.0\n'
    MODS = ',0\n0,\n1,1.0\n2,\n3,-1.0\n'
    CSV_EXTRA = ',a,b\n1,2,3\n4,5,6\n'
    DUP_FIRST = ',0\n0,False\n1,True\n2,False\n3,False\n'
    DUP_NONE = ',0\n0,True\n1,True\n2,False\n3,False\n'
    DUP_SUB_LAST = ',0\n0,True\n1,True\n2,False\n3,False\n'
    DD_SUB_LAST_IGN = ',a,b\n0,2,4\n1,1,5\n'
    DD_DEFAULT = ',a,b\n0,1,3\n2,2,4\n3,1,5\n'
    SDUP_LAST = ',0\n0,True\n1,False\n2,False\n3,True\n4,False\n'
    SDD_FALSE = ',0\n1,2.0\n'
    rs = pd.Series([3.0, 1.0, NAN, 3.0, 2.0, 3.0])
    exp = {"average": (RK_AVERAGE, RK_AVERAGE_DESC_TOP, RK_AVERAGE_PCT_BOT), "min": (RK_MIN, RK_MIN_DESC_TOP, RK_MIN_PCT_BOT),
           "max": (RK_MAX, RK_MAX_DESC_TOP, RK_MAX_PCT_BOT), "first": (RK_FIRST, RK_FIRST_DESC_TOP, RK_FIRST_PCT_BOT),
           "dense": (RK_DENSE, RK_DENSE_DESC_TOP, RK_DENSE_PCT_BOT)}
    for m, e in exp.items():
        assert rs.rank(method=m).to_csv() == e[0]
        assert rs.rank(method=m, ascending=False, na_option="top").to_csv() == e[1]
        assert rs.rank(method=m, pct=True, na_option="bottom").to_csv() == e[2]
    assert _raises_value(lambda: rs.rank(na_option="x"), "na_option must be one of 'keep', 'top', or 'bottom'")
    rdf = pd.DataFrame({"a": [1, 1, 2, 1], "b": [3, 3, 4, 5]})
    assert rdf.rank(axis=1, method="min").to_csv() == RK_DF_AX1
    assert rdf.rank(method="dense", ascending=False).to_csv() == RK_DF_DENSE
    assert pd.Series([0, 0, 1]).pct_change().to_csv() == PCT_INT0
    assert pd.Series([0, -2, 0, 3]).pct_change().to_csv() == PCT_NEG0
    a = pd.Series([5, -5, 0, 7])
    b = pd.Series([0, 2, 0, -2])
    assert (a // 0).to_csv() == FDIV0
    assert (a % 0).to_csv() == MOD0
    assert (a // b).to_csv() == FDIVS
    assert (a % b).to_csv() == MODS
    assert _raises_value(lambda: rs.quantile(1.5), "percentiles should all be in the interval [0, 1]")
    assert _raises_value(lambda: rs.quantile([0.5, -1.0]), "percentiles should all be in the interval [0, 1]")
    assert pd.read_csv_string("a,b\n1,2,3\n4,5,6\n").to_csv() == CSV_EXTRA
    dd = pd.DataFrame({"a": [1, 1, 2, 1], "b": [3, 3, 4, 5]})
    assert dd.duplicated().to_csv() == DUP_FIRST
    assert dd.duplicated(keep=False).to_csv() == DUP_NONE
    assert dd.duplicated(subset=["a"], keep="last").to_csv() == DUP_SUB_LAST
    assert dd.drop_duplicates(subset="a", keep="last", ignore_index=True).to_csv() == DD_SUB_LAST_IGN
    assert dd.drop_duplicates().to_csv() == DD_DEFAULT
    sd = pd.Series([1.0, 2.0, 1.0, NAN, NAN])
    assert sd.duplicated(keep="last").to_csv() == SDUP_LAST
    assert sd.drop_duplicates(keep=False).to_csv() == SDD_FALSE
    print("test_fixes: ok")

test_fixes()
