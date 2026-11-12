import csv

@test
def test_csv():
    r = csv.reader(['a,b,"c,d"\r\n', 'x,"he said ""hi""",\n', '\n', '"multi\n', 'line",z\n'])
    rows = [x for x in r]
    assert rows == [["a", "b", "c,d"], ["x", 'he said "hi"', ""], [], ["multi\nline", "z"]]
    assert csv.reader(["a;b"], delimiter=";").rows == [["a", "b"]]
    assert csv.reader(["a, b"], skipinitialspace=True).rows == [["a", "b"]]
    w = csv.writer()
    w.writerow(["a", "b c", "d,e", 'q"r'])
    w.writerow([""])
    assert w.getvalue() == 'a,b c,"d,e","q""r"\r\n""\r\n'
    w2 = csv.writer(quoting=csv.QUOTE_ALL, lineterminator="\n")
    w2.writerow(["1", "x"])
    assert w2.getvalue() == '"1","x"\n'
    dr = csv.DictReader(["n,v\n", "a,1\n", "b,2\n"])
    got = [d["n"] + d["v"] for d in dr]
    assert got == ["a1", "b2"]
    dw = csv.DictWriter(["n", "v"], lineterminator="\n")
    dw.writeheader()
    dw.writerow({"n": "a", "v": "1"})
    assert dw.getvalue() == "n,v\na,1\n"
test_csv()
