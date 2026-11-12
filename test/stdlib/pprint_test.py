import pprint

@test
def test_pprint():
    assert pprint.pformat([1, 2, 3]) == "[1, 2, 3]"
    l = [i * 1000 for i in range(30)]
    assert pprint.pformat(l, width=40) == """[0,
 1000,
 2000,
 3000,
 4000,
 5000,
 6000,
 7000,
 8000,
 9000,
 10000,
 11000,
 12000,
 13000,
 14000,
 15000,
 16000,
 17000,
 18000,
 19000,
 20000,
 21000,
 22000,
 23000,
 24000,
 25000,
 26000,
 27000,
 28000,
 29000]"""
    d = {"banana": [1, 2, 3, 4, 5, 6], "apple": [10, 20, 30, 40, 50, 60]}
    assert pprint.pformat(d, width=30) == """{'apple': [10,
           20,
           30,
           40,
           50,
           60],
 'banana': [1, 2, 3, 4, 5, 6]}"""
    assert pprint.pformat({"b": 1, "a": 2}) == "{'a': 2, 'b': 1}"
    assert pprint.pformat({"b": 1, "a": 2}, sort_dicts=False) == "{'b': 1, 'a': 2}"
    nested = [["alpha", "beta", "gamma"], ["delta", "epsilon", "zeta"]]
    assert pprint.pformat(nested, width=30) == """[['alpha', 'beta', 'gamma'],
 ['delta', 'epsilon', 'zeta']]"""
    t = ("aaaaaaaaaaaa", "bbbbbbbbbbbb", "cccccccccccc")
    assert pprint.pformat(t, width=30) == """('aaaaaaaaaaaa',
 'bbbbbbbbbbbb',
 'cccccccccccc')"""
test_pprint()
