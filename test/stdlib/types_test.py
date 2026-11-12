import types


@test
def test_types():
    ns = types.SimpleNamespace(a=1, b="x")
    assert ns.a == 1 and ns.b == "x"
    assert isinstance(None, types.NoneType)
    assert isinstance((i for i in range(2)), types.GeneratorType)
test_types()
