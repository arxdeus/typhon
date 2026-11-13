# Expected values from CPython 3.14
import string

@test
def test_template():
    t = string.Template("$a-${b}$$ $a")
    assert t.substitute(a=1, b="x") == "1-x$ 1"
    assert string.Template("$a $zz ${q").safe_substitute(a=1) == "1 $zz ${q"
    assert string.Template("$a $zz").safe_substitute({"a": "A"}) == "A $zz"
    assert t.get_identifiers() == ["a", "b"]
    assert t.is_valid()
    assert not string.Template("$").is_valid()
    try:
        string.Template("$a $zz").substitute(a=1)
        assert False
    except KeyError:
        pass
    try:
        string.Template("x\n $ y").substitute(a=1)
        assert False
    except ValueError as e:
        assert str(e) == "Invalid placeholder in string: line 2, col 2"
