import fnmatch
import shlex
import html

@test
def test_fnmatch():
    assert fnmatch.fnmatchcase("foo.py", "*.py")
    assert not fnmatch.fnmatchcase("foo.pyc", "*.py")
    assert fnmatch.fnmatchcase("abc", "a?c")
    assert fnmatch.fnmatchcase("abc", "[a-c]bc")
    assert not fnmatch.fnmatchcase("dbc", "[!d]bc")
    assert fnmatch.fnmatchcase("a*b", "a[*]b")
    assert fnmatch.fnmatchcase("axxbyyb", "a*b")
    assert fnmatch.fnmatch("FOO.PY", "*.py")
    assert fnmatch.filter(["a.py", "b.c", "c.py"], "*.py") == ["a.py", "c.py"]
    assert fnmatch.translate("*.py") == "(?s:.*\\.py)\\Z"
test_fnmatch()

@test
def test_shlex():
    assert shlex.split("a 'b c' \"d e\" f\\ g") == ["a", "b c", "d e", "f g"]
    assert shlex.split("x # c", comments=True) == ["x"]
    assert shlex.split("a '' b") == ["a", "", "b"]
    assert shlex.quote("abc") == "abc"
    assert shlex.quote("a b") == "'a b'"
    assert shlex.quote("it's") == "'it'\"'\"'s'"
    assert shlex.quote("") == "''"
    assert shlex.join(["a", "b c"]) == "a 'b c'"
    try:
        shlex.split("'abc")
        assert False
    except ValueError:
        pass
test_shlex()

@test
def test_html():
    assert html.escape("<a href=\"x\">&'</a>") == "&lt;a href=&quot;x&quot;&gt;&amp;&#x27;&lt;/a&gt;"
    assert html.escape("\"'", quote=False) == "\"'"
    assert html.unescape("&lt;b&gt; &amp; &#65;&#x42; &quot;x&quot; &copy;") == "<b> & AB \"x\" \u00a9"
    assert html.unescape("&unknown; &amp") == "&unknown; &"
test_html()
