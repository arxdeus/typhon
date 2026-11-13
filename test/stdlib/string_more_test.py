# Ported from CPython 3.14 string (PSF)
import string

@test
def test_capwords():
    assert string.capwords("  hello   wORLD foo ") == "Hello World Foo"
    assert string.capwords("a-b-c", "-") == "A-B-C"

test_capwords()
