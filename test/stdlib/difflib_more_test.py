import difflib
from difflib import SequenceMatcher

@test
def test_difflib_more():
    s = SequenceMatcher("abcd bcd", "bcde abc")
    assert s.quick_ratio() == 0.875
    assert s.real_quick_ratio() == 1.0
    assert s.ratio() == 0.375
    a = ["one\n", "two\n", "three\n", "x\n"]
    b = ["one\n", "tree\n", "three\n", "four\n"]
    d = difflib.context_diff(a, b, "a", "b")
    assert d == ['*** a\n', '--- b\n', '***************\n', '*** 1,4 ****\n', '  one\n', '! two\n', '  three\n', '! x\n', '--- 1,4 ----\n', '  one\n', '! tree\n', '  three\n', '! four\n']
test_difflib_more()
