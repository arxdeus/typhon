import difflib
from difflib import SequenceMatcher

@test
def test_difflib():
    s = SequenceMatcher("abxcd", "abcd")
    assert s.get_matching_blocks() == [(0, 0, 2), (3, 2, 2), (5, 4, 0)]
    assert s.get_opcodes() == [("equal", 0, 2, 0, 2), ("delete", 2, 3, 2, 2), ("equal", 3, 5, 2, 4)]
    assert s.ratio() == 8.0 / 9.0
    assert SequenceMatcher("", "").ratio() == 1.0
    assert SequenceMatcher("abcd", "bcde").ratio() == 0.75
    assert SequenceMatcher("private Thread currentThread;", "private volatile Thread currentThread;").find_longest_match(0, 29, 0, 38) == (6, 15, 23)
    assert difflib.get_close_matches("appel", ["ape", "apple", "peach", "puppy"]) == ["apple", "ape"]
    a = ["one\n", "two\n", "three\n"]
    b = ["one\n", "tree\n", "three\n", "four\n"]
    d = difflib.unified_diff(a, b, "a.txt", "b.txt")
    assert d == ["--- a.txt\n", "+++ b.txt\n", "@@ -1,3 +1,4 @@\n", " one\n", "-two\n", "+tree\n", " three\n", "+four\n"]
    assert difflib.ndiff(["a", "b"], ["a", "c"]) == ["  a", "- b", "+ c"]
test_difflib()
