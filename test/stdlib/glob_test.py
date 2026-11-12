import glob
import os
import shutil

BASE = "build/glob-test-" + str(os.getpid())


def _touch(p: str):
    with open(p, "w") as f:
        f.write("x")


def _setup():
    os.makedirs(BASE + "/a/b/c", exist_ok=True)
    os.makedirs(BASE + "/d", exist_ok=True)
    _touch(BASE + "/x.py")
    _touch(BASE + "/y.txt")
    _touch(BASE + "/.hidden.py")
    _touch(BASE + "/a/one.py")
    _touch(BASE + "/a/b/two.py")
    _touch(BASE + "/a/b/c/three.py")
    _touch(BASE + "/d/[x].py")


def _rel(xs: List[str]) -> List[str]:
    n = len(BASE) + 1
    return sorted([x[n:] for x in xs])


@test
def test_glob():
    _setup()
    assert _rel(glob.glob(BASE + "/*.py")) == ["x.py"]
    assert _rel(glob.glob(BASE + "/*")) == ["a", "d", "x.py", "y.txt"]
    assert _rel(glob.glob(BASE + "/?.*")) == ["x.py", "y.txt"]
    assert _rel(glob.glob(BASE + "/.*")) == [".hidden.py"]
    assert _rel(glob.glob(BASE + "/*.py", include_hidden=True)) == [".hidden.py", "x.py"]
    assert _rel(glob.glob(BASE + "/*/")) == ["a/", "d/"]
    assert _rel(glob.glob(BASE + "/*/*.py")) == ["a/one.py", "d/[x].py"]
    assert _rel(glob.glob(BASE + "/**/*.py")) == ["a/one.py", "d/[x].py"]
    assert _rel(glob.glob(BASE + "/**/*.py", recursive=True)) == [
        "a/b/c/three.py", "a/b/two.py", "a/one.py", "d/[x].py", "x.py"]
    assert _rel(glob.glob(BASE + "/**", recursive=True)) == [
        "", "a", "a/b", "a/b/c", "a/b/c/three.py", "a/b/two.py", "a/one.py",
        "d", "d/[x].py", "x.py", "y.txt"]
    assert _rel(glob.glob(BASE + "/a/**/", recursive=True)) == ["a/", "a/b/", "a/b/c/"]
    assert _rel(glob.glob(BASE + "/[xy].*")) == ["x.py", "y.txt"]
    assert _rel(glob.glob(BASE + "/d/" + glob.escape("[x].py"))) == ["d/[x].py"]
    assert _rel(glob.glob(BASE + "/x.py")) == ["x.py"]
    assert glob.glob(BASE + "/nope.py") == List[str]()
    assert sorted(glob.glob("*.py", root_dir=BASE + "/a")) == ["one.py"]
    assert sorted(glob.glob("**/*.py", root_dir=BASE + "/a", recursive=True)) == [
        "b/c/three.py", "b/two.py", "one.py"]
    assert _rel(list(glob.iglob(BASE + "/*.txt"))) == ["y.txt"]
    assert glob.escape("a[b]*?.py") == "a[[]b][*][?].py"
    assert glob.has_magic("a*") and not glob.has_magic("abc")
    shutil.rmtree(BASE)
    assert not os.path.exists(BASE)
test_glob()
