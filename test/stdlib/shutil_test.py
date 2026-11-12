import os
import shutil

BASE = "build/shutil-test-" + str(os.getpid())


def _write(p: str, s: str):
    with open(p, "w") as f:
        f.write(s)


def _read(p: str) -> str:
    with open(p) as f:
        return f.read()


@test
def test_copy_and_move():
    os.makedirs(BASE + "/src/sub", exist_ok=True)
    _write(BASE + "/src/a.txt", "alpha")
    _write(BASE + "/src/sub/b.py", "beta")
    _write(BASE + "/src/sub/c.log", "gamma")
    os.chmod(BASE + "/src/a.txt", 0o640)

    assert shutil.copyfile(BASE + "/src/a.txt", BASE + "/a2.txt") == BASE + "/a2.txt"
    assert _read(BASE + "/a2.txt") == "alpha"
    try:
        shutil.copyfile(BASE + "/a2.txt", BASE + "/a2.txt")
        assert False
    except shutil.SameFileError:
        pass

    os.makedirs(BASE + "/dst", exist_ok=True)
    assert shutil.copy(BASE + "/src/a.txt", BASE + "/dst") == BASE + "/dst/a.txt"
    assert os.stat(BASE + "/dst/a.txt").st_mode & 0o777 == 0o640
    shutil.copy2(BASE + "/src/a.txt", BASE + "/a3.txt")
    assert int(os.stat(BASE + "/a3.txt").st_mtime) == int(os.stat(BASE + "/src/a.txt").st_mtime)

    shutil.copytree(BASE + "/src", BASE + "/tree")
    assert sorted(os.listdir(BASE + "/tree")) == ["a.txt", "sub"]
    assert _read(BASE + "/tree/sub/b.py") == "beta"
    shutil.copytree(BASE + "/src", BASE + "/tree2", ignore=shutil.ignore_patterns("*.log"))
    assert sorted(os.listdir(BASE + "/tree2/sub")) == ["b.py"]
    try:
        shutil.copytree(BASE + "/src", BASE + "/tree")
        assert False
    except OSError:
        pass
    shutil.copytree(BASE + "/src", BASE + "/tree", dirs_exist_ok=True)

    assert shutil.move(BASE + "/a2.txt", BASE + "/dst") == BASE + "/dst/a2.txt"
    assert not os.path.exists(BASE + "/a2.txt")
    shutil.move(BASE + "/tree2", BASE + "/moved")
    assert os.path.isfile(BASE + "/moved/sub/b.py")

    shutil.rmtree(BASE + "/tree")
    assert not os.path.exists(BASE + "/tree")
    shutil.rmtree(BASE + "/missing", ignore_errors=True)
    try:
        shutil.rmtree(BASE + "/missing")
        assert False
    except FileNotFoundError:
        pass
    shutil.rmtree(BASE)
    assert not os.path.exists(BASE)
test_copy_and_move()


@test
def test_which_and_usage():
    sh = shutil.which("sh")
    assert sh is not None and os.path.basename(sh) == "sh"
    assert shutil.which("definitely-not-a-command-xyz") is None
    assert shutil.which("sh", path="/nonexistent") is None
    u = shutil.disk_usage(".")
    assert u.total > 0 and u.free >= 0 and u.used >= 0 and u.used <= u.total
test_which_and_usage()
