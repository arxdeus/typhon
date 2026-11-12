import os
import sys

# Child mode: write without close() and exit; the parent reads the file back.
_child = os.environ.get("TYPHON_IO_EXIT_FLUSH", "")
if _child:
    f = open(_child, "w")
    f.write("flushed at exit\n")
    sys.exit(0)


@test
def test_unclosed_file_flushed_at_exit():
    path = f"/tmp/typhon_io_exit_flush_{os.getpid()}.txt"
    rc = os.system(f"TYPHON_IO_EXIT_FLUSH={path} '{sys.argv[0]}' >/dev/null 2>&1")
    assert rc == 0
    with open(path) as g:
        assert g.read() == "flushed at exit\n"
    os.remove(path)
test_unclosed_file_flushed_at_exit()
