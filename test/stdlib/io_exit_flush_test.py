import os
from internal.file import _file_flush_at_exit


@test
def test_unclosed_file_flushed_at_exit():
    # The atexit hook flushes buffered data of files that were never closed.
    path = f"/tmp/typhon_io_exit_flush_{os.getpid()}.txt"
    f = open(path, "w")
    f.write("flushed at exit\n")
    _file_flush_at_exit()
    with open(path) as g:
        assert g.read() == "flushed at exit\n"
    f.close()
    # Running the hook again (closed file, nothing pending) must be a no-op.
    _file_flush_at_exit()
    os.remove(path)
test_unclosed_file_flushed_at_exit()
