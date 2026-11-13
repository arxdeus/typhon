# Ported from CPython 3.14 zlib/timeit (PSF)
import zlib
import timeit

def noop():
    pass

@test
def test_checksums():
    assert zlib.crc32(b"hello world") == 222957957
    assert zlib.crc32(b"") == 0
    assert zlib.crc32(b"world", zlib.crc32(b"hello ")) == 222957957
    assert zlib.adler32(b"hello world") == 436929629
    assert zlib.adler32(b"") == 1

@test
def test_timeit():
    assert timeit.timeit(noop, 10) >= 0.0
    assert len(timeit.repeat(noop, 3, 5)) == 3

test_checksums()
test_timeit()
