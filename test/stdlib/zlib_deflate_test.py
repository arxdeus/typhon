# Ported from CPython 3.14 zlib (PSF); streams produced by CPython zlib
import zlib

@test
def test_inflate_cpython():
    assert zlib.decompress(bytes.fromhex("789ccb48cdc9c957c8c020cbf38b7252308547254725312401eb20db61")) == b"hello hello hello hello world" * 20
    assert zlib.decompress(bytes.fromhex("789c6360646266616563e7e0e4e2e6e1e5e3171014121611151397909492969195935750545256515553d7d0d4d2d6d1d5d33730343236313533b7b0b4b2b6b1b5b37770747276717573f7f0f4f2f6f1f5f30f080c0a0e090d0b8f888c8a8e898d8b4f484c4a4e494d4bcfc8cccacec9cdcb2f282c2a2e292d2bafa8acaaaea9adab6f686c6a6e696d6befe8eceaeee9edeb9f3071d2e42953a74d9f3173d6ec3973e7cd5fb070d1e2254b972d5fb172d5ea356bd7addfb071d3e62d5bb76ddfb173d7ee3d7bf7ed3f70f0d0e123478f1d3f71f2d4e93367cf9dbf70f1d2e52b57af5dbf71f3d6ed3b77efdd7ff0f0d1e3274f9f3d7ff1f2d5eb376fdfbdfff0f1d3e72f5fbf7dfff1f317c3a8d747bd3eeaf551af8f7a7dd4eba35e1ff5faa8d747bd3e54bc0e00b8a0b38a")) == bytes([i % 251 for i in range(3000)])
    assert zlib.decompress(bytes.fromhex("7801010300fcff616263024d0127")) == b"abc"

@test
def test_roundtrip():
    for lvl in [0, 1, 6, 9]:
        d = b"the quick brown fox " * 50 + bytes([i % 7 for i in range(500)])
        assert zlib.decompress(zlib.compress(d, lvl)) == d
        assert zlib.decompress(zlib.compress(b"", lvl)) == b""
    d2 = b"abcabcabc" * 100
    assert len(zlib.compress(d2)) < len(d2)
    assert zlib.decompress(zlib.compress(d2, 6, 31), 31) == d2

test_inflate_cpython()
test_roundtrip()
