# Ported from CPython 3.14 struct (PSF)
import struct

@test
def test_struct_class():
    s = struct.Struct("<iH")
    assert s.size == 6
    b = s.pack(-5, 7)
    assert b == b"\xfb\xff\xff\xff\x07\x00"
    r = s.unpack(b)
    assert r[0] == -5 and r[1] == 7

@test
def test_iter_pack_into():
    data = struct.pack("<HH", 1, 2) + struct.pack("<HH", 3, 4)
    it = struct.iter_unpack("<HH", data)
    assert len(it) == 2
    assert it[1][0] == 3 and it[1][1] == 4
    buf = bytearray(8)
    struct.pack_into("<I", buf, 2, 0x01020304)
    assert bytes(buf) == b"\x00\x00\x04\x03\x02\x01\x00\x00"

test_struct_class()
test_iter_pack_into()
