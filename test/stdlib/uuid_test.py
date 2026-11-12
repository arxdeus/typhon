import uuid
from uuid import UUID

@test
def test_uuid():
    u = UUID("12345678-1234-5678-1234-567812345678")
    assert str(u) == "12345678-1234-5678-1234-567812345678"
    assert u.hex == "12345678123456781234567812345678"
    assert repr(u) == "UUID('12345678-1234-5678-1234-567812345678')"
    assert u.urn == "urn:uuid:12345678-1234-5678-1234-567812345678"
    assert u.hi == 0x1234567812345678 and u.lo == 0x1234567812345678
    assert UUID.from_ints(u.hi, u.lo) == u
    assert UUID("{12345678-1234-5678-1234-567812345678}") == u
    assert UUID(u.bytes) == u
    assert str(uuid.uuid5(uuid.NAMESPACE_DNS, "python.org")) == "886313e1-3b8a-5372-9b90-0c9aee199e5d"
    assert str(uuid.uuid3(uuid.NAMESPACE_DNS, "python.org")) == "6fa459ea-ee8a-3ca4-894e-db77e160355e"
    a = uuid.uuid4()
    b = uuid.uuid4()
    assert a != b and a.version == 4 and (int(a.bytes[8]) >> 6) == 2
    assert len(str(a)) == 36
    try:
        UUID("abc")
        assert False
    except ValueError:
        pass
test_uuid()
