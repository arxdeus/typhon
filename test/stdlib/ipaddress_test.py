# Ported from CPython 3.14 ipaddress (PSF)
import ipaddress

@test
def test_v4():
    a = ipaddress.ip_address("192.168.1.10")
    assert str(a) == "192.168.1.10"
    assert int(a) == 3232235786
    assert a.is_private and not a.is_global
    assert str(a + 5) == "192.168.1.15"
    assert a < ipaddress.ip_address("192.168.2.0")
    n = ipaddress.ip_network("10.0.0.0/30")
    assert n.num_addresses == 4
    assert str(n.netmask) == "255.255.255.252"
    assert str(n.broadcast_address) == "10.0.0.3"
    assert ipaddress.ip_address("10.0.0.2") in n
    assert ipaddress.ip_address("10.0.1.2") not in n
    assert len(n.hosts()) == 2
    assert str(ipaddress.ip_network("10.1.2.3/24", False)) == "10.1.2.0/24"
    assert ipaddress.ip_address("8.8.8.8").is_global
    try:
        ipaddress.ip_address("1.2.3.256")
        assert False
    except ipaddress.AddressValueError:
        pass

test_v4()
