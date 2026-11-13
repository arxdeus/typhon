# Expected values from CPython 3.14
import ipaddress as ip

_out = List[str]()

def _p(*args):
    _out.append(" ".join([str(a) for a in args]))

@test
def test_ipv6():
    for s in ["2001:db8::1", "::", "::1", "fe80::1:2", "1:0:0:2:0:0:0:3", "::ffff:1.2.3.4", "2001:DB8:0:0:1:0:0:1", "1:2:3:4:5:6:7:8"]:
        a = ip.IPv6Address(s)
        _p(a, a.exploded, a.is_loopback, a.is_link_local, a.is_private, a.is_multicast, a.ipv4_mapped, a.packed.hex())
    _p(repr(ip.IPv6Address("::1") + 5), ip.IPv6Address("2001:db8::") < ip.IPv6Address("2001:db9::"))
    n = ip.IPv6Network("2001:db8::/32")
    _p(n, n.netmask, n.hostmask, n.broadcast_address, n.num_addresses if False else "", ip.IPv6Address("2001:db8:1::1") in n, repr(n))
    n = ip.IPv6Network("2001:db8::/126")
    _p(n.num_addresses, n.netmask)
    for bad in ["1::2::3", "12345::", ":1:2:3:4:5:6:7", "1:2:3", "g::"]:
        try:
            ip.IPv6Address(bad)
        except ip.AddressValueError as e:
            _p(e)
    try:
        ip.IPv6Network("2001:db8::1/32")
    except ValueError as e:
        _p(e)

    expected = "2001:db8::1 2001:0db8:0000:0000:0000:0000:0000:0001 False False True False None 20010db8000000000000000000000001\n:: 0000:0000:0000:0000:0000:0000:0000:0000 False False True False None 00000000000000000000000000000000\n::1 0000:0000:0000:0000:0000:0000:0000:0001 True False True False None 00000000000000000000000000000001\nfe80::1:2 fe80:0000:0000:0000:0000:0000:0001:0002 False True True False None fe800000000000000000000000010002\n1:0:0:2::3 0001:0000:0000:0002:0000:0000:0000:0003 False False False False None 00010000000000020000000000000003\n::ffff:1.2.3.4 0000:0000:0000:0000:0000:ffff:1.2.3.4 False False False False 1.2.3.4 00000000000000000000ffff01020304\n2001:db8::1:0:0:1 2001:0db8:0000:0000:0001:0000:0000:0001 False False True False None 20010db8000000000001000000000001\n1:2:3:4:5:6:7:8 0001:0002:0003:0004:0005:0006:0007:0008 False False False False None 00010002000300040005000600070008\nIPv6Address('::6') True\n2001:db8::/32 ffff:ffff:: ::ffff:ffff:ffff:ffff:ffff:ffff 2001:db8:ffff:ffff:ffff:ffff:ffff:ffff  True IPv6Network('2001:db8::/32')\n4 ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffc\nAt most one '::' permitted in '1::2::3'\nAt most 4 characters permitted in '12345' in '12345::'\nLeading ':' only permitted as part of '::' in ':1:2:3:4:5:6:7'\nExactly 8 parts expected without '::' in '1:2:3'\nOnly hex digits permitted in 'g' in 'g::'\n2001:db8::1/32 has host bits set\n"
    assert "\n".join(_out) + "\n" == expected

@test
def test_ip_dispatch():
    import ipaddress
    a = ipaddress.ip_address("::1")
    b = ipaddress.ip_address("10.0.0.1")
    assert str(a) == "::1"
    assert str(b) == "10.0.0.1"
    assert str(ipaddress.ip_network("2001:db8::/32")) == "2001:db8::/32"
    assert str(ipaddress.ip_network("10.0.0.0/8")) == "10.0.0.0/8"
test_ip_dispatch()
