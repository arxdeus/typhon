# Ported from CPython 3.14 ipaddress (PSF license), IPv4 subset

class AddressValueError(ValueError):
    pass

class NetmaskValueError(ValueError):
    pass

class IPv4Address:
    _ip: int

    def __init__(self, address: str):
        parts = address.split(".")
        if len(parts) != 4:
            raise AddressValueError(f"Expected 4 octets in {address!r}")
        v = 0
        for p in parts:
            if len(p) == 0 or len(p) > 3 or not p.isdigit():
                raise AddressValueError(f"Octet {p} of address {address!r} is not valid")
            o = int(p)
            if o > 255:
                raise AddressValueError(f"Octet {o} (> 255) not permitted in {address!r}")
            v = (v << 8) | o
        self._ip = v

    def __init__(self, address: int):
        if address < 0 or address > 0xFFFFFFFF:
            raise AddressValueError(f"{address} (>= 2**32) is not permitted as an IPv4 address")
        self._ip = address

    def __int__(self) -> int:
        return self._ip

    def __str__(self) -> str:
        return f"{(self._ip >> 24) & 255}.{(self._ip >> 16) & 255}.{(self._ip >> 8) & 255}.{self._ip & 255}"

    def __repr__(self) -> str:
        return "IPv4Address('" + self.__str__() + "')"

    def __eq__(self, o: IPv4Address) -> bool:
        return self._ip == o._ip

    def __ne__(self, o: IPv4Address) -> bool:
        return self._ip != o._ip

    def __lt__(self, o: IPv4Address) -> bool:
        return self._ip < o._ip

    def __le__(self, o: IPv4Address) -> bool:
        return self._ip <= o._ip

    def __gt__(self, o: IPv4Address) -> bool:
        return self._ip > o._ip

    def __ge__(self, o: IPv4Address) -> bool:
        return self._ip >= o._ip

    def __hash__(self) -> int:
        return hash(self._ip)

    def __add__(self, n: int) -> IPv4Address:
        return IPv4Address(self._ip + n)

    def __sub__(self, n: int) -> IPv4Address:
        return IPv4Address(self._ip - n)

    @property
    def packed(self) -> bytes:
        return bytes([(self._ip >> 24) & 255, (self._ip >> 16) & 255, (self._ip >> 8) & 255, self._ip & 255])

    @property
    def version(self) -> int:
        return 4

    @property
    def max_prefixlen(self) -> int:
        return 32

    @property
    def is_loopback(self) -> bool:
        return (self._ip >> 24) == 127

    @property
    def is_multicast(self) -> bool:
        return (self._ip >> 28) == 14

    @property
    def is_unspecified(self) -> bool:
        return self._ip == 0

    @property
    def is_link_local(self) -> bool:
        return (self._ip >> 16) == 0xA9FE

    @property
    def is_private(self) -> bool:
        ip = self._ip
        return ((ip >> 24) == 10 or (ip >> 20) == 0xAC1 or (ip >> 16) == 0xC0A8
                or (ip >> 24) == 127 or (ip >> 16) == 0xA9FE or ip == 0 or (ip >> 24) == 0
                or (ip >> 16) == 0xC000 and False)

    @property
    def is_global(self) -> bool:
        return not self.is_private and not self.is_multicast and self._ip != 0xFFFFFFFF and (self._ip >> 22) != 0x191

    @property
    def is_reserved(self) -> bool:
        return (self._ip >> 28) == 15

class IPv4Network:
    network_address: IPv4Address
    prefixlen: int

    def __init__(self, address: str, strict: bool = True):
        i = address.find("/")
        base = address
        pl = 32
        if i >= 0:
            base = address[:i]
            ps = address[i + 1:]
            if len(ps) == 0 or not ps.isdigit():
                raise NetmaskValueError(f"{ps!r} is not a valid netmask")
            pl = int(ps)
            if pl > 32:
                raise NetmaskValueError(f"{ps!r} is not a valid netmask")
        a = IPv4Address(base)
        mask = ((0xFFFFFFFF << (32 - pl)) & 0xFFFFFFFF)
        if strict and (a._ip & ~mask & 0xFFFFFFFF) != 0:
            raise ValueError(f"{address} has host bits set")
        self.network_address = IPv4Address(a._ip & mask)
        self.prefixlen = pl

    @property
    def netmask(self) -> IPv4Address:
        return IPv4Address((0xFFFFFFFF << (32 - self.prefixlen)) & 0xFFFFFFFF)

    @property
    def hostmask(self) -> IPv4Address:
        return IPv4Address(((0xFFFFFFFF << (32 - self.prefixlen)) & 0xFFFFFFFF) ^ 0xFFFFFFFF)

    @property
    def broadcast_address(self) -> IPv4Address:
        return IPv4Address(self.network_address._ip | self.hostmask._ip)

    @property
    def num_addresses(self) -> int:
        return 1 << (32 - self.prefixlen)

    def __contains__(self, a: IPv4Address) -> bool:
        m = (0xFFFFFFFF << (32 - self.prefixlen)) & 0xFFFFFFFF
        return (a._ip & m) == self.network_address._ip

    def hosts(self) -> List[IPv4Address]:
        out = List[IPv4Address]()
        lo = self.network_address._ip
        hi = self.broadcast_address._ip
        if self.prefixlen >= 31:
            for v in range(lo, hi + 1):
                out.append(IPv4Address(v))
            return out
        for v in range(lo + 1, hi):
            out.append(IPv4Address(v))
        return out

    def __str__(self) -> str:
        return str(self.network_address) + "/" + str(self.prefixlen)

    def __repr__(self) -> str:
        return "IPv4Network('" + self.__str__() + "')"

    def __eq__(self, o: IPv4Network) -> bool:
        return self.network_address._ip == o.network_address._ip and self.prefixlen == o.prefixlen

def ip_address(address: str) -> IPv4Address:
    return IPv4Address(address)

def ip_network(address: str, strict: bool = True) -> IPv4Network:
    return IPv4Network(address, strict)
