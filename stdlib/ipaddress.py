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

# ip_address/ip_network return a Union when the family is only known at run time.
def ip_address(address: str) -> Union[IPv4Address, IPv6Address]:
    if ":" in address:
        return IPv6Address(address)
    return IPv4Address(address)

def ip_network(address: str, strict: bool = True) -> Union[IPv4Network, IPv6Network]:
    if ":" in address:
        return IPv6Network(address, strict)
    return IPv4Network(address, strict)


def _hex4(v: int) -> str:
    return ("%x" % v)


class IPv6Address:
    _h: List[int]  # eight 16-bit groups, big endian

    def __init__(self, address: str):
        if len(address) == 0:
            raise AddressValueError("Address cannot be empty")
        if "%" in address:
            address = address[:address.find("%")]
        parts = address.split(":")
        if len(parts) < 3:
            raise AddressValueError(f"At least 3 parts expected in {address!r}")
        if "." in parts[-1]:
            v4 = IPv4Address(parts[-1])._ip
            parts[-1] = "%x" % (v4 >> 16)
            parts.append("%x" % (v4 & 0xFFFF))
        if len(parts) > 9:
            raise AddressValueError(f"At most 8 colons permitted in {address!r}")
        skip = -1
        for i in range(1, len(parts) - 1):
            if parts[i] == "":
                if skip != -1:
                    raise AddressValueError(f"At most one '::' permitted in {address!r}")
                skip = i
        head = List[int]()
        tail = List[int]()
        if skip != -1:
            ph = skip
            pt = len(parts) - skip - 1
            if parts[0] == "":
                ph -= 1
                if ph != 0:
                    raise AddressValueError(f"Leading ':' only permitted as part of '::' in {address!r}")
            if parts[-1] == "":
                pt -= 1
                if pt != 0:
                    raise AddressValueError(f"Trailing ':' only permitted as part of '::' in {address!r}")
            if ph + pt > 7:
                raise AddressValueError(f"Expected at most 7 other parts with '::' in {address!r}")
            for i in range(ph):
                head.append(IPv6Address._group(parts[i], address))
            for i in range(len(parts) - pt, len(parts)):
                tail.append(IPv6Address._group(parts[i], address))
            self._h = head
            for _ in range(8 - len(head) - len(tail)):
                self._h.append(0)
            for g in tail:
                self._h.append(g)
        else:
            if len(parts) != 8:
                raise AddressValueError(f"Exactly 8 parts expected without '::' in {address!r}")
            if parts[0] == "":
                raise AddressValueError(f"Leading ':' only permitted as part of '::' in {address!r}")
            if parts[-1] == "":
                raise AddressValueError(f"Trailing ':' only permitted as part of '::' in {address!r}")
            self._h = List[int]()
            for p in parts:
                self._h.append(IPv6Address._group(p, address))

    def _group(p: str, address: str) -> int:
        if len(p) == 0:
            raise AddressValueError(f"Only hex digits permitted in {p!r} in {address!r}")
        if len(p) > 4:
            raise AddressValueError(f"At most 4 characters permitted in {p!r} in {address!r}")
        v = 0
        for c in p:
            if "0" <= c <= "9":
                v = v * 16 + (ord(c) - 48)
            elif "a" <= c <= "f":
                v = v * 16 + (ord(c) - 87)
            elif "A" <= c <= "F":
                v = v * 16 + (ord(c) - 55)
            else:
                raise AddressValueError(f"Only hex digits permitted in {p!r} in {address!r}")
        return v

    def _from_groups(h: List[int]) -> IPv6Address:
        a = IPv6Address("::")
        a._h = h
        return a

    @property
    def exploded(self) -> str:
        if self.ipv4_mapped is not None:
            v4 = self.ipv4_mapped.__str__()
            return ":".join([("%04x" % self._h[k]) for k in range(6)]) + ":" + v4
        return ":".join([("%04x" % g) for g in self._h])

    @property
    def compressed(self) -> str:
        if self.ipv4_mapped is not None:
            return "::ffff:" + str(self.ipv4_mapped.__str__())
        best_start = -1
        best_len = 0
        i = 0
        while i < 8:
            if self._h[i] == 0:
                j = i
                while j < 8 and self._h[j] == 0:
                    j += 1
                if j - i > best_len:
                    best_start = i
                    best_len = j - i
                i = j
            else:
                i += 1
        if best_len < 2:
            return ":".join([_hex4(g) for g in self._h])
        left = ":".join([_hex4(self._h[k]) for k in range(best_start)])
        right = ":".join([_hex4(self._h[k]) for k in range(best_start + best_len, 8)])
        return left + "::" + right

    def __str__(self) -> str:
        return self.compressed

    def __repr__(self) -> str:
        return f"IPv6Address({self.compressed!r})"

    def __eq__(self, o: IPv6Address) -> bool:
        return self._h == o._h

    def __ne__(self, o: IPv6Address) -> bool:
        return self._h != o._h

    def __lt__(self, o: IPv6Address) -> bool:
        return self._h < o._h

    def __le__(self, o: IPv6Address) -> bool:
        return self._h <= o._h

    def __gt__(self, o: IPv6Address) -> bool:
        return self._h > o._h

    def __ge__(self, o: IPv6Address) -> bool:
        return self._h >= o._h

    def __hash__(self) -> int:
        h = 1469598103934665603
        for g in self._h:
            h = (h ^ g) * 1099511628211
        return h

    @property
    def packed(self) -> bytes:
        out = bytearray()
        for g in self._h:
            out.append(g >> 8)
            out.append(g & 255)
        return bytes(out)

    @property
    def version(self) -> int:
        return 6

    @property
    def max_prefixlen(self) -> int:
        return 128

    @property
    def is_unspecified(self) -> bool:
        return self == IPv6Address("::")

    @property
    def is_loopback(self) -> bool:
        return self == IPv6Address("::1")

    @property
    def is_multicast(self) -> bool:
        return (self._h[0] & 0xFF00) == 0xFF00

    @property
    def is_link_local(self) -> bool:
        return (self._h[0] & 0xFFC0) == 0xFE80

    @property
    def is_site_local(self) -> bool:
        return (self._h[0] & 0xFFC0) == 0xFEC0

    @property
    def is_private(self) -> bool:
        h = self._h
        if self.is_loopback or self.is_unspecified:
            return True
        if (h[0] & 0xFE00) == 0xFC00 or self.is_link_local:
            return True
        if h[0] == 0x2001 and h[1] == 0x0db8:
            return True
        if h[0] == 0 and h[1] == 0 and h[2] == 0 and h[3] == 0 and h[4] == 0 and h[5] == 0xFFFF:
            return False
        return False

    @property
    def is_global(self) -> bool:
        return not self.is_private

    @property
    def ipv4_mapped(self) -> Optional[IPv4Address]:
        h = self._h
        if h[0] == 0 and h[1] == 0 and h[2] == 0 and h[3] == 0 and h[4] == 0 and h[5] == 0xFFFF:
            return IPv4Address((h[6] << 16) | h[7])
        return None

    def __add__(self, n: int) -> IPv6Address:
        return IPv6Address._from_groups(IPv6Address._addn(self._h, n))

    def __sub__(self, n: int) -> IPv6Address:
        return IPv6Address._from_groups(IPv6Address._addn(self._h, -n))

    def _addn(h: List[int], n: int) -> List[int]:
        out = [0] * 8
        carry = n
        for i in range(7, -1, -1):
            t = h[i] + carry
            out[i] = t & 0xFFFF
            carry = t >> 16
        if carry != 0:
            raise AddressValueError("address out of range")
        return out


class IPv6Network:
    network_address: IPv6Address
    prefixlen: int

    def __init__(self, address: str, strict: bool = True):
        i = address.find("/")
        base = address
        pl = 128
        if i >= 0:
            base = address[:i]
            ps = address[i + 1:]
            if len(ps) == 0 or not ps.isdigit():
                raise NetmaskValueError(f"{ps!r} is not a valid netmask")
            pl = int(ps)
            if pl > 128:
                raise NetmaskValueError(f"{ps!r} is not a valid netmask")
        a = IPv6Address(base)
        h = List[int]()
        for k in range(8):
            bits = pl - 16 * k
            m = 0
            if bits >= 16:
                m = 0xFFFF
            elif bits > 0:
                m = (0xFFFF << (16 - bits)) & 0xFFFF
            if strict and (a._h[k] & ~m & 0xFFFF) != 0:
                raise ValueError(f"{address} has host bits set")
            h.append(a._h[k] & m)
        self.network_address = IPv6Address._from_groups(h)
        self.prefixlen = pl

    def _mask(self, k: int) -> int:
        bits = self.prefixlen - 16 * k
        if bits >= 16:
            return 0xFFFF
        if bits > 0:
            return (0xFFFF << (16 - bits)) & 0xFFFF
        return 0

    @property
    def netmask(self) -> IPv6Address:
        return IPv6Address._from_groups([self._mask(k) for k in range(8)])

    @property
    def hostmask(self) -> IPv6Address:
        return IPv6Address._from_groups([self._mask(k) ^ 0xFFFF for k in range(8)])

    @property
    def broadcast_address(self) -> IPv6Address:
        return IPv6Address._from_groups(
            [self.network_address._h[k] | (self._mask(k) ^ 0xFFFF) for k in range(8)])

    @property
    def num_addresses(self) -> int:
        host = 128 - self.prefixlen
        if host > 62:
            raise OverflowError("num_addresses exceeds 63 bits")
        return 1 << host

    def __contains__(self, a: IPv6Address) -> bool:
        for k in range(8):
            if (a._h[k] & self._mask(k)) != self.network_address._h[k]:
                return False
        return True

    def subnet_of(self, o: IPv6Network) -> bool:
        return o.prefixlen <= self.prefixlen and self.network_address in o

    def __str__(self) -> str:
        return f"{self.network_address}/{self.prefixlen}"

    def __repr__(self) -> str:
        return f"IPv6Network({str(self)!r})"

    def __eq__(self, o: IPv6Network) -> bool:
        return self.network_address == o.network_address and self.prefixlen == o.prefixlen



def ip_address_v6(address: str) -> IPv6Address:
    return IPv6Address(address)


def ip_network_v6(address: str, strict: bool = True) -> IPv6Network:
    return IPv6Network(address, strict)


def is_ipv6(address: str) -> bool:
    return ":" in address
