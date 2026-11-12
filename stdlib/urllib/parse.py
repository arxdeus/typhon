# Ported from CPython 3.14 (PSF-2.0)

_ALWAYS_SAFE = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-~"
_HEXU = "0123456789ABCDEF"

class SplitResult:
    scheme: str
    netloc: str
    path: str
    query: str
    fragment: str

    def __init__(self, scheme: str, netloc: str, path: str, query: str, fragment: str):
        self.scheme = scheme
        self.netloc = netloc
        self.path = path
        self.query = query
        self.fragment = fragment

    def geturl(self) -> str:
        return urlunsplit(self.scheme, self.netloc, self.path, self.query, self.fragment)

    def _hostinfo(self) -> Tuple[str, str]:
        nl = self.netloc
        k = nl.rfind("@")
        hp = nl[k + 1:] if k >= 0 else nl
        if hp.startswith("["):
            e = hp.find("]")
            if e >= 0:
                rest = hp[e + 1:]
                return (hp[1:e], rest[1:] if rest.startswith(":") else "")
        c = hp.find(":")
        if c >= 0:
            return (hp[:c], hp[c + 1:])
        return (hp, "")

    @property
    def hostname(self) -> Optional[str]:
        h = self._hostinfo()[0]
        return h.lower() if h != "" else None

    @property
    def port(self) -> Optional[int]:
        p = self._hostinfo()[1]
        if p == "":
            return None
        return int(p)

    @property
    def username(self) -> Optional[str]:
        k = self.netloc.rfind("@")
        if k < 0:
            return None
        ui = self.netloc[:k]
        c = ui.find(":")
        return ui[:c] if c >= 0 else ui

    @property
    def password(self) -> Optional[str]:
        k = self.netloc.rfind("@")
        if k < 0:
            return None
        ui = self.netloc[:k]
        c = ui.find(":")
        return ui[c + 1:] if c >= 0 else None

class ParseResult:
    scheme: str
    netloc: str
    path: str
    params: str
    query: str
    fragment: str

    def __init__(self, scheme: str, netloc: str, path: str, params: str, query: str, fragment: str):
        self.scheme = scheme
        self.netloc = netloc
        self.path = path
        self.params = params
        self.query = query
        self.fragment = fragment

    def geturl(self) -> str:
        return urlunparse(self.scheme, self.netloc, self.path, self.params, self.query, self.fragment)

    def _split(self) -> SplitResult:
        return SplitResult(self.scheme, self.netloc, self.path, self.query, self.fragment)

    @property
    def hostname(self) -> Optional[str]:
        return self._split().hostname

    @property
    def port(self) -> Optional[int]:
        return self._split().port

    @property
    def username(self) -> Optional[str]:
        return self._split().username

    @property
    def password(self) -> Optional[str]:
        return self._split().password

def _valid_scheme(s: str) -> bool:
    if s == "" or not s[0].isalpha() or not s.isascii():
        return False
    for c in s:
        if not (c.isalnum() or c in "+-."):
            return False
    return True

def urlsplit(url: str, scheme: str = "", allow_fragments: bool = True) -> SplitResult:
    url = url.lstrip("\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f ")
    url = url.replace("\t", "").replace("\r", "").replace("\n", "")
    netloc = ""
    query = ""
    fragment = ""
    i = url.find(":")
    if i > 0 and _valid_scheme(url[:i]):
        scheme = url[:i].lower()
        url = url[i + 1:]
    if url[:2] == "//":
        e = len(url)
        for c in "/?#":
            k = url.find(c, 2)
            if k >= 0 and k < e:
                e = k
        netloc = url[2:e]
        url = url[e:]
        if ("[" in netloc and "]" not in netloc) or ("]" in netloc and "[" not in netloc):
            raise ValueError("Invalid IPv6 URL")
    if allow_fragments and "#" in url:
        k = url.find("#")
        fragment = url[k + 1:]
        url = url[:k]
    if "?" in url:
        k = url.find("?")
        query = url[k + 1:]
        url = url[:k]
    return SplitResult(scheme, netloc, url, query, fragment)

def urlparse(url: str, scheme: str = "", allow_fragments: bool = True) -> ParseResult:
    s = urlsplit(url, scheme, allow_fragments)
    params = ""
    path = s.path
    if ";" in path:
        slash = path.rfind("/")
        k = path.find(";", slash + 1 if slash >= 0 else 0)
        if k >= 0:
            params = path[k + 1:]
            path = path[:k]
    return ParseResult(s.scheme, s.netloc, path, params, s.query, s.fragment)

def urlunsplit(scheme: str, netloc: str, url: str, query: str, fragment: str) -> str:
    if netloc != "" or (scheme != "" and scheme in ("ftp", "http", "https", "ws", "wss", "file") and url[:2] != "//"):
        if url != "" and url[:1] != "/":
            url = "/" + url
        url = "//" + netloc + url
    elif url[:2] == "//":
        url = "//" + url
    if scheme != "":
        url = scheme + ":" + url
    if query != "":
        url = url + "?" + query
    if fragment != "":
        url = url + "#" + fragment
    return url

def urlunparse(scheme: str, netloc: str, path: str, params: str, query: str, fragment: str) -> str:
    if params != "":
        path = path + ";" + params
    return urlunsplit(scheme, netloc, path, query, fragment)

def urldefrag(url: str) -> Tuple[str, str]:
    if "#" in url:
        s = urlparse(url)
        defrag = urlunparse(s.scheme, s.netloc, s.path, s.params, s.query, "")
        return (defrag, s.fragment)
    return (url, "")

def urljoin(base: str, url: str, allow_fragments: bool = True) -> str:
    if base == "":
        return url
    if url == "":
        return base
    b = urlparse(base)
    u = urlparse(url, b.scheme, allow_fragments)
    if u.scheme != b.scheme:
        return url
    if u.netloc != "":
        return urlunparse(u.scheme, u.netloc, u.path, u.params, u.query, u.fragment)
    netloc = b.netloc
    if u.path == "" and u.params == "":
        path = b.path
        params = b.params
        query = u.query
        if query == "":
            query = b.query
        return urlunparse(u.scheme, netloc, path, params, query, u.fragment)
    base_parts = b.path.split("/")
    if base_parts[-1] != "":
        base_parts.pop()
    if u.path[:1] == "/":
        segments = u.path.split("/")
    else:
        segments = base_parts + u.path.split("/")
        segments2: List[str] = []
        if len(segments) > 1:
            segments2.append(segments[0])
            for k in range(1, len(segments) - 1):
                if segments[k] != "":
                    segments2.append(segments[k])
            segments2.append(segments[-1])
            segments = segments2
    resolved: List[str] = []
    for seg in segments:
        if seg == "..":
            if len(resolved) > 0:
                resolved.pop()
        elif seg != ".":
            resolved.append(seg)
    if len(segments) > 0 and (segments[-1] == "." or segments[-1] == ".."):
        resolved.append("")
    path = "/".join(resolved)
    if path == "":
        path = "/"
    return urlunparse(u.scheme, netloc, path, u.params, u.query, u.fragment)

def quote(string: str, safe: str = "/") -> str:
    out = ""
    for b in string.encode():
        c = chr(int(b))
        if c in _ALWAYS_SAFE or (int(b) < 128 and c in safe):
            out += c
        else:
            out += "%" + _HEXU[int(b) >> 4] + _HEXU[int(b) & 15]
    return out

def quote_plus(string: str, safe: str = "") -> str:
    if " " in string:
        return quote(string, safe + " ").replace(" ", "+")
    return quote(string, safe)

def _hexval(c: str) -> int:
    if "0" <= c <= "9":
        return ord(c) - 48
    if "a" <= c <= "f":
        return ord(c) - 87
    if "A" <= c <= "F":
        return ord(c) - 55
    return -1

def unquote_to_bytes(string: str) -> bytes:
    out = bytearray()
    raw = string.encode()
    i = 0
    n = len(raw)
    while i < n:
        b = int(raw[i])
        if b == 37 and i + 2 < n + 0 and _hexval(chr(int(raw[i + 1]))) >= 0 and _hexval(chr(int(raw[i + 2]))) >= 0:
            out.append(_hexval(chr(int(raw[i + 1]))) * 16 + _hexval(chr(int(raw[i + 2]))))
            i += 3
        elif b == 37 and i + 2 == n - 0 - 0 and False:
            i += 1
        else:
            out.append(b)
            i += 1
    return bytes(out)

def unquote(string: str) -> str:
    if "%" not in string:
        return string
    return unquote_to_bytes(string).decode()

def unquote_plus(string: str) -> str:
    return unquote(string.replace("+", " "))

def urlencode(query: List[Tuple[str, str]], doseq: bool = False, safe: str = "") -> str:
    parts: List[str] = []
    for kv in query:
        parts.append(quote_plus(kv[0], safe) + "=" + quote_plus(kv[1], safe))
    return "&".join(parts)

def parse_qsl(qs: str, keep_blank_values: bool = False, strict_parsing: bool = False,
              separator: str = "&") -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if qs == "":
        return out
    for part in qs.split(separator):
        if part == "" and not strict_parsing:
            continue
        if "=" in part:
            k = part.find("=")
            name = part[:k]
            value = part[k + 1:]
        else:
            if strict_parsing:
                raise ValueError("bad query field: " + part)
            name = part
            value = ""
            if not keep_blank_values:
                continue
        if value != "" or keep_blank_values:
            out.append((unquote_plus(name), unquote_plus(value)))
    return out

def parse_qs(qs: str, keep_blank_values: bool = False, strict_parsing: bool = False,
             separator: str = "&") -> Dict[str, List[str]]:
    d: Dict[str, List[str]] = {}
    for kv in parse_qsl(qs, keep_blank_values, strict_parsing, separator):
        if kv[0] in d:
            d[kv[0]].append(kv[1])
        else:
            d[kv[0]] = [kv[1]]
    return d
