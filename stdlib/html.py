# Ported from CPython 3.14 (PSF-2.0)

_ENT = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'", "nbsp": "\u00a0",
        "copy": "\u00a9", "reg": "\u00ae", "hellip": "\u2026", "mdash": "\u2014",
        "ndash": "\u2013", "euro": "\u20ac", "eacute": "\u00e9", "agrave": "\u00e0"}

def escape(s: str, quote: bool = True) -> str:
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if quote:
        s = s.replace('"', "&quot;").replace("'", "&#x27;")
    return s

def unescape(s: str) -> str:
    if "&" not in s:
        return s
    out = ""
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c != "&":
            out += c
            i += 1
            continue
        j = i + 1
        if j < n and s[j] == "#":
            j += 1
            base = 10
            if j < n and (s[j] == "x" or s[j] == "X"):
                base = 16
                j += 1
            k = j
            while k < n and (s[k].isdigit() or (base == 16 and s[k] in "abcdefABCDEF")):
                k += 1
            if k > j:
                num = int(s[j:k], base)
                if k < n and s[k] == ";":
                    k += 1
                if num == 0 or num > 0x10FFFF or (0xD800 <= num <= 0xDFFF):
                    out += "\ufffd"
                elif num == 0x80:
                    out += "\u20ac"
                else:
                    out += chr(num)
                i = k
                continue
        else:
            k = j
            while k < n and s[k].isalnum():
                k += 1
            name = s[j:k]
            if k < n and s[k] == ";" and name in _ENT:
                out += _ENT[name]
                i = k + 1
                continue
            if name in ("amp", "lt", "gt", "quot"):
                out += _ENT[name]
                i = k
                continue
        out += "&"
        i += 1
    return out
