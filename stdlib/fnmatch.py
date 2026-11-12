# Ported from CPython 3.14 (PSF-2.0)

def _match(name: str, pat: str) -> bool:
    n = len(name)
    m = len(pat)
    i = 0
    j = 0
    star_j = -1
    star_i = 0
    while i < n:
        if j < m and pat[j] == "*":
            star_j = j
            j += 1
            star_i = i
            continue
        if j < m and pat[j] == "[":
            k = j + 1
            if k < m and pat[k] == "!":
                k += 1
            if k < m and pat[k] == "]":
                k += 1
            while k < m and pat[k] != "]":
                k += 1
            if k < m:
                body = pat[j + 1:k]
                neg = False
                if len(body) > 0 and body[0] == "!":
                    neg = True
                    body = body[1:]
                found = False
                p = 0
                c = name[i]
                while p < len(body):
                    if p + 2 < len(body) and body[p + 1] == "-":
                        if body[p] <= c <= body[p + 2]:
                            found = True
                        p += 3
                    else:
                        if body[p] == c:
                            found = True
                        p += 1
                if found != neg:
                    i += 1
                    j = k + 1
                    continue
            elif name[i] == "[":
                i += 1
                j += 1
                continue
        elif j < m and (pat[j] == "?" or pat[j] == name[i]):
            i += 1
            j += 1
            continue
        if star_j >= 0:
            j = star_j + 1
            star_i += 1
            i = star_i
        else:
            return False
    while j < m and pat[j] == "*":
        j += 1
    return j == m

def fnmatchcase(name: str, pat: str) -> bool:
    return _match(name, pat)

def fnmatch(name: str, pat: str) -> bool:
    return _match(name.lower(), pat.lower())

def filter(names: List[str], pat: str) -> List[str]:
    return [n for n in names if _match(n, pat)]

def translate(pat: str) -> str:
    res = ""
    i = 0
    n = len(pat)
    while i < n:
        c = pat[i]
        i += 1
        if c == "*":
            while i < n and pat[i] == "*":
                i += 1
            res += ".*"
        elif c == "?":
            res += "."
        elif c == "[":
            j = i
            if j < n and pat[j] == "!":
                j += 1
            if j < n and pat[j] == "]":
                j += 1
            while j < n and pat[j] != "]":
                j += 1
            if j >= n:
                res += "\\["
            else:
                body = pat[i:j].replace("\\", "\\\\")
                i = j + 1
                if body[0] == "!":
                    body = "^" + body[1:]
                elif body[0] == "^":
                    body = "\\" + body
                res += "[" + body + "]"
        else:
            if c.isalnum() or c == "_":
                res += c
            else:
                res += "\\" + c
    return "(?s:" + res + ")\\Z"
