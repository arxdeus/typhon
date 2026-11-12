# Ported from CPython 3.14 (PSF-2.0)

def quote(s: str) -> str:
    if s == "":
        return "''"
    safe = True
    for c in s:
        if not (("a" <= c <= "z") or ("A" <= c <= "Z") or ("0" <= c <= "9") or c in "@%+=:,./-_"):
            safe = False
            break
    if safe:
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"

def join(split_command: List[str]) -> str:
    return " ".join([quote(a) for a in split_command])

def split(s: str, comments: bool = False, posix: bool = True) -> List[str]:
    out: List[str] = []
    cur = ""
    have = False
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == " " or c == "\t" or c == "\n" or c == "\r":
            if have:
                out.append(cur)
                cur = ""
                have = False
            i += 1
        elif c == "#" and comments and not have:
            while i < n and s[i] != "\n":
                i += 1
        elif c == "'":
            i += 1
            have = True
            while i < n and s[i] != "'":
                cur += s[i]
                i += 1
            if i >= n:
                raise ValueError("No closing quotation")
            i += 1
        elif c == '"':
            i += 1
            have = True
            while i < n and s[i] != '"':
                if s[i] == "\\" and i + 1 < n and s[i + 1] in '\\"':
                    i += 1
                cur += s[i]
                i += 1
            if i >= n:
                raise ValueError("No closing quotation")
            i += 1
        elif c == "\\":
            i += 1
            if i >= n:
                raise ValueError("No escaped character")
            cur += s[i]
            have = True
            i += 1
        else:
            cur += c
            have = True
            i += 1
    if have:
        out.append(cur)
    return out
