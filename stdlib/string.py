
ascii_letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
ascii_lowercase = "abcdefghijklmnopqrstuvwxyz"
ascii_uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

digits = "0123456789"
hexdigits = "0123456789abcdefABCDEF"
octdigits = "01234567"

punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
printable = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c"
whitespace = " \t\n\r\x0b\x0c"

def capwords(s: str, sep: str = "") -> str:
    parts = s.split() if sep == "" else s.split(sep)
    out = List[str]()
    for w in parts:
        out.append(w.capitalize())
    return (" " if sep == "" else sep).join(out)


def _is_idstart(c: str) -> bool:
    return ("a" <= c <= "z") or ("A" <= c <= "Z") or c == "_"


def _is_idchar(c: str) -> bool:
    return _is_idstart(c) or ("0" <= c <= "9")


class Template:
    template: str
    delimiter: str

    def __init__(self, template: str):
        self.template = template
        self.delimiter = "$"

    def _line_info(self, pos: int) -> Tuple[int, int]:
        lineno = 1
        last = 0
        for i in range(pos):
            if self.template[i] == "\n":
                lineno += 1
                last = i + 1
        return lineno, pos - last + 1

    def _invalid(self, pos: int):
        lineno, colno = self._line_info(pos)
        raise ValueError(f"Invalid placeholder in string: line {lineno}, col {colno}")

    def _expand(self, mapping: Dict[str, str], safe: bool) -> str:
        t = self.template
        n = len(t)
        out = List[str]()
        i = 0
        while i < n:
            c = t[i]
            if c != "$":
                out.append(c)
                i += 1
                continue
            if i + 1 >= n:
                if safe:
                    out.append("$")
                    i += 1
                    continue
                self._invalid(i)
            d = t[i + 1]
            if d == "$":
                out.append("$")
                i += 2
            elif _is_idstart(d):
                j = i + 1
                while j < n and _is_idchar(t[j]):
                    j += 1
                name = t[i + 1:j]
                if name in mapping:
                    out.append(mapping[name])
                elif safe:
                    out.append(t[i:j])
                else:
                    raise KeyError(name)
                i = j
            elif d == "{":
                j = i + 2
                if j < n and _is_idstart(t[j]):
                    while j < n and _is_idchar(t[j]):
                        j += 1
                if j < n and t[j] == "}" and j > i + 2:
                    name = t[i + 2:j]
                    if name in mapping:
                        out.append(mapping[name])
                    elif safe:
                        out.append(t[i:j + 1])
                    else:
                        raise KeyError(name)
                    i = j + 1
                elif safe:
                    out.append("$")
                    i += 1
                else:
                    self._invalid(i)
            else:
                if safe:
                    out.append("$")
                    i += 1
                else:
                    self._invalid(i)
        return "".join(out)

    def substitute(self, mapping: Dict[str, str]) -> str:
        return self._expand(mapping, False)

    def safe_substitute(self, mapping: Dict[str, str]) -> str:
        return self._expand(mapping, True)

    def substitute(self, **kwargs) -> str:
        return self._expand(Template._to_dict(kwargs), False)

    def safe_substitute(self, **kwargs) -> str:
        return self._expand(Template._to_dict(kwargs), True)

    def _to_dict(kwargs) -> Dict[str, str]:
        d = Dict[str, str]()
        keys = kwargs.__keys__()
        for index in static.range(static.len(kwargs.args)):
            d[keys[index]] = str(kwargs.args[index])
        return d

    def is_valid(self) -> bool:
        t = self.template
        n = len(t)
        i = 0
        while i < n:
            if t[i] != "$":
                i += 1
                continue
            if i + 1 >= n:
                return False
            d = t[i + 1]
            if d == "$":
                i += 2
            elif _is_idstart(d):
                j = i + 1
                while j < n and _is_idchar(t[j]):
                    j += 1
                i = j
            elif d == "{":
                j = i + 2
                if j < n and _is_idstart(t[j]):
                    while j < n and _is_idchar(t[j]):
                        j += 1
                    if j < n and t[j] == "}":
                        i = j + 1
                        continue
                return False
            else:
                return False
        return True

    def get_identifiers(self) -> List[str]:
        t = self.template
        n = len(t)
        out = List[str]()
        i = 0
        while i < n:
            if t[i] != "$" or i + 1 >= n:
                i += 1
                continue
            d = t[i + 1]
            name = ""
            if d == "$":
                i += 2
                continue
            elif _is_idstart(d):
                j = i + 1
                while j < n and _is_idchar(t[j]):
                    j += 1
                name = t[i + 1:j]
                i = j
            elif d == "{" and i + 2 < n and _is_idstart(t[i + 2]):
                j = i + 2
                while j < n and _is_idchar(t[j]):
                    j += 1
                if j < n and t[j] == "}":
                    name = t[i + 2:j]
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1
            if name != "" and name not in out:
                out.append(name)
        return out
