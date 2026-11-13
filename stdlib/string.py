
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
