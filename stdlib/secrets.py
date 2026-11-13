# Ported from CPython 3.14 secrets (PSF license)
import os
import base64
from hmac import compare_digest

DEFAULT_ENTROPY = 32

def token_bytes(nbytes: int = DEFAULT_ENTROPY) -> bytes:
    return os.urandom(nbytes)

def token_hex(nbytes: int = DEFAULT_ENTROPY) -> str:
    return token_bytes(nbytes).hex()

def token_urlsafe(nbytes: int = DEFAULT_ENTROPY) -> str:
    tok = token_bytes(nbytes)
    s = base64.urlsafe_b64encode(tok).decode("ascii")
    while s.endswith("="):
        s = s[:-1]
    return s

def randbits(k: int) -> int:
    if k == 0:
        return 0
    n = (k + 7) // 8
    b = token_bytes(n)
    v = 0
    for i in range(n):
        v = (v << 8) | int(b[i])
    return v >> (n * 8 - k)

def randbelow(exclusive_upper_bound: int) -> int:
    if exclusive_upper_bound <= 0:
        raise ValueError("Upper bound must be positive.")
    k = exclusive_upper_bound.bit_length()
    r = randbits(k)
    while r >= exclusive_upper_bound:
        r = randbits(k)
    return r

def choice(seq):
    if len(seq) == 0:
        raise IndexError("Cannot choose from an empty sequence")
    return seq[randbelow(len(seq))]
