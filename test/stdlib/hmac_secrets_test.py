# Ported from CPython 3.14 hmac/secrets (PSF)
import hmac
import secrets

@test
def test_hmac():
    assert hmac.new(b"key", b"The quick brown fox jumps over the lazy dog", "sha256").hexdigest() == "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"
    assert hmac.new(b"key", b"The quick brown fox jumps over the lazy dog", "md5").hexdigest() == "80070713463e7749b90c2dc24911e275"
    assert hmac.digest(b"k" * 200, b"msg", "sha1").hex() == "b2e64fb68e7096463a9d23749f90679809fdb8fa"
    h = hmac.new(b"key", b"", "sha256")
    h.update(b"The quick brown fox ")
    h.update(b"jumps over the lazy dog")
    assert h.hexdigest() == "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"
    assert hmac.compare_digest(b"abc", b"abc")
    assert not hmac.compare_digest(b"abc", b"abd")

@test
def test_secrets():
    assert len(secrets.token_bytes(16)) == 16
    assert len(secrets.token_hex(8)) == 16
    for _ in range(50):
        assert 0 <= secrets.randbelow(10) < 10
        assert 0 <= secrets.randbits(5) < 32
    assert secrets.choice([7]) == 7

test_hmac()
test_secrets()
