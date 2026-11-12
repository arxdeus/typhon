import textwrap
import base64
import binascii

T = "The quick brown fox jumps over the lazy dog. Pneumonoultramicroscopicsilico is long, well-known text."

@test
def test_textwrap():
    assert textwrap.wrap(T, 20) == ['The quick brown fox', 'jumps over the lazy', 'dog. Pneumonoultrami', 'croscopicsilico is', 'long, well-known', 'text.']
    assert textwrap.fill("a b c d", 3) == "a b\nc d"
    assert textwrap.shorten("Hello  world! this is long", 12) == "Hello [...]"
    assert textwrap.shorten("Hello world", 20) == "Hello world"
    assert textwrap.dedent("   a\n    b\n\n   c\n") == "a\n b\n\nc\n"
    assert textwrap.indent("a\n\nb\n", "+ ") == "+ a\n\n+ b\n"
    assert textwrap.wrap("abcdefgh", 3, break_long_words=False) == ["abcdefgh"]
test_textwrap()

@test
def test_base64():
    assert base64.b64encode(b"hello world") == b"aGVsbG8gd29ybGQ="
    assert base64.b64encode(b"a") == b"YQ=="
    assert base64.b64encode(b"") == b""
    assert base64.b64decode(b"aGVsbG8gd29ybGQ=") == b"hello world"
    assert base64.urlsafe_b64encode(b"\xfb\xff\xfe") == b"-__-"
    assert base64.urlsafe_b64decode(b"-__-") == b"\xfb\xff\xfe"
    assert base64.b32encode(b"hello") == b"NBSWY3DP"
    assert base64.b32encode(b"hi") == b"NBUQ===="
    assert base64.b32decode(b"NBUQ====") == b"hi"
    assert base64.b16encode(b"\x01\xab") == b"01AB"
    assert base64.b16decode(b"01AB") == b"\x01\xab"
    assert base64.encodebytes(b"hi") == b"aGk=\n"
    try:
        base64.b64decode(b"abc")
        assert False
    except binascii.Error:
        pass

test_base64()

@test
def test_binascii():
    assert binascii.hexlify(b"\x00\xffab") == b"00ff6162"
    assert binascii.unhexlify(b"00FF6162") == b"\x00\xffab"
    assert binascii.crc32(b"hello") == 907060870
    try:
        binascii.unhexlify(b"abc")
        assert False
    except binascii.Error:
        pass
test_binascii()
