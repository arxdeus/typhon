from urllib.parse import *

@test
def test_urlparse():
    p = urlparse("http://user:pw@Example.com:8080/a/b;x=1?q=1&r=2#frag")
    assert p.scheme == "http" and p.netloc == "user:pw@Example.com:8080"
    assert p.path == "/a/b" and p.params == "x=1" and p.query == "q=1&r=2" and p.fragment == "frag"
    assert p.hostname == "example.com" and p.port == 8080
    assert p.username == "user" and p.password == "pw"
    assert p.geturl() == "http://user:pw@Example.com:8080/a/b;x=1?q=1&r=2#frag"
    s = urlsplit("mailto:a@b.c")
    assert s.scheme == "mailto" and s.path == "a@b.c" and s.netloc == ""
    assert urlsplit("//host/p").netloc == "host"
    assert urlsplit("/just/path?x#y").path == "/just/path"
    assert urlunsplit("https", "h", "p", "", "") == "https://h/p"
    assert urldefrag("http://a/b#c") == ("http://a/b", "c")
    assert urlsplit("http://h:80").port == 80
    assert urlsplit("http://h").port is None
test_urlparse()

@test
def test_urljoin():
    b = "http://a/b/c/d;p?q"
    assert urljoin(b, "g") == "http://a/b/c/g"
    assert urljoin(b, "./g") == "http://a/b/c/g"
    assert urljoin(b, "g/") == "http://a/b/c/g/"
    assert urljoin(b, "/g") == "http://a/g"
    assert urljoin(b, "//g") == "http://g"
    assert urljoin(b, "?y") == "http://a/b/c/d;p?y"
    assert urljoin(b, "g?y#s") == "http://a/b/c/g?y#s"
    assert urljoin(b, "..") == "http://a/b/"
    assert urljoin(b, "../g") == "http://a/b/g"
    assert urljoin(b, "../../g") == "http://a/g"
    assert urljoin(b, "#s") == "http://a/b/c/d;p?q#s"
    assert urljoin(b, "https://x/y") == "https://x/y"
test_urljoin()

@test
def test_quote():
    assert quote("a b/c&d~é") == "a%20b/c%26d~%C3%A9"
    assert quote("a/b", safe="") == "a%2Fb"
    assert quote_plus("a b&c") == "a+b%26c"
    assert unquote("a%20b%C3%A9%zz%4") == "a bé%zz%4"
    assert unquote_plus("a+b%26") == "a b&"
    assert urlencode([("a", "1 2"), ("b", "x&y")]) == "a=1+2&b=x%26y"
    assert parse_qsl("a=1&b=2&a=3&c=") == [("a", "1"), ("b", "2"), ("a", "3")]
    assert parse_qsl("c=&d", keep_blank_values=True) == [("c", ""), ("d", "")]
    assert parse_qs("a=1&b=2&a=3") == {"a": ["1", "3"], "b": ["2"]}
test_quote()
