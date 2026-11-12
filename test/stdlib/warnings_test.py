import warnings


@test
def test_record_and_filters():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.warn("first")
        warnings.warn("second", warnings.DeprecationWarning)
        warnings.warn("third", warnings.RuntimeWarning)
        assert w is not None
        log = w if w is not None else List[warnings.WarningMessage]()
        assert len(log) == 3
        assert str(log[0].message) == "first"
        assert log[0].category_name == "UserWarning"
        assert log[1].category_name == "DeprecationWarning"
        assert isinstance(log[2].message, warnings.RuntimeWarning)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("ignore")
        warnings.warn("hidden")
        assert len(w if w is not None else List[warnings.WarningMessage]()) == 0

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("default")
        warnings.warn("dup")
        warnings.warn("dup")
        warnings.warn("other")
        assert len(w if w is not None else List[warnings.WarningMessage]()) == 2

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message="skip", category=warnings.UserWarning)
        warnings.warn("Skip me")
        warnings.warn("keep me")
        log = w if w is not None else List[warnings.WarningMessage]()
        assert len(log) == 1 and str(log[0].message) == "keep me"
test_record_and_filters()


@test
def test_error():
    with warnings.catch_warnings():
        warnings.simplefilter("error", warnings.UserWarning)
        caught = False
        try:
            warnings.warn("boom")
        except warnings.UserWarning as e:
            caught = str(e) == "boom"
        assert caught
        # other categories still pass through the default path
        warnings.simplefilter("ignore", warnings.RuntimeWarning)
        warnings.warn("quiet", warnings.RuntimeWarning)
    try:
        warnings.simplefilter("bogus")
        assert False
    except ValueError as e:
        assert str(e) == "invalid action: 'bogus'"
test_error()


@test
def test_format():
    assert warnings.formatwarning("msg", "UserWarning", "f.py", 3) == "f.py:3: UserWarning: msg\n"
    assert warnings.formatwarning("m", "DeprecationWarning", "a.py", 7, "  x = 1  ") == "a.py:7: DeprecationWarning: m\n  x = 1\n"
test_format()
