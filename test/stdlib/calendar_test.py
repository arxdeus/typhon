# Ported from CPython 3.14 calendar (PSF)
import calendar

@test
def test_calendar():
    assert calendar.isleap(2000) and not calendar.isleap(1900) and calendar.isleap(2024)
    assert calendar.leapdays(2000, 2025) == 7
    assert calendar.weekday(2026, 11, 13) == 4
    assert calendar.weekday(2000, 1, 1) == 5
    assert calendar.monthrange(2024, 2) == (3, 29)
    assert calendar.monthrange(2026, 11) == (6, 30)
    mc = calendar.monthcalendar(2026, 11)
    assert mc[0] == [0, 0, 0, 0, 0, 0, 1] and mc[-1] == [30, 0, 0, 0, 0, 0, 0]
    assert calendar.timegm((2001, 9, 9, 1, 46, 40)) == 1000000000

test_calendar()
