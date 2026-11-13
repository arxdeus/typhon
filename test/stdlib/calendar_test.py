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


@test
def test_text_calendar():
    import calendar
    assert calendar.TextCalendar(6).formatmonth(2024, 2, 3, 2) == '       February 2024\n\nSun Mon Tue Wed Thu Fri Sat\n\n                  1   2   3\n\n  4   5   6   7   8   9  10\n\n 11  12  13  14  15  16  17\n\n 18  19  20  21  22  23  24\n\n 25  26  27  28  29\n\n'
    assert calendar.weekheader(3) == "Mon Tue Wed Thu Fri Sat Sun"
    assert calendar.calendar(2024).startswith("                                  2024\n\n      January")
