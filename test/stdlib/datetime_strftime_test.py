# Expected values from CPython 3.14
from datetime import datetime, date, time

@test
def test_strftime():
    d = datetime(2024, 3, 5, 7, 8, 9, 123456)
    assert d.strftime("%Y-%m-%d %H:%M:%S.%f %a %A %b %B %j %y %I %p %U %W %w %% %F %T %G-%V") == "2024-03-05 07:08:09.123456 Tue Tuesday Mar March 065 24 07 AM 09 10 2 % 2024-03-05 07:08:09 2024-10"
    assert date(2024, 3, 5).strftime("%d.%m.%Y") == "05.03.2024"
    assert f"{d:%Y/%m/%d}" == "2024/03/05"

@test
def test_strptime():
    assert str(datetime.strptime("2024-03-05 07:08:09", "%Y-%m-%d %H:%M:%S")) == "2024-03-05 07:08:09"
    assert str(datetime.strptime("05/Mar/24 07PM", "%d/%b/%y %I%p")) == "2024-03-05 19:00:00"
    assert str(datetime.strptime("Tuesday, March 5 2024 12:00:01.5", "%A, %B %d %Y %H:%M:%S.%f")) == "2024-03-05 12:00:01.500000"

@test
def test_iso():
    d = datetime(2024, 3, 5, 7, 8, 9, 123456)
    assert d.isoformat(sep=' ', timespec='milliseconds') == "2024-03-05 07:08:09.123"
    assert str(time.fromisoformat("01:02:03.5")) == "01:02:03.500000"
    assert d.ctime() == "Tue Mar  5 07:08:09 2024"
