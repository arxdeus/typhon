# Ported from CPython 3.14 calendar (PSF license), core subset

MONDAY = 0
TUESDAY = 1
WEDNESDAY = 2
THURSDAY = 3
FRIDAY = 4
SATURDAY = 5
SUNDAY = 6

mdays = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
day_abbr = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
month_name = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
month_abbr = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

class IllegalMonthError(ValueError):
    pass

_firstweekday = 0

def isleap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def leapdays(y1: int, y2: int) -> int:
    y1 -= 1
    y2 -= 1
    return (y2 // 4 - y1 // 4) - (y2 // 100 - y1 // 100) + (y2 // 400 - y1 // 400)

def _days_from_civil(y: int, m: int, d: int) -> int:
    y -= 1 if m <= 2 else 0
    era = y // 400
    yoe = y - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468

def weekday(year: int, month: int, day: int) -> int:
    # 1970-01-01 is a Thursday (3)
    return (_days_from_civil(year, month, day) + 3) % 7

def monthrange(year: int, month: int) -> Tuple[int, int]:
    if month < 1 or month > 12:
        raise IllegalMonthError(f"bad month number {month}; must be 1-12")
    ndays = mdays[month] + (1 if month == 2 and isleap(year) else 0)
    return (weekday(year, month, 1), ndays)

def setfirstweekday(fw: int):
    global _firstweekday
    _firstweekday = fw

def firstweekday() -> int:
    return _firstweekday

def monthcalendar(year: int, month: int) -> List[List[int]]:
    r = monthrange(year, month)
    first = (r[0] - _firstweekday) % 7
    out = List[List[int]]()
    row = [0] * first
    for d in range(1, r[1] + 1):
        row.append(d)
        if len(row) == 7:
            out.append(row)
            row = []
    if len(row) > 0:
        while len(row) < 7:
            row.append(0)
        out.append(row)
    return out

def timegm(t: Tuple[int, int, int, int, int, int]) -> int:
    days = _days_from_civil(t[0], t[1], t[2])
    return days * 86400 + t[3] * 3600 + t[4] * 60 + t[5]

def month(theyear: int, themonth: int, w: int = 0, l: int = 0) -> str:
    w = max(2, w)
    l = max(1, l)
    s = (month_name[themonth] + " " + str(theyear)).center(7 * (w + 1) - 1).rstrip() + "\n" * l
    names = List[str]()
    for i in range(7):
        names.append(day_abbr[(i + _firstweekday) % 7][:w].center(w))
    s += " ".join(names).rstrip() + "\n" * l
    for week in monthcalendar(theyear, themonth):
        cells = List[str]()
        for d in week:
            cells.append(("" if d == 0 else str(d)).rjust(w).center(w) if False else ("" if d == 0 else str(d)).center(w))
        s += " ".join(cells).rstrip() + "\n" * l
    return s
