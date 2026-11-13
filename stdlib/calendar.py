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

class TextCalendar:
    firstweekday: int

    def __init__(self, firstweekday: int = 0):
        self.firstweekday = firstweekday

    def monthdayscalendar(self, year: int, month: int) -> List[List[int]]:
        r = monthrange(year, month)
        first = (r[0] - self.firstweekday) % 7
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

    def formatday(self, day: int, weekday: int, width: int) -> str:
        if day == 0:
            return " " * width
        return str(day).rjust(width)

    def formatweek(self, theweek: List[int], width: int) -> str:
        cells = List[str]()
        for i in range(len(theweek)):
            cells.append(self.formatday(theweek[i], i, width))
        return " ".join(cells)

    def formatweekday(self, day: int, width: int) -> str:
        name = day_name[day] if width >= 9 else day_abbr[day]
        return name[:width].center(width)

    def formatweekheader(self, width: int) -> str:
        names = List[str]()
        for i in range(7):
            names.append(self.formatweekday((i + self.firstweekday) % 7, width))
        return " ".join(names)

    def formatmonthname(self, theyear: int, themonth: int, width: int,
                        withyear: bool = True) -> str:
        s = month_name[themonth]
        if withyear:
            s = s + " " + str(theyear)
        return s.center(width)

    def formatmonth(self, theyear: int, themonth: int, w: int = 0, l: int = 0) -> str:
        w = max(2, w)
        l = max(1, l)
        s = self.formatmonthname(theyear, themonth, 7 * (w + 1) - 1).rstrip()
        s += "\n" * l
        s += self.formatweekheader(w).rstrip()
        s += "\n" * l
        for week in self.monthdayscalendar(theyear, themonth):
            s += self.formatweek(week, w).rstrip()
            s += "\n" * l
        return s

    def prmonth(self, theyear: int, themonth: int, w: int = 0, l: int = 0):
        print(self.formatmonth(theyear, themonth, w, l), end="")

    def formatyear(self, theyear: int, w: int = 2, l: int = 1, c: int = 6, m: int = 3) -> str:
        w = max(2, w)
        l = max(1, l)
        c = max(2, c)
        colwidth = (w + 1) * 7 - 1
        v = List[str]()
        v.append(str(theyear).center(colwidth * m + c * (m - 1)).rstrip())
        v.append("\n" * l)
        header = self.formatweekheader(w)
        for row_start in range(0, 12, m):
            months = List[int]()
            for k in range(row_start + 1, min(row_start + m, 12) + 1):
                months.append(k)
            v.append("\n" * l)
            names = List[str]()
            for k in months:
                names.append(self.formatmonthname(theyear, k, colwidth, False))
            v.append((" " * c).join(names).rstrip())
            v.append("\n" * l)
            heads = List[str]()
            for k in months:
                heads.append(header)
            v.append((" " * c).join(heads).rstrip())
            v.append("\n" * l)
            cals = List[List[List[int]]]()
            height = 0
            for k in months:
                cal = self.monthdayscalendar(theyear, k)
                cals.append(cal)
                height = max(height, len(cal))
            for j in range(height):
                weeks = List[str]()
                for cal in cals:
                    if j >= len(cal):
                        weeks.append(" " * colwidth)
                    else:
                        weeks.append(self.formatweek(cal[j], w))
                v.append((" " * c).join(weeks).rstrip())
                v.append("\n" * l)
        return "".join(v)

    def pryear(self, theyear: int, w: int = 0, l: int = 0, c: int = 6, m: int = 3):
        print(self.formatyear(theyear, w, l, c, m), end="")

def month(theyear: int, themonth: int, w: int = 0, l: int = 0) -> str:
    return TextCalendar(_firstweekday).formatmonth(theyear, themonth, w, l)

def prmonth(theyear: int, themonth: int, w: int = 0, l: int = 0):
    print(month(theyear, themonth, w, l), end="")

def calendar(theyear: int, w: int = 0, l: int = 0, c: int = 6, m: int = 3) -> str:
    return TextCalendar(_firstweekday).formatyear(theyear, w, l, c, m)

def prcal(theyear: int, w: int = 0, l: int = 0, c: int = 6, m: int = 3):
    print(calendar(theyear, w, l, c, m), end="")

def weekheader(n: int) -> str:
    return TextCalendar(_firstweekday).formatweekheader(n)

def monthdayscalendar(year: int, month: int) -> List[List[int]]:
    return monthcalendar(year, month)
