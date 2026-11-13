# Copyright (C) 2026 the Typhon authors
# Ported from pandas 3.1.1 (BSD-3-Clause)

# Datetime support: calendar arithmetic on int64 nanoseconds since the Unix
# epoch (naive timestamps), Timestamp / Timedelta scalars, offset aliases and
# date_range generation.

NS_SEC: Literal[int] = 1000000000
NS_MIN: Literal[int] = 60000000000
NS_HOUR: Literal[int] = 3600000000000
NS_DAY: Literal[int] = 86400000000000
NAT: Literal[int] = -9223372036854775807 - 1

_DIM = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December"]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

def is_leap(y: int) -> bool:
    return (y % 4 == 0 and y % 100 != 0) or y % 400 == 0

def days_in_month(y: int, m: int) -> int:
    if m == 2 and is_leap(y):
        return 29
    return _DIM[m - 1]

def days_from_civil(y0: int, m: int, d: int) -> int:
    y = y0 - 1 if m <= 2 else y0
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    mp = m - 3 if m > 2 else m + 9
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468

def civil_from_days(z0: int) -> Tuple[int, int, int]:
    z = z0 + 719468
    era = (z if z >= 0 else z - 146096) // 146097
    doe = z - era * 146097
    yoe = (doe - doe // 1460 + doe // 36524 - doe // 146096) // 365
    y = yoe + era * 400
    doy = doe - (365 * yoe + yoe // 4 - yoe // 100)
    mp = (5 * doy + 2) // 153
    d = doy - (153 * mp + 2) // 5 + 1
    m = mp + 3 if mp < 10 else mp - 9
    return (y + 1 if m <= 2 else y, m, d)

def make_ns(y: int, mo: int, d: int, h: int = 0, mi: int = 0, s: int = 0, frac_ns: int = 0) -> int:
    return days_from_civil(y, mo, d) * NS_DAY + h * NS_HOUR + mi * NS_MIN + s * NS_SEC + frac_ns

def floor_day(ns: int) -> int:
    return (ns // NS_DAY) * NS_DAY

def dayofweek(ns: int) -> int:
    return (ns // NS_DAY + 3) % 7

def fields(ns: int) -> Tuple[int, int, int, int, int, int, int]:
    days = ns // NS_DAY
    rem = ns - days * NS_DAY
    y, m, d = civil_from_days(days)
    return (y, m, d, rem // NS_HOUR, (rem % NS_HOUR) // NS_MIN, (rem % NS_MIN) // NS_SEC, rem % NS_SEC)

def dayofyear(ns: int) -> int:
    y, m, d, _, _, _, _ = fields(ns)
    return ns // NS_DAY - days_from_civil(y, 1, 1) + 1

def add_months(ns: int, k: int) -> int:
    y, m, d, _, _, _, _ = fields(ns)
    tot = y * 12 + (m - 1) + k
    ny = tot // 12
    nm = tot - ny * 12 + 1
    nd = min(d, days_in_month(ny, nm))
    return make_ns(ny, nm, nd) + (ns - floor_day(ns))

def _p2(x: int) -> str:
    return str(x).zfill(2)

def _digits(s: str) -> bool:
    if len(s) == 0:
        return False
    for ch in s:
        if not ch.isdigit():
            return False
    return True

def parse_ts(s0: str) -> int:
    s = s0.strip()
    date = s
    tm = ""
    for sep in (" ", "T"):
        p = s.find(sep)
        if p > 0:
            date = s[:p]
            tm = s[p + 1:]
            break
    parts = date.split("-")
    if len(parts) == 1:
        parts = date.split("/")
    if len(parts) != 3 or not _digits(parts[0]) or not _digits(parts[1]) or not _digits(parts[2]):
        raise ValueError(f"Unknown datetime string format, unable to parse: {s0}")
    y = int(parts[0])
    mo = int(parts[1])
    d = int(parts[2])
    if mo < 1 or mo > 12 or d < 1 or d > days_in_month(y, mo):
        raise ValueError(f"day is out of range for month: {s0}")
    h = 0
    mi = 0
    sec = 0
    frac = 0
    if tm != "":
        if tm.endswith("Z"):
            tm = tm[:-1]
        tp = tm.split(":")
        h = int(tp[0])
        if len(tp) > 1:
            mi = int(tp[1])
        if len(tp) > 2:
            sp = tp[2].split(".")
            sec = int(sp[0])
            if len(sp) > 1:
                f = sp[1]
                while len(f) < 9:
                    f = f + "0"
                frac = int(f[:9])
    return make_ns(y, mo, d, h, mi, sec, frac)

def fmt_ts(ns: int, mode: int) -> str:
    # mode 0: date only, 1: date + time, 2: + milliseconds, 3: + microseconds, 4: + nanoseconds
    y, m, d, h, mi, s, fr = fields(ns)
    out = str(y).zfill(4) + "-" + _p2(m) + "-" + _p2(d)
    if mode >= 1:
        out += " " + _p2(h) + ":" + _p2(mi) + ":" + _p2(s)
    if mode == 2:
        out += "." + str(fr // 1000000).zfill(3)
    elif mode == 3:
        out += "." + str(fr // 1000).zfill(6)
    elif mode >= 4:
        out += "." + str(fr).zfill(9)
    return out

def mode_for(vals: List[int]) -> int:
    mode = 0
    for v in vals:
        if v == NAT:
            continue
        r = v - floor_day(v)
        if r != 0 and mode < 1:
            mode = 1
        sub = r % NS_SEC
        if sub != 0:
            if sub % 1000000 == 0:
                if mode < 2:
                    mode = 2
            elif sub % 1000 == 0:
                if mode < 3:
                    mode = 3
            else:
                mode = 4
    return mode

def fmt_ts_full(ns: int) -> str:
    sub = ns % NS_SEC
    mode = 1
    if sub != 0:
        mode = 3 if sub % 1000 == 0 else 4
    return fmt_ts(ns, mode)

def fmt_td(ns: int) -> str:
    neg = ns < 0
    a = -ns if neg else ns
    days = a // NS_DAY
    rem = a - days * NS_DAY
    h = rem // NS_HOUR
    mi = (rem % NS_HOUR) // NS_MIN
    s = (rem % NS_MIN) // NS_SEC
    fr = rem % NS_SEC
    t = _p2(h) + ":" + _p2(mi) + ":" + _p2(s)
    if fr != 0:
        t += "." + (str(fr // 1000).zfill(6) if fr % 1000 == 0 else str(fr).zfill(9))
    return ("-" if neg else "") + str(days) + " days " + t

def parse_td(s0: str) -> int:
    s = s0.strip()
    neg = False
    if s.startswith("-"):
        neg = True
        s = s[1:].strip()
    days = 0
    rest = s
    p = s.find("day")
    if p > 0:
        days = int(s[:p].strip())
        rest = s[p:]
        q = rest.find(" ")
        rest = rest[q + 1:].strip() if q >= 0 else ""
    ns = days * NS_DAY
    if rest != "":
        tp = rest.split(":")
        if len(tp) == 3:
            sp = tp[2].split(".")
            ns += int(tp[0]) * NS_HOUR + int(tp[1]) * NS_MIN + int(sp[0]) * NS_SEC
            if len(sp) > 1:
                f = sp[1]
                while len(f) < 9:
                    f = f + "0"
                ns += int(f[:9])
        else:
            raise ValueError(f"unit abbreviation w/o a number: {s0}")
    return -ns if neg else ns

class Timedelta:
    value: int

    def __init__(self, value = None, days: int = 0, hours: int = 0, minutes: int = 0, seconds: int = 0,
                 milliseconds: int = 0, microseconds: int = 0, nanoseconds: int = 0, weeks: int = 0):
        v = 0
        if value is not None:
            x = value
            if isinstance(x, str):
                v = parse_td(x)
            elif isinstance(x, Timedelta):
                v = x.value
            else:
                v = int(x)
        self.value = (v + (days + 7 * weeks) * NS_DAY + hours * NS_HOUR + minutes * NS_MIN + seconds * NS_SEC
                      + milliseconds * 1000000 + microseconds * 1000 + nanoseconds)

    @property
    def days(self) -> int:
        return self.value // NS_DAY

    @property
    def seconds(self) -> int:
        return (self.value % NS_DAY) // NS_SEC

    @property
    def microseconds(self) -> int:
        return (self.value % NS_SEC) // 1000

    @property
    def nanoseconds(self) -> int:
        return self.value % 1000

    def total_seconds(self) -> float:
        return float(self.value) / 1e9

    def __add__(self, o: Timedelta) -> Timedelta:
        return Timedelta(self.value + o.value)

    def __sub__(self, o: Timedelta) -> Timedelta:
        return Timedelta(self.value - o.value)

    def __neg__(self) -> Timedelta:
        return Timedelta(-self.value)

    def __mul__(self, k: int) -> Timedelta:
        return Timedelta(self.value * k)

    def __eq__(self, o: Timedelta) -> bool:
        return self.value == o.value

    def __ne__(self, o: Timedelta) -> bool:
        return self.value != o.value

    def __lt__(self, o: Timedelta) -> bool:
        return self.value < o.value

    def __le__(self, o: Timedelta) -> bool:
        return self.value <= o.value

    def __gt__(self, o: Timedelta) -> bool:
        return self.value > o.value

    def __ge__(self, o: Timedelta) -> bool:
        return self.value >= o.value

    def __str__(self) -> str:
        return fmt_td(self.value)

    def __repr__(self) -> str:
        return "Timedelta('" + fmt_td(self.value) + "')"

class Timestamp:
    value: int

    def __init__(self, ts = None, year: int = 1970, month: int = 1, day: int = 1, hour: int = 0,
                 minute: int = 0, second: int = 0, microsecond: int = 0, nanosecond: int = 0):
        if ts is None:
            self.value = make_ns(year, month, day, hour, minute, second, microsecond * 1000 + nanosecond)
        else:
            x = ts
            if isinstance(x, str):
                self.value = parse_ts(x)
            elif isinstance(x, Timestamp):
                self.value = x.value
            else:
                self.value = int(x)

    @property
    def year(self) -> int:
        return fields(self.value)[0]

    @property
    def month(self) -> int:
        return fields(self.value)[1]

    @property
    def day(self) -> int:
        return fields(self.value)[2]

    @property
    def hour(self) -> int:
        return fields(self.value)[3]

    @property
    def minute(self) -> int:
        return fields(self.value)[4]

    @property
    def second(self) -> int:
        return fields(self.value)[5]

    @property
    def microsecond(self) -> int:
        return (self.value % NS_SEC) // 1000

    @property
    def nanosecond(self) -> int:
        return self.value % 1000

    @property
    def dayofweek(self) -> int:
        return dayofweek(self.value)

    @property
    def day_of_week(self) -> int:
        return dayofweek(self.value)

    @property
    def dayofyear(self) -> int:
        return dayofyear(self.value)

    @property
    def quarter(self) -> int:
        return (self.month - 1) // 3 + 1

    @property
    def is_leap_year(self) -> bool:
        return is_leap(self.year)

    @property
    def days_in_month(self) -> int:
        return days_in_month(self.year, self.month)

    def day_name(self) -> str:
        return _DAYS[dayofweek(self.value)]

    def month_name(self) -> str:
        return _MONTHS[self.month - 1]

    def weekday(self) -> int:
        return dayofweek(self.value)

    def normalize(self) -> Timestamp:
        return Timestamp(floor_day(self.value))

    def date(self) -> Tuple[int, int, int]:
        f = fields(self.value)
        return (f[0], f[1], f[2])

    def isoformat(self) -> str:
        return fmt_ts_full(self.value).replace(" ", "T")

    def strftime(self, fmt: str) -> str:
        return strftime_ns(self.value, fmt)

    def __add__(self, o: Timedelta) -> Timestamp:
        return Timestamp(self.value + o.value)

    def __sub__(self, o: Timedelta) -> Timestamp:
        return Timestamp(self.value - o.value)

    def __sub__(self, o: Timestamp) -> Timedelta:
        return Timedelta(self.value - o.value)

    def __eq__(self, o: Timestamp) -> bool:
        return self.value == o.value

    def __ne__(self, o: Timestamp) -> bool:
        return self.value != o.value

    def __lt__(self, o: Timestamp) -> bool:
        return self.value < o.value

    def __le__(self, o: Timestamp) -> bool:
        return self.value <= o.value

    def __gt__(self, o: Timestamp) -> bool:
        return self.value > o.value

    def __ge__(self, o: Timestamp) -> bool:
        return self.value >= o.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __str__(self) -> str:
        return fmt_ts_full(self.value)

    def __repr__(self) -> str:
        return "Timestamp('" + fmt_ts_full(self.value) + "')"

def strftime_ns(ns: int, fmt: str) -> str:
    y, mo, d, h, mi, s, fr = fields(ns)
    out = List[str]()
    i = 0
    n = len(fmt)
    while i < n:
        ch = fmt[i]
        if ch == "%" and i + 1 < n:
            c = fmt[i + 1]
            i += 2
            if c == "Y":
                out.append(str(y).zfill(4))
            elif c == "y":
                out.append(_p2(y % 100))
            elif c == "m":
                out.append(_p2(mo))
            elif c == "d":
                out.append(_p2(d))
            elif c == "H":
                out.append(_p2(h))
            elif c == "M":
                out.append(_p2(mi))
            elif c == "S":
                out.append(_p2(s))
            elif c == "f":
                out.append(str(fr // 1000).zfill(6))
            elif c == "j":
                out.append(str(dayofyear(ns)).zfill(3))
            elif c == "A":
                out.append(_DAYS[dayofweek(ns)])
            elif c == "a":
                out.append(_DAYS[dayofweek(ns)][:3])
            elif c == "B":
                out.append(_MONTHS[mo - 1])
            elif c == "b":
                out.append(_MONTHS[mo - 1][:3])
            elif c == "w":
                out.append(str((dayofweek(ns) + 1) % 7))
            elif c == "%":
                out.append("%")
            else:
                out.append("%" + c)
        else:
            out.append(ch)
            i += 1
    return "".join(out)

# ---------------------------------------------------------------------------
# offsets

# unit codes: 0 ns-tick (value in ns), 1 business day, 2 week (anchor weekday),
# 3 month end, 4 month start, 5 quarter end, 6 quarter start, 7 year end, 8 year start
def parse_freq(freq: str) -> Tuple[int, int, int]:
    j = 0
    while j < len(freq) and freq[j].isdigit():
        j += 1
    n = int(freq[:j]) if j > 0 else 1
    u = freq[j:]
    if u == "D":
        return (0, n * NS_DAY, 0)
    if u == "h" or u == "H":
        return (0, n * NS_HOUR, 0)
    if u == "min" or u == "T":
        return (0, n * NS_MIN, 0)
    if u == "s" or u == "S":
        return (0, n * NS_SEC, 0)
    if u == "ms" or u == "L":
        return (0, n * 1000000, 0)
    if u == "us" or u == "U":
        return (0, n * 1000, 0)
    if u == "ns" or u == "N":
        return (0, n, 0)
    if u == "B":
        return (1, n, 0)
    if u == "W" or u == "W-SUN":
        return (2, n, 6)
    if u == "W-MON":
        return (2, n, 0)
    if u == "W-TUE":
        return (2, n, 1)
    if u == "W-WED":
        return (2, n, 2)
    if u == "W-THU":
        return (2, n, 3)
    if u == "W-FRI":
        return (2, n, 4)
    if u == "W-SAT":
        return (2, n, 5)
    if u == "ME" or u == "M":
        return (3, n, 0)
    if u == "MS":
        return (4, n, 0)
    if u == "QE" or u == "Q" or u == "QE-DEC":
        return (5, n, 0)
    if u == "QS" or u == "QS-JAN":
        return (6, n, 0)
    if u == "YE" or u == "A" or u == "Y" or u == "YE-DEC":
        return (7, n, 0)
    if u == "YS" or u == "AS" or u == "YS-JAN":
        return (8, n, 0)
    raise ValueError(f"Invalid frequency: {freq}")

def month_end_ns(ns: int) -> int:
    y, m, d, _, _, _, _ = fields(ns)
    return make_ns(y, m, days_in_month(y, m)) + (ns - floor_day(ns))

def month_start_ns(ns: int) -> int:
    y, m, d, _, _, _, _ = fields(ns)
    return make_ns(y, m, 1) + (ns - floor_day(ns))

def roll_forward(ns: int, kind: int, arg: int) -> int:
    # first anchored date on or after ns (time of day kept)
    tod = ns - floor_day(ns)
    if kind == 1:
        d = dayofweek(ns)
        if d < 5:
            return ns
        return ns + (7 - d) * NS_DAY
    if kind == 2:
        d = dayofweek(ns)
        return ns + ((arg - d) % 7) * NS_DAY
    y, m, dd, _, _, _, _ = fields(ns)
    if kind == 3:
        return month_end_ns(ns)
    if kind == 4:
        if dd == 1:
            return ns
        return add_months(month_start_ns(ns), 1)
    if kind == 5:
        qm = ((m - 1) // 3 + 1) * 3
        return make_ns(y, qm, days_in_month(y, qm)) + tod
    if kind == 6:
        qm = ((m - 1) // 3) * 3 + 1
        base = make_ns(y, qm, 1) + tod
        if base >= ns:
            return base
        return add_months(base, 3)
    if kind == 7:
        return make_ns(y, 12, 31) + tod
    if kind == 8:
        base = make_ns(y, 1, 1) + tod
        if base >= ns:
            return base
        return make_ns(y + 1, 1, 1) + tod
    return ns

def step_anchor(ns: int, kind: int, n: int, arg: int) -> int:
    tod = ns - floor_day(ns)
    if kind == 1:
        cur = ns
        k = n
        while k > 0:
            cur += NS_DAY
            while dayofweek(cur) >= 5:
                cur += NS_DAY
            k -= 1
        return cur
    if kind == 2:
        return ns + 7 * n * NS_DAY
    if kind == 3:
        return month_end_ns(add_months(month_start_ns(ns), n))
    if kind == 4:
        return add_months(ns, n)
    if kind == 5:
        return month_end_ns(add_months(month_start_ns(ns), 3 * n))
    if kind == 6:
        return add_months(ns, 3 * n)
    if kind == 7:
        y = fields(ns)[0]
        return make_ns(y + n, 12, 31) + tod
    y = fields(ns)[0]
    return make_ns(y + n, 1, 1) + tod

def date_range_ns(start: Optional[int], end: Optional[int], periods: Optional[int], freq: str,
                  normalize: bool = False) -> List[int]:
    kind, n, arg = parse_freq(freq)
    out = List[int]()
    if start is None:
        # end + periods: generate backwards
        e = end.__val__()
        p = periods.__val__()
        cur = e
        if kind == 0:
            for i in range(p):
                out.append(e - (p - 1 - i) * n)
            return out
        tmp = List[int]()
        if kind == 1:
            while dayofweek(cur) >= 5:
                cur -= NS_DAY
        for _ in range(p):
            tmp.append(cur)
            cur = step_anchor(cur, kind, -n, arg) if kind != 1 else _prev_bday(cur, n)
        for i in range(len(tmp) - 1, -1, -1):
            out.append(tmp[i])
        return out
    s = start.__val__()
    if normalize:
        s = floor_day(s)
    if kind == 0:
        cur = s
        if periods is not None:
            p = periods.__val__()
            for i in range(p):
                out.append(s + i * n)
        else:
            e = end.__val__()
            while cur <= e:
                out.append(cur)
                cur += n
        return out
    cur = roll_forward(s, kind, arg)
    if periods is not None:
        p = periods.__val__()
        for _ in range(p):
            out.append(cur)
            cur = step_anchor(cur, kind, n, arg)
    else:
        e = end.__val__()
        while cur <= e:
            out.append(cur)
            cur = step_anchor(cur, kind, n, arg)
    return out

def _prev_bday(ns: int, n: int) -> int:
    cur = ns
    k = n
    while k > 0:
        cur -= NS_DAY
        while dayofweek(cur) >= 5:
            cur -= NS_DAY
        k -= 1
    return cur
