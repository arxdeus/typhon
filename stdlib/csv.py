# Ported from CPython 3.14 (PSF-2.0)
# Subset: reader over an iterable of str lines, writer collecting lines,
# DictReader/DictWriter with str values. Dialect = excel (delimiter/quotechar configurable).

QUOTE_MINIMAL = 0
QUOTE_ALL = 1
QUOTE_NONNUMERIC = 2
QUOTE_NONE = 3

class Error(Exception):
    pass

class reader:
    rows: List[List[str]]
    line_num: int
    _pos: int

    def __init__(self, lines: List[str], delimiter: str = ",", quotechar: str = '"',
                 skipinitialspace: bool = False, strict: bool = False):
        self.rows = []
        self.line_num = 0
        self._pos = 0
        row: List[str] = []
        field = ""
        in_q = False
        started = False  # row has content
        quoted_field = False
        i = 0
        for raw in lines:
            self.line_num += 1
            line = raw
            if not in_q:
                row = []
                field = ""
                started = False
                quoted_field = False
            n = len(line)
            i = 0
            while i < n:
                c = line[i]
                if in_q:
                    if c == quotechar:
                        if i + 1 < n and line[i + 1] == quotechar:
                            field += quotechar
                            i += 1
                        else:
                            in_q = False
                    else:
                        field += c
                elif c == quotechar and field == "" and not quoted_field:
                    in_q = True
                    quoted_field = True
                    started = True
                elif c == delimiter:
                    row.append(field)
                    field = ""
                    quoted_field = False
                    started = True
                    if skipinitialspace:
                        while i + 1 < n and line[i + 1] == " ":
                            i += 1
                elif c == "\n" or c == "\r":
                    pass
                else:
                    field += c
                    started = True
                i += 1
            if in_q:
                continue
            if started or field != "":
                row.append(field)
            self.rows.append(row)
        if in_q and strict:
            raise Error("unexpected end of data")
        if in_q:
            row.append(field)
            self.rows.append(row)

    def __iter__(self):
        for r in self.rows:
            yield r

    def __next__(self) -> List[str]:
        if self._pos >= len(self.rows):
            raise StopIteration()
        self._pos += 1
        return self.rows[self._pos - 1]

def _fmt_field(f: str, delimiter: str, quotechar: str, quoting: int, lt: str) -> str:
    need = quoting == QUOTE_ALL
    if not need:
        if f.find(delimiter) >= 0 or f.find(quotechar) >= 0 or f.find("\n") >= 0 or f.find("\r") >= 0:
            need = True
    if quoting == QUOTE_NONE:
        need = False
    if need:
        return quotechar + f.replace(quotechar, quotechar + quotechar) + quotechar
    return f

class writer:
    lines: List[str]
    delimiter: str
    quotechar: str
    quoting: int
    lineterminator: str

    def __init__(self, delimiter: str = ",", quotechar: str = '"', quoting: int = QUOTE_MINIMAL,
                 lineterminator: str = "\r\n"):
        self.lines = []
        self.delimiter = delimiter
        self.quotechar = quotechar
        self.quoting = quoting
        self.lineterminator = lineterminator

    def writerow(self, row: List[str]) -> str:
        if len(row) == 1 and row[0] == "":
            line = self.quotechar + self.quotechar + self.lineterminator
        else:
            parts = [_fmt_field(f, self.delimiter, self.quotechar, self.quoting, self.lineterminator) for f in row]
            line = self.delimiter.join(parts) + self.lineterminator
        self.lines.append(line)
        return line

    def writerows(self, rows: List[List[str]]):
        for r in rows:
            self.writerow(r)

    def getvalue(self) -> str:
        return "".join(self.lines)

class DictReader:
    fieldnames: List[str]
    rows: List[Dict[str, str]]

    def __init__(self, lines: List[str], delimiter: str = ",", quotechar: str = '"'):
        r = reader(lines, delimiter, quotechar)
        self.fieldnames = []
        self.rows = []
        first = True
        for row in r.rows:
            if first:
                self.fieldnames = row
                first = False
                continue
            if len(row) == 0:
                continue
            d: Dict[str, str] = {}
            for k in range(len(self.fieldnames)):
                d[self.fieldnames[k]] = row[k] if k < len(row) else ""
            self.rows.append(d)

    def __iter__(self):
        for r in self.rows:
            yield r

class DictWriter:
    fieldnames: List[str]
    w: writer

    def __init__(self, fieldnames: List[str], delimiter: str = ",", quotechar: str = '"',
                 lineterminator: str = "\r\n"):
        self.fieldnames = fieldnames
        self.w = writer(delimiter, quotechar, QUOTE_MINIMAL, lineterminator)

    def writeheader(self):
        self.w.writerow(self.fieldnames)

    def writerow(self, rowdict: Dict[str, str]):
        self.w.writerow([rowdict.get(k, "") for k in self.fieldnames])

    def writerows(self, rows: List[Dict[str, str]]):
        for r in rows:
            self.writerow(r)

    def getvalue(self) -> str:
        return self.w.getvalue()
