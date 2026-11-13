# Ported from pandas 3.1.1 (BSD-3-Clause)

# Text layout of DataFrame/Series repr (pandas/io/formats/format.py and
# string.py) with the default display options of a non-interactive run:
# display.width=80, max_columns=0 (fit terminal of 80x60), max_rows=60,
# min_rows=10, precision=6.

from .scalar import K_FLOAT, K_INT, K_BOOL, K_STR, NA_STR, _isnan
from .core import Column

DISPLAY_WIDTH: Literal[int] = 80
TERM_HEIGHT: Literal[int] = 60
MAX_ROWS: Literal[int] = 60
MIN_ROWS: Literal[int] = 10
PRECISION: Literal[int] = 6

def _is_decimal_number(t0: str) -> bool:
    # ^\s*[+-]?[0-9]+\.[0-9]*$
    t = t0.lstrip()
    j = 0
    if len(t) > 0 and (t[0] == "+" or t[0] == "-"):
        j = 1
    nd = 0
    while j < len(t) and t[j].isdigit():
        j += 1
        nd += 1
    if nd == 0 or j >= len(t) or t[j] != ".":
        return False
    j += 1
    while j < len(t) and t[j].isdigit():
        j += 1
    return j == len(t)

def _trim_zeros_float(strs: List[str]) -> List[str]:
    isnum = [_is_decimal_number(x) for x in strs]
    anyn = False
    for b in isnum:
        if b:
            anyn = True
    if not anyn:
        return strs
    tr = strs.copy()
    while True:
        allz = True
        for k in range(len(tr)):
            if isnum[k] and not tr[k].endswith("0"):
                allz = False
                break
        if not allz:
            break
        for k in range(len(tr)):
            if isnum[k]:
                tr[k] = tr[k][:-1]
    for k in range(len(tr)):
        if isnum[k] and tr[k].endswith("."):
            tr[k] = tr[k] + "0"
    return tr

def _fmt_f(x: float, digits: int, sci: bool) -> str:
    # "{value: .{digits}f}" / "{value: .{digits}e}"
    if x == float("inf"):
        return " inf"
    if x == -float("inf"):
        return "-inf"
    s = ("%." + str(digits) + ("e" if sci else "f")) % x
    if not s.startswith("-"):
        s = " " + s
    return s

def format_floats(vals: List[float], na_rep: str = "NaN") -> List[str]:
    digits = PRECISION
    def with_fmt(sci: bool) -> List[str]:
        out = [na_rep if _isnan(x) else _fmt_f(x, digits, sci) for x in vals]
        return _trim_zeros_float(out)

    fv = with_fmt(False)
    maxlen = 0
    for x in fv:
        if len(x) > maxlen:
            maxlen = len(x)
    too_long = maxlen > digits + 6
    has_large = False
    has_small = False
    for x in vals:
        if _isnan(x):
            continue
        a = abs(x)
        if a > 1e6:
            has_large = True
        if a < 10.0 ** (-digits) and a > 0:
            has_small = True
    if has_small or (too_long and has_large):
        fv = with_fmt(True)
    # pandas then pads NaN entries to the formatted width (left side)
    return fv

def format_column(c: Column) -> List[str]:
    if c.kind == K_FLOAT:
        return format_floats(c.f)
    if c.kind == K_INT:
        return [(" " + str(x)) if x >= 0 else str(x) for x in c.i]
    if c.kind == K_BOOL:
        return [" True" if x else " False" for x in c.i]
    return [" " + (x if x != NA_STR else "NaN") for x in c.s]

def format_index(c: Column) -> List[str]:
    # index labels are formatted without the leading sign space
    if c.kind == K_FLOAT:
        return [s.strip() for s in format_floats(c.f)]
    if c.kind == K_INT:
        return [str(x) for x in c.i]
    if c.kind == K_BOOL:
        return ["True" if x else "False" for x in c.i]
    return [x if x != NA_STR else "NaN" for x in c.s]

def _width(col: List[str]) -> int:
    w = 0
    for s in col:
        if len(s) > w:
            w = len(s)
    return w

def adjoin(cols: List[List[str]], left_first: bool) -> str:
    # pandas adjoin(1, ...): every column but the last padded to width + 1,
    # first column (index) left-justified, others right-justified (they are
    # already justified), last column not padded on the right
    if len(cols) == 0:
        return ""
    nrows = len(cols[0])
    widths = [_width(c) for c in cols]
    lines = List[str](nrows)
    for r in range(nrows):
        parts = List[str](len(cols))
        for k in range(len(cols)):
            s = cols[k][r]
            if k == len(cols) - 1:
                parts.append(s)
            elif k == 0 and left_first:
                parts.append(s.ljust(widths[k] + 1))
            else:
                parts.append(s.ljust(widths[k] + 1))
        lines.append("".join(parts))
    return "\n".join(lines)

def sparsify_levels(levels: List[Column]) -> List[List[str]]:
    # MultiIndex display: a label equal to the previous row's (with all outer
    # levels equal too) is blanked.
    out = List[List[str]]()
    n = len(levels[0])
    prev_same = [True for _ in range(n)]
    for lv in levels:
        strs = format_index(lv)
        col = List[str](n)
        cur_same = [False for _ in range(n)]
        for i in range(n):
            eq = i > 0 and prev_same[i] and lv.get(i) == lv.get(i - 1)
            cur_same[i] = eq
            col.append("" if eq else strs[i])
        out.append(col)
        prev_same = cur_same
    return out

def frame_repr(cols: List[Column], col_labels: List[str], idx: Column, idx_name: Optional[str], nrows: int, fit: bool,
               idx_levels: List[Column] = List[Column](), idx_names: List[Optional[str]] = List[Optional[str]]()) -> str:
    ncols = len(cols)
    # vertical truncation
    row_sel = list(range(nrows))
    trunc_v = False
    tr_row_num = 0
    if nrows > MAX_ROWS:
        trunc_v = True
        tr_row_num = MIN_ROWS // 2
        row_sel = list(range(tr_row_num)) + list(range(nrows - tr_row_num, nrows))
    # format every column (full), truncation picks from these
    idx_strs = format_index(idx.take(row_sel))
    lvl_strs = List[List[str]]()
    multi = len(idx_levels) > 0
    if multi:
        lvl_strs = sparsify_levels([c.take(row_sel) for c in idx_levels])
    data_strs = List[List[str]]()
    for j in range(ncols):
        data_strs.append(format_column(cols[j].take(row_sel)))
    has_idx_name = idx_name is not None
    if multi:
        has_idx_name = False
        for nm in idx_names:
            if nm is not None:
                has_idx_name = True

    def build(sel: List[int]) -> List[List[str]]:
        strcols = List[List[str]]()
        if multi:
            for lj in range(len(lvl_strs)):
                ic = List[str]()
                ic.append("")
                if has_idx_name:
                    ic.append(idx_names[lj].__val__() if idx_names[lj] is not None else "")
                ic.extend(lvl_strs[lj])
                iw = _width(ic)
                strcols.append([s.ljust(iw) for s in ic])
        else:
            ic = List[str]()
            ic.append("")
            if has_idx_name:
                ic.append(idx_name.__val__())
            ic.extend(idx_strs)
            iw = _width(ic)
            strcols.append([s.ljust(iw) for s in ic])
        for j in sel:
            vals = data_strs[j]
            hdr = col_labels[j] if cols[j].kind == K_STR else " " + col_labels[j]
            w = max(_width(vals), len(hdr))
            col = List[str]()
            col.append(hdr.rjust(w))
            if has_idx_name:
                col.append("".rjust(w))
            for s in vals:
                col.append(s.rjust(w))
            strcols.append(col)
        return strcols

    all_sel = list(range(ncols))
    strcols = build(all_sel)
    trunc_h = False
    tr_col_num = 0
    # _fit_strcols_to_terminal_width (strcols include index)
    if not fit:
        return adjoin(strcols, True)
    if trunc_v:
        strcols = _insert_vdots(strcols, tr_row_num, -1, 1 + (1 if has_idx_name else 0))
    col_lens = [_width(c) for c in strcols]
    total = 0
    for w in col_lens:
        total += w
    total += len(col_lens) - 1
    adj_dif = total - DISPLAY_WIDTH + 1
    if adj_dif > 0:
        adj_dif += 5
        n_cols = len(col_lens)
        lens = col_lens.copy()
        while adj_dif > 0 and n_cols > 1:
            mid = _py_round_half_even(n_cols / 2)
            adj_dif -= lens.pop(mid) + 1
            n_cols -= 1
        max_fit = n_cols - 1
        if max_fit < 2:
            max_fit = 2
        if max_fit < ncols:
            trunc_h = True
            tr_col_num = max_fit // 2
            sel = list(range(tr_col_num)) + list(range(ncols - tr_col_num, ncols))
            strcols = build(sel)
            if trunc_v:
                strcols = _insert_vdots(strcols, tr_row_num, -1, 1 + (1 if has_idx_name else 0))
            strcols.insert(tr_col_num + 1, [" ..." for _ in range(len(strcols[0]))])
            if trunc_v:
                # the dots column gets "..." at the truncated row too
                hdr_rows = 1 + (1 if has_idx_name else 0)
                strcols[tr_col_num + 1][tr_row_num + hdr_rows] = " ..."
    out = adjoin(strcols, True)
    if trunc_h or trunc_v:
        out += f"\n\n[{nrows} rows x {ncols} columns]"
    return out

def _insert_vdots(strcols: List[List[str]], row_num: int, dot_col: int, hdr_rows: int) -> List[List[str]]:
    for ix in range(len(strcols)):
        col = strcols[ix]
        cw = len(col[row_num + hdr_rows]) if row_num + hdr_rows < len(col) else 0
        dots = "..." if cw > 3 or ix == dot_col else ".."
        if ix == 0:
            s = dots.ljust(cw)
        else:
            s = dots.rjust(cw)
        col.insert(row_num + hdr_rows, s)
    return strcols

def _py_round_half_even(x: float) -> int:
    f = int(x)
    d = x - float(f)
    if d > 0.5:
        return f + 1
    if d < 0.5:
        return f
    return f if f % 2 == 0 else f + 1

def series_repr(c: Column, idx: Column, idx_name: Optional[str], name: Optional[str], dtype: str, footer_name: bool = True,
                idx_levels: List[Column] = List[Column](), idx_names: List[Optional[str]] = List[Optional[str]]()) -> str:
    n = len(c)
    sel = list(range(n))
    trunc = False
    half = 0
    if n > MAX_ROWS:
        trunc = True
        half = MIN_ROWS // 2
        sel = list(range(half)) + list(range(n - half, n))
    vals = format_column(c.take(sel))
    labs = format_index(idx.take(sel))
    header = ""
    if len(idx_levels) > 0:
        lv = sparsify_levels([c.take(sel) for c in idx_levels])
        widths = List[int]()
        for j in range(len(lv)):
            nm = idx_names[j].__val__() if idx_names[j] is not None else ""
            widths.append(max(_width(lv[j]), len(nm)))
        labs = List[str]()
        for k in range(len(sel)):
            labs.append("  ".join([lv[j][k].ljust(widths[j]) for j in range(len(lv))]))
        has_nm = False
        for nm in idx_names:
            if nm is not None:
                has_nm = True
        if has_nm:
            header = "  ".join([(idx_names[j].__val__() if idx_names[j] is not None else "").ljust(widths[j]) for j in range(len(lv))])
    iw = _width(labs)
    vw = _width(vals)
    lines = List[str]()
    if len(idx_levels) > 0:
        if header != "":
            lines.append(header)
    elif idx_name is not None:
        lines.append(idx_name.__val__())
    for k in range(len(sel)):
        if trunc and k == half:
            lines.append("..".ljust(iw) + "   " + "...".rjust(vw))
        lines.append(labs[k].ljust(iw) + "   " + vals[k].rjust(vw))
    footer = List[str]()
    if name is not None:
        footer.append(f"Name: {name.__val__()}")
    if trunc:
        footer.append(f"Length: {n}")
    footer.append(f"dtype: {dtype}")
    return "\n".join(lines) + "\n" + ", ".join(footer)
