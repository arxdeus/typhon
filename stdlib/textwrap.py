# Ported from CPython 3.14 (PSF-2.0)

def _is_ws(c: str) -> bool:
    return c == " " or c == "\t" or c == "\n" or c == "\r" or c == "\x0b" or c == "\x0c"

def _chunks(text: str, break_on_hyphens: bool) -> List[str]:
    # split into whitespace runs and words; words optionally split after hyphens
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = i
        if _is_ws(text[i]):
            while j < n and _is_ws(text[j]):
                j += 1
            out.append(text[i:j])
        else:
            while j < n and not _is_ws(text[j]):
                j += 1
            w = text[i:j]
            if break_on_hyphens and "-" in w:
                s = 0
                k = 0
                m = len(w)
                while k < m:
                    if w[k] == "-" and k > s and k + 1 < m and w[k + 1].isalpha() and w[k - 1].isalpha():
                        # require letter run of 2+ before hyphen like CPython
                        if k - 1 > s and w[k - 2].isalpha():
                            out.append(w[s:k + 1])
                            s = k + 1
                    k += 1
                out.append(w[s:])
            else:
                out.append(w)
        i = j
    return out

def _wrap_chunks(chunks: List[str], width: int, initial_indent: str, subsequent_indent: str,
                 expand_tabs: bool, replace_whitespace: bool, fix_sentence_endings: bool,
                 break_long_words: bool, drop_whitespace: bool, max_lines: int,
                 placeholder: str) -> List[str]:
    lines: List[str] = []
    if width <= 0:
        raise ValueError("invalid width " + str(width) + " (must be > 0)")
    chunks = chunks[:]
    chunks.reverse()
    while len(chunks) > 0:
        cur: List[str] = []
        cur_len = 0
        indent = subsequent_indent if len(lines) > 0 else initial_indent
        w = width - len(indent)
        if drop_whitespace and chunks[-1].strip() == "" and len(lines) > 0:
            chunks.pop()
        while len(chunks) > 0:
            l = len(chunks[-1])
            if cur_len + l <= w:
                cur.append(chunks.pop())
                cur_len += l
            else:
                break
        if len(chunks) > 0 and len(chunks[-1]) > w:
            if break_long_words:
                space_left = 1 if w < 1 else w - cur_len
                if space_left > 0 or len(cur) == 0:
                    ch = chunks[-1]
                    end = space_left
                    if ch[:0] == "" and False:
                        pass
                    cur.append(ch[:end])
                    chunks[-1] = ch[end:]
                    cur_len += end
            elif len(cur) == 0:
                cur.append(chunks.pop())
                cur_len += len(cur[-1])
        if drop_whitespace and len(cur) > 0 and cur[-1].strip() == "":
            cur_len -= len(cur[-1])
            cur.pop()
        if len(cur) > 0:
            if max_lines <= 0 or len(lines) + 1 < max_lines or ((len(chunks) == 0 or (drop_whitespace and len(chunks) == 1 and chunks[0].strip() == "")) and cur_len <= w):
                lines.append(indent + "".join(cur))
            else:
                done = False
                while len(cur) > 0:
                    if cur[-1].strip() != "" and cur_len + len(placeholder) <= w:
                        cur.append(placeholder)
                        lines.append(indent + "".join(cur))
                        done = True
                        break
                    cur_len -= len(cur[-1])
                    cur.pop()
                if not done:
                    if len(lines) > 0:
                        prev = lines[-1].rstrip()
                        if len(prev) + len(placeholder) <= width:
                            lines[-1] = prev + placeholder
                            break
                    lines.append(indent + placeholder.lstrip())
                break
    return lines

def _munge(text: str, expand_tabs: bool, replace_whitespace: bool) -> str:
    if expand_tabs:
        text = text.expandtabs(8)
    if replace_whitespace:
        out = ""
        for c in text:
            out += " " if (c == "\t" or c == "\n" or c == "\x0b" or c == "\x0c" or c == "\r") else c
        text = out
    return text

def wrap(text: str, width: int = 70, initial_indent: str = "", subsequent_indent: str = "",
         expand_tabs: bool = True, replace_whitespace: bool = True,
         fix_sentence_endings: bool = False, break_long_words: bool = True,
         drop_whitespace: bool = True, break_on_hyphens: bool = True,
         tabsize: int = 8, max_lines: int = 0, placeholder: str = " [...]") -> List[str]:
    if max_lines > 0:
        indent = subsequent_indent if max_lines > 1 else initial_indent
        if len(indent) + len(placeholder.lstrip()) > width:
            raise ValueError("placeholder too large for max width")
    if expand_tabs:
        text = text.expandtabs(tabsize)
    text = _munge(text, False, replace_whitespace)
    chunks = _chunks(text, break_on_hyphens)
    if fix_sentence_endings:
        i = 0
        while i + 1 < len(chunks):
            if chunks[i + 1] == " " and (chunks[i].endswith(".") or chunks[i].endswith("!") or chunks[i].endswith("?")) and len(chunks[i]) > 1 and chunks[i][-2].islower():
                chunks[i + 1] = "  "
                i += 2
            else:
                i += 1
    return _wrap_chunks(chunks, width, initial_indent, subsequent_indent, expand_tabs,
                        replace_whitespace, fix_sentence_endings, break_long_words,
                        drop_whitespace, max_lines, placeholder)

def fill(text: str, width: int = 70, initial_indent: str = "", subsequent_indent: str = "",
         expand_tabs: bool = True, replace_whitespace: bool = True,
         fix_sentence_endings: bool = False, break_long_words: bool = True,
         drop_whitespace: bool = True, break_on_hyphens: bool = True,
         tabsize: int = 8, max_lines: int = 0, placeholder: str = " [...]") -> str:
    return "\n".join(wrap(text, width, initial_indent, subsequent_indent, expand_tabs,
                          replace_whitespace, fix_sentence_endings, break_long_words,
                          drop_whitespace, break_on_hyphens, tabsize, max_lines, placeholder))

def shorten(text: str, width: int, placeholder: str = " [...]") -> str:
    parts = text.split()
    return fill(" ".join(parts), width, max_lines=1, placeholder=placeholder)

def dedent(text: str) -> str:
    lines = text.split("\n")
    margin: Optional[str] = None
    for line in lines:
        if line.strip() == "":
            continue
        k = 0
        while k < len(line) and (line[k] == " " or line[k] == "\t"):
            k += 1
        ind = line[:k]
        if margin is None:
            margin = ind
        else:
            m = margin
            a = 0
            while a < len(m) and a < len(ind) and m[a] == ind[a]:
                a += 1
            margin = m[:a]
    out: List[str] = []
    for line in lines:
        if line.strip() == "":
            out.append("")
        elif margin is not None:
            out.append(line[len(margin):])
        else:
            out.append(line)
    return "\n".join(out)

def indent(text: str, prefix: str, predicate: Optional[function[[str], bool]] = None) -> str:
    out = ""
    for line in text.splitlines(True):
        if predicate is None:
            ok = line.strip() != ""
        else:
            ok = predicate(line)
        out += prefix + line if ok else line
    return out
