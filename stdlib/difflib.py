# Ported from CPython 3.14 (PSF-2.0)
# SequenceMatcher works on List[str] (use list(s) for characters; str is accepted too).
# Unsupported: isjunk callback, HtmlDiff, Differ.compare, context_diff, get_close_matches key.

def _chars(s: str) -> List[str]:
    return [c for c in s]

class SequenceMatcher:
    a: List[str]
    b: List[str]
    autojunk: bool
    b2j: Dict[str, List[int]]
    popular: Set[str]
    matching_blocks: Optional[List[Tuple[int, int, int]]]
    opcodes: Optional[List[Tuple[str, int, int, int, int]]]

    def __init__(self, a: List[str], b: List[str], autojunk: bool = True):
        self.a = a
        self.b = b
        self.autojunk = autojunk
        self.b2j = {}
        self.popular = set()
        self.matching_blocks = None
        self.opcodes = None
        self._chain_b()

    def __init__(self, a: str, b: str, autojunk: bool = True):
        self.a = _chars(a)
        self.b = _chars(b)
        self.autojunk = autojunk
        self.b2j = {}
        self.popular = set()
        self.matching_blocks = None
        self.opcodes = None
        self._chain_b()

    def set_seqs(self, a: List[str], b: List[str]):
        self.a = a
        self.b = b
        self.matching_blocks = None
        self.opcodes = None
        self._chain_b()

    def _chain_b(self):
        b = self.b
        self.b2j = {}
        for i in range(len(b)):
            elt = b[i]
            if elt in self.b2j:
                self.b2j[elt].append(i)
            else:
                self.b2j[elt] = [i]
        self.popular = set()
        n = len(b)
        if self.autojunk and n >= 200:
            ntest = n // 100 + 1
            for elt in list(self.b2j.keys()):
                if len(self.b2j[elt]) > ntest:
                    self.popular.add(elt)
            for elt in self.popular:
                del self.b2j[elt]

    def find_longest_match(self, alo: int = 0, ahi: Optional[int] = None, blo: int = 0,
                           bhi: Optional[int] = None) -> Tuple[int, int, int]:
        a = self.a
        b = self.b
        ahi_ = len(a) if ahi is None else ahi
        bhi_ = len(b) if bhi is None else bhi
        besti = alo
        bestj = blo
        bestsize = 0
        j2len: Dict[int, int] = {}
        for i in range(alo, ahi_):
            newj2len: Dict[int, int] = {}
            if a[i] in self.b2j:
                for j in self.b2j[a[i]]:
                    if j < blo:
                        continue
                    if j >= bhi_:
                        break
                    k = j2len.get(j - 1, 0) + 1
                    newj2len[j] = k
                    if k > bestsize:
                        besti = i - k + 1
                        bestj = j - k + 1
                        bestsize = k
            j2len = newj2len
        # extend with popular elements (no junk support)
        while besti > alo and bestj > blo and b[bestj - 1] not in self.popular and a[besti - 1] == b[bestj - 1] and False:
            pass
        while besti > alo and bestj > blo and a[besti - 1] == b[bestj - 1]:
            besti -= 1
            bestj -= 1
            bestsize += 1
        while besti + bestsize < ahi_ and bestj + bestsize < bhi_ and a[besti + bestsize] == b[bestj + bestsize]:
            bestsize += 1
        return (besti, bestj, bestsize)

    def get_matching_blocks(self) -> List[Tuple[int, int, int]]:
        if self.matching_blocks is not None:
            return self.matching_blocks
        la = len(self.a)
        lb = len(self.b)
        queue: List[Tuple[int, int, int, int]] = [(0, la, 0, lb)]
        matching: List[Tuple[int, int, int]] = []
        while len(queue) > 0:
            alo, ahi, blo, bhi = queue.pop()
            m = self.find_longest_match(alo, ahi, blo, bhi)
            i, j, k = m
            if k > 0:
                matching.append(m)
                if alo < i and blo < j:
                    queue.append((alo, i, blo, j))
                if i + k < ahi and j + k < bhi:
                    queue.append((i + k, ahi, j + k, bhi))
        matching.sort()
        i1 = 0
        j1 = 0
        k1 = 0
        non_adjacent: List[Tuple[int, int, int]] = []
        for blk in matching:
            i2, j2, k2 = blk
            if i1 + k1 == i2 and j1 + k1 == j2:
                k1 += k2
            else:
                if k1 > 0:
                    non_adjacent.append((i1, j1, k1))
                i1 = i2
                j1 = j2
                k1 = k2
        if k1 > 0:
            non_adjacent.append((i1, j1, k1))
        non_adjacent.append((la, lb, 0))
        self.matching_blocks = non_adjacent
        return non_adjacent

    def get_opcodes(self) -> List[Tuple[str, int, int, int, int]]:
        if self.opcodes is not None:
            return self.opcodes
        i = 0
        j = 0
        answer: List[Tuple[str, int, int, int, int]] = []
        for blk in self.get_matching_blocks():
            ai, bj, size = blk
            tag = ""
            if i < ai and j < bj:
                tag = "replace"
            elif i < ai:
                tag = "delete"
            elif j < bj:
                tag = "insert"
            if tag != "":
                answer.append((tag, i, ai, j, bj))
            i = ai + size
            j = bj + size
            if size > 0:
                answer.append(("equal", ai, i, bj, j))
        self.opcodes = answer
        return answer

    def get_grouped_opcodes(self, n: int = 3) -> List[List[Tuple[str, int, int, int, int]]]:
        codes = self.get_opcodes()[:]
        if len(codes) == 0:
            codes = [("equal", 0, 1, 0, 1)]
        if codes[0][0] == "equal":
            t = codes[0]
            codes[0] = (t[0], max(t[1], t[2] - n), t[2], max(t[3], t[4] - n), t[4])
        if codes[-1][0] == "equal":
            t = codes[-1]
            codes[-1] = (t[0], t[1], min(t[2], t[1] + n), t[3], min(t[4], t[3] + n))
        nn = n + n
        groups: List[List[Tuple[str, int, int, int, int]]] = []
        group: List[Tuple[str, int, int, int, int]] = []
        for c in codes:
            tag, i1, i2, j1, j2 = c
            if tag == "equal" and i2 - i1 > nn:
                group.append((tag, i1, min(i2, i1 + n), j1, min(j2, j1 + n)))
                groups.append(group)
                group = []
                i1 = max(i1, i2 - n)
                j1 = max(j1, j2 - n)
            group.append((tag, i1, i2, j1, j2))
        if len(group) > 0 and not (len(group) == 1 and group[0][0] == "equal"):
            groups.append(group)
        return groups

    def quick_ratio(self) -> float:
        return _quick_ratio_of(self)

    def real_quick_ratio(self) -> float:
        la = len(self.a)
        lb = len(self.b)
        if la + lb == 0:
            return 1.0
        return 2.0 * min(la, lb) / (la + lb)

    def ratio(self) -> float:
        matches = 0
        for blk in self.get_matching_blocks():
            matches += blk[2]
        length = len(self.a) + len(self.b)
        if length == 0:
            return 1.0
        return 2.0 * matches / length

def _quick_ratio_of(sm: SequenceMatcher) -> float:
    la = len(sm.a)
    lb = len(sm.b)
    if la + lb == 0:
        return 1.0
    avail: Dict[str, int] = {}
    matches = 0
    for e in sm.b:
        avail[e] = avail.get(e, 0) + 1
    used: Dict[str, int] = {}
    for e in sm.a:
        u = used.get(e, 0)
        if u < avail.get(e, 0):
            matches += 1
        used[e] = u + 1
    return 2.0 * matches / (la + lb)

def get_close_matches(word: str, possibilities: List[str], n: int = 3, cutoff: float = 0.6) -> List[str]:
    if n <= 0:
        raise ValueError("n must be > 0: " + str(n))
    scored: List[Tuple[float, str]] = []
    for x in possibilities:
        s = SequenceMatcher(x, word)
        if s.real_quick_ratio() >= cutoff and s.quick_ratio() >= cutoff and s.ratio() >= cutoff:
            scored.append((s.ratio(), x))
    scored.sort(reverse=True)
    out: List[str] = []
    for t in scored[:n]:
        out.append(t[1])
    return out

def _format_range_unified(start: int, stop: int) -> str:
    beginning = start + 1
    length = stop - start
    if length == 1:
        return str(beginning)
    if length == 0:
        beginning -= 1
    return str(beginning) + "," + str(length)

def unified_diff(a: List[str], b: List[str], fromfile: str = "", tofile: str = "",
                 fromfiledate: str = "", tofiledate: str = "", n: int = 3,
                 lineterm: str = "\n") -> List[str]:
    out: List[str] = []
    started = False
    for group in SequenceMatcher(a, b).get_grouped_opcodes(n):
        if not started:
            started = True
            fd = "\t" + fromfiledate if fromfiledate != "" else ""
            td = "\t" + tofiledate if tofiledate != "" else ""
            out.append("--- " + fromfile + fd + lineterm)
            out.append("+++ " + tofile + td + lineterm)
        first = group[0]
        last = group[-1]
        out.append("@@ -" + _format_range_unified(first[1], last[2]) + " +" +
                   _format_range_unified(first[3], last[4]) + " @@" + lineterm)
        for g in group:
            tag, i1, i2, j1, j2 = g
            if tag == "equal":
                for line in a[i1:i2]:
                    out.append(" " + line)
                continue
            if tag == "replace" or tag == "delete":
                for line in a[i1:i2]:
                    out.append("-" + line)
            if tag == "replace" or tag == "insert":
                for line in b[j1:j2]:
                    out.append("+" + line)
    return out

def ndiff(a: List[str], b: List[str]) -> List[str]:
    # Simplified line-level ndiff: no intraline '?' hints.
    out: List[str] = []
    for g in SequenceMatcher(a, b, False).get_opcodes():
        tag, i1, i2, j1, j2 = g
        if tag == "equal":
            for line in a[i1:i2]:
                out.append("  " + line)
        else:
            if tag == "replace" or tag == "delete":
                for line in a[i1:i2]:
                    out.append("- " + line)
            if tag == "replace" or tag == "insert":
                for line in b[j1:j2]:
                    out.append("+ " + line)
    return out

def _format_range_context(start: int, stop: int) -> str:
    beginning = start + 1
    length = stop - start
    if length == 0:
        beginning -= 1
    if length <= 1:
        return str(beginning)
    return str(beginning) + "," + str(beginning + length - 1)

def _ctx_prefix(tag: str) -> str:
    if tag == "insert":
        return "+ "
    if tag == "delete":
        return "- "
    if tag == "replace":
        return "! "
    return "  "

def context_diff(a: List[str], b: List[str], fromfile: str = "", tofile: str = "",
                 fromfiledate: str = "", tofiledate: str = "", n: int = 3,
                 lineterm: str = "\n") -> List[str]:
    out: List[str] = []
    started = False
    for group in SequenceMatcher(a, b).get_grouped_opcodes(n):
        if not started:
            started = True
            fd = "\t" + fromfiledate if fromfiledate != "" else ""
            td = "\t" + tofiledate if tofiledate != "" else ""
            out.append("*** " + fromfile + fd + lineterm)
            out.append("--- " + tofile + td + lineterm)
        first = group[0]
        last = group[-1]
        out.append("***************" + lineterm)
        out.append("*** " + _format_range_context(first[1], last[2]) + " ****" + lineterm)
        has_from = False
        has_to = False
        for g in group:
            if g[0] == "replace" or g[0] == "delete":
                has_from = True
            if g[0] == "replace" or g[0] == "insert":
                has_to = True
        if has_from:
            for g in group:
                if g[0] != "insert":
                    for line in a[g[1]:g[2]]:
                        out.append(_ctx_prefix(g[0]) + line)
        out.append("--- " + _format_range_context(first[3], last[4]) + " ----" + lineterm)
        if has_to:
            for g in group:
                if g[0] != "delete":
                    for line in b[g[3]:g[4]]:
                        out.append(_ctx_prefix(g[0]) + line)
    return out
