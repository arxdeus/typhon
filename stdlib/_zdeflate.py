# Ported from zlib 1.2.12 deflate.c / trees.c (zlib license): byte-identical output to zlib's
# deflate for levels 1-9 with the default strategy, memLevel 8, any window size 9..15.

_EXTRA_L = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0]
_EXTRA_D = [0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13]
_EXTRA_BL = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7]
_BL_ORDER = [16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15]

_HEAP_SIZE = 573

class _Tree:
    freq: List[int]
    code: List[int]
    ln: List[int]
    dad: List[int]

    def __init__(self, n: int):
        self.freq = [0] * n
        self.code = [0] * n
        self.ln = [0] * n
        self.dad = [0] * n

def _bi_reverse(code: int, ln: int) -> int:
    res = 0
    while True:
        res |= code & 1
        code >>= 1
        res <<= 1
        ln -= 1
        if ln <= 0:
            break
    return res >> 1

def _gen_codes(tree: _Tree, max_code: int, bl_count: List[int]):
    next_code = [0] * 16
    code = 0
    for bits in range(1, 16):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code
    for n in range(max_code + 1):
        ln = tree.ln[n]
        if ln == 0:
            continue
        tree.code[n] = _bi_reverse(next_code[ln], ln)
        next_code[ln] += 1

def _build_static():
    length_code = [0] * 256
    base_length = [0] * 29
    dist_code = [0] * 512
    base_dist = [0] * 30
    length = 0
    code = 0
    while code < 28:
        base_length[code] = length
        for _ in range(1 << _EXTRA_L[code]):
            length_code[length] = code
            length += 1
        code += 1
    length_code[length - 1] = code
    dist = 0
    code = 0
    while code < 16:
        base_dist[code] = dist
        for _ in range(1 << _EXTRA_D[code]):
            dist_code[dist] = code
            dist += 1
        code += 1
    dist >>= 7
    while code < 30:
        base_dist[code] = dist << 7
        for _ in range(1 << (_EXTRA_D[code] - 7)):
            dist_code[256 + dist] = code
            dist += 1
        code += 1
    bl_count = [0] * 16
    slt = _Tree(288)
    n = 0
    while n <= 143:
        slt.ln[n] = 8
        n += 1
        bl_count[8] += 1
    while n <= 255:
        slt.ln[n] = 9
        n += 1
        bl_count[9] += 1
    while n <= 279:
        slt.ln[n] = 7
        n += 1
        bl_count[7] += 1
    while n <= 287:
        slt.ln[n] = 8
        n += 1
        bl_count[8] += 1
    _gen_codes(slt, 287, bl_count)
    sdt = _Tree(30)
    for i in range(30):
        sdt.ln[i] = 5
        sdt.code[i] = _bi_reverse(i, 5)
    return (length_code, base_length, dist_code, base_dist, slt, sdt)

_ST = _build_static()

# level -> (good_length, max_lazy, nice_length, max_chain, fast)
_CFG = [
    (0, 0, 0, 0, 0), (4, 4, 8, 4, 1), (4, 5, 16, 8, 1), (4, 6, 32, 32, 1),
    (4, 4, 16, 16, 0), (8, 16, 32, 32, 0), (8, 32, 128, 128, 0),
    (8, 32, 128, 256, 0), (32, 128, 258, 1024, 0), (32, 258, 258, 4096, 0),
]

class _Deflate:
    level: int
    wsize: int
    wmask: int
    window_size: int
    hash_mask: int
    window: Ptr[u8]
    prev: List[int]
    head: List[int]
    ins_h: int
    strstart: int
    lookahead: int
    match_start: int
    match_length: int
    prev_length: int
    prev_match: int
    match_available: int
    block_start: int
    insert: int
    max_chain: int
    good_match: int
    nice_match: int
    max_lazy: int
    data: bytes
    inpos: int
    out: List[int]
    bitbuf: int
    bitcnt: int
    sym_d: List[int]
    sym_lc: List[int]
    sym_n: int
    dyn_l: _Tree
    dyn_d: _Tree
    bl: _Tree
    l_max: int
    d_max: int
    heap: List[int]
    heap_len: int
    heap_max: int
    depth: List[int]
    bl_count: List[int]
    opt_len: int
    static_len: int

    def __init__(self, data: bytes, level: int, w_bits: int):
        self.level = level
        self.wsize = 1 << w_bits
        self.wmask = self.wsize - 1
        self.window_size = 2 * self.wsize
        self.hash_mask = 32767
        self.window = Ptr[u8](self.window_size)
        str.memset(self.window.as_byte(), u8(0), self.window_size)
        self.prev = [0] * self.wsize
        self.head = [0] * 32768
        self.ins_h = 0
        self.strstart = 0
        self.lookahead = 0
        self.match_start = 0
        self.match_length = 2
        self.prev_length = 2
        self.prev_match = 0
        self.match_available = 0
        self.block_start = 0
        self.insert = 0
        c = _CFG[level]
        self.good_match = c[0]
        self.max_lazy = c[1]
        self.nice_match = c[2]
        self.max_chain = c[3]
        self.data = data
        self.inpos = 0
        self.out = []
        self.bitbuf = 0
        self.bitcnt = 0
        self.sym_d = [0] * 16384
        self.sym_lc = [0] * 16384
        self.sym_n = 0
        self.dyn_l = _Tree(_HEAP_SIZE)
        self.dyn_d = _Tree(61)
        self.bl = _Tree(39)
        self.l_max = 0
        self.d_max = 0
        self.heap = [0] * _HEAP_SIZE
        self.heap_len = 0
        self.heap_max = 0
        self.depth = [0] * _HEAP_SIZE
        self.bl_count = [0] * 16
        self.opt_len = 0
        self.static_len = 0
        self._init_block()

    # ---- bit output ----
    def _send_bits(self, value: int, length: int):
        self.bitbuf |= value << self.bitcnt
        self.bitcnt += length
        while self.bitcnt >= 8:
            self.out.append(self.bitbuf & 255)
            self.bitbuf >>= 8
            self.bitcnt -= 8

    def _windup(self):
        if self.bitcnt > 0:
            self.out.append(self.bitbuf & 255)
        self.bitbuf = 0
        self.bitcnt = 0

    # ---- trees ----
    def _init_block(self):
        for n in range(286):
            self.dyn_l.freq[n] = 0
        for n in range(30):
            self.dyn_d.freq[n] = 0
        for n in range(19):
            self.bl.freq[n] = 0
        self.dyn_l.freq[256] = 1
        self.opt_len = 0
        self.static_len = 0
        self.sym_n = 0

    def _smaller(self, t: _Tree, n: int, m: int) -> bool:
        return t.freq[n] < t.freq[m] or (t.freq[n] == t.freq[m] and self.depth[n] <= self.depth[m])

    def _pqdownheap(self, t: _Tree, k: int):
        v = self.heap[k]
        j = k << 1
        while j <= self.heap_len:
            if j < self.heap_len and self._smaller(t, self.heap[j + 1], self.heap[j]):
                j += 1
            if self._smaller(t, v, self.heap[j]):
                break
            self.heap[k] = self.heap[j]
            k = j
            j <<= 1
        self.heap[k] = v

    def _gen_bitlen(self, t: _Tree, max_code: int, has_stree: bool, stree: _Tree,
                    extra: List[int], base: int, max_length: int):
        overflow = 0
        for b in range(16):
            self.bl_count[b] = 0
        t.ln[self.heap[self.heap_max]] = 0
        h = self.heap_max + 1
        while h < _HEAP_SIZE:
            n = self.heap[h]
            bits = t.ln[t.dad[n]] + 1
            if bits > max_length:
                bits = max_length
                overflow += 1
            t.ln[n] = bits
            h += 1
            if n > max_code:
                continue
            self.bl_count[bits] += 1
            xbits = 0
            if n >= base:
                xbits = extra[n - base]
            f = t.freq[n]
            self.opt_len += f * (bits + xbits)
            if has_stree:
                self.static_len += f * (stree.ln[n] + xbits)
        if overflow == 0:
            return
        while True:
            bits = max_length - 1
            while self.bl_count[bits] == 0:
                bits -= 1
            self.bl_count[bits] -= 1
            self.bl_count[bits + 1] += 2
            self.bl_count[max_length] -= 1
            overflow -= 2
            if overflow <= 0:
                break
        h = _HEAP_SIZE
        bits = max_length
        while bits != 0:
            n = self.bl_count[bits]
            while n != 0:
                h -= 1
                m = self.heap[h]
                if m > max_code:
                    continue
                if t.ln[m] != bits:
                    self.opt_len += (bits - t.ln[m]) * t.freq[m]
                    t.ln[m] = bits
                n -= 1
            bits -= 1

    def _build_tree(self, t: _Tree, has_stree: bool, stree: _Tree, extra: List[int],
                    base: int, elems: int, max_length: int) -> int:
        max_code = -1
        self.heap_len = 0
        self.heap_max = _HEAP_SIZE
        for n in range(elems):
            if t.freq[n] != 0:
                self.heap_len += 1
                self.heap[self.heap_len] = n
                max_code = n
                self.depth[n] = 0
            else:
                t.ln[n] = 0
        while self.heap_len < 2:
            node = 0
            if max_code < 2:
                max_code += 1
                node = max_code
            self.heap_len += 1
            self.heap[self.heap_len] = node
            t.freq[node] = 1
            self.depth[node] = 0
            self.opt_len -= 1
            if has_stree:
                self.static_len -= stree.ln[node]
        n = self.heap_len // 2
        while n >= 1:
            self._pqdownheap(t, n)
            n -= 1
        node = elems
        while True:
            n = self.heap[1]
            self.heap[1] = self.heap[self.heap_len]
            self.heap_len -= 1
            self._pqdownheap(t, 1)
            m = self.heap[1]
            self.heap_max -= 1
            self.heap[self.heap_max] = n
            self.heap_max -= 1
            self.heap[self.heap_max] = m
            t.freq[node] = t.freq[n] + t.freq[m]
            self.depth[node] = (self.depth[n] if self.depth[n] >= self.depth[m] else self.depth[m]) + 1
            t.dad[n] = node
            t.dad[m] = node
            self.heap[1] = node
            node += 1
            self._pqdownheap(t, 1)
            if self.heap_len < 2:
                break
        self.heap_max -= 1
        self.heap[self.heap_max] = self.heap[1]
        self._gen_bitlen(t, max_code, has_stree, stree, extra, base, max_length)
        _gen_codes(t, max_code, self.bl_count)
        return max_code

    def _scan_tree(self, t: _Tree, max_code: int):
        prevlen = -1
        nextlen = t.ln[0]
        count = 0
        max_count = 7
        min_count = 4
        if nextlen == 0:
            max_count = 138
            min_count = 3
        t.ln[max_code + 1] = 0xFFFF
        for n in range(max_code + 1):
            curlen = nextlen
            nextlen = t.ln[n + 1]
            count += 1
            if count < max_count and curlen == nextlen:
                continue
            elif count < min_count:
                self.bl.freq[curlen] += count
            elif curlen != 0:
                if curlen != prevlen:
                    self.bl.freq[curlen] += 1
                self.bl.freq[16] += 1
            elif count <= 10:
                self.bl.freq[17] += 1
            else:
                self.bl.freq[18] += 1
            count = 0
            prevlen = curlen
            if nextlen == 0:
                max_count = 138
                min_count = 3
            elif curlen == nextlen:
                max_count = 6
                min_count = 3
            else:
                max_count = 7
                min_count = 4

    def _send_code(self, c: int, t: _Tree):
        self._send_bits(t.code[c], t.ln[c])

    def _send_tree(self, t: _Tree, max_code: int):
        prevlen = -1
        nextlen = t.ln[0]
        count = 0
        max_count = 7
        min_count = 4
        if nextlen == 0:
            max_count = 138
            min_count = 3
        for n in range(max_code + 1):
            curlen = nextlen
            nextlen = t.ln[n + 1]
            count += 1
            if count < max_count and curlen == nextlen:
                continue
            elif count < min_count:
                while True:
                    self._send_code(curlen, self.bl)
                    count -= 1
                    if count == 0:
                        break
            elif curlen != 0:
                if curlen != prevlen:
                    self._send_code(curlen, self.bl)
                    count -= 1
                self._send_code(16, self.bl)
                self._send_bits(count - 3, 2)
            elif count <= 10:
                self._send_code(17, self.bl)
                self._send_bits(count - 3, 3)
            else:
                self._send_code(18, self.bl)
                self._send_bits(count - 11, 7)
            count = 0
            prevlen = curlen
            if nextlen == 0:
                max_count = 138
                min_count = 3
            elif curlen == nextlen:
                max_count = 6
                min_count = 3
            else:
                max_count = 7
                min_count = 4

    def _build_bl_tree(self) -> int:
        self._scan_tree(self.dyn_l, self.l_max)
        self._scan_tree(self.dyn_d, self.d_max)
        dummy = _ST[5]
        self._build_tree(self.bl, False, dummy, _EXTRA_BL, 0, 19, 7)
        mb = 18
        while mb >= 3:
            if self.bl.ln[_BL_ORDER[mb]] != 0:
                break
            mb -= 1
        self.opt_len += 3 * (mb + 1) + 5 + 5 + 4
        return mb

    def _send_all_trees(self, lcodes: int, dcodes: int, blcodes: int):
        self._send_bits(lcodes - 257, 5)
        self._send_bits(dcodes - 1, 5)
        self._send_bits(blcodes - 4, 4)
        for rank in range(blcodes):
            self._send_bits(self.bl.ln[_BL_ORDER[rank]], 3)
        self._send_tree(self.dyn_l, lcodes - 1)
        self._send_tree(self.dyn_d, dcodes - 1)

    def _compress_block(self, lt: _Tree, dt: _Tree):
        for sx in range(self.sym_n):
            dist = self.sym_d[sx]
            lc = self.sym_lc[sx]
            if dist == 0:
                self._send_code(lc, lt)
            else:
                code = _ST[0][lc]
                self._send_code(code + 257, lt)
                extra = _EXTRA_L[code]
                if extra != 0:
                    lc -= _ST[1][code]
                    self._send_bits(lc, extra)
                dist -= 1
                dcode = _ST[2][dist] if dist < 256 else _ST[2][256 + (dist >> 7)]
                self._send_code(dcode, dt)
                extra = _EXTRA_D[dcode]
                if extra != 0:
                    dist -= _ST[3][dcode]
                    self._send_bits(dist, extra)
        self._send_code(256, lt)

    def _stored_block(self, buf: int, stored_len: int, last: int):
        self._send_bits(last, 3)
        self._windup()
        self.out.append(stored_len & 255)
        self.out.append((stored_len >> 8) & 255)
        self.out.append((~stored_len) & 255)
        self.out.append(((~stored_len) >> 8) & 255)
        for i in range(stored_len):
            self.out.append(int(self.window[buf + i]))

    def _tr_flush_block(self, buf: int, stored_len: int, last: int):
        slt = _ST[4]
        sdt = _ST[5]
        self.l_max = self._build_tree(self.dyn_l, True, slt, _EXTRA_L, 257, 286, 15)
        self.d_max = self._build_tree(self.dyn_d, True, sdt, _EXTRA_D, 0, 30, 15)
        max_blindex = self._build_bl_tree()
        opt_lenb = (self.opt_len + 3 + 7) >> 3
        static_lenb = (self.static_len + 3 + 7) >> 3
        if static_lenb <= opt_lenb:
            opt_lenb = static_lenb
        if stored_len + 4 <= opt_lenb and buf >= 0:
            self._stored_block(buf, stored_len, last)
        elif static_lenb == opt_lenb:
            self._send_bits(2 + last, 3)
            self._compress_block(slt, sdt)
        else:
            self._send_bits(4 + last, 3)
            self._send_all_trees(self.l_max + 1, self.d_max + 1, max_blindex + 1)
            self._compress_block(self.dyn_l, self.dyn_d)
        self._init_block()
        if last:
            self._windup()

    def _tally_lit(self, c: int) -> bool:
        self.sym_d[self.sym_n] = 0
        self.sym_lc[self.sym_n] = c
        self.sym_n += 1
        self.dyn_l.freq[c] += 1
        return self.sym_n == 16383

    def _tally_dist(self, dist: int, lc: int) -> bool:
        self.sym_d[self.sym_n] = dist
        self.sym_lc[self.sym_n] = lc
        self.sym_n += 1
        dist -= 1
        self.dyn_l.freq[_ST[0][lc] + 257] += 1
        dcode = _ST[2][dist] if dist < 256 else _ST[2][256 + (dist >> 7)]
        self.dyn_d.freq[dcode] += 1
        return self.sym_n == 16383

    def _flush_block(self, last: int):
        bs = self.block_start
        self._tr_flush_block(bs, self.strstart - bs, last)
        self.block_start = self.strstart

    # ---- deflate.c ----
    def _update_hash(self, c: int):
        self.ins_h = ((self.ins_h << 5) ^ c) & self.hash_mask

    def _insert_string(self, s: int) -> int:
        self._update_hash(int(self.window[s + 2]))
        mh = self.head[self.ins_h]
        self.prev[s & self.wmask] = mh
        self.head[self.ins_h] = s
        return mh

    def _slide_hash(self):
        ws = self.wsize
        for n in range(32768):
            m = self.head[n]
            self.head[n] = m - ws if m >= ws else 0
        for n in range(ws):
            m = self.prev[n]
            self.prev[n] = m - ws if m >= ws else 0

    def _fill_window(self):
        ws = self.wsize
        total = len(self.data)
        while True:
            more = self.window_size - self.lookahead - self.strstart
            if self.strstart >= ws + (ws - 262):
                cnt = ws - more
                if cnt > 0:
                    str.memcpy(self.window.as_byte(), (self.window + ws).as_byte(), cnt)
                self.match_start -= ws
                self.strstart -= ws
                self.block_start -= ws
                if self.insert > self.strstart:
                    self.insert = self.strstart
                self._slide_hash()
                more += ws
            if self.inpos >= total:
                break
            n = total - self.inpos
            if n > more:
                n = more
            base = self.strstart + self.lookahead
            for i in range(n):
                self.window[base + i] = u8(int(self.data[self.inpos + i]))
            self.inpos += n
            self.lookahead += n
            if self.lookahead + self.insert >= 3:
                st = self.strstart - self.insert
                self.ins_h = int(self.window[st])
                self._update_hash(int(self.window[st + 1]))
                while self.insert > 0:
                    self._update_hash(int(self.window[st + 2]))
                    self.prev[st & self.wmask] = self.head[self.ins_h]
                    self.head[self.ins_h] = st
                    st += 1
                    self.insert -= 1
                    if self.lookahead + self.insert < 3:
                        break
            if not (self.lookahead < 262 and self.inpos < total):
                break

    def _longest_match(self, cur_match: int) -> int:
        chain_length = self.max_chain
        w = self.window
        scan = self.strstart
        best_len = self.prev_length
        nice = self.nice_match
        maxd = self.wsize - 262
        limit = scan - maxd if scan > maxd else 0
        if self.prev_length >= self.good_match:
            chain_length >>= 2
        if nice > self.lookahead:
            nice = self.lookahead
        scan_end1 = int(w[scan + best_len - 1])
        scan_end = int(w[scan + best_len])
        w0 = int(w[scan])
        w1 = int(w[scan + 1])
        while True:
            m = cur_match
            if int(w[m + best_len]) == scan_end and int(w[m + best_len - 1]) == scan_end1 and int(w[m]) == w0 and int(w[m + 1]) == w1:
                ln = 3
                while ln < 258 and w[scan + ln] == w[m + ln]:
                    ln += 1
                if ln > best_len:
                    self.match_start = cur_match
                    best_len = ln
                    if ln >= nice:
                        break
                    scan_end1 = int(w[scan + best_len - 1])
                    scan_end = int(w[scan + best_len])
            cur_match = self.prev[cur_match & self.wmask]
            if cur_match <= limit:
                break
            chain_length -= 1
            if chain_length == 0:
                break
        if best_len <= self.lookahead:
            return best_len
        return self.lookahead

    def _deflate_slow(self):
        maxd = self.wsize - 262
        while True:
            if self.lookahead < 262:
                self._fill_window()
                if self.lookahead == 0:
                    break
            hash_head = 0
            if self.lookahead >= 3:
                hash_head = self._insert_string(self.strstart)
            self.prev_length = self.match_length
            self.prev_match = self.match_start
            self.match_length = 2
            if hash_head != 0 and self.prev_length < self.max_lazy and self.strstart - hash_head <= maxd:
                self.match_length = self._longest_match(hash_head)
                if self.match_length <= 5 and (self.match_length == 3 and self.strstart - self.match_start > 4096):
                    self.match_length = 2
            if self.prev_length >= 3 and self.match_length <= self.prev_length:
                max_insert = self.strstart + self.lookahead - 3
                bflush = self._tally_dist(self.strstart - 1 - self.prev_match, self.prev_length - 3)
                self.lookahead -= self.prev_length - 1
                self.prev_length -= 2
                while True:
                    self.strstart += 1
                    if self.strstart <= max_insert:
                        hash_head = self._insert_string(self.strstart)
                    self.prev_length -= 1
                    if self.prev_length == 0:
                        break
                self.match_available = 0
                self.match_length = 2
                self.strstart += 1
                if bflush:
                    self._flush_block(0)
            elif self.match_available != 0:
                bflush = self._tally_lit(int(self.window[self.strstart - 1]))
                if bflush:
                    self._flush_block(0)
                self.strstart += 1
                self.lookahead -= 1
            else:
                self.match_available = 1
                self.strstart += 1
                self.lookahead -= 1
        if self.match_available != 0:
            self._tally_lit(int(self.window[self.strstart - 1]))
            self.match_available = 0
        self._flush_block(1)

    def _deflate_fast(self):
        maxd = self.wsize - 262
        while True:
            if self.lookahead < 262:
                self._fill_window()
                if self.lookahead == 0:
                    break
            hash_head = 0
            if self.lookahead >= 3:
                hash_head = self._insert_string(self.strstart)
            if hash_head != 0 and self.strstart - hash_head <= maxd:
                self.match_length = self._longest_match(hash_head)
            if self.match_length >= 3:
                bflush = self._tally_dist(self.strstart - self.match_start, self.match_length - 3)
                self.lookahead -= self.match_length
                if self.match_length <= self.max_lazy and self.lookahead >= 3:
                    self.match_length -= 1
                    while True:
                        self.strstart += 1
                        hash_head = self._insert_string(self.strstart)
                        self.match_length -= 1
                        if self.match_length == 0:
                            break
                    self.strstart += 1
                else:
                    self.strstart += self.match_length
                    self.match_length = 0
                    self.ins_h = int(self.window[self.strstart])
                    self._update_hash(int(self.window[self.strstart + 1]))
            else:
                bflush = self._tally_lit(int(self.window[self.strstart]))
                self.lookahead -= 1
                self.strstart += 1
            if bflush:
                self._flush_block(0)
        self._flush_block(1)

    def run(self) -> List[int]:
        if _CFG[self.level][4] == 1:
            self._deflate_fast()
        else:
            self._deflate_slow()
        return self.out
