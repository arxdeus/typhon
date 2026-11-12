# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# Multiple minimum degree ordering (SuperLU mmd.c: genmmd / mmdint /
# mmdelm / mmdupd / mmdnum, itself an f2c translation of SPARSPAK's GENMMD)
# and the A'A / A'+A structure builders of SuperLU get_perm_c.c. All arrays
# are 1-based (index 0 unused) exactly as in the C code; the goto-based
# control flow is rewritten as structured loops with the same visiting order.

def _mmdint(neqns: int, xadj: List[int], dhead: List[int], dforw: List[int],
            dbakw: List[int], qsize: List[int], llist: List[int],
            marker: List[int]):
    for node in range(1, neqns + 1):
        dhead[node] = 0
        qsize[node] = 1
        marker[node] = 0
        llist[node] = 0
    for node in range(1, neqns + 1):
        ndeg = xadj[node + 1] - xadj[node] + 1
        fnode = dhead[ndeg]
        dforw[node] = fnode
        dhead[ndeg] = node
        if fnode > 0:
            dbakw[fnode] = node
        dbakw[node] = -ndeg

def _mmdelm(mdnode: int, xadj: List[int], adjncy: List[int], dhead: List[int],
            dforw: List[int], dbakw: List[int], qsize: List[int],
            llist: List[int], marker: List[int], maxint: int, tag: int):
    marker[mdnode] = tag
    istrt = xadj[mdnode]
    istop = xadj[mdnode + 1] - 1
    elmnt = 0
    rloc = istrt
    rlmt = istop
    for i in range(istrt, istop + 1):
        nabor = adjncy[i]
        if nabor == 0:
            break
        if marker[nabor] >= tag:
            continue
        marker[nabor] = tag
        if dforw[nabor] < 0:
            llist[nabor] = elmnt
            elmnt = nabor
        else:
            adjncy[rloc] = nabor
            rloc += 1
    # L300: merge with reachable nodes from generalized elements
    while elmnt > 0:
        adjncy[rlmt] = -elmnt
        link = elmnt
        done = False
        while not done:
            jstrt = xadj[link]
            jstop = xadj[link + 1] - 1
            restart = False
            for j in range(jstrt, jstop + 1):
                node = adjncy[j]
                link = -node
                if node < 0:
                    restart = True
                    break
                elif node == 0:
                    break
                if marker[node] >= tag or dforw[node] < 0:
                    continue
                marker[node] = tag
                while rloc >= rlmt:
                    link2 = -adjncy[rlmt]
                    rloc = xadj[link2]
                    rlmt = xadj[link2 + 1] - 1
                adjncy[rloc] = node
                rloc += 1
            if not restart:
                done = True
        elmnt = llist[elmnt]
    # L1000
    if rloc <= rlmt:
        adjncy[rloc] = 0
    link = mdnode
    done = False
    while not done:
        istrt = xadj[link]
        istop = xadj[link + 1] - 1
        restart = False
        for i in range(istrt, istop + 1):
            rnode = adjncy[i]
            link = -rnode
            if rnode < 0:
                restart = True
                break
            elif rnode == 0:
                break
            pvnode = dbakw[rnode]
            if pvnode != 0 and pvnode != -maxint:
                nxnode = dforw[rnode]
                if nxnode > 0:
                    dbakw[nxnode] = pvnode
                if pvnode > 0:
                    dforw[pvnode] = nxnode
                npv = -pvnode
                if pvnode < 0:
                    dhead[npv] = nxnode
            jstrt = xadj[rnode]
            jstop = xadj[rnode + 1] - 1
            xqnbr = jstrt
            for j in range(jstrt, jstop + 1):
                nabor = adjncy[j]
                if nabor == 0:
                    break
                if marker[nabor] >= tag:
                    continue
                adjncy[xqnbr] = nabor
                xqnbr += 1
            nqnbrs = xqnbr - jstrt
            if nqnbrs <= 0:
                qsize[mdnode] += qsize[rnode]
                qsize[rnode] = 0
                marker[rnode] = maxint
                dforw[rnode] = -mdnode
                dbakw[rnode] = -maxint
            else:
                dforw[rnode] = nqnbrs + 1
                dbakw[rnode] = 0
                adjncy[xqnbr] = mdnode
                xqnbr += 1
                if xqnbr <= jstop:
                    adjncy[xqnbr] = 0
        if not restart:
            done = True

def _reset_marker(neqns: int, marker: List[int], maxint: int):
    for i in range(1, neqns + 1):
        if marker[i] < maxint:
            marker[i] = 0

def _mmdupd(ehead: int, neqns: int, xadj: List[int], adjncy: List[int],
            delta: int, mdeg: int, dhead: List[int], dforw: List[int],
            dbakw: List[int], qsize: List[int], llist: List[int],
            marker: List[int], maxint: int, tag: int):
    # returns (mdeg, tag)
    mdeg0 = mdeg + delta
    elmnt = ehead
    while elmnt > 0:
        mtag = tag + mdeg0
        if mtag >= maxint:
            tag = 1
            _reset_marker(neqns, marker, maxint)
            mtag = tag + mdeg0
        q2head = 0
        qxhead = 0
        deg0 = 0
        link = elmnt
        done = False
        while not done:
            istrt = xadj[link]
            istop = xadj[link + 1] - 1
            restart = False
            for i in range(istrt, istop + 1):
                enode = adjncy[i]
                link = -enode
                if enode < 0:
                    restart = True
                    break
                elif enode == 0:
                    break
                if qsize[enode] == 0:
                    continue
                deg0 += qsize[enode]
                marker[enode] = mtag
                if dbakw[enode] != 0:
                    continue
                if dforw[enode] == 2:
                    llist[enode] = q2head
                    q2head = enode
                else:
                    llist[enode] = qxhead
                    qxhead = enode
            if not restart:
                done = True
        # L800 / L1500: two passes, first the q2 list then the qx list
        for iq2 in range(1, -1, -1):
            enode = q2head if iq2 == 1 else qxhead
            while enode > 0:
                if dbakw[enode] != 0:
                    enode = llist[enode]
                    continue
                tag += 1
                deg = deg0
                if iq2 == 1:
                    istrt = xadj[enode]
                    nabor = adjncy[istrt]
                    if nabor == elmnt:
                        nabor = adjncy[istrt + 1]
                    link = nabor
                    if dforw[nabor] >= 0:
                        deg += qsize[nabor]
                    else:
                        done = False
                        while not done:
                            istrt = xadj[link]
                            istop = xadj[link + 1] - 1
                            restart = False
                            for i in range(istrt, istop + 1):
                                node = adjncy[i]
                                link = -node
                                if node == enode:
                                    continue
                                if node < 0:
                                    restart = True
                                    break
                                elif node == 0:
                                    break
                                if qsize[node] == 0:
                                    continue
                                if marker[node] < tag:
                                    marker[node] = tag
                                    deg += qsize[node]
                                    continue
                                if dbakw[node] != 0:
                                    continue
                                if dforw[node] == 2:
                                    qsize[enode] += qsize[node]
                                    qsize[node] = 0
                                    marker[node] = maxint
                                    dforw[node] = -enode
                                    dbakw[node] = -maxint
                                    continue
                                if dbakw[node] == 0:
                                    dbakw[node] = -maxint
                            if not restart:
                                done = True
                else:
                    istrt = xadj[enode]
                    istop = xadj[enode + 1] - 1
                    for i in range(istrt, istop + 1):
                        nabor = adjncy[i]
                        if nabor == 0:
                            break
                        if marker[nabor] >= tag:
                            continue
                        marker[nabor] = tag
                        link = nabor
                        if dforw[nabor] >= 0:
                            deg += qsize[nabor]
                            continue
                        done = False
                        while not done:
                            jstrt = xadj[link]
                            jstop = xadj[link + 1] - 1
                            restart = False
                            for j in range(jstrt, jstop + 1):
                                node = adjncy[j]
                                link = -node
                                if node < 0:
                                    restart = True
                                    break
                                elif node == 0:
                                    break
                                if marker[node] >= tag:
                                    continue
                                marker[node] = tag
                                deg += qsize[node]
                            if not restart:
                                done = True
                # L2100
                deg = deg - qsize[enode] + 1
                fnode = dhead[deg]
                dforw[enode] = fnode
                dbakw[enode] = -deg
                if fnode > 0:
                    dbakw[fnode] = enode
                dhead[deg] = enode
                if deg < mdeg:
                    mdeg = deg
                enode = llist[enode]
        tag = mtag
        elmnt = llist[elmnt]
    return mdeg, tag

def _mmdnum(neqns: int, perm: List[int], invp: List[int], qsize: List[int]):
    for node in range(1, neqns + 1):
        if qsize[node] <= 0:
            perm[node] = invp[node]
        else:
            perm[node] = -invp[node]
    for node in range(1, neqns + 1):
        if perm[node] > 0:
            continue
        father = node
        while perm[father] <= 0:
            father = -perm[father]
        root = father
        num = perm[root] + 1
        invp[node] = -num
        perm[root] = num
        father = node
        while True:
            nextf = -perm[father]
            if nextf <= 0:
                break
            perm[father] = -root
            father = nextf
    for node in range(1, neqns + 1):
        num = -invp[node]
        invp[node] = num
        perm[num] = node

def genmmd(neqns: int, xadj: List[int], adjncy: List[int], delta: int = 0,
           maxint: int = 2147483647) -> List[int]:
    # xadj / adjncy 1-based (index 0 unused, values 1-based). Returns the
    # 0-based perm_c as computed by get_perm_c (perm_c[i] = invp[i+1] - 1).
    # Note: SuperLU passes (perm_c, invp) as genmmd's (invp, perm) arguments.
    invp = [0] * (neqns + 2)
    perm = [0] * (neqns + 2)
    dhead = [0] * (neqns + 2)
    qsize = [0] * (neqns + 2)
    llist = [0] * (neqns + 2)
    marker = [0] * (neqns + 2)
    if neqns <= 0:
        return List[int]()
    _mmdint(neqns, xadj, dhead, invp, perm, qsize, llist, marker)
    num = 1
    nextmd = dhead[1]
    while nextmd > 0:
        mdnode = nextmd
        nextmd = invp[mdnode]
        marker[mdnode] = maxint
        invp[mdnode] = -num
        num += 1
    if num <= neqns:
        tag = 1
        dhead[1] = 0
        mdeg = 2
        finished = False
        while not finished:
            while dhead[mdeg] <= 0:
                mdeg += 1
            mdlmt = mdeg + delta
            ehead = 0
            while True:
                mdnode = dhead[mdeg]
                if mdnode <= 0:
                    mdeg += 1
                    if mdeg > mdlmt:
                        break
                    continue
                nextmd = invp[mdnode]
                dhead[mdeg] = nextmd
                if nextmd > 0:
                    perm[nextmd] = -mdeg
                invp[mdnode] = -num
                if num + qsize[mdnode] > neqns:
                    finished = True
                    break
                tag += 1
                if tag >= maxint:
                    tag = 1
                    _reset_marker(neqns, marker, maxint)
                _mmdelm(mdnode, xadj, adjncy, dhead, invp, perm, qsize, llist,
                        marker, maxint, tag)
                num += qsize[mdnode]
                llist[mdnode] = ehead
                ehead = mdnode
                if delta < 0:
                    break
            if finished:
                break
            if num > neqns:
                break
            mdeg, tag = _mmdupd(ehead, neqns, xadj, adjncy, delta, mdeg, dhead,
                                invp, perm, qsize, llist, marker, maxint, tag)
    _mmdnum(neqns, perm, invp, qsize)
    out = [0] * neqns
    for i in range(neqns):
        out[i] = invp[i + 1] - 1
    return out

def _transpose_struct(m: int, n: int, colptr, rowind):
    # column-oriented form of T = A' (SuperLU getata / at_plus_a)
    marker = [0] * max(m, n, 1)
    for j in range(n):
        for i in range(colptr[j], colptr[j + 1]):
            marker[rowind[i]] += 1
    t_colptr = [0] * (m + 1)
    for i in range(m):
        t_colptr[i + 1] = t_colptr[i] + marker[i]
        marker[i] = t_colptr[i]
    t_rowind = [0] * max(colptr[n], 1)
    for j in range(n):
        for i in range(colptr[j], colptr[j + 1]):
            col = rowind[i]
            t_rowind[marker[col]] = j
            marker[col] += 1
    return t_colptr, t_rowind

def getata(m: int, n: int, colptr, rowind):
    # structure of A'A without the diagonal
    t_colptr, t_rowind = _transpose_struct(m, n, colptr, rowind)
    marker = [-1] * max(m, n, 1)
    b_colptr = [0] * (n + 1)
    b_rowind = List[int]()
    for j in range(n):
        b_colptr[j] = len(b_rowind)
        marker[j] = j
        for i in range(colptr[j], colptr[j + 1]):
            k = rowind[i]
            for ti in range(t_colptr[k], t_colptr[k + 1]):
                trow = t_rowind[ti]
                if marker[trow] != j:
                    marker[trow] = j
                    b_rowind.append(trow)
    b_colptr[n] = len(b_rowind)
    return b_colptr, b_rowind

def at_plus_a(n: int, colptr, rowind):
    # structure of A + A' without the diagonal
    t_colptr, t_rowind = _transpose_struct(n, n, colptr, rowind)
    marker = [-1] * max(n, 1)
    b_colptr = [0] * (n + 1)
    b_rowind = List[int]()
    for j in range(n):
        b_colptr[j] = len(b_rowind)
        marker[j] = j
        for i in range(colptr[j], colptr[j + 1]):
            k = rowind[i]
            if marker[k] != j:
                marker[k] = j
                b_rowind.append(k)
        for i in range(t_colptr[j], t_colptr[j + 1]):
            k = t_rowind[i]
            if marker[k] != j:
                marker[k] = j
                b_rowind.append(k)
    b_colptr[n] = len(b_rowind)
    return b_colptr, b_rowind

def mmd_order(n: int, b_colptr: List[int], b_rowind: List[int]) -> List[int]:
    # get_perm_c tail: GENMMD on the (1-based) adjacency structure
    bnz = b_colptr[n]
    if bnz == 0:
        return list(range(n))
    xadj = [0] * (n + 2)
    for i in range(n + 1):
        xadj[i + 1] = b_colptr[i] + 1
    adjncy = [0] * (bnz + 2)
    for i in range(bnz):
        adjncy[i + 1] = b_rowind[i] + 1
    return genmmd(n, xadj, adjncy)
