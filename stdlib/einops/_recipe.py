# Ported from einops 0.8.2 (MIT)
#
# Backend independent pattern parsing and recipe construction. Patterns are
# parsed at runtime. Elementary axes are identified by strings; anonymous axes
# (numbers > 1) get unique names "#<k>" so that they never compare equal.

ELLIPSIS = "\u2026"
ELLIPSIS_G = "\u2026g"  # ellipsis inside a parenthesized group

class EinopsError(Exception):
    _f_EinopsError: int
    def __init__(self, message: str = ""):
        super().__init__(message)
        self._f_EinopsError = 0

class _Parsed:
    has_ellipsis: bool
    has_ellipsis_parenthesized: bool
    identifiers: List[str]          # ordered, unique
    has_non_unitary_anonymous_axes: bool
    composition: List[List[str]]    # ellipsis appears as [ELLIPSIS] (top level) or inside a group
    anon: Dict[str, int]

    def __init__(self, expression: str, counter: List[int]):
        self.has_ellipsis = False
        self.has_ellipsis_parenthesized = False
        self.identifiers = []
        self.has_non_unitary_anonymous_axes = False
        self.composition = []
        self.anon = {}
        expr = expression
        if "." in expr:
            if "..." not in expr:
                raise EinopsError("Expression may contain dots only inside ellipsis (...)")
            if expr.count("...") != 1 or expr.count(".") != 3:
                raise EinopsError("Expression may contain dots only inside ellipsis (...); only one ellipsis for tensor ")
            expr = expr.replace("...", "\x01")
            self.has_ellipsis = True

        in_group = False
        group = List[str]()
        cur = ""
        for ch in expr:
            if ch == "(" or ch == ")" or ch == " ":
                if cur:
                    self._add(cur, in_group, group, counter)
                cur = ""
                if ch == "(":
                    if in_group:
                        raise EinopsError("Axis composition is one-level (brackets inside brackets not allowed)")
                    in_group = True
                    group = List[str]()
                elif ch == ")":
                    if not in_group:
                        raise EinopsError("Brackets are not balanced")
                    self.composition.append(group)
                    in_group = False
            elif ch.isalnum() or ch == "_" or ch == "\x01":
                cur += ch
            else:
                raise EinopsError(f"Unknown character '{ch}'")
        if in_group:
            raise EinopsError(f'Imbalanced parentheses in expression: "{expression}"')
        if cur:
            self._add(cur, in_group, group, counter)

    def _add(self, x: str, in_group: bool, group: List[str], counter: List[int]):
        if x == "\x01":
            self.identifiers.append(ELLIPSIS)
            if in_group:
                group.append(ELLIPSIS_G)
                self.has_ellipsis_parenthesized = True
            else:
                self.composition.append([ELLIPSIS])
            return
        if x in self.identifiers:
            raise EinopsError(f'Indexing expression contains duplicate dimension "{x}"')
        is_number = x.isdigit()
        if is_number:
            v = int(x)
            if v == 1:
                if not in_group:
                    self.composition.append(List[str]())
                return
            if v <= 0:
                raise EinopsError(f"Anonymous axis should have positive length, not {v}")
            name = "#" + str(counter[0])
            counter[0] += 1
            self.anon[name] = v
            self.has_non_unitary_anonymous_axes = True
            x = name
        else:
            if not (x[0].isalpha() or x[0] == "_") or x[0] == "_" or x[len(x) - 1] == "_":
                raise EinopsError(f"Invalid axis identifier: {x}")
        self.identifiers.append(x)
        if in_group:
            group.append(x)
        else:
            self.composition.append([x])

class Plan:
    init_shape: List[int]      # empty list + need_init False means skip
    need_init: bool
    perm: List[int]            # empty => identity
    reduced: List[int]
    added: List[Tuple[int, int]]   # (position, length), sorted by position
    n_after_add: int
    final_shape: List[int]
    need_final: bool

    def __init__(self):
        self.init_shape = []
        self.need_init = False
        self.perm = []
        self.reduced = []
        self.added = []
        self.n_after_add = 0
        self.final_shape = []
        self.need_final = False

def _flat(c: List[List[str]]):
    r = List[str]()
    for g in c:
        for a in g:
            r.append(a)
    return r

def make_plan(pattern: str, operation: str, axes_lengths: Dict[str, int], shape: List[int]) -> Plan:
    ndim = len(shape)
    if "->" not in pattern:
        raise EinopsError(f"Pattern should contain '->': {pattern}")
    sides = pattern.split("->")
    counter = [0]
    left = _Parsed(sides[0], counter)
    rght = _Parsed(sides[1], counter)
    if not left.has_ellipsis and rght.has_ellipsis:
        raise EinopsError(f"Ellipsis found in right side, but not left side of a pattern {pattern}")
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise EinopsError(f"Ellipsis inside parenthesis in the left side is not allowed: {pattern}")
    if operation == "rearrange":
        if left.has_non_unitary_anonymous_axes or rght.has_non_unitary_anonymous_axes:
            raise EinopsError("Non-unitary anonymous axes are not supported in rearrange (exception is length 1)")
        for a in left.identifiers:
            if a not in rght.identifiers:
                raise EinopsError(f"Identifiers only on one side of expression (should be on both): {a}")
        for a in rght.identifiers:
            if a not in left.identifiers:
                raise EinopsError(f"Identifiers only on one side of expression (should be on both): {a}")
    elif operation == "repeat":
        for a in left.identifiers:
            if a not in rght.identifiers:
                raise EinopsError(f"Unexpected identifiers on the left side of repeat: {a}")
        for a in rght.identifiers:
            if not a.startswith("#") and a not in left.identifiers and a not in axes_lengths:
                raise EinopsError(f"Specify sizes for new axes in repeat: {a}")
    elif operation in ("min", "max", "sum", "mean", "prod", "any", "all"):
        for a in rght.identifiers:
            if a not in left.identifiers:
                raise EinopsError(f"Unexpected identifiers on the right side of reduce {operation}: {a}")
    else:
        raise EinopsError(f"Unknown reduction {operation}")

    left_comp = List[List[str]]()
    rght_comp = List[List[str]]()
    left_ids = List[str]()
    rght_ids = List[str]()
    if left.has_ellipsis:
        n_other = len(left.composition) - 1
        if ndim < n_other:
            raise EinopsError(f"Wrong shape: expected >={n_other} dims. Received {ndim}-dim tensor.")
        ell = [ELLIPSIS + str(i) for i in range(ndim - n_other)]
        for g in left.composition:
            if len(g) == 1 and g[0] == ELLIPSIS:
                for a in ell:
                    left_comp.append([a])
            else:
                left_comp.append(g)
        for g in rght.composition:
            if len(g) == 1 and g[0] == ELLIPSIS:
                for a in ell:
                    rght_comp.append([a])
            else:
                ng = List[str]()
                for a in g:
                    if a == ELLIPSIS or a == ELLIPSIS_G:
                        ng.extend(ell)
                    else:
                        ng.append(a)
                rght_comp.append(ng)
        for a in left.identifiers:
            if a != ELLIPSIS:
                left_ids.append(a)
        left_ids.extend(ell)
        for a in rght.identifiers:
            if a != ELLIPSIS:
                rght_ids.append(a)
        if rght.has_ellipsis:
            rght_ids.extend(ell)
    else:
        if ndim != len(left.composition):
            raise EinopsError(f"Wrong shape: expected {len(left.composition)} dims. Received {ndim}-dim tensor.")
        left_comp = left.composition
        rght_comp = rght.composition
        left_ids = left.identifiers
        rght_ids = rght.identifiers

    UNKNOWN = -999999
    EXPECTED = -99999
    names = List[str]()
    lengths = List[int]()
    pos = Dict[str, int]()
    def _reg(a: str, names: List[str], lengths: List[int], pos: Dict[str, int], left: _Parsed, rght: _Parsed):
        if a in pos:
            return
        pos[a] = len(names)
        names.append(a)
        if a in left.anon:
            lengths.append(left.anon[a])
        elif a in rght.anon:
            lengths.append(rght.anon[a])
        else:
            lengths.append(UNKNOWN)
    for g in left_comp:
        for a in g:
            _reg(a, names, lengths, pos, left, rght)
    for a in rght_ids:
        _reg(a, names, lengths, pos, left, rght)
    for k, v in axes_lengths.items():
        if k not in pos:
            raise EinopsError(f"Axis {k} is not used in transform")
        lengths[pos[k]] = v

    # infer from shape
    need_init = False
    for i in range(len(left_comp)):
        g = left_comp[i]
        length = shape[i]
        known_prod = 1
        unknown = List[int]()
        for a in g:
            p = pos[a]
            if lengths[p] == UNKNOWN:
                unknown.append(p)
            else:
                known_prod *= lengths[p]
        if len(unknown) > 1:
            raise EinopsError(f"Could not infer sizes for {g}")
        if len(unknown) == 0:
            if length != known_prod:
                raise EinopsError(f"Shape mismatch, {length} != {known_prod}")
        else:
            if length % known_prod != 0:
                raise EinopsError(f"Shape mismatch, can't divide axis of length {length} in chunks of {known_prod}")
            lengths[unknown[0]] = length // known_prod
        if len(g) != 1:
            need_init = True

    ordered_left = _flat(left_comp)
    ordered_rght = _flat(rght_comp)
    reduced_axes = [a for a in ordered_left if a not in rght_ids]
    order_after = [a for a in ordered_rght if a in left_ids] + reduced_axes
    plan = Plan()
    plan.need_init = need_init
    n_left = len(ordered_left)
    if need_init:
        plan.init_shape = lengths[:n_left]
    perm = [ordered_left.index(a) for a in order_after]
    ident = True
    for i in range(len(perm)):
        if perm[i] != i:
            ident = False
    if not ident:
        plan.perm = perm
    first_red = len(order_after) - len(reduced_axes)
    plan.reduced = list(range(first_red, n_left))
    for i in range(len(ordered_rght)):
        a = ordered_rght[i]
        if a not in left_ids:
            plan.added.append((i, lengths[pos[a]]))
    plan.n_after_add = len(plan.added) + n_left
    need_final = False
    for g in rght_comp:
        p = 1
        for a in g:
            p *= lengths[pos[a]]
        plan.final_shape.append(p)
        if len(g) != 1:
            need_final = True
    plan.need_final = need_final
    return plan
