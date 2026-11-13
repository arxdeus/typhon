# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# export_text (sklearn/tree/_export.py).

import math
from ._tree import TREE_UNDEFINED, TREE_LEAF

def _fmt(v: float, decimals: int) -> str:
    return f"{v:.{decimals}f}"

def _compute_depth(t, node: int) -> int:
    best = 0
    st = [(node, 1)]
    while len(st) > 0:
        nd, d = st.pop()
        if d > best:
            best = d
        if t.left[nd] != TREE_LEAF:
            st.append((t.left[nd], d + 1))
            st.append((t.right[nd], d + 1))
    return best

def export_text(decision_tree, *, feature_names: Optional[List[str]] = None,
                class_names: Optional[List[str]] = None, max_depth: int = 10,
                spacing: int = 3, decimals: int = 2, show_weights: bool = False) -> str:
    t = decision_tree.tree_
    is_clf = decision_tree._is_clf
    names = List[str]()
    if feature_names is not None:
        fn = feature_names
        if len(fn) != t.n_features:
            raise ValueError(f"feature_names must contain {t.n_features} elements, got {len(fn)}")
        names = fn
    else:
        names = [f"feature_{i}" for i in range(t.n_features)]
    cnames = List[str]()
    if is_clf:
        if class_names is not None:
            cnames = class_names
        else:
            cnames = [str(c) for c in decision_tree.classes_]
    out = List[str]()
    _rec(t, is_clf, names, cnames, out, 0, 1, max_depth, spacing, decimals, show_weights)
    return "".join(out)

def _add_leaf(t, is_clf: bool, cnames: List[str], out: List[str], node: int, indent: str,
              decimals: int, show_weights: bool):
    stride = t.value_stride
    val = ""
    if is_clf:
        nc = t.max_n_classes
        if show_weights:
            w = t.wnns[node]
            val = "[" + ", ".join([_fmt(t.val[node * stride + c] * w, decimals) for c in range(nc)]) + "]"
        best = 0
        for c in range(1, nc):
            if t.val[node * stride + c] > t.val[node * stride + best]:
                best = c
        val += " class: " + cnames[best]
        if show_weights:
            out.append(f"{indent} weights: {val}\n")
        else:
            out.append(f"{indent}{val}\n")
    else:
        val = "[" + _fmt(t.val[node * stride], decimals) + "]"
        out.append(f"{indent} value: {val}\n")

def _rec(t, is_clf: bool, names: List[str], cnames: List[str], out: List[str], node: int,
         depth: int, max_depth: int, spacing: int, decimals: int, show_weights: bool):
    indent = ("|" + " " * spacing) * depth
    indent = indent[:len(indent) - spacing] + "-" * spacing
    if depth <= max_depth + 1:
        if t.feat[node] != TREE_UNDEFINED:
            name = names[t.feat[node]]
            thr = _fmt(t.thr[node], decimals)
            out.append(f"{indent} {name} <= {thr}\n")
            _rec(t, is_clf, names, cnames, out, t.left[node], depth + 1, max_depth, spacing, decimals, show_weights)
            out.append(f"{indent} {name} >  {thr}\n")
            _rec(t, is_clf, names, cnames, out, t.right[node], depth + 1, max_depth, spacing, decimals, show_weights)
        else:
            _add_leaf(t, is_clf, cnames, out, node, indent, decimals, show_weights)
    else:
        sd = _compute_depth(t, node)
        if sd == 1:
            _add_leaf(t, is_clf, cnames, out, node, indent, decimals, show_weights)
        else:
            out.append(f"{indent} truncated branch of depth {sd}\n")

# ------------------------------------------------------------ export_graphviz

def _py_round(x: float, n: int) -> float:
    return round(x, n)

def _np_around(x: float, n: int) -> float:
    s = 10.0 ** float(n)
    return float(round(x * s)) / s

def _hex2(v: int) -> str:
    d = "0123456789abcdef"
    return d[v // 16] + d[v % 16]

def _color_brew(n: int) -> List[List[int]]:
    out = List[List[int]]()
    s = 0.75
    v = 0.9
    c = s * v
    m = v - c
    step = 360.0 / float(n)
    cnt = int(math.ceil((385.0 - 25.0) / step))
    for i in range(cnt):
        h = int(25.0 + float(i) * step)
        h_bar = float(h) / 60.0
        x = c * (1.0 - abs((h_bar % 2.0) - 1.0))
        k = int(h_bar)
        r = 0.0
        g = 0.0
        b = 0.0
        if k == 0 or k == 6:
            r = c
            g = x
        elif k == 1:
            r = x
            g = c
        elif k == 2:
            g = c
            b = x
        elif k == 3:
            g = x
            b = c
        elif k == 4:
            r = x
            b = c
        else:
            r = c
            b = x
        out.append([int(255.0 * (r + m)), int(255.0 * (g + m)), int(255.0 * (b + m))])
    return out

class _DotExporter:
    out: List[str]
    max_depth: int
    has_max_depth: bool
    names: List[str]
    has_names: bool
    cnames: List[str]
    has_cnames: bool
    cnames_true: bool
    label: str
    filled: bool
    leaves_parallel: bool
    impurity: bool
    node_ids: bool
    proportion: bool
    rotate: bool
    rounded: bool
    precision: int
    fontname: str
    chars: List[str]
    ranks_keys: List[str]
    ranks_vals: List[List[str]]
    leaves: List[str]
    rgb: List[List[int]]
    have_rgb: bool
    has_bounds: bool
    b0: float
    b1: float

    def __init__(self):
        self.out = List[str]()
        self.max_depth = 0
        self.has_max_depth = False
        self.names = List[str]()
        self.has_names = False
        self.cnames = List[str]()
        self.has_cnames = False
        self.cnames_true = False
        self.label = "all"
        self.filled = False
        self.leaves_parallel = False
        self.impurity = True
        self.node_ids = False
        self.proportion = False
        self.rotate = False
        self.rounded = False
        self.precision = 3
        self.fontname = "helvetica"
        self.chars = List[str]()
        self.ranks_keys = List[str]()
        self.ranks_vals = List[List[str]]()
        self.leaves = List[str]()
        self.rgb = List[List[int]]()
        self.have_rgb = False
        self.has_bounds = False
        self.b0 = 0.0
        self.b1 = 0.0

    def esc(self, s: str) -> str:
        return s.replace('"', '\\"')

    def _color(self, alpha: float, c: List[int]) -> str:
        r = int(round(alpha * float(c[0]) + (1.0 - alpha) * 255.0))
        g = int(round(alpha * float(c[1]) + (1.0 - alpha) * 255.0))
        b = int(round(alpha * float(c[2]) + (1.0 - alpha) * 255.0))
        return "#" + _hex2(r) + _hex2(g) + _hex2(b)

    def fill_color(self, t, is_clf: bool, node: int) -> str:
        stride = t.value_stride
        if not self.have_rgb:
            nc = t.max_n_classes if is_clf else 1
            self.rgb = _color_brew(nc)
            self.have_rgb = True
            if not is_clf:
                mn = t.val[0]
                mx = t.val[0]
                uniq = False
                for v in t.val:
                    if v < mn:
                        mn = v
                    if v > mx:
                        mx = v
                    if v != t.val[0]:
                        uniq = True
                if uniq:
                    self.has_bounds = True
                    self.b0 = mn
                    self.b1 = mx
        if is_clf or not self.has_bounds:
            nc = t.max_n_classes if is_clf else 1
            best = 0
            for c in range(1, nc):
                if t.val[node * stride + c] > t.val[node * stride + best]:
                    best = c
            vals = [t.val[node * stride + c] for c in range(nc)]
            vals.sort(reverse=True)
            alpha = 0.0
            if len(vals) > 1:
                alpha = (vals[0] - vals[1]) / (1.0 - vals[1])
            return self._color(alpha, self.rgb[best])
        alpha = (t.val[node * stride] - self.b0) / (self.b1 - self.b0)
        return self._color(alpha, self.rgb[0])

    def node_to_str(self, t, is_clf: bool, criterion: str, node: int) -> str:
        stride = t.value_stride
        labels = (self.label == "root" and node == 0) or self.label == "all"
        ch = self.chars
        s = ch[6]
        if self.node_ids:
            if labels:
                s += "node "
            s += ch[0] + str(node) + ch[4]
        if t.left[node] != TREE_LEAF:
            feature = ""
            if self.has_names:
                feature = self.esc(self.names[t.feat[node]])
            else:
                feature = "x" + ch[1] + str(t.feat[node]) + ch[2]
            s += feature + " " + ch[3] + " " + str(_py_round(t.thr[node], self.precision)) + ch[4]
        if self.impurity:
            if labels:
                s += criterion + " = "
            s += str(_py_round(t.imp[node], self.precision)) + ch[4]
        if labels:
            s += "samples = "
        if self.proportion:
            percent = 100.0 * float(t.nns[node]) / float(t.nns[0])
            s += str(_py_round(percent, 1)) + "%" + ch[4]
        else:
            s += str(t.nns[node]) + ch[4]
        nc = t.max_n_classes if is_clf else 1
        vals = [t.val[node * stride + c] for c in range(nc)]
        if not self.proportion and is_clf:
            w = t.wnns[node]
            vals = [v * w for v in vals]
        if labels:
            s += "value = "
        vt = ""
        if not is_clf:
            vt = str(_np_around(vals[0], self.precision))
        elif self.proportion:
            vt = "[" + ", ".join([str(_np_around(v, self.precision)) for v in vals]) + "]"
        else:
            integral = True
            for v in vals:
                if v - math.floor(v) != 0.0:
                    integral = False
            if integral:
                vt = "[" + ", ".join([str(int(v)) for v in vals]) + "]"
            else:
                vt = "[" + ", ".join([str(_np_around(v, self.precision)) for v in vals]) + "]"
        s += vt + ch[4]
        if (self.has_cnames or self.cnames_true) and is_clf:
            if labels:
                s += "class = "
            best = 0
            for c in range(1, nc):
                if vals[c] > vals[best]:
                    best = c
            if not self.cnames_true:
                s += self.esc(self.cnames[best])
            else:
                s += "y" + ch[1] + str(best) + ch[2]
        if s.endswith(ch[4]):
            s = s[:len(s) - len(ch[4])]
        return s + ch[5]

    def _add_rank(self, key: str, node: int):
        for i in range(len(self.ranks_keys)):
            if self.ranks_keys[i] == key:
                self.ranks_vals[i].append(str(node))
                return
        self.ranks_keys.append(key)
        self.ranks_vals.append([str(node)])

    def recurse(self, t, is_clf: bool, criterion: str, node: int, parent: int, depth: int):
        left = t.left[node]
        right = t.right[node]
        if (not self.has_max_depth) or depth <= self.max_depth:
            if left == TREE_LEAF:
                self.leaves.append(str(node))
            else:
                self._add_rank(str(depth), node)
            self.out.append(f"{node} [label=" + self.node_to_str(t, is_clf, criterion, node))
            if self.filled:
                self.out.append(', fillcolor="' + self.fill_color(t, is_clf, node) + '"')
            self.out.append("] ;\n")
            if parent >= 0:
                self.out.append(f"{parent} -> {node}")
                if parent == 0:
                    a0 = int(45.0 * ((float(self.rotate) - 0.5) * -2.0))
                    a1 = int(-45.0 * ((float(self.rotate) - 0.5) * -2.0))
                    self.out.append(" [labeldistance=2.5, labelangle=")
                    if node == 1:
                        self.out.append(f'{a0}, headlabel="True"]')
                    else:
                        self.out.append(f'{a1}, headlabel="False"]')
                self.out.append(" ;\n")
            if left != TREE_LEAF:
                self.recurse(t, is_clf, criterion, left, node, depth + 1)
                self.recurse(t, is_clf, criterion, right, node, depth + 1)
        else:
            self.leaves.append(str(node))
            self.out.append(f'{node} [label="(...)"')
            if self.filled:
                self.out.append(', fillcolor="#C0C0C0"')
            self.out.append("] ;\n")
            if parent >= 0:
                self.out.append(f"{parent} -> {node} ;\n")

    def head(self):
        o = self.out
        o.append("digraph Tree {\n")
        o.append("node [shape=box")
        rf = List[str]()
        if self.filled:
            rf.append("filled")
        if self.rounded:
            rf.append("rounded")
        if len(rf) > 0:
            o.append(', style="' + ", ".join(rf) + '", color="black"')
        o.append(', fontname="' + self.fontname + '"')
        o.append("] ;\n")
        if self.leaves_parallel:
            o.append("graph [ranksep=equally, splines=polyline] ;\n")
        o.append('edge [fontname="' + self.fontname + '"] ;\n')
        if self.rotate:
            o.append("rankdir=LR ;\n")

    def tail(self):
        if self.leaves_parallel:
            keys = List[str]()
            keys.append("leaves")
            for k in self.ranks_keys:
                keys.append(k)
            keys.sort()
            for k in keys:
                if k == "leaves":
                    self.out.append("{rank=same ; " + "; ".join(self.leaves) + "} ;\n")
                else:
                    for i in range(len(self.ranks_keys)):
                        if self.ranks_keys[i] == k:
                            self.out.append("{rank=same ; " + "; ".join(self.ranks_vals[i]) + "} ;\n")
        self.out.append("}")

def export_graphviz(decision_tree, out_file = None, *, max_depth: Optional[int] = None,
                    feature_names: Optional[List[str]] = None, class_names = None,
                    label: str = "all", filled: bool = False, leaves_parallel: bool = False,
                    impurity: bool = True, node_ids: bool = False, proportion: bool = False,
                    rotate: bool = False, rounded: bool = False, special_characters: bool = False,
                    precision: int = 3, fontname: str = "helvetica"):
    """Returns the DOT text when out_file is None, else writes it to the path out_file."""
    t = decision_tree.tree_
    is_clf = decision_tree._is_clf
    e = _DotExporter()
    if max_depth is not None:
        e.has_max_depth = True
        e.max_depth = max_depth.__val__()
    if feature_names is not None:
        if len(feature_names) != decision_tree.n_features_in_:
            raise ValueError(f"Length of feature_names, {len(feature_names)} does not match number of features, {decision_tree.n_features_in_}")
        e.has_names = True
        e.names = feature_names
    if class_names is not None:
        if isinstance(class_names, bool):
            e.cnames_true = class_names
        else:
            e.has_cnames = True
            e.cnames = [str(c) for c in class_names]
    e.label = label
    e.filled = filled
    e.leaves_parallel = leaves_parallel
    e.impurity = impurity
    e.node_ids = node_ids
    e.proportion = proportion
    e.rotate = rotate
    e.rounded = rounded
    e.precision = precision
    e.fontname = fontname
    if special_characters:
        e.chars = ["&#35;", "<SUB>", "</SUB>", "&le;", "<br/>", ">", "<"]
    else:
        e.chars = ["#", "[", "]", "<=", "\\n", '"', '"']
    crit = decision_tree.criterion
    e.head()
    e.recurse(t, is_clf, crit, 0, -1, 0)
    e.tail()
    text = "".join(e.out)
    if isinstance(out_file, str):
        with open(out_file, "w") as f:
            f.write(text)
        return ""
    return text
