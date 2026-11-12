# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# export_text (sklearn/tree/_export.py).

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
