# Copyright (C) 2026 the Typhon authors
# Ported from mne 1.13.2 (BSD-3-Clause)
#
# Label operations: read_label / write_label, label_sign_flip, extract_label_time_course,
# labels_to_stc, grow_labels, random_parcellation, label_adjacency, morph_labels (with given
# morph maps), spatial_tris_adjacency, mesh_dist. grow_labels / random_parcellation take the
# surfaces explicitly through `surfaces` (dict hemi -> (rr, tris)); otherwise they read
# <subjects_dir>/<subject>/surf/<hemi>.<surface> with read_surface.

import numpy as np
from scipy.sparse import coo_array, csr_array, coo_matrix
from scipy.sparse.csgraph import dijkstra
from .simulation._labels import Label
from .source_space import SourceSpaces
from .source_estimate import _STC
from .io._bytes import _read_text


def _fmt6(v: float) -> str:
    return f"{v:f}"


def read_label(filename: str, subject = None, color = None, verbose = None) -> Label:
    """mne.read_label (FreeSurfer .label text file)."""
    base = filename[filename.rfind("/") + 1:]
    hemi = ""
    if base.endswith("lh.label") or base.startswith("lh."):
        hemi = "lh"
    elif base.endswith("rh.label") or base.startswith("rh."):
        hemi = "rh"
    else:
        raise ValueError(f"Cannot find which hemisphere it is. File should end with lh.label or rh.label: {base}")
    base_ = ""
    if base.startswith("lh.") or base.startswith("rh."):
        base_ = base[3:]
        if base.endswith(".label"):
            base_ = base[:-6]
    else:
        base_ = base[:-9]
    name = f"{base_}-{hemi}"
    lines = _read_text(filename).split("\n")
    comment = lines[0][1:] if len(lines) > 0 else ""
    nv = int(lines[1].strip())
    vert = np.zeros(nv, dtype=int)
    pos = np.zeros((nv, 3))
    val = np.zeros(nv)
    for i in range(nv):
        p = lines[2 + i].split()
        vert[i] = int(float(p[0]))
        pos[i, 0] = float(p[1]) * 1e-3
        pos[i, 1] = float(p[2]) * 1e-3
        pos[i, 2] = float(p[3]) * 1e-3
        val[i] = float(p[4])
    order = np.argsort(vert)
    return Label(vert[order], val[order], hemi, name, pos[order], subject, comment, filename, color)


def write_label(filename: str, label: Label, verbose = None):
    """mne.write_label."""
    head = filename[:filename.rfind("/") + 1]
    name = filename[filename.rfind("/") + 1:]
    if name.endswith(".label"):
        name = name[:-6]
    if not (name.startswith(label.hemi) or name.endswith(label.hemi)):
        name += "-" + label.hemi
    path = head + name + ".label"
    out = List[str]()
    n = label.vertices.shape[0]
    out.append(f"#{label.comment}")
    out.append(f"{n}")
    for i in range(n):
        out.append(f"{int(label.vertices[i])} {_fmt6(1e3 * label.pos[i, 0])} {_fmt6(1e3 * label.pos[i, 1])} {_fmt6(1e3 * label.pos[i, 2])} {_fmt6(label.values[i])}")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def _intersect_sorted(a: np.ndarray[int, 1], b: np.ndarray[int, 1]) -> np.ndarray[int, 1]:
    s = Set[int]([int(v) for v in b])
    r = sorted([int(v) for v in a if int(v) in s])
    return np.array(r, dtype=int)


def label_sign_flip(label: Label, src: SourceSpaces) -> np.ndarray[float, 1]:
    """mne.label.label_sign_flip."""
    if len(src.spaces) != 2:
        raise ValueError("Only source spaces with 2 hemisphers are accepted")
    hi = 0 if label.hemi == "lh" else 1
    s = src.spaces[hi]
    sel = _intersect_sorted(s.vertno, label.vertices)
    if sel.shape[0] == 0:
        return np.zeros(0)
    ori = s.nn[sel]
    U, sv, Vh = np.linalg.svd(ori, full_matrices=False)
    dots = np.dot(ori, Vh[0])
    if np.mean(dots) < 0:
        dots = -dots
    return np.sign(dots)


def _label_vertidx(label: Label, vertno: List[np.ndarray[int, 1]], nvert: List[int]) -> np.ndarray[int, 1]:
    if label.hemi == "lh":
        tv = _intersect_sorted(vertno[0], label.vertices)
        return np.searchsorted(vertno[0], tv)
    tv = _intersect_sorted(vertno[1], label.vertices)
    return nvert[0] + np.searchsorted(vertno[1], tv)


def extract_label_time_course(stcs, labels, src: SourceSpaces, mode: str = "auto", allow_empty = False,
                              return_generator: bool = False, *, mri_resolution: bool = True, verbose = None):
    """mne.extract_label_time_course for a SourceEstimate (surface, scalar data) and a list of
    Labels. Modes: mean, mean_flip, pca_flip, max, auto (= mean_flip)."""
    single = not isinstance(stcs, List)
    sl = List[_STC[np.ndarray[float, 2]]]()
    if single:
        sl.append(stcs)
    else:
        for s in stcs:
            sl.append(s)
    res = List[np.ndarray[float, 2]]()
    for stc in sl:
        md = "mean_flip" if mode == "auto" else mode
        vertno = stc.vertices
        nvert = [v.shape[0] for v in vertno]
        data = stc.data
        tc = np.zeros((len(labels), data.shape[1]))
        bad = List[str]()
        for li in range(len(labels)):
            lab = labels[li]
            vi = _label_vertidx(lab, vertno, nvert)
            if vi.shape[0] == 0:
                bad.append(lab.name)
                continue
            d = data[vi]
            if md == "mean":
                tc[li] = np.mean(d, axis=0)
            elif md == "max":
                tc[li] = np.max(np.abs(d), axis=0)
            else:
                s2 = src.copy()
                for k in range(2):
                    s2.spaces[k].vertno = vertno[k]
                flip = label_sign_flip(lab, s2)
                if md == "mean_flip":
                    tc[li] = np.mean(d * flip[:, None], axis=0)
                elif md == "pca_flip":
                    U, sv, V = np.linalg.svd(d, full_matrices=False)
                    sign = np.sign(np.dot(U[:, 0], flip))
                    scale = np.linalg.norm(sv) / np.sqrt(d.shape[0])
                    tc[li] = sign * scale * V[0]
                else:
                    raise ValueError(f"Invalid mode {md}")
        if len(bad) > 0 and not allow_empty:
            raise ValueError(f"source space does not contain any vertices for {len(bad)} label{'s' if len(bad) != 1 else ''}:\n{bad}")
        res.append(tc)
    if single:
        return res[0]
    return res


def labels_to_stc(labels, values, tmin: float = 0.0, tstep: float = 1.0, subject = None, src = None,
                  verbose = None):
    """mne.labels_to_stc: surface case (src=None). Vertices in several labels get the mean."""
    v = np.asarray(values, dtype=float)
    if v.ndim == 1:
        v = v.reshape((v.shape[0], 1))
    if v.shape[0] != len(labels):
        raise ValueError(f"values.shape[0] ({v.shape[0]}) must match the number of labels ({len(labels)})")
    verts = List[np.ndarray[int, 1]]()
    datas = List[np.ndarray[float, 2]]()
    for hemi in ("lh", "rh"):
        idx = [li for li in range(len(labels)) if labels[li].hemi == hemi]
        allv = Set[int]()
        for li in idx:
            for x in labels[li].vertices:
                allv.add(int(x))
        these = np.array(sorted(list(allv)), dtype=int)
        acc = np.zeros((these.shape[0], v.shape[1]))
        cnt = np.zeros(these.shape[0])
        for li in idx:
            pos = np.searchsorted(these, labels[li].vertices)
            for p in pos:
                acc[p] += v[li]
                cnt[p] += 1.0
        for p in range(these.shape[0]):
            acc[p] /= cnt[p]
        verts.append(these)
        datas.append(acc)
    return _STC[np.ndarray[float, 2]](np.concatenate(datas), verts, tmin, tstep, subject, "surface")


# ------------------------------------------------------------------ meshes / adjacency

def _edge_set(tris: np.ndarray[int, 2]) -> List[Tuple[int, int]]:
    s = Set[Tuple[int, int]]()
    for t in range(tris.shape[0]):
        for k in range(3):
            a = int(tris[t, k])
            b = int(tris[t, (k + 1) % 3])
            if a != b:
                s.add((min(a, b), max(a, b)))
    return sorted(list(s))


def spatial_tris_adjacency(tris, remap_vertices: bool = False, verbose = None):
    """mne.spatial_tris_adjacency: coo adjacency (ones) of the mesh."""
    t = np.asarray(tris).astype(int)
    if remap_vertices:
        u = np.unique(t)
        m = Dict[int, int]()
        for i in range(u.shape[0]):
            m[int(u[i])] = i
        t = t.copy()
        for i in range(t.shape[0]):
            for k in range(3):
                t[i, k] = m[int(t[i, k])]
    n = int(np.max(t)) + 1
    es = _edge_set(t)
    r = List[int]()
    c = List[int]()
    for a, b in es:
        r.append(a)
        c.append(b)
        r.append(b)
        c.append(a)
    return coo_array((np.ones(len(r)), (np.array(r, dtype=int), np.array(c, dtype=int))), shape=(n, n))


def mesh_dist(tris, vert):
    """mne.surface.mesh_dist: csr graph with Euclidean edge lengths."""
    t = np.asarray(tris).astype(int)
    v = np.asarray(vert, dtype=float)
    n = v.shape[0]
    es = _edge_set(t)
    r = List[int]()
    c = List[int]()
    d = List[float]()
    for a, b in es:
        w = float(np.linalg.norm(v[a] - v[b]))
        r.append(a)
        c.append(b)
        d.append(w)
        r.append(b)
        c.append(a)
        d.append(w)
    return csr_array(coo_array((np.array(d), (np.array(r, dtype=int), np.array(c, dtype=int))), shape=(n, n)))


def label_adjacency(labels, src: SourceSpaces, tris):
    """mne.label_adjacency. tris: [lh_tris, rh_tris] (full vertex indices of the used triangles)."""
    n_lh = src.spaces[0].vertno.shape[0]
    n = n_lh + src.spaces[1].vertno.shape[0]
    adj = np.zeros((n, n), dtype=bool)
    for hi in range(2):
        s = src.spaces[hi]
        m = Dict[int, int]()
        for i in range(s.vertno.shape[0]):
            m[int(s.vertno[i])] = i + (n_lh if hi == 1 else 0)
        t = np.asarray(tris[hi]).astype(int)
        for a, b in _edge_set(t):
            if a in m and b in m:
                adj[m[a], m[b]] = True
                adj[m[b], m[a]] = True
    nl = len(labels)
    inds = List[List[int]]()
    for lab in labels:
        hi = 0 if lab.hemi == "lh" else 1
        s = src.spaces[hi]
        used = _intersect_sorted(s.vertno, lab.vertices)
        pos = np.searchsorted(s.vertno, used)
        inds.append([int(p) + (n_lh if hi == 1 else 0) for p in pos])
    r = List[int]()
    c = List[int]()
    for i in range(nl):
        for j in range(nl):
            hit = False
            for a in inds[i]:
                for b in inds[j]:
                    if adj[a, b]:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                r.append(i)
                c.append(j)
    return coo_matrix((np.ones(len(r)), (np.array(r, dtype=int), np.array(c, dtype=int))), shape=(nl, nl))


# ------------------------------------------------------------------ growing / parcellation

def _surfaces_for(hemis_used: List[str], subject: str, subjects_dir, surface: str, surfaces):
    rr = Dict[str, np.ndarray[float, 2]]()
    tr = Dict[str, np.ndarray[int, 2]]()
    for h in hemis_used:
        if not isinstance(surfaces, NoneType):
            rr[h] = np.asarray(surfaces[h][0], dtype=float)
            tr[h] = np.asarray(surfaces[h][1]).astype(int)
        else:
            from ._surface import read_surface
            a, b = read_surface(f"{subjects_dir}/{subject}/surf/{h}.{surface}")
            rr[h] = a
            tr[h] = b
    return rr, tr


def _hue_colors(n: int) -> np.ndarray[float, 2]:
    """hsv colormap sampled at linspace(0, 1, n, endpoint=False) (matplotlib hsv, 256 entries)."""
    out = np.zeros((n, 4))
    for i in range(n):
        x = i / n
        idx = int(x * 256)
        h = idx / 255.0
        h6 = h * 6.0
        k = int(np.floor(h6))
        f = h6 - k
        q = 1.0 - f
        r = 0.0
        g = 0.0
        b = 0.0
        kk = k % 6
        if kk == 0:
            r, g, b = 1.0, f, 0.0
        elif kk == 1:
            r, g, b = q, 1.0, 0.0
        elif kk == 2:
            r, g, b = 0.0, 1.0, f
        elif kk == 3:
            r, g, b = 0.0, q, 1.0
        elif kk == 4:
            r, g, b = f, 0.0, 1.0
        else:
            r, g, b = 1.0, 0.0, q
        out[i, 0] = r
        out[i, 1] = g
        out[i, 2] = b
        out[i, 3] = 1.0
    return out


def grow_labels(subject: str, seeds, extents, hemis, subjects_dir = None, n_jobs = None, overlap: bool = True,
                names = None, surface: str = "white", colors = None, surfaces = None, verbose = None):
    """mne.grow_labels. extents in mm (surface units), hemis 0 = lh / 1 = rh (one, or one per seed);
    seeds is a list of vertex-id lists."""
    sd: List[List[int]] = List[List[int]]()
    for s in seeds:
        if isinstance(s, int):
            sd.append([s])
        else:
            sd.append([int(x) for x in s])
    n = len(sd)
    ext = List[float]()
    ea = np.atleast_1d(np.asarray(extents, dtype=float))
    for i in range(n):
        ext.append(float(ea[0] if ea.shape[0] == 1 else ea[i]))
    ha = np.atleast_1d(np.asarray(hemis).astype(int))
    hm = List[str]()
    for i in range(n):
        h = int(ha[0] if ha.shape[0] == 1 else ha[i])
        hm.append("lh" if h == 0 else "rh")
    nm = List[str]()
    if isinstance(names, NoneType):
        for i in range(n):
            nm.append(f"Label_{i}-{hm[i]}")
    else:
        nl0 = [names] if isinstance(names, str) else [str(x) for x in names]
        for i in range(n):
            x = nl0[i]
            nm.append(x if x.endswith(hm[i]) else x + "-" + hm[i])
    used = sorted(list(Set[str](hm)))
    vert, tris = _surfaces_for(used, subject, subjects_dir, surface, surfaces)
    dist = Dict[str, csr_array[float]]()
    for h in used:
        dist[h] = mesh_dist(tris[h], vert[h])
    labels = List[Label]()
    if overlap:
        for i in range(n):
            d = dijkstra(dist[hm[i]], indices=sd[i], min_only=True, limit=ext[i])
            vs = List[int]()
            vd = List[float]()
            for k in range(d.shape[0]):
                if np.isfinite(d[k]):
                    vs.append(k)
                    vd.append(float(int(d[k])))
            va = np.array(vs, dtype=int)
            seed_repr = str(sd[i]) if len(sd[i]) == 1 else ",".join([str(x) for x in sd[i]])
            comment = f"Circular label: seed={seed_repr}, extent={ext[i]:0.1f}mm"
            labels.append(Label(va, np.array(vd), hm[i], nm[i], vert[hm[i]][va], subject, comment))
    else:
        for h in used:
            idx = [i for i in range(n) if hm[i] == h]
            nv = vert[h].shape[0]
            seed_label = np.full(nv, -1, dtype=int)
            for k in range(len(idx)):
                for sv in sd[idx[k]]:
                    if seed_label[sv] >= 0:
                        raise ValueError("Overlapping seeds")
                    seed_label[sv] = k
            srcs = np.array([j for j in range(nv) if seed_label[j] >= 0], dtype=int)
            dd, pp, so = dijkstra(dist[h], indices=srcs, min_only=True, return_predecessors=True)
            parc = np.full(nv, -1, dtype=int)
            for j in range(nv):
                if np.isfinite(dd[j]):
                    lab = int(seed_label[int(so[j])])
                    if dd[j] <= ext[idx[lab]]:
                        parc[j] = lab
            for k in range(len(idx)):
                va = np.array([j for j in range(nv) if parc[j] == k], dtype=int)
                labels.append(Label(va, None, h, nm[idx[k]], None, subject))
    cols = _hue_colors(len(labels)) if isinstance(colors, NoneType) else np.asarray(colors, dtype=float)
    for k in range(len(labels)):
        if cols.ndim == 1:
            labels[k].color = cols
        elif isinstance(colors, NoneType) or cols.shape[0] > 1:
            labels[k].color = cols[k]
        else:
            labels[k].color = cols[0]
    return labels


def random_parcellation(subject: str, n_parcel: int, hemi: str, subjects_dir = None, surface: str = "white",
                        random_state = None, surfaces = None, verbose = None):
    """mne.label.random_parcellation (region growing + merging of the smallest labels)."""
    hemis = ["lh", "rh"] if hemi == "both" else [hemi]
    vert, tris = _surfaces_for(hemis, subject, subjects_dir, surface, surfaces)
    rng = np.random.RandomState(None if isinstance(random_state, NoneType) else int(random_state))
    labels = List[Label]()
    for h in sorted(list(Set[str](hemis))):
        graph = mesh_dist(tris[h], vert[h])
        nvert = vert[h].shape[0]
        parcel_size = len(hemis) * nvert // n_parcel
        parc = np.full(nvert, -1, dtype=int)
        s = int(rng.randint(0, nvert))
        label_idx = 0
        edge = [s]
        parc[s] = label_idx
        label_size = 1
        rest = nvert - 1
        while rest > 0:
            if len(edge) == 0:
                cand = np.array([j for j in range(nvert) if parc[j] < 0], dtype=int)
                s = int(cand[int(rng.randint(0, cand.shape[0]))])
                edge = [s]
                label_idx += 1
                label_size = 1
                parc[s] = label_idx
                rest -= 1
            vf = edge.pop(0)
            for k in range(int(graph.indptr[vf]), int(graph.indptr[vf + 1])):
                vt = int(graph.indices[k])
                if parc[vt] >= 0:
                    continue
                if label_size > parcel_size:
                    label_idx += 1
                    label_size = 1
                    edge = [vt]
                    parc[vt] = label_idx
                    rest -= 1
                    break
                parc[vt] = label_idx
                label_size += 1
                edge.append(vt)
                rest -= 1
        n_labels = label_idx + 1
        sizes = [0 for _ in range(n_labels)]
        for j in range(nvert):
            sizes[int(parc[j])] += 1
        conn = [[False for _ in range(n_labels)] for _ in range(n_labels)]
        for a in range(nvert):
            for k in range(int(graph.indptr[a]), int(graph.indptr[a + 1])):
                b = int(graph.indices[k])
                if parc[a] != parc[b]:
                    conn[int(parc[a])][int(parc[b])] = True
        label_id = [i for i in range(n_labels)]
        target = n_parcel // len(hemis)
        while n_labels > target:
            i = 0
            for q in range(1, len(sizes)):
                if sizes[q] < sizes[i]:
                    i = q
            nb = [q for q in range(len(sizes)) if conn[i][q]]
            j = nb[0]
            for q in nb:
                if sizes[q] < sizes[j]:
                    j = q
            for q in range(len(sizes)):
                if conn[i][q]:
                    conn[j][q] = True
                if conn[q][i]:
                    conn[q][j] = True
            conn.pop(i)
            for row in conn:
                row.pop(i)
            conn[j][j] = False
            sizes[j] += sizes[i]
            sizes.pop(i)
            n_labels -= 1
            for q in range(nvert):
                if parc[q] == label_id[i]:
                    parc[q] = label_id[j]
            label_id.pop(i)
        for i in range(n_labels):
            va = np.array([q for q in range(nvert) if parc[q] == label_id[i]], dtype=int)
            labels.append(Label(va, None, h, "label_" + str(i), None, subject))
    cols = _hue_colors(len(labels))
    for k in range(len(labels)):
        labels[k].color = cols[k]
    return labels


def morph_labels(labels, subject_to: str, subject_from = None, subjects_dir = None, surf_name: str = "white",
                 verbose = None, morph_maps = None, positions = None):
    """mne.morph_labels with explicit morph maps: morph_maps = [lh_map, rh_map] dense (n_to, n_from)
    matrices (argmax over columns gives the source vertex); positions = [lh_pos, rh_pos] of subject_to."""
    if isinstance(morph_maps, NoneType):
        raise NotImplementedError("morph_labels needs morph_maps (read_morph_map requires the FreeSurfer subject tree)")
    out = List[Label]()
    for lab in labels:
        li = 0 if lab.hemi == "lh" else 1
        mm = np.asarray(morph_maps[li], dtype=float)
        idx = np.argmax(mm, axis=1)
        keep = Set[int]([int(v) for v in lab.vertices])
        vs = List[int]()
        for r in range(mm.shape[0]):
            if int(idx[r]) in keep:
                vs.append(r)
        va = np.array(vs, dtype=int)
        pos = None
        if not isinstance(positions, NoneType):
            pos = np.asarray(positions[li], dtype=float)[va]
        out.append(Label(va, None, lab.hemi, lab.name, pos, subject_to, lab.comment, None, lab.color))
    return out
