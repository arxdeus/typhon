# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# extract_patches_2d / reconstruct_from_patches_2d / PatchExtractor and
# img_to_graph / grid_to_graph (dense adjacency, returned as a scipy.sparse
# COO array or a dense ndarray with dense=True).

import numpy as np
from numpy.random.legacy import check_random_state

def _compute_n_patches(i_h: int, i_w: int, p_h: int, p_w: int, max_patches) -> int:
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    allp = n_h * n_w
    if isinstance(max_patches, NoneType):
        return allp
    elif isinstance(max_patches, int):
        if max_patches == 0:
            return allp
        return max_patches if max_patches < allp else allp
    else:
        mp = float(max_patches)
        if 0 < mp < 1:
            return int(mp * allp)
        raise ValueError(f"Invalid value for max_patches: {max_patches}")

def _patches3(image, patch_size, max_patches, random_state) -> np.ndarray[float, 4]:
    img = np.asarray(image, dtype=float)
    i_h = img.shape[0]; i_w = img.shape[1]
    p_h = int(patch_size[0]); p_w = int(patch_size[1])
    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height of the image.")
    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width of the image.")
    I = np.ascontiguousarray(img.reshape((i_h, i_w, -1)))
    nc = I.shape[2]
    n = _compute_n_patches(i_h, i_w, p_h, p_w, max_patches)
    out = np.empty((n, p_h, p_w, nc))
    use_rand = not isinstance(max_patches, NoneType)
    if use_rand:
        if isinstance(max_patches, int):
            use_rand = max_patches != 0
    if use_rand:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + 1, size=n)
        j_s = rng.randint(i_w - p_w + 1, size=n)
        for t in range(n):
            for a in range(p_h):
                for b in range(p_w):
                    for c in range(nc):
                        out[t, a, b, c] = I[i_s[t] + a, j_s[t] + b, c]
    else:
        t = 0
        for i in range(i_h - p_h + 1):
            for j in range(i_w - p_w + 1):
                for a in range(p_h):
                    for b in range(p_w):
                        for c in range(nc):
                            out[t, a, b, c] = I[i + a, j + b, c]
                t += 1
    return out

def extract_patches_2d(image, patch_size, *, max_patches = None, random_state = None):
    """2-D image -> (n_patches, p_h, p_w); 3-D (colour) image -> 4-D."""
    img = np.asarray(image, dtype=float)
    P = _patches3(img, patch_size, max_patches, random_state)
    if static.len(img.shape) == 2:
        return np.ascontiguousarray(P.reshape((P.shape[0], P.shape[1], P.shape[2])))
    else:
        return P

def reconstruct_from_patches_2d(patches, image_size):
    P = np.asarray(patches, dtype=float)
    i_h = int(image_size[0]); i_w = int(image_size[1])
    p_h = P.shape[1]; p_w = P.shape[2]
    img = np.zeros(image_size)
    n_h = i_h - p_h + 1
    n_w = i_w - p_w + 1
    t = 0
    for i in range(n_h):
        for j in range(n_w):
            img[i:i + p_h, j:j + p_w] += P[t]
            t += 1
    for i in range(i_h):
        for j in range(i_w):
            img[i, j] /= float(min(i + 1, p_h, n_h, i_h - i) * min(j + 1, p_w, n_w, i_w - j))
    return img

class PatchExtractor:
    patch_size: Optional[Tuple[int, int]]
    max_patches: Optional[float]
    random_state: Optional[int]

    def __init__(self, *, patch_size: Optional[Tuple[int, int]] = None, max_patches = None,
                 random_state: Optional[int] = None):
        self.patch_size = patch_size
        self.max_patches = None if isinstance(max_patches, NoneType) else float(max_patches)
        self.random_state = random_state

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        Xa = np.asarray(X, dtype=float)
        n, i_h, i_w = Xa.shape[0], Xa.shape[1], Xa.shape[2]
        ps = (max(i_h // 10, 1), max(i_w // 10, 1)) if self.patch_size is None else self.patch_size.__val__()
        out = List[np.ndarray[float, 2]]()
        for k in range(n):
            mp = self.max_patches
            P = np.empty((0, 0, 0))
            if mp is None:
                P = extract_patches_2d(np.ascontiguousarray(Xa[k]), ps, random_state=self.random_state)
            elif mp.__val__() >= 1.0:
                P = extract_patches_2d(np.ascontiguousarray(Xa[k]), ps, max_patches=int(mp.__val__()), random_state=self.random_state)
            else:
                P = extract_patches_2d(np.ascontiguousarray(Xa[k]), ps, max_patches=mp.__val__(), random_state=self.random_state)
            for t in range(P.shape[0]):
                out.append(np.ascontiguousarray(P[t]))
        R = np.empty((len(out), ps[0], ps[1]))
        for t in range(len(out)):
            R[t] = out[t]
        return R

def _vid(x: int, y: int, z: int, n_y: int, n_z: int) -> int:
    return (x * n_y + y) * n_z + z

def _to_graph_dense(n_x: int, n_y: int, n_z: int, mask, img) -> np.ndarray[float, 2]:
    nv_all = n_x * n_y * n_z
    e0 = List[int](); e1 = List[int]()
    for x in range(n_x):
        for y in range(n_y):
            for z in range(n_z - 1):
                e0.append(_vid(x, y, z, n_y, n_z)); e1.append(_vid(x, y, z + 1, n_y, n_z))
    for x in range(n_x):
        for y in range(n_y - 1):
            for z in range(n_z):
                e0.append(_vid(x, y, z, n_y, n_z)); e1.append(_vid(x, y + 1, z, n_y, n_z))
    for x in range(n_x - 1):
        for y in range(n_y):
            for z in range(n_z):
                e0.append(_vid(x, y, z, n_y, n_z)); e1.append(_vid(x + 1, y, z, n_y, n_z))
    flat = np.ones(nv_all) if isinstance(img, NoneType) else np.ascontiguousarray(np.asarray(img, dtype=float)).ravel()
    weighted = not isinstance(img, NoneType)
    keep = [True] * nv_all
    if not isinstance(mask, NoneType):
        mk = np.asarray(mask).ravel()
        for i in range(nv_all):
            keep[i] = bool(mk[i])
    newid = [-1] * nv_all
    nv = 0
    for i in range(nv_all):
        if keep[i]:
            newid[i] = nv
            nv += 1
    G = np.zeros((nv, nv))
    for t in range(len(e0)):
        a = e0[t]; b = e1[t]
        if keep[a] and keep[b]:
            w = abs(flat[a] - flat[b]) if weighted else 1.0
            G[newid[a], newid[b]] += w
            G[newid[b], newid[a]] += w
    for i in range(nv_all):
        if keep[i]:
            G[newid[i], newid[i]] += flat[i] if weighted else 1.0
    return G

def img_to_graph(img, *, mask = None, dense: Literal[bool] = False):
    import scipy.sparse as sp
    A = np.asarray(img, dtype=float)
    nx = A.shape[0]
    ny = A.shape[1] if static.len(A.shape) > 1 else 1
    nz = A.shape[2] if static.len(A.shape) > 2 else 1
    G = _to_graph_dense(nx, ny, nz, mask, A)
    if dense:
        return G
    else:
        return sp.coo_array(G)

def grid_to_graph(n_x: int, n_y: int, n_z: int = 1, *, mask = None, dense: Literal[bool] = False):
    import scipy.sparse as sp
    G = _to_graph_dense(n_x, n_y, n_z, mask, None)
    if dense:
        return G.astype(int)
    else:
        return sp.coo_array(G.astype(int))
