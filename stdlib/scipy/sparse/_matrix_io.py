# Copyright (C) 2026 the Typhon authors
# Ported from scipy 1.18.1 (BSD-3-Clause)

# save_npz / load_npz (scipy.sparse._matrix_io). Files use the same layout
# as scipy (data, indices, indptr / row, col, format, shape, _is_array).
# Typhon is statically typed, so load_npz returns a CSR matrix/array (the
# stored format is converted); float64 data only.

import numpy as np
from numpy.npio import _save, _npz_entry, _le, NpzFile
from io import BytesIO

def _npy_scalar_bytes(text: str) -> bytes:
    # 0-d '|S<n>' array holding `text`
    n = len(text)
    hdr = "{'descr': '|S" + str(n) + "', 'fortran_order': False, 'shape': (), }"
    total = 10 + len(hdr) + 1
    pad = (64 - total % 64) % 64
    hdr = hdr + " " * pad + "\n"
    out = bytearray(b'\x93NUMPY\x01\x00')
    out.append(len(hdr) & 0xFF)
    out.append((len(hdr) >> 8) & 0xFF)
    out.extend(hdr.encode())
    out.extend(text.encode())
    return bytes(out)

def _npy_bool_scalar(v: bool) -> bytes:
    hdr = "{'descr': '|b1', 'fortran_order': False, 'shape': (), }"
    total = 10 + len(hdr) + 1
    pad = (64 - total % 64) % 64
    hdr = hdr + " " * pad + "\n"
    out = bytearray(b'\x93NUMPY\x01\x00')
    out.append(len(hdr) & 0xFF)
    out.append((len(hdr) >> 8) & 0xFF)
    out.extend(hdr.encode())
    out.append(1 if v else 0)
    return bytes(out)

def _arr_bytes(a) -> bytes:
    bio = BytesIO()
    _save(bio, a)
    return bio.getvalue()

def save_npz(file: str, matrix, compressed: bool = True):
    """Save a csr/csc/coo sparse matrix or array to an .npz file."""
    if not file.endswith('.npz'):
        file += '.npz'
    out = List[bytes]()
    cd = List[bytes]()
    offset = [0]
    if matrix._spfmt() == "csr" or matrix._spfmt() == "csc":
        _npz_entry(out, cd, offset, 'indices', _arr_bytes(matrix.indices), compressed)
        _npz_entry(out, cd, offset, 'indptr', _arr_bytes(matrix.indptr), compressed)
    elif matrix._spfmt() == "coo":
        _npz_entry(out, cd, offset, 'row', _arr_bytes(matrix.row), compressed)
        _npz_entry(out, cd, offset, 'col', _arr_bytes(matrix.col), compressed)
    else:
        raise NotImplementedError(f"Save is not implemented for sparse matrix of format {matrix._spfmt()}.")
    _npz_entry(out, cd, offset, 'format', _npy_scalar_bytes(matrix._spfmt()), compressed)
    shp = np.array([matrix.shape[0], matrix.shape[1]], dtype=int)
    _npz_entry(out, cd, offset, 'shape', _arr_bytes(shp), compressed)
    _npz_entry(out, cd, offset, 'data', _arr_bytes(matrix.data), compressed)
    if not matrix._is_matrix:
        _npz_entry(out, cd, offset, '_is_array', _npy_bool_scalar(True), compressed)
    with open(file, 'wb') as f:
        for b in out:
            f.write(b)
        cdsize = 0
        for b in cd:
            f.write(b)
            cdsize += len(b)
        f.write(b'PK\x05\x06' + _le(0, 2) + _le(0, 2) + _le(len(cd), 2) +
                _le(len(cd), 2) + _le(cdsize, 4) + _le(offset[0], 4) + _le(0, 2))

def load_npz(file: str, as_array: Literal[bool] = False):
    """Load a matrix saved with save_npz as CSR (matrix, or array when
    as_array=True). csr/csc/coo files are supported."""
    from . import csr_matrix, csr_array, csc_matrix, csc_array, coo_matrix, coo_array
    loaded = NpzFile(open(file, 'rb').read())
    if 'format' not in loaded:
        raise ValueError(f"The file {file} does not contain a sparse array or matrix.")
    fmt = ""
    for i in range(len(loaded.names)):
        if loaded.names[i] == 'format':
            raw = loaded.blobs[i]
            # payload follows the header (header length at bytes 8..9)
            hl = int(raw[8]) + (int(raw[9]) << 8)
            fmt = str(raw[10 + hl:])
    data = loaded.get('data', 1, float)
    shp = loaded.get('shape', 1, int)
    shape = (int(shp[0]), int(shp[1]))
    if fmt == "csr" or fmt == "csc":
        ind = loaded.get('indices', 1, int)
        ptr = loaded.get('indptr', 1, int)
        if fmt == "csr":
            m = csr_array((data, ind, ptr), shape=shape) if as_array else csr_matrix((data, ind, ptr), shape=shape)
            return m
        else:
            c = csc_array((data, ind, ptr), shape=shape) if as_array else csc_matrix((data, ind, ptr), shape=shape)
            return c.tocsr()
    elif fmt == "coo":
        r = loaded.get('row', 1, int)
        c = loaded.get('col', 1, int)
        if as_array:
            return coo_array((data, (r, c)), shape=shape).tocsr()
        return coo_matrix((data, (r, c)), shape=shape).tocsr()
    raise NotImplementedError(f"Load is not implemented for sparse matrix of format {fmt}.")
