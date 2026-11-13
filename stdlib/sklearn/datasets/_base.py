# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Toy dataset loaders reading sklearn's bundled CSV files (copied to
# datasets/data/, the .csv.gz ones decompressed). Returns (data, target)
# with return_X_y=True, else a _Dataset record with the main Bunch fields.

import numpy as np

def _data_dir() -> str:
    f = __file__
    return f[:f.rfind("/") + 1] + "data/"

class _Dataset:
    data: np.ndarray[float, 2]
    target: np.ndarray[int, 1]
    target_names: List[str]
    feature_names: List[str]
    filename: str

    def __init__(self, data: np.ndarray[float, 2], target: np.ndarray[int, 1],
                 target_names: List[str], feature_names: List[str], filename: str):
        self.data = data
        self.target = target
        self.target_names = target_names
        self.feature_names = feature_names
        self.filename = filename

class _RegDataset:
    data: np.ndarray[float, 2]
    target: np.ndarray[float, 1]
    feature_names: List[str]

    def __init__(self, data: np.ndarray[float, 2], target: np.ndarray[float, 1],
                 feature_names: List[str]):
        self.data = data
        self.target = target
        self.feature_names = feature_names

def load_csv_data(data_file_name: str):
    """header 'n_samples,n_features,name...' then rows 'x...,label'."""
    path = _data_dir() + data_file_name
    names = List[str]()
    data = np.empty((0, 0))
    target = np.empty(0, dtype=int)
    with open(path) as f:
        first = True
        i = 0
        nf = 0
        for line in f:
            s = line.strip()
            if s == "":
                continue
            parts = s.split(",")
            if first:
                n = int(parts[0])
                nf = int(parts[1])
                for k in range(2, len(parts)):
                    names.append(parts[k])
                data = np.empty((n, nf))
                target = np.empty(n, dtype=int)
                first = False
                continue
            for k in range(nf):
                data[i, k] = float(parts[k])
            target[i] = int(parts[nf])
            i += 1
    return data, target, names

def _read_matrix(path: str, sep: str) -> np.ndarray[float, 2]:
    rows = List[List[float]]()
    with open(path) as f:
        for line in f:
            s = line.strip()
            if s == "":
                continue
            r = List[float]()
            if sep == " ":
                for t in s.split():
                    r.append(float(t))
            else:
                for t in s.split(sep):
                    r.append(float(t))
            rows.append(r)
    n = len(rows)
    m = len(rows[0]) if n > 0 else 0
    out = np.empty((n, m))
    for i in range(n):
        for j in range(m):
            out[i, j] = rows[i][j]
    return out

def load_iris(*, return_X_y: Literal[bool] = False, as_frame: bool = False):
    data, target, names = load_csv_data("iris.csv")
    if return_X_y:
        return data, target
    else:
        return _Dataset(data, target, names, ["sepal length (cm)", "sepal width (cm)",
                        "petal length (cm)", "petal width (cm)"], "iris.csv")

def load_wine(*, return_X_y: Literal[bool] = False, as_frame: bool = False):
    data, target, names = load_csv_data("wine_data.csv")
    if return_X_y:
        return data, target
    else:
        return _Dataset(data, target, names, ["alcohol", "malic_acid", "ash", "alcalinity_of_ash",
                        "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
                        "proanthocyanins", "color_intensity", "hue",
                        "od280/od315_of_diluted_wines", "proline"], "wine_data.csv")

def load_breast_cancer(*, return_X_y: Literal[bool] = False, as_frame: bool = False):
    data, target, names = load_csv_data("breast_cancer.csv")
    if return_X_y:
        return data, target
    else:
        fn = List[str]()
        base = ["radius", "texture", "perimeter", "area", "smoothness", "compactness",
                "concavity", "concave points", "symmetry", "fractal dimension"]
        for b in base:
            fn.append("mean " + b)
        for b in base:
            fn.append(b + " error")
        for b in base:
            fn.append("worst " + b)
        return _Dataset(data, target, names, fn, "breast_cancer.csv")

def load_digits(*, n_class: int = 10, return_X_y: Literal[bool] = False, as_frame: bool = False):
    raw = _read_matrix(_data_dir() + "digits.csv", ",")
    n = raw.shape[0]
    keep = List[int]()
    for i in range(n):
        if int(raw[i, 64]) < n_class:
            keep.append(i)
    data = np.empty((len(keep), 64))
    target = np.empty(len(keep), dtype=int)
    for a in range(len(keep)):
        for j in range(64):
            data[a, j] = raw[keep[a], j]
        target[a] = int(raw[keep[a], 64])
    if return_X_y:
        return data, target
    else:
        fn = [f"pixel_{r}_{c}" for r in range(8) for c in range(8)]
        return _Dataset(data, target, [str(i) for i in range(10)], fn, "digits.csv")

def load_diabetes(*, return_X_y: Literal[bool] = False, as_frame: bool = False, scaled: bool = True):
    data = _read_matrix(_data_dir() + "diabetes_data_raw.csv", " ")
    t2 = _read_matrix(_data_dir() + "diabetes_target.csv", " ")
    target = np.ascontiguousarray(t2[:, 0])
    if scaled:
        from ..preprocessing import scale
        data = scale(data, copy=False)
        data /= float(data.shape[0]) ** 0.5
    if return_X_y:
        return data, target
    else:
        return _RegDataset(data, target, ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"])

class _Linnerud:
    data: np.ndarray[float, 2]
    target: np.ndarray[float, 2]
    feature_names: List[str]
    target_names: List[str]

    def __init__(self, data: np.ndarray[float, 2], target: np.ndarray[float, 2],
                 feature_names: List[str], target_names: List[str]):
        self.data = data
        self.target = target
        self.feature_names = feature_names
        self.target_names = target_names

def _read_header_matrix(path: str):
    with open(path) as f:
        header = f.readline().split()
    return _read_matrix_skip(path, 1), header

def _read_matrix_skip(path: str, skip: int) -> np.ndarray[float, 2]:
    rows = List[List[float]]()
    i = 0
    with open(path) as f:
        for line in f:
            if i < skip:
                i += 1
                continue
            s = line.strip()
            if s == "":
                continue
            r = List[float]()
            for t in s.split():
                r.append(float(t))
            rows.append(r)
    n = len(rows)
    m = len(rows[0]) if n > 0 else 0
    out = np.empty((n, m))
    for a in range(n):
        for j in range(m):
            out[a, j] = rows[a][j]
    return out

def load_linnerud(*, return_X_y: Literal[bool] = False, as_frame: bool = False):
    """Exercise (data) vs physiological (target) measurements; 20 x 3 each."""
    d, dn = _read_header_matrix(_data_dir() + "linnerud_exercise.csv")
    t, tn = _read_header_matrix(_data_dir() + "linnerud_physiological.csv")
    if return_X_y:
        return d, t
    else:
        return _Linnerud(d, t, dn, tn)

class _Files:
    data: List[str]
    filenames: List[str]
    target: np.ndarray[int, 1]
    target_names: List[str]

    def __init__(self, data: List[str], filenames: List[str], target: np.ndarray[int, 1],
                 target_names: List[str]):
        self.data = data
        self.filenames = filenames
        self.target = target
        self.target_names = target_names

def load_files(container_path: str, *, description: Optional[str] = None,
               categories: Optional[List[str]] = None, load_content: bool = True,
               shuffle: bool = True, encoding: Optional[str] = None,
               decode_error: str = "strict", random_state: int = 0, allowed_extensions = None):
    """Load text files with categories as subfolder names (sorted). shuffle
    uses RandomState(random_state) like sklearn."""
    import os
    folders = List[str]()
    for name in sorted(os.listdir(container_path)):
        if os.path.isdir(os.path.join(container_path, name)):
            folders.append(name)
    if categories is not None:
        keep = List[str]()
        for c in folders:
            if c in categories:
                keep.append(c)
        folders = keep
    target_l = List[int]()
    files = List[str]()
    for label in range(len(folders)):
        folder = os.path.join(container_path, folders[label])
        for d in sorted(os.listdir(folder)):
            files.append(os.path.join(folder, d))
            target_l.append(label)
    target = np.array(target_l, dtype=int)
    order = np.arange(len(files))
    if shuffle:
        from . import _shuf
        order = _shuf(len(files), random_state)
    fs = List[str]()
    tg = np.empty(len(files), dtype=int)
    data = List[str]()
    for k in range(len(files)):
        j = int(order[k])
        fs.append(files[j])
        tg[k] = target[j]
        if load_content:
            with open(files[j]) as fh:
                data.append(fh.read())
    return _Files(data, fs, tg, folders)
