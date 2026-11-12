# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# RawDataset, EEGWindowsDataset and BaseConcatDataset. Typhon is statically
# typed, so:
# - a description is an ordered list of (key, pandas Scalar) pairs, exposed as
#   a pandas Series (`ds.description`) like braindecode;
# - targets are ints (`y`), as produced by every windower in braindecode when
#   a mapping or integer labels are used;
# - windows are served from an in-memory mne RawArray (preload=True path).

import numpy as np
import pandas as pd
from pandas import Scalar
from mne.io import RawArray

WindowTransform = Callable[[np.ndarray[np.float32, 2]], np.ndarray[np.float32, 2]]


class Description:
    """Ordered key -> pandas Scalar mapping (braindecode description Series)."""
    keys: List[str]
    values: List[Scalar]

    def __init__(self):
        self.keys = List[str]()
        self.values = List[Scalar]()

    def __init__(self, d: Description):
        self.keys = list(d.keys)
        self.values = list(d.values)

    def __init__(self, d: Dict[str, T], T: type):
        self.keys = List[str]()
        self.values = List[Scalar]()
        for k, v in d.items():
            self.keys.append(k)
            self.values.append(Scalar.of(v))

    def __init__(self, s: pd.Series):
        self.keys = [str(k) for k in s.index]
        self.values = s.tolist()

    def __init__(self, d: NoneType):
        self.keys = List[str]()
        self.values = List[Scalar]()

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, k: str) -> bool:
        return k in self.keys

    def __getitem__(self, k: str) -> Scalar:
        i = self.keys.index(k) if k in self.keys else -1
        if i < 0:
            raise KeyError(k)
        return self.values[i]

    def __setitem__(self, k: str, v):
        sv = Scalar.of(v)
        if k in self.keys:
            self.values[self.keys.index(k)] = sv
        else:
            self.keys.append(k)
            self.values.append(sv)

    def items(self):
        for i in range(len(self.keys)):
            yield (self.keys[i], self.values[i])

    def copy(self) -> Description:
        return Description(self)

    def to_series(self) -> pd.Series:
        return pd.Series(self.values, index=self.keys)

    def __repr__(self) -> str:
        return self.to_series().__repr__()


def _dict_or_none(description) -> Description:
    if isinstance(description, Description):
        return description.copy()
    return Description(description)


class RawDataset:
    """Continuous recording with a description (braindecode RawDataset)."""
    raw: RawArray
    _description: Description
    target_name: Optional[str]
    transform: Optional[Callable[[np.ndarray[float, 2]], np.ndarray[float, 2]]]
    raw_preproc_kwargs: List[str]

    def __init__(self, raw: RawArray, description = None, target_name: Optional[str] = None,
                 transform = None):
        self.raw = raw
        self._description = _dict_or_none(description)
        self.target_name = target_name
        self.transform = transform
        self.raw_preproc_kwargs = List[str]()

    @property
    def description(self) -> pd.Series:
        return self._description.to_series()

    def set_description(self, description, overwrite: bool = False):
        d = _dict_or_none(description)
        for k, v in d.items():
            if k in self._description and not overwrite:
                raise AssertionError(f"'{k}' already in description. Please rename or set overwrite to True.")
            self._description[k] = v

    def __len__(self) -> int:
        return self.raw.n_times

    def __getitem__(self, index: int):
        X = self.raw._data[:, index:index + 1].copy()
        if self.transform is not None:
            X = self.transform(X)
        y: Optional[Scalar] = None
        if self.target_name is not None:
            y = self._description[self.target_name]
        return X, y

    def __repr__(self) -> str:
        return f"RawDataset({self.raw.info.nchan} ch, {self.raw.info.sfreq} Hz, {self.raw.n_times} samples)"


BaseDataset = RawDataset


class EEGWindowsDataset:
    """Windows served from a continuous recording (braindecode EEGWindowsDataset).
    __getitem__ returns (X float32 (n_ch, n_times), y int, [i_window, start, stop])."""
    raw: RawArray
    metadata: pd.DataFrame
    _description: Description
    crop_inds: np.ndarray[int, 2]
    y: List[int]
    transform: Optional[WindowTransform]
    targets_from: str
    last_target_only: bool
    raw_preproc_kwargs: List[str]
    window_kwargs: List[str]
    _read_offset: int  # sample offset of raw relative to crop_inds (create_from_mne_epochs)

    def __init__(self, raw: RawArray, metadata: pd.DataFrame, description = None,
                 transform = None, targets_from: str = "metadata", last_target_only: bool = True):
        self.raw = raw
        self.metadata = metadata
        self._description = _dict_or_none(description)
        self.transform = transform
        if targets_from not in ("metadata", "channels"):
            raise ValueError("Wrong value for parameter `targets_from`.")
        if targets_from == "channels":
            raise NotImplementedError("targets_from='channels' is not supported in Typhon braindecode")
        self.targets_from = targets_from
        self.last_target_only = last_target_only
        self.crop_inds = metadata.loc[:, ["i_window_in_trial", "i_start_in_trial", "i_stop_in_trial"]].to_numpy(dtype=int)
        self.y = [int(v) for v in metadata["target"].to_list()]
        self.raw_preproc_kwargs = List[str]()
        self.window_kwargs = List[str]()
        self._read_offset = 0

    @property
    def description(self) -> pd.Series:
        return self._description.to_series()

    def set_description(self, description, overwrite: bool = False):
        d = _dict_or_none(description)
        for k, v in d.items():
            if k in self._description and not overwrite:
                raise AssertionError(f"'{k}' already in description. Please rename or set overwrite to True.")
            self._description[k] = v

    def __len__(self) -> int:
        return self.crop_inds.shape[0]

    def _crop(self, index: int) -> List[int]:
        return [int(self.crop_inds[index, 0]), int(self.crop_inds[index, 1]), int(self.crop_inds[index, 2])]

    def get_window(self, index: int) -> np.ndarray[np.float32, 2]:
        s = int(self.crop_inds[index, 1]) - self._read_offset
        e = int(self.crop_inds[index, 2]) - self._read_offset
        X = self.raw._data[:, s:e].astype(np.float32)
        if self.transform is not None:
            X = self.transform(X)
        return X

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        return self.get_window(index), self.y[index], self._crop(index)

    def __repr__(self) -> str:
        return f"EEGWindowsDataset({len(self)} windows, {self.raw.info.nchan} ch, {self.raw.info.sfreq} Hz)"


WindowsDataset = EEGWindowsDataset


class BaseConcatDataset[T]:
    """Concatenation of RawDataset or EEGWindowsDataset objects."""
    datasets: List[T]
    cumulative_sizes: List[int]
    target_transform: Optional[Callable[[int], int]]

    def __init__(self, list_of_ds: List[T], target_transform = None, *, lazy: bool = False,
                 coalesce_reads: bool = False):
        if len(list_of_ds) == 0:
            raise ValueError("datasets should not be an empty iterable")
        self.datasets = list(list_of_ds)
        self.target_transform = target_transform
        self._recompute()

    def __init__(self, list_of_ds: List[BaseConcatDataset[T]], target_transform = None, *,
                 lazy: bool = False, coalesce_reads: bool = False):
        flat = List[T]()
        for c in list_of_ds:
            flat.extend(c.datasets)
        if len(flat) == 0:
            raise ValueError("datasets should not be an empty iterable")
        self.datasets = flat
        self.target_transform = target_transform
        self._recompute()

    def _recompute(self):
        cs = List[int]()
        t = 0
        for d in self.datasets:
            t += len(d)
            cs.append(t)
        self.cumulative_sizes = cs

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if len(self.cumulative_sizes) else 0

    def _locate(self, idx: int) -> Tuple[int, int]:
        n = len(self)
        if idx < 0:
            if -idx > n:
                raise ValueError("absolute value of index should not exceed dataset length")
            idx += n
        if idx >= n:
            raise IndexError("index out of range")
        lo = 0
        hi = len(self.cumulative_sizes)
        while lo < hi:  # bisect_right
            mid = (lo + hi) // 2
            if idx < self.cumulative_sizes[mid]:
                hi = mid
            else:
                lo = mid + 1
        k = lo
        s = idx if k == 0 else idx - self.cumulative_sizes[k - 1]
        return k, s

    def __getitem__(self, idx: int):
        k, s = self._locate(idx)
        item = self.datasets[k][s]
        if self.target_transform is not None:
            if isinstance(item[1], int):
                return item[0], self.target_transform(item[1]), item[2]
        return item

    def get_xy(self, indices: List[int]):
        """Stacked windows (float32) and int targets for the given indices."""
        X0, _, _ = self[indices[0]]
        out = np.empty((len(indices), X0.shape[0], X0.shape[1]), dtype=np.float32)
        y = np.empty(len(indices), dtype=int)
        for i in range(len(indices)):
            X, t, _ = self[indices[i]]
            out[i] = X
            y[i] = t
        return out, y

    @property
    def description(self) -> pd.DataFrame:
        keys = List[str]()
        for d in self.datasets:
            for k in d._description.keys:
                if k not in keys:
                    keys.append(k)
        n = len(self.datasets)
        if len(keys) == 0:
            return pd.DataFrame(index=list(range(n)))
        cols = Dict[str, pd.Series]()
        for k in keys:
            vals = List[Scalar]()
            for ds in self.datasets:
                vals.append(ds._description[k] if k in ds._description else Scalar.none())
            cols[k] = pd.Series(vals)
        return pd.DataFrame(cols)

    def set_description(self, description: Dict[str, List[V]], overwrite: bool = False, V: type):
        for k, v in description.items():
            if len(v) != len(self.datasets):
                raise ValueError(f"Length of values ({len(v)}) does not match number of datasets ({len(self.datasets)}).")
            for i in range(len(self.datasets)):
                d = Description()
                d[k] = v[i]
                self.datasets[i].set_description(d, overwrite=overwrite)

    def _subset(self, ids: List[int]):
        return BaseConcatDataset([self.datasets[i] for i in ids], target_transform=self.target_transform)

    def split(self, by: str) -> Dict[str, BaseConcatDataset[T]]:
        """Split on a description column (groups sorted like pandas groupby)."""
        groups = Dict[Scalar, List[int]]()
        order = List[Scalar]()
        for i in range(len(self.datasets)):
            v = self.datasets[i]._description[by]
            if v not in groups:
                groups[v] = List[int]()
                order.append(v)
            groups[v].append(i)
        order.sort()
        out = Dict[str, BaseConcatDataset[T]]()
        for v in order:
            out[str(v)] = self._subset(groups[v])
        return out

    def split(self, by: List[int]) -> Dict[str, BaseConcatDataset[T]]:
        return self.split([by])

    def split(self, by: List[List[int]]) -> Dict[str, BaseConcatDataset[T]]:
        out = Dict[str, BaseConcatDataset[T]]()
        for i in range(len(by)):
            out[str(i)] = self._subset(by[i])
        return out

    def split(self, by: Dict[str, List[int]]) -> Dict[str, BaseConcatDataset[T]]:
        out = Dict[str, BaseConcatDataset[T]]()
        for k, v in by.items():
            out[k] = self._subset(v)
        return out

    def get_metadata(self) -> pd.DataFrame:
        """Window metadata of all datasets plus description columns and i_dataset."""
        if not isinstance(self.datasets[0], EEGWindowsDataset):
            compile_error("Metadata dataframe can only be computed when all datasets are WindowsDataset.")
        dfs = List[pd.DataFrame]()
        for i in range(len(self.datasets)):
            ds = self.datasets[i]
            df = ds.metadata.copy()
            if "i_dataset" in ds._description or "i_dataset" in df.columns:
                raise ValueError("Window metadata and dataset descriptions cannot contain the key 'i_dataset'.")
            for k, v in ds._description.items():
                df[k] = v
            df["i_dataset"] = i
            dfs.append(df)
        return pd.concat(dfs)

    @property
    def transform(self):
        return [d.transform for d in self.datasets]

    @transform.setter
    def transform(self, fn):
        for d in self.datasets:
            d.transform = fn

    def set_target(self, column: str):
        for ds in self.datasets:
            n = len(ds)
            md = ds.metadata
            vals = List[int]()
            if column in md.columns:
                vals = [int(v) for v in md[column].to_list()]
            elif column in ds._description:
                vals = [int(ds._description[column]) for _ in range(n)]
            else:
                raise KeyError(f"Column {column} not found in metadata or description")
            md["target"] = vals
            ds.y = vals
        return self

    def __repr__(self) -> str:
        return f"BaseConcatDataset({len(self.datasets)} datasets, {len(self)} items)"
