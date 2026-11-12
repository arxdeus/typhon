# Ported from braindecode 2.0.0 / skorch 1.4.0 (BSD-3-Clause)
#
# Native replacement for the skorch-based EEGClassifier / EEGRegressor: the
# default skorch training path (no validation split or a fraction split,
# shuffled training batches with drop_last=True, mean train loss per epoch
# weighted by batch size), predict / predict_proba / score and a history.
#
# Shuffling follows torch.utils.data.RandomSampler: every epoch draws a seed
# from the global torch generator (torch.empty((), int64).random_()) and takes
# torch.randperm(n, generator=Generator().manual_seed(seed)), after the
# DataLoader iterator base_seed draw.

import numpy as np
import torch
from .datasets.base import BaseConcatDataset


class EpochRecord:
    epoch: int
    train_loss: float
    train_accuracy: Optional[float]
    valid_loss: Optional[float]
    valid_accuracy: Optional[float]
    lr: float

    def __init__(self, epoch: int, lr: float):
        self.epoch = epoch
        self.train_loss = 0.0
        self.train_accuracy = None
        self.valid_loss = None
        self.valid_accuracy = None
        self.lr = lr

    def __getitem__(self, key: str) -> float:
        if key == "epoch":
            return float(self.epoch)
        if key == "train_loss":
            return self.train_loss
        if key == "lr":
            return self.lr
        if key == "train_accuracy" and self.train_accuracy is not None:
            return self.train_accuracy.__val__()
        if key == "valid_loss" and self.valid_loss is not None:
            return self.valid_loss.__val__()
        if key == "valid_accuracy" and self.valid_accuracy is not None:
            return self.valid_accuracy.__val__()
        raise KeyError(key)


class History:
    records: List[EpochRecord]

    def __init__(self):
        self.records = List[EpochRecord]()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> EpochRecord:
        return self.records[i]

    def __iter__(self):
        for r in self.records:
            yield r

    def __getitem__(self, key: Tuple[Slice, str]) -> List[float]:
        sl, name = key
        n = len(self.records)
        start, stop, step, _ = sl.adjust_indices(n)
        return [self.records[i][name] for i in range(start, stop, step)]


def _sampler_perm(n: int):
    """DataLoader iter: base_seed draw, then RandomSampler seed draw, then randperm."""
    from torch.utils.data.sampler import _rand_int64_seed
    _rand_int64_seed(None)
    seed = _rand_int64_seed(None)
    g = torch.Generator()
    g.manual_seed(seed)
    return [int(v) for v in torch.randperm(n, generator=g).tolist()]


class _XY:
    """In-memory (X float32, y int) data, from arrays or a windows dataset."""
    X: np.ndarray[np.float32, 3]
    y: np.ndarray[int, 1]

    def __init__(self, X, y):
        if isinstance(X, BaseConcatDataset):
            Xs, ys = X.get_xy(list(range(len(X))))
            self.X = Xs
            self.y = ys
        else:
            self.X = np.asarray(X).astype(np.float32)
            self.y = np.asarray(y).astype(int)

    def __len__(self) -> int:
        return self.X.shape[0]

    def batch(self, idx: List[int]):
        sel = np.array(idx, dtype=int)
        xb = torch.tensor(np.ascontiguousarray(self.X[sel]))
        yb = torch.tensor(np.ascontiguousarray(self.y[sel]))
        return xb, yb

    def take(self, idx: List[int]) -> _XY:
        sel = np.array(idx, dtype=int)
        return _XY(self.X[sel].copy(), self.y[sel].copy())


def _make_crit(criterion):
    return criterion()

def _make_opt(optimizer, params, lr: float):
    return optimizer(params, lr=lr)


class _EEGClassifier[M, C, O]:
    """Training loop with skorch semantics for the default path.

    module: an initialized torch.nn.Module; criterion: a loss class
    (e.g. nn.CrossEntropyLoss); optimizer: an optimizer class
    (e.g. torch.optim.AdamW) built as optimizer(module.parameters(), lr=lr).
    train_split=None trains on everything; a float in (0, 1) keeps that
    fraction as a validation set (stratification is not applied).
    callbacks: list of "accuracy" to record train/valid accuracy.
    """
    module_: M
    criterion_: C
    optimizer_: O
    lr: float
    batch_size: int
    max_epochs: int
    train_split: Optional[float]
    shuffle: bool
    drop_last: bool
    callbacks: List[str]
    history: History
    classes_: List[int]

    def __init__(self, module: M, crit: C, opt: O, lr0: float, lr: Optional[float],
                 train_split: Optional[float], batch_size: int, max_epochs: int,
                 callbacks: Optional[List[str]], shuffle: bool, drop_last: bool):
        self.module_ = module
        self.criterion_ = crit
        self.optimizer_ = opt
        self.lr = lr0 if lr is None else lr.__val__()
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.train_split = train_split
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.callbacks = List[str]() if callbacks is None else callbacks.__val__()
        self.history = History()
        self.classes_ = List[int]()

    def _forward(self, data: _XY) -> np.ndarray[np.float32, 2]:
        self.module_.eval()
        outs = List[np.ndarray[np.float32, 2]]()
        with torch.no_grad():
            k = 0
            n = len(data)
            while k < n:
                e = min(k + self.batch_size, n)
                xb, _ = data.batch(list(range(k, e)))
                outs.append(self.module_(xb).numpy(np.float32, 2).copy())
                k = e
        return np.concatenate(outs, axis=0)

    def _eval_loss(self, data: _XY):
        self.module_.eval()
        tot = 0.0
        correct = 0
        with torch.no_grad():
            k = 0
            n = len(data)
            while k < n:
                e = min(k + self.batch_size, n)
                xb, yb = data.batch(list(range(k, e)))
                out = self.module_(xb)
                tot += float(self.criterion_(out, yb).item()) * (e - k)
                pr = out.numpy(np.float32, 2)
                for i in range(e - k):
                    if int(np.argmax(pr[i])) == int(data.y[k + i]):
                        correct += 1
                k = e
        return tot / max(len(data), 1), correct / max(len(data), 1)

    def fit(self, X, y = None, epochs: Optional[int] = None):
        data = _XY(X, y)
        cl = sorted(list(Set[int](data.y.tolist())))
        self.classes_ = cl
        train = data
        valid: Optional[_XY] = None
        if self.train_split is not None:
            n = len(data)
            n_valid = int(np.ceil(n * self.train_split.__val__()))
            train = data.take(list(range(n - n_valid)))
            valid = data.take(list(range(n - n_valid, n)))
        n_ep = self.max_epochs if epochs is None else epochs.__val__()
        for ep in range(n_ep):
            self.module_.train()
            n = len(train)
            order = _sampler_perm(n) if self.shuffle else list(range(n))
            rec = EpochRecord(len(self.history) + 1, self.lr)
            tot = 0.0
            cnt = 0
            k = 0
            while k < n:
                e = min(k + self.batch_size, n)
                if self.drop_last and e - k < self.batch_size:
                    break
                xb, yb = train.batch(order[k:e])
                self.optimizer_.zero_grad()
                out = self.module_(xb)
                loss = self.criterion_(out, yb)
                loss.backward()
                self.optimizer_.step()
                tot += float(loss.item()) * (e - k)
                cnt += e - k
                k = e
            rec.train_loss = tot / cnt if cnt > 0 else float("nan")
            if "accuracy" in self.callbacks:
                _, acc = self._eval_loss(train)
                rec.train_accuracy = acc
            if valid is not None:
                vl, va = self._eval_loss(valid.__val__())
                rec.valid_loss = vl
                if "accuracy" in self.callbacks:
                    rec.valid_accuracy = va
            self.history.records.append(rec)
        return self

    def partial_fit(self, X, y = None):
        return self.fit(X, y, epochs=1)

    def predict_proba(self, X, y = None) -> np.ndarray[np.float32, 2]:
        """Module outputs (braindecode does not apply softmax)."""
        return self._forward(_XY(X, y if not isinstance(y, NoneType) else np.zeros(0, dtype=int)))

    def predict(self, X) -> np.ndarray[int, 1]:
        p = self.predict_proba(X)
        out = np.empty(p.shape[0], dtype=int)
        for i in range(p.shape[0]):
            out[i] = int(np.argmax(p[i]))
        return out

    def score(self, X, y = None) -> float:
        d = _XY(X, y)
        p = self.predict(X)
        c = 0
        for i in range(len(d)):
            if p[i] == d.y[i]:
                c += 1
        return c / len(d)


def EEGClassifier(module, criterion = None, optimizer = None, optimizer__lr: float = 0.01,
                  lr: Optional[float] = None, train_split: Optional[float] = None,
                  batch_size: int = 128, max_epochs: int = 10, callbacks: Optional[List[str]] = None,
                  iterator_train__shuffle: bool = True, iterator_train__drop_last: bool = True,
                  device: str = "cpu", cropped: bool = False, verbose: int = 0):
    """braindecode.EEGClassifier (skorch default training path)."""
    if isinstance(criterion, NoneType):
        from torch import nn
        return EEGClassifier(module, nn.CrossEntropyLoss, optimizer, optimizer__lr, lr, train_split,
                             batch_size, max_epochs, callbacks, iterator_train__shuffle,
                             iterator_train__drop_last)
    else:
        lr_ = optimizer__lr if lr is None else lr.__val__()
        crit = criterion()
        opt = optimizer(module.parameters(), lr=lr_)
        return _EEGClassifier(module, crit, opt, optimizer__lr, lr, train_split, batch_size,
                              max_epochs, callbacks, iterator_train__shuffle, iterator_train__drop_last)


EEGRegressor = EEGClassifier
