# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# CountVectorizer / TfidfTransformer / TfidfVectorizer / HashingVectorizer on
# lists of str (input='content'); analyzers 'word', 'char', 'char_wb';
# strip_accents None/'ascii'/'unicode'; stop_words None/'english'/list;
# max_df/min_df (int or float), max_features, vocabulary, binary.
# transform() returns scipy.sparse CSR arrays (built from dense rows).

import re
import numpy as np

ENGLISH_STOP_WORDS = set([
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all',
    'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among',
    'amongst', 'amoungst', 'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone',
    'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became',
    'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'behind',
    'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom',
    'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 'cry', 'de',
    'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight',
    'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every',
    'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fifty', 'fill',
    'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four',
    'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have',
    'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers',
    'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in',
    'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter',
    'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might',
    'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my',
    'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 'next', 'nine', 'no',
    'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often',
    'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours',
    'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather',
    're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she',
    'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow',
    'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system',
    'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence',
    'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they',
    'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout',
    'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty',
    'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well',
    'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas',
    'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whither',
    'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without',
    'would', 'yet', 'you', 'your', 'yours', 'yourself', 'yourselves',])

def strip_accents_unicode(s: str) -> str:
    import unicodedata
    ascii_ok = True
    for ch in s:
        if ord(ch) > 127:
            ascii_ok = False
            break
    if ascii_ok:
        return s
    nf = unicodedata.normalize("NFKD", s)
    return "".join([c for c in nf if not unicodedata.combining(c)])

def strip_accents_ascii(s: str) -> str:
    import unicodedata
    nf = unicodedata.normalize("NFKD", s)
    return "".join([c for c in nf if ord(c) < 128])

_WHITE_SPACES = re.compile(r"\s\s+")

class _Analyzer:
    analyzer: str
    lowercase: bool
    strip_accents: str
    token_pattern: str
    ngram_range: Tuple[int, int]
    stop_words: Optional[Set[str]]

    def __init__(self, analyzer: str, lowercase: bool, strip_accents: str, token_pattern: str,
                 ngram_range: Tuple[int, int], stop_words: Optional[Set[str]]):
        if analyzer not in ("word", "char", "char_wb"):
            raise ValueError(f"{analyzer} is not a valid tokenization scheme/analyzer")
        if ngram_range[0] > ngram_range[1]:
            raise ValueError(f"Invalid value for ngram_range={ngram_range} lower boundary larger than the upper boundary.")
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.stop_words = stop_words

    def preprocess(self, doc: str) -> str:
        d = doc.lower() if self.lowercase else doc
        if self.strip_accents == "ascii":
            d = strip_accents_ascii(d)
        elif self.strip_accents == "unicode":
            d = strip_accents_unicode(d)
        return d

    def _word_ngrams(self, tokens: List[str]) -> List[str]:
        if self.stop_words is not None:
            sw = self.stop_words.__val__()
            tokens = [w for w in tokens if w not in sw]
        min_n, max_n = self.ngram_range
        if max_n == 1:
            return tokens
        orig = tokens
        out = List[str]()
        if min_n == 1:
            out = orig.copy()
            min_n += 1
        n0 = len(orig)
        for n in range(min_n, min(max_n + 1, n0 + 1)):
            for i in range(n0 - n + 1):
                out.append(" ".join(orig[i:i + n]))
        return out

    def _char_ngrams(self, text: str) -> List[str]:
        t = _WHITE_SPACES.sub(" ", text)
        L = len(t)
        min_n, max_n = self.ngram_range
        out = List[str]()
        if min_n == 1:
            out = [c for c in t]
            min_n += 1
        for n in range(min_n, min(max_n + 1, L + 1)):
            for i in range(L - n + 1):
                out.append(t[i:i + n])
        return out

    def _char_wb_ngrams(self, text: str) -> List[str]:
        t = _WHITE_SPACES.sub(" ", text)
        min_n, max_n = self.ngram_range
        out = List[str]()
        for w0 in t.split():
            w = " " + w0 + " "
            wl = len(w)
            for n in range(min_n, max_n + 1):
                off = 0
                out.append(w[off:off + n])
                while off + n < wl:
                    off += 1
                    out.append(w[off:off + n])
                if off == 0:
                    break
        return out

    def __call__(self, doc: str) -> List[str]:
        d = self.preprocess(doc)
        if self.analyzer == "word":
            return self._word_ngrams(re.compile(self.token_pattern).findall(d))
        elif self.analyzer == "char":
            return self._char_ngrams(d)
        else:
            return self._char_wb_ngrams(d)

def _stop_set(stop_words) -> Optional[Set[str]]:
    if isinstance(stop_words, NoneType):
        return None
    elif isinstance(stop_words, str):
        if stop_words == "english":
            return set(ENGLISH_STOP_WORDS)
        raise ValueError(f"not a built-in stop list: {stop_words}")
    else:
        return set([str(w) for w in stop_words])

class _CountBase:
    strip_accents: str
    lowercase: bool
    token_pattern: str
    ngram_range: Tuple[int, int]
    analyzer: str
    max_df: float
    _max_df_int: bool
    min_df: float
    _min_df_int: bool
    max_features: Optional[int]
    binary: bool
    _stop: Optional[Set[str]]
    _fixed: Optional[List[str]]
    vocabulary_: Dict[str, int]
    fixed_vocabulary_: bool

    def __init__(self, *, input: str = "content", encoding: str = "utf-8",
                 decode_error: str = "strict", strip_accents = None, lowercase: bool = True,
                 preprocessor = None, tokenizer = None, stop_words = None,
                 token_pattern: str = r"(?u)\b\w\w+\b", ngram_range: Tuple[int, int] = (1, 1),
                 analyzer: str = "word", max_df = 1.0, min_df = 1, max_features: Optional[int] = None,
                 vocabulary = None, binary: bool = False, dtype: type = int):
        self.strip_accents = "" if isinstance(strip_accents, NoneType) else str(strip_accents)
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self._max_df_int = isinstance(max_df, int)
        self.max_df = float(max_df)
        self._min_df_int = isinstance(min_df, int)
        self.min_df = float(min_df)
        self.max_features = max_features
        self.binary = binary
        self._stop = _stop_set(stop_words)
        if isinstance(vocabulary, NoneType):
            self._fixed = None
        else:
            self._fixed = [str(t) for t in vocabulary]
        self.vocabulary_ = {}
        self.fixed_vocabulary_ = False

    def build_analyzer(self) -> _Analyzer:
        return _Analyzer(self.analyzer, self.lowercase, self.strip_accents, self.token_pattern,
                         self.ngram_range, self._stop if self.analyzer == "word" else None)

    def _count(self, docs, vocab: Dict[str, int], grow: bool) -> List[Dict[int, int]]:
        an = self.build_analyzer()
        rows = List[Dict[int, int]]()
        for doc in docs:
            if isinstance(doc, List):
                raise ValueError("Iterable over raw text documents expected, string object received.")
            cnt = Dict[int, int]()
            for feat in an(str(doc)):
                idx = -1
                if feat in vocab:
                    idx = vocab[feat]
                elif grow:
                    idx = len(vocab)
                    vocab[feat] = idx
                if idx >= 0:
                    cnt[idx] = cnt.get(idx, 0) + 1
            rows.append(cnt)
        return rows

    def _matrix(self, rows: List[Dict[int, int]], nf: int) -> np.ndarray[float, 2]:
        X = np.zeros((len(rows), nf))
        for i in range(len(rows)):
            for j, c in rows[i].items():
                X[i, j] = 1.0 if self.binary else float(c)
        return X

    def _fit_dense(self, raw_documents) -> np.ndarray[float, 2]:
        if isinstance(raw_documents, str):
            raise ValueError("Iterable over raw text documents expected, string object received.")
        if self._fixed is not None:
            fv = self._fixed.__val__()
            vocab = Dict[str, int]()
            for i in range(len(fv)):
                if fv[i] in vocab:
                    raise ValueError(f"Duplicate term in vocabulary: {fv[i]!r}")
                vocab[fv[i]] = i
            if len(vocab) == 0:
                raise ValueError("empty vocabulary passed to fit")
            self.vocabulary_ = vocab
            self.fixed_vocabulary_ = True
            return self._matrix(self._count(raw_documents, vocab, False), len(vocab))
        vocab = Dict[str, int]()
        rows = self._count(raw_documents, vocab, True)
        if len(vocab) == 0:
            raise ValueError("empty vocabulary; perhaps the documents only contain stop words")
        X = self._matrix(rows, len(vocab))
        n_doc = X.shape[0]
        max_dc = self.max_df if self._max_df_int else self.max_df * n_doc
        min_dc = self.min_df if self._min_df_int else self.min_df * n_doc
        if max_dc < min_dc:
            raise ValueError("max_df corresponds to < documents than min_df")
        # _sort_features: column order by sorted term
        terms = sorted(vocab.keys())
        order = [vocab[t] for t in terms]
        X = np.ascontiguousarray(X[:, np.array(order)]) if len(order) > 0 else X
        nf = len(terms)
        dfs = np.zeros(nf)
        for j in range(nf):
            for i in range(n_doc):
                if X[i, j] != 0:
                    dfs[j] += 1.0
        mask = [True] * nf
        for j in range(nf):
            if dfs[j] > max_dc or dfs[j] < min_dc:
                mask[j] = False
        if self.max_features is not None:
            lim = self.max_features.__val__()
            if sum([1 for m in mask if m]) > lim:
                tfs = X.sum(axis=0)
                cand = [j for j in range(nf) if mask[j]]
                neg = np.array([-tfs[j] for j in cand])
                sel = np.argsort(neg, kind="quicksort")[:lim]
                mask = [False] * nf
                for s in sel:
                    mask[cand[int(s)]] = True
        kept = [j for j in range(nf) if mask[j]]
        if len(kept) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower min_df or a higher max_df.")
        self.vocabulary_ = {terms[kept[t]]: t for t in range(len(kept))}
        self.fixed_vocabulary_ = False
        return np.ascontiguousarray(X[:, np.array(kept)])


    def _transform_dense(self, raw_documents) -> np.ndarray[float, 2]:
        if len(self.vocabulary_) == 0:
            raise ValueError("Vocabulary not fitted or provided")
        return self._matrix(self._count(raw_documents, self.vocabulary_, False), len(self.vocabulary_))

    def get_feature_names_out(self, input_features = None) -> List[str]:
        out = [""] * len(self.vocabulary_)
        for t, i in self.vocabulary_.items():
            out[i] = t
        return out

    def inverse_transform(self, X) -> List[List[str]]:
        A = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        names = self.get_feature_names_out()
        return [[names[j] for j in range(A.shape[1]) if A[i, j] != 0] for i in range(A.shape[0])]

class CountVectorizer(_CountBase):
    def fit(self, raw_documents, y = None):
        self._fit_dense(raw_documents)
        return self

    def fit_transform(self, raw_documents, y = None):
        import scipy.sparse as sp
        return sp.csr_array(self._fit_dense(raw_documents)).astype(int)

    def transform(self, raw_documents):
        import scipy.sparse as sp
        return sp.csr_array(self._transform_dense(raw_documents)).astype(int)

def _normalize_rows(X: np.ndarray[float, 2], norm: str):
    for i in range(X.shape[0]):
        s = 0.0
        if norm == "l2":
            for j in range(X.shape[1]):
                s += X[i, j] * X[i, j]
            s = np.sqrt(s)
        elif norm == "l1":
            for j in range(X.shape[1]):
                s += abs(X[i, j])
        else:
            for j in range(X.shape[1]):
                s = max(s, abs(X[i, j]))
        if s != 0:
            for j in range(X.shape[1]):
                X[i, j] /= s

class TfidfTransformer:
    norm: str
    use_idf: bool
    smooth_idf: bool
    sublinear_tf: bool
    idf_: np.ndarray[float, 1]

    def __init__(self, *, norm = "l2", use_idf: bool = True, smooth_idf: bool = True,
                 sublinear_tf: bool = False):
        self.norm = "" if isinstance(norm, NoneType) else str(norm)
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.idf_ = np.empty(0)

    def _dense(self, X) -> np.ndarray[float, 2]:
        if hasattr(X, "toarray"):
            return np.asarray(X.toarray(), dtype=float)
        return np.array(np.asarray(X, dtype=float))

    def fit(self, X, y = None):
        A = self._dense(X)
        if self.use_idf:
            n, p = A.shape
            df = np.zeros(p)
            for j in range(p):
                for i in range(n):
                    if A[i, j] != 0:
                        df[j] += 1.0
            df += float(self.smooth_idf)
            ns = n + int(self.smooth_idf)
            idf = np.full(p, float(ns))
            idf /= df
            np.log(idf, out=idf)
            idf += 1.0
            self.idf_ = idf
        return self

    def _transform_dense(self, A: np.ndarray[float, 2]) -> np.ndarray[float, 2]:
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if A[i, j] != 0:
                    if self.sublinear_tf:
                        A[i, j] = np.log(A[i, j]) + 1.0
                    if self.use_idf:
                        A[i, j] *= self.idf_[j]
        if self.norm != "":
            _normalize_rows(A, self.norm)
        return A

    def transform(self, X):
        import scipy.sparse as sp
        return sp.csr_array(self._transform_dense(self._dense(X)))

    def fit_transform(self, X, y = None):
        return self.fit(X).transform(X)

class TfidfVectorizer(_CountBase):
    _tfidf: TfidfTransformer

    def __init__(self, *, input: str = "content", encoding: str = "utf-8",
                 decode_error: str = "strict", strip_accents = None, lowercase: bool = True,
                 preprocessor = None, tokenizer = None, analyzer: str = "word", stop_words = None,
                 token_pattern: str = r"(?u)\b\w\w+\b", ngram_range: Tuple[int, int] = (1, 1),
                 max_df = 1.0, min_df = 1, max_features: Optional[int] = None, vocabulary = None,
                 binary: bool = False, dtype: type = float, norm = "l2", use_idf: bool = True,
                 smooth_idf: bool = True, sublinear_tf: bool = False):
        super().__init__(strip_accents=strip_accents, lowercase=lowercase, stop_words=stop_words,
                         token_pattern=token_pattern, ngram_range=ngram_range, analyzer=analyzer,
                         max_df=max_df, min_df=min_df, max_features=max_features,
                         vocabulary=vocabulary, binary=binary)
        self._tfidf = TfidfTransformer(norm=norm, use_idf=use_idf, smooth_idf=smooth_idf,
                                       sublinear_tf=sublinear_tf)

    @property
    def idf_(self):
        return self._tfidf.idf_

    def fit(self, raw_documents, y = None):
        self._tfidf.fit(self._fit_dense(raw_documents))
        return self

    def fit_transform(self, raw_documents, y = None):
        import scipy.sparse as sp
        X = self._fit_dense(raw_documents)
        self._tfidf.fit(X)
        return sp.csr_array(self._tfidf._transform_dense(X))

    def transform(self, raw_documents):
        import scipy.sparse as sp
        return sp.csr_array(self._tfidf._transform_dense(self._transform_dense(raw_documents)))

class HashingVectorizer:
    _an: _Analyzer
    n_features: int
    binary: bool
    norm: str
    alternate_sign: bool

    def __init__(self, *, input: str = "content", encoding: str = "utf-8",
                 decode_error: str = "strict", strip_accents = None, lowercase: bool = True,
                 preprocessor = None, tokenizer = None, stop_words = None,
                 token_pattern: str = r"(?u)\b\w\w+\b", ngram_range: Tuple[int, int] = (1, 1),
                 analyzer: str = "word", n_features: int = 1048576, binary: bool = False,
                 norm = "l2", alternate_sign: bool = True, dtype: type = float):
        sa = "" if isinstance(strip_accents, NoneType) else str(strip_accents)
        self._an = _Analyzer(analyzer, lowercase, sa, token_pattern, ngram_range,
                             _stop_set(stop_words) if analyzer == "word" else None)
        self.n_features = n_features
        self.binary = binary
        self.norm = "" if isinstance(norm, NoneType) else str(norm)
        self.alternate_sign = alternate_sign

    def fit(self, X, y = None):
        return self

    def partial_fit(self, X, y = None):
        return self

    def transform(self, X):
        import scipy.sparse as sp
        from ._hash import _hash_index
        if isinstance(X, str):
            raise ValueError("Iterable over raw text documents expected, string object received.")
        docs = [str(d) for d in X]
        D = np.zeros((len(docs), self.n_features))
        for i in range(len(docs)):
            for f in self._an(docs[i]):
                idx, h = _hash_index(f, self.n_features)
                v = 1.0
                if self.alternate_sign:
                    v *= float((1 if h >= 0 else 0) * 2 - 1)
                D[i, idx] += v
        if self.binary:
            for i in range(D.shape[0]):
                for j in range(D.shape[1]):
                    if D[i, j] != 0:
                        D[i, j] = 1.0
        if self.norm != "":
            _normalize_rows(D, self.norm)
        return sp.csr_array(D)

    def fit_transform(self, X, y = None):
        return self.transform(X)
