# Generates the expected literals for test/sklearn/test_feature_extraction.py (scikit-learn 1.9.1).
#   <venv>/bin/python test/sklearn/gen_feature_extraction_expected.py
import numpy as np
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer, HashingVectorizer
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d, img_to_graph, grid_to_graph

def L(name, a):
    a = a.toarray() if hasattr(a, "toarray") else a
    print(f"{name} = {[float(v) for v in np.asarray(a, dtype=float).ravel()]!r}")

def S(name, a):
    print(f"{name} = {[str(v) for v in a]!r}")

docs = ["The quick brown fox jumps over the lazy dog.", "Never jump over the lazy dog quickly!",
        "A fox, a dog and a café: naïve résumé.", "the the THE dog dog fox", "Brown foxes are quick; dogs are lazy."]
cv = CountVectorizer(); L("cv_X", cv.fit_transform(docs)); S("cv_names", cv.get_feature_names_out())
cv = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=2); L("cv2_X", cv.fit_transform(docs)); S("cv2_names", cv.get_feature_names_out())
cv = CountVectorizer(max_features=5, binary=True, max_df=0.8); L("cv3_X", cv.fit_transform(docs)); S("cv3_names", cv.get_feature_names_out())
cv = CountVectorizer(analyzer="char", ngram_range=(2, 3), strip_accents="unicode", min_df=4); L("cv4_X", cv.fit_transform(docs)); S("cv4_names", cv.get_feature_names_out())
cv = CountVectorizer(analyzer="char_wb", ngram_range=(2, 3), strip_accents="ascii", min_df=3); L("cv5_X", cv.fit_transform(docs)); S("cv5_names", cv.get_feature_names_out())
cv = CountVectorizer(vocabulary=["dog", "fox", "lazy", "cat"]); L("cv6_X", cv.fit_transform(docs)); L("cv6_T", cv.transform(["dog cat cat"]))
cv = CountVectorizer(lowercase=False, token_pattern=r"\b\w+\b"); cv.fit(docs[:2]); L("cv7_T", cv.transform(docs)); S("cv7_names", cv.get_feature_names_out())
C = CountVectorizer().fit_transform(docs)
t = TfidfTransformer().fit(C); L("tt_idf", t.idf_); L("tt_X", t.transform(C))
t = TfidfTransformer(norm="l1", smooth_idf=False, sublinear_tf=True).fit(C); L("tt2_X", t.transform(C))
tv = TfidfVectorizer(ngram_range=(1, 2), max_df=0.9); L("tv_X", tv.fit_transform(docs)); L("tv_idf", tv.idf_)
tv = TfidfVectorizer(use_idf=False, norm=None, analyzer="char_wb", ngram_range=(3, 3)); L("tv2_X", tv.transform(docs[:2]) if False else tv.fit_transform(docs[:2]))
hv = HashingVectorizer(n_features=16); L("hv_X", hv.transform(docs))
hv = HashingVectorizer(n_features=10, alternate_sign=False, norm="l1", ngram_range=(1, 2)); L("hv2_X", hv.transform(docs))
D = [{"city": "Dubai", "temp": 33.0}, {"city": "London", "temp": 12.0, "rain": 1.0}, {"city": "Paris", "temp": 18.5}]
dv = DictVectorizer(sparse=False); L("dv_X", dv.fit_transform(D)); S("dv_names", dv.get_feature_names_out())
L("dv_T", dv.transform([{"city": "Paris", "wind": 3.0, "temp": 1.0}]))
dv = DictVectorizer(sort=False); L("dv2_X", dv.fit_transform(D)); S("dv2_names", dv.get_feature_names_out())
fh = FeatureHasher(n_features=12); L("fh_X", fh.transform([{"dog": 1.0, "cat": 2.0, "elephant": 4.0}, {"dog": 2.0, "run": 5.0, "city": "red"}]))
fh = FeatureHasher(n_features=8, input_type="string", alternate_sign=False); L("fh2_X", fh.transform([["a", "b", "a"], ["hello", "world"]]))
img = (np.arange(30.0).reshape(5, 6) * 7) % 11
P = extract_patches_2d(img, (2, 3)); L("pt", P[:4]); print("pt_n =", P.shape[0])
P = extract_patches_2d(img, (3, 2), max_patches=5, random_state=0); L("ptr", P)
img3 = np.stack([img, img * 2], axis=2); P3 = extract_patches_2d(img3, (2, 2), max_patches=0.3, random_state=1); L("pt3", P3); print("pt3_n =", P3.shape[0])
P = extract_patches_2d(img, (2, 3)); L("rec", reconstruct_from_patches_2d(P * 1.5, (5, 6)))
L("i2g", img_to_graph(img[:3, :3]))
m = np.ones((3, 4), dtype=bool); m[1, 2] = False; L("g2g", grid_to_graph(3, 4, mask=m))
L("g2g3", grid_to_graph(2, 2, 2))
