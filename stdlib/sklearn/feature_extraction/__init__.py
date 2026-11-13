# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

from ._dict_vectorizer import DictVectorizer
from ._hash import FeatureHasher
from . import text
from . import image
from .image import extract_patches_2d, reconstruct_from_patches_2d, img_to_graph, grid_to_graph
