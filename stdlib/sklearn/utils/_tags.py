# Copyright (C) 2026 the Typhon authors
# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# Estimator tags (sklearn/utils/_tags.py). Tags are derived statically from
# the estimator's class: mixins give estimator_type, and an estimator may
# define __sklearn_tags__() returning a Tags to override.

class InputTags:
    one_d_array: bool
    two_d_array: bool
    three_d_array: bool
    sparse: bool
    categorical: bool
    string: bool
    dict: bool
    positive_only: bool
    allow_nan: bool
    pairwise: bool

    def __init__(self, one_d_array: bool = False, two_d_array: bool = True,
                 three_d_array: bool = False, sparse: bool = False,
                 categorical: bool = False, string: bool = False, dict: bool = False,
                 positive_only: bool = False, allow_nan: bool = False,
                 pairwise: bool = False):
        self.one_d_array = one_d_array
        self.two_d_array = two_d_array
        self.three_d_array = three_d_array
        self.sparse = sparse
        self.categorical = categorical
        self.string = string
        self.dict = dict
        self.positive_only = positive_only
        self.allow_nan = allow_nan
        self.pairwise = pairwise

class TargetTags:
    required: bool
    one_d_labels: bool
    two_d_labels: bool
    positive_only: bool
    multi_output: bool
    single_output: bool

    def __init__(self, required: bool = False, one_d_labels: bool = False,
                 two_d_labels: bool = False, positive_only: bool = False,
                 multi_output: bool = False, single_output: bool = True):
        self.required = required
        self.one_d_labels = one_d_labels
        self.two_d_labels = two_d_labels
        self.positive_only = positive_only
        self.multi_output = multi_output
        self.single_output = single_output

class TransformerTags:
    preserves_dtype: List[str]

    def __init__(self):
        self.preserves_dtype = ["float64"]

class ClassifierTags:
    poor_score: bool
    multi_class: bool
    multi_label: bool

    def __init__(self, poor_score: bool = False, multi_class: bool = True,
                 multi_label: bool = False):
        self.poor_score = poor_score
        self.multi_class = multi_class
        self.multi_label = multi_label

class RegressorTags:
    poor_score: bool

    def __init__(self, poor_score: bool = False):
        self.poor_score = poor_score

class Tags:
    estimator_type: Optional[str]
    target_tags: TargetTags
    transformer_tags: Optional[TransformerTags]
    classifier_tags: Optional[ClassifierTags]
    regressor_tags: Optional[RegressorTags]
    array_api_support: bool
    no_validation: bool
    non_deterministic: bool
    requires_fit: bool
    _skip_test: bool
    input_tags: InputTags

    def __init__(self, estimator_type: Optional[str], target_tags: TargetTags,
                 transformer_tags: Optional[TransformerTags] = None,
                 classifier_tags: Optional[ClassifierTags] = None,
                 regressor_tags: Optional[RegressorTags] = None,
                 array_api_support: bool = False, no_validation: bool = False,
                 non_deterministic: bool = False, requires_fit: bool = True,
                 _skip_test: bool = False, input_tags: Optional[InputTags] = None):
        self.estimator_type = estimator_type
        self.target_tags = target_tags
        self.transformer_tags = transformer_tags
        self.classifier_tags = classifier_tags
        self.regressor_tags = regressor_tags
        self.array_api_support = array_api_support
        self.no_validation = no_validation
        self.non_deterministic = non_deterministic
        self.requires_fit = requires_fit
        self._skip_test = _skip_test
        self.input_tags = input_tags if input_tags is not None else InputTags()

def default_tags(estimator) -> Tags:
    """Tags of a plain estimator: type from the *Mixin classes (the
    _estimator_type class attribute), transformer tags if it transforms."""
    et: Optional[str] = None
    if hasattr(estimator, "_estimator_type"):
        et = str(estimator._estimator_type)
    has_fit = hasattr(estimator, "fit")
    tt = TargetTags(required=False)
    ct: Optional[ClassifierTags] = None
    rt: Optional[RegressorTags] = None
    if et is not None:
        if et == "classifier":
            tt = TargetTags(required=True)
            ct = ClassifierTags()
        elif et == "regressor":
            tt = TargetTags(required=True)
            rt = RegressorTags()
    xt: Optional[TransformerTags] = None
    if hasattr(estimator, "transform") or hasattr(estimator, "fit_transform"):
        xt = TransformerTags()
    return Tags(et, tt, xt, ct, rt, requires_fit=has_fit)

def get_tags(estimator) -> Tags:
    if hasattr(estimator, "__sklearn_tags__"):
        return estimator.__sklearn_tags__()
    return default_tags(estimator)
