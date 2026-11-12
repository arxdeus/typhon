# Ported from scikit-learn 1.9.1 (BSD-3-Clause)

# sklearn.ensemble (dense input, single output).

from ._forest import (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier,
                      ExtraTreesRegressor)
from ._gb import GradientBoostingClassifier, GradientBoostingRegressor
from ._weight_boosting import AdaBoostClassifier, AdaBoostRegressor
from ._bagging import BaggingClassifier, BaggingRegressor, IsolationForest
from ._voting import VotingClassifier, VotingRegressor
from ._hist_gradient_boosting.gradient_boosting import (HistGradientBoostingClassifier,
                                                        HistGradientBoostingRegressor)
