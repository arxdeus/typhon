# Ported from braindecode 2.0.0 (BSD-3-Clause)
#
# Native Typhon implementation of braindecode. See docs/libraries/braindecode.md.

__version__: Literal[str] = "2.0.0"

import datasets
import preprocessing
import augmentation
import training
from .classifier import EEGClassifier, EEGRegressor
