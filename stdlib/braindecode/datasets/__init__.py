# Ported from braindecode 2.0.0 (BSD-3-Clause)

from .base import BaseConcatDataset, RawDataset, BaseDataset, EEGWindowsDataset, WindowsDataset, Description
from .xy import create_from_X_y
from ._mne import create_from_mne_raw, create_from_mne_epochs


def _no_download(name: str):
    raise NotImplementedError(f"{name} needs network access and file readers (EDF/GDF/FIF); "
                              "it is not available in Typhon braindecode. Build datasets with "
                              "create_from_X_y / create_from_mne_raw instead.")


class MOABBDataset:
    def __init__(self, dataset_name: str = "", subject_ids = None, **kwargs):
        _no_download("MOABBDataset")


class TUH:
    def __init__(self, path: str = "", **kwargs):
        _no_download("TUH")


class TUHAbnormal:
    def __init__(self, path: str = "", **kwargs):
        _no_download("TUHAbnormal")


class SleepPhysionet:
    def __init__(self, subject_ids = None, **kwargs):
        _no_download("SleepPhysionet")


class BCICompetitionIVDataset4:
    def __init__(self, subject_ids = None):
        _no_download("BCICompetitionIVDataset4")
