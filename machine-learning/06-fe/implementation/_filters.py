from ._base import FeatureSelector

import numpy as np

from sklearn.feature_selection import chi2


class Chi2Filter(FeatureSelector):
    def __init__(self, n_features_to_select: int):
        super().__init__()
        self.n_features_to_select = n_features_to_select

    def fit(self, X, y, **kwargs):
        statistics, _ = chi2(X, y)
        scores = np.array(statistics)
        scores[np.isnan(scores)] = np.finfo(scores.dtype).min
        self.selected = np.zeros(scores.shape, dtype=bool)
        self.selected[np.argsort(scores, kind="mergesort")[-self.n_features_to_select:]] = 1
        return self
