from ._base import FeatureSelector

import numpy as np
import math

from sklearn.base import clone
from tqdm import tqdm


class RFE(FeatureSelector):
    def __init__(self, estimator, n_features_to_select: int, step: int = 1):
        super().__init__()
        self.estimator = clone(estimator)
        self.n_features_to_select = n_features_to_select
        self.step = step

    def fit(self, X, y, **kwargs):
        n_features = X.shape[1]
        self.selected = np.ones(n_features, dtype=bool)
        steps_count = math.ceil((n_features - self.n_features_to_select) / self.step)
        for _ in tqdm(range(steps_count)):
            self.fit_estimator(X, y)
            importance = np.abs(np.ravel(self.estimator.coef_.toarray()))
            ranks = np.argsort(importance)
            threshold = min(self.step, np.sum(self.selected) - self.n_features_to_select)
            self.selected[np.arange(n_features)[self.selected][ranks][:threshold]] = False
        self.fit_estimator(X, y)
        return self

    def fit_estimator(self, X, y):
        n_features = X.shape[1]
        features = np.arange(n_features)[self.selected]
        self.estimator.fit(X[:, features], y)
