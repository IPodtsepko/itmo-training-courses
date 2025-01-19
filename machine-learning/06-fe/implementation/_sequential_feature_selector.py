from ._base import FeatureSelector

import numpy as np

from concurrent.futures import ThreadPoolExecutor
from sklearn.base import clone
from tqdm import tqdm


class SFS(FeatureSelector):
    def __init__(self, estimator, n_features_to_select: int, cv: int = 5):
        super().__init__()
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.cv = cv

    def fit(self, X_train, y_train, **kwargs):
        X_test = kwargs["X_test"]
        y_test = kwargs["y_test"]
        n_features = X_train.shape[1]
        self.selected = np.zeros(shape=n_features, dtype=bool)
        n_iterations = self.n_features_to_select
        for _ in tqdm(range(n_iterations)):
            new_feature_idx = self._get_best_new_feature_score(
                X_train, y_train, X_test, y_test
            )
            self.selected[new_feature_idx] = True
        return self

    def _get_best_new_feature_score(self, X_train, y_train, X_test, y_test):
        candidate_feature_indices = np.flatnonzero(~self.selected)
        scores = {}
        executor = ThreadPoolExecutor()
        for idx, score in executor.map(
                lambda feature_idx: self._get_feature_score(
                    feature_idx, X_train, y_train, X_test, y_test
                ),
                candidate_feature_indices,
        ):
            scores[idx] = score
        new_feature_idx = max(scores, key=lambda feature_idx: scores[feature_idx])
        return new_feature_idx

    def _get_feature_score(self, feature_idx, X_train, y_train, X_test, y_test):
        estimator = clone(self.estimator)
        candidate_mask = self.selected.copy()
        candidate_mask[feature_idx] = True
        estimator.fit(X_train[:, candidate_mask], y_train)
        return feature_idx, estimator.score(X_test[:, candidate_mask], y_test)
