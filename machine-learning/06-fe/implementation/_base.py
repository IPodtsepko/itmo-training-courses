from abc import ABCMeta, abstractmethod


class FeatureSelector:
    __metaclass__ = ABCMeta

    def __init__(self):
        self.selected = None

    def get_selected(self):
        return self.selected

    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    def transform(self, X):
        return X[:, self.selected]

    def fit_transform(self, X, y, **kwargs):
        return self.fit(X, y, **kwargs).transform(X)
