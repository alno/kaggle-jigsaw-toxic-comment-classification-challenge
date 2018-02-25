from sklearn.base import BaseEstimator


class OnColumn(BaseEstimator):

    def __init__(self, column, transformer):
        self.column = column
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def fit_transform(self, X, y=None):
        return self.transformer.fit_transform(X[self.column], y)

    def transform(self, X):
        return self.transformer.transform(X[self.column])
