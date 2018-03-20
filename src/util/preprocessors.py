from sklearn.base import BaseEstimator
import pandas as pd


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


class DropColumns(BaseEstimator):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return X.drop(self.columns, axis=1)


class SelectColumns(BaseEstimator):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return X[self.columns]


class AvgGroupsColumns(BaseEstimator):

    def __init__(self, groups, columns):
        self.groups = groups
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        res = pd.DataFrame(index=X.index)

        for group_name, group_content in self.groups:
            for col in self.columns:
                res["%s__%s" % (group_name, col)] = X[["%s__%s" % (g, col) for g in group_content]].mean(axis=1)

        return res


class Union:

    def __init__(self, *estimators):
        self.estimators = estimators

    def fit(self, X, y=None):
        for e in self.estimators:
            e.fit(X)
        return self

    def fit_transform(self, X, y=None):
        return pd.concat([e.fit_transform(X) for e in self.estimators], axis=1)

    def transform(self, X):
        return pd.concat([e.transform(X) for e in self.estimators], axis=1)
