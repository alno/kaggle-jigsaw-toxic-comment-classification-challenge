import numpy as np
import pandas as pd

import re

from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import resample


class MultiProba(MultiOutputClassifier):

    def predict(self, X):
        return self.predict_proba(X)

    def predict_proba(self, X):
        return np.array(super().predict_proba(X))[:, :, 1].T


class SimpleAverage:

    def fit(self, X, y):
        assert X.shape[1] % 6 == 0
        self.n_groups = X.shape[1] // 6
        return self

    def predict(self, X):
        return sum(X.iloc[:, i*6:i*6 + 6].values for i in range(self.n_groups)) / self.n_groups


class WeightedAverage:

    def __init__(self, weights):
        self.weights = weights

    def fit(self, X, y):
        assert X.shape[1] == len(self.weights) * 6
        return self

    def predict(self, X):
        return sum(X.iloc[:, i*6:i*6 + 6].values * w for i, w in enumerate(self.weights))


class OnExtendedData:

    def __init__(self, model, n_samples=50000, max_len=None, decay=1):
        self.model = model
        self.n_samples = n_samples
        self.max_len = max_len
        self.decay = decay

    def fit_eval(self, train_X, train_y, val_X, val_y):
        new_train_X, new_train_y = self._extend_train_data(train_X, train_y)
        return self.model.fit_eval(new_train_X, new_train_y, val_X, val_y)

    def fit(self, train_X, train_y):
        self.model.fit(*self._extend_train_data(train_X, train_y))
        return self

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    def _extend_train_data(self, train_X, train_y):
        assert all(d == np.object for d in train_X.dtypes.values)

        cand_X = train_X
        cand_y = train_y

        if self.max_len is not None:
            selected = (cand_X.applymap(lambda t: len(re.split('\W+', t))).max(axis=1) < self.max_len).values
            cand_X = cand_X.loc[selected]
            cand_y = cand_y.loc[selected]

        print("Selected %d candidates" % cand_X.shape[0])

        left_X, left_y = resample(cand_X, cand_y, n_samples=self.n_samples)
        right_X, right_y = resample(cand_X, cand_y, n_samples=self.n_samples)

        generated_X = left_X.reset_index(drop=True) + ' ' + right_X.reset_index(drop=True)
        generated_y = ((left_y.reset_index(drop=True) + right_y.reset_index(drop=True)) * self.decay).clip(upper=1)

        pd.concat((generated_X, generated_y), axis=1).to_csv('generated.csv')

        return pd.concat((train_X, generated_X)), pd.concat((train_y, generated_y))
