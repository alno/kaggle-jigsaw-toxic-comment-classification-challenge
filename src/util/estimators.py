import numpy as np
import pandas as pd

import re

from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils import resample

from copy import deepcopy


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

    def __init__(self, weights, renorm=False):
        self.weights = np.asarray(weights) / sum(weights) if renorm else weights

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

        return pd.concat((train_X, generated_X)), pd.concat((train_y, generated_y))


class Pipeline:

    def __init__(self, *steps):
        self.steps = steps

    def fit_eval(self, train_X, train_y, eval_X, eval_y):
        for step in self.steps[:-1]:
            if hasattr(step, 'fit_transform'):
                train_X = step.fit_transform(train_X)
            else:
                train_X = step.transform(train_X)

            eval_X = step.transform(eval_X)

        return self.steps[-1].fit_eval(train_X, train_y, eval_X, eval_y)

    def predict(self, X):
        for step in self.steps[:-1]:
            X = step.transform(X)

        return self.steps[-1].predict(X)


class Bagged:

    def __init__(self, n, model, sample_size=1.0, sample_replace=True):
        self.n = n
        self.sample_size = sample_size
        self.sample_replace = sample_replace
        self.model = model

    def fit_eval(self, train_X, train_y, eval_X, eval_y):
        self.fitted_models = []
        for i in range(self.n):
            bag_train_X, bag_train_y = resample(train_X, train_y, n_samples=int(self.sample_size * len(train_X)), replace=self.sample_replace)
            model = deepcopy(self.model)
            model.fit_eval(bag_train_X, bag_train_y, eval_X, eval_y)
            self.fitted_models.append(model)

    def predict(self, X):
        return sum(m.predict(X) for m in self.fitted_models) / len(self.fitted_models)
