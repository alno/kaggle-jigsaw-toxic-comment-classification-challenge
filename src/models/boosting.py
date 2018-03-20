import numpy as np
import lightgbm as lgb
import xgboost as xgb

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class LgbOnKBestModel:

    default_params = {
        'learning_rate': 0.2,
        'application': 'binary',
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'auc',
        'data_random_seed': 2,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.6,
        'nthread': 4,
        'lambda_l1': 1,
        'lambda_l2': 1
    }

    default_rounds = {
        'toxic': 140,
        'severe_toxic': 50,
        'obscene': 80,
        'threat': 80,
        'insult': 70,
        'identity_hate': 80
    }

    def __init__(self, params={}, rounds={}, feature_threshold=0.2):
        self.params = {**self.default_params, **params}
        self.rounds = {**self.default_rounds, **rounds}
        self.feature_threshold = feature_threshold

    def fit(self, train_X, train_y):
        self.label_columns = list(train_y.columns)
        self.label_transformers = {}
        self.label_models = {}

        for label in self.label_columns:
            label_y = train_y[label]

            self.label_transformers[label] = SelectFromModel(LogisticRegression(solver='sag'), threshold=self.feature_threshold)
            label_train_X = self.label_transformers[label].fit_transform(train_X, label_y)

            label_train_X, label_valid_X, label_train_y, label_valid_y = train_test_split(label_train_X, label_y, test_size=0.05, random_state=144)

            dtrain = lgb.Dataset(label_train_X, label=label_train_y)
            dvalid = lgb.Dataset(label_valid_X, label=label_valid_y)

            self.label_models[label] = lgb.train(self.params, train_set=dtrain, num_boost_round=self.rounds[label], valid_sets=[dtrain, dvalid], verbose_eval=10)

        return self

    def predict(self, X):
        res = np.zeros((X.shape[0], len(self.label_columns)))

        for li, label in enumerate(self.label_columns):
            label_X = self.label_transformers[label].transform(X)
            res[:, li] = self.label_models[label].predict(label_X)

        return res


class LgbModel:

    default_params = {
        'learning_rate': 0.2,
        'application': 'binary',
        'num_leaves': 31,
        'verbosity': -1,
        'metric': 'auc',
        'bagging_fraction': 0.8,
        'feature_fraction': 0.6,
        'nthread': 4,
        'lambda_l1': 1,
        'lambda_l2': 1
    }

    default_rounds = {
        'toxic': 140,
        'severe_toxic': 50,
        'obscene': 80,
        'threat': 80,
        'insult': 70,
        'identity_hate': 80
    }

    def __init__(self, params={}, rounds={}, verbose_eval=10):
        self.params = {**self.default_params, **params}
        self.rounds = {**self.default_rounds, **rounds}
        self.verbose_eval = verbose_eval

    def fit_eval(self, train_X, train_y, eval_X, eval_y):
        self.label_columns = list(train_y.columns)
        self.label_models = {}

        for label in self.label_columns:
            print("Training model for %s..." % label)
            dtrain = lgb.Dataset(train_X, label=train_y[label])
            dvalid = lgb.Dataset(eval_X, label=eval_y[label])

            self.label_models[label] = lgb.train(self.params, train_set=dtrain, num_boost_round=self.rounds[label], valid_sets=[dtrain, dvalid], verbose_eval=self.verbose_eval)

        return self

    def predict(self, X):
        res = np.zeros((X.shape[0], len(self.label_columns)))

        for li, label in enumerate(self.label_columns):
            res[:, li] = self.label_models[label].predict(X)

        return res


class XgbModel:

    default_params = dict(
        objective='binary:logistic',
        eta=0.01,
        max_depth=4,
        subsample=0.9,
        tree_method='exact',
        eval_metric='auc',
        min_child_weight=1,
        silent=1,
    )

    default_rounds = {
        'toxic': 140,
        'severe_toxic': 50,
        'obscene': 80,
        'threat': 80,
        'insult': 70,
        'identity_hate': 80
    }

    def __init__(self, params={}, rounds={}, verbose_eval=10):
        self.params = {**self.default_params, **params}
        self.rounds = {**self.default_rounds, **rounds}
        self.verbose_eval = verbose_eval

    def fit_eval(self, train_X, train_y, eval_X, eval_y):
        self.label_columns = list(train_y.columns)
        self.label_models = {}

        for label in self.label_columns:
            print("Training model for %s..." % label)
            dtrain = xgb.DMatrix(train_X, label=train_y[label])
            dvalid = xgb.DMatrix(eval_X, label=eval_y[label])

            self.label_models[label] = xgb.train(self.params, dtrain, self.rounds[label], [(dtrain, 'train'), (dvalid, 'valid')], verbose_eval=self.verbose_eval)

        return self

    def predict(self, X):
        res = np.zeros((X.shape[0], len(self.label_columns)))

        for li, label in enumerate(self.label_columns):
            res[:, li] = self.label_models[label].predict(X)

        return res
