import random
import re

import numpy as np


class RandomCrop:

    def __init__(self, min_len=1.0, max_len=1.0):
        self.min_len = min_len
        self.max_len = max_len

    def transform(self, X):
        X = X.copy()
        X['comment_text'] = X['comment_text'].map(self._transform_text)
        return X

    def _transform_text(self, text):
        words = text.split()

        min_len = self.min_len if isinstance(self.min_len, int) else int(np.ceil(self.min_len * len(words)))
        max_len = self.max_len if isinstance(self.max_len, int) else int(np.ceil(self.max_len * len(words)))

        max_len = min(max_len, len(words))
        min_len = min(min_len, max_len)

        length = random.randint(min_len, max_len)
        offset = random.randint(0, len(words) - length)

        return ' '.join(words[offset:offset+length])


class RandomTranslation:

    def __init__(self, prob=0.2):
        self.prob = prob
        self.langs = ['de', 'fr', 'es']

    def transform(self, X):
        replace = np.random.rand(len(X))
        langs = np.random.choice(self.langs, size=len(X))

        res = X.copy()
        for i, r, lang in zip(X.index, replace, langs):
            if r < self.prob:
                res.loc[i, 'comment_text'] = X.loc[i, 'comment_text__%s' % lang]
        return res


class RandomConcat:

    def __init__(self, prob=0.1, max_len=30, orig_weight=0.9, new_weight=0.7):
        self.prob = prob
        self.max_len = max_len
        self.orig_weight = orig_weight
        self.new_weight = new_weight

    def fit(self, X, y):
        if self.max_len is not None:
            selected = (X['comment_text'].map(lambda t: len(re.split('\W+', t))) < self.max_len).values
            self.cand_X = X.loc[selected]
            self.cand_y = y.loc[selected]
        else:
            self.cand_X = X
            self.cand_y = y

    def transform(self, X, y):
        X = X.copy()
        y = y.copy().astype(np.float32)

        for idx, r in zip(X.index, np.random.rand(len(X))):
            if r >= self.prob:
                continue
            cand_idx = np.random.choice(self.cand_X.index)

            if np.random.rand() > 0.5:
                X.loc[idx, 'comment_text'] = '%s %s' % (X.loc[idx, 'comment_text'], self.cand_X.loc[cand_idx, 'comment_text'])
            else:
                X.loc[idx, 'comment_text'] = '%s %s' % (self.cand_X.loc[cand_idx, 'comment_text'], X.loc[idx, 'comment_text'])

            y.loc[idx] = (y.loc[idx] * self.orig_weight + self.cand_y.loc[cand_idx] * self.new_weight).clip(lower=0.0, upper=1.0)

        return X, y
