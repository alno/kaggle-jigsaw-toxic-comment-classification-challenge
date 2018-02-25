import numpy as np

from sklearn.multioutput import MultiOutputClassifier


class MultiProba(MultiOutputClassifier):

    def predict(self, X):
        return self.predict_proba(X)

    def predict_proba(self, X):
        return np.array(super().predict_proba(X))[:, :, 1].T
