# Authors: Borja Ayerdi [ayerdi.borja -at- gmail -dot- com]
# Copyright(c) 2016
# License: Simple BSD

"""
This module implements Rotation Forest

References
----------
.. [1] Juan J. Rodriguez, et al, "Rotation Forest: A NewClassifier
          Ensemble Method", IEEE Transactions on Pattern Analysis and
          Machine Intelligence, 2006.

"""

import random
import numpy as np

from copy import deepcopy
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier


__all__ = ["RotationForest"]


class RotationForest(object):
    """
    Rotation Forest
    """

    def __init__(self, n, model):
        self._n = n
        self._model = model
        self._fitted_models = []
        self._inforotar = []
        self._std = []
        self._med = []
        self._noise = []

    @staticmethod
    def _apply_pca(data, n_comps=1):
        """
        Applies PCA to the data

        :param data: ndarray
        A MxN array with M samples of dimension N

        :return: sklearn.decomposition.PCA

        """
        # PCA
        pca = PCA(n_components=n_comps, whiten=False, copy=True)
        pca.fit(data)

        return pca

    def fit(self, X, y):
        """
        Fit the model using X, y as training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
        Target values (class labels in classification, real numbers in
        regression)

        Returns
        -------
        self : object

        Returns an instance of self.
        """

        if hasattr(X, 'values'):
            X = X.values

        n_samps, NF = X.shape

        # Compute mean, std and noise for z-score
        self._std = np.std(X, axis=0)
        self._med = np.mean(X, axis=0)
        self._noise = [random.uniform(-0.000005, 0.000005) for p in range(0, X.shape[1])]

        # Apply Z-score
        Xz = (X-self._med)/(self._std+self._noise)

        for i in range(self._n):
            # For each classifier in the ensemble
            # Given:
            # X: the objects in the training data set (an N x n matrix)
            # Y: the labels of the training set (an N x 1 matrix)
            # K: the number of subsets
            # NF: the number of total features
            # {w1,w2,.., wc}: the set of class labels
            #L

            # Prepare the rotaton matrix R:
            # Split F (the feature set) into K subsets Fij (for j=1,..,K/4)
            # K is a random value between 1 and NF/4.
            # We want at least 1 feature for each subset.
            K = int(round(1 + NF/4*random.random()))

            FK = np.zeros((K, NF))
            for j in range(K):
                numSelecFeatures = int(1 + round((NF-1)*random.random()))
                rp = np.random.permutation(NF)
                v = [rp[k] for k in range(0, numSelecFeatures)]
                FK[j, v] = 1

            R = np.zeros((NF, NF))
            for l in range(K):
                # Let Xzij be the data set X for the features in Fij
                pos = np.nonzero(FK[l, :])[0]

                vpos = [pos[m] for m in range(0, len(pos))]

                Xzij = Xz[:, vpos]
                pca = self._apply_pca(Xzij, len(pos))

                for indI in range(0, len(pca.components_)):
                    for indJ in range(0, len(pca.components_)):
                        R[pos[indI], pos[indJ]] = pca.components_[indI, indJ]

            self._inforotar.append(R)
            Xrot = Xz.dot(R)

            model = deepcopy(self._model)
            model.fit(Xrot, y)
            self._fitted_models.append(model)

        return self

    def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """

        if hasattr(X, 'values'):
            X = X.values

        # Z-score
        Xz = (X-self._med)/(self._std+self._noise)

        return sum(self._fitted_models[i].predict(Xz.dot(self._inforotar[i])) for i in range(self._n)) / self._n
