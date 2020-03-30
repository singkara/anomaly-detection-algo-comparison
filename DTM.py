import numpy as np
import warnings
import numbers
from scipy.sparse import issparse

from sklearn.neighbors.base import NeighborsBase
from sklearn.neighbors.base import KNeighborsMixin
from sklearn.neighbors.base import UnsupervisedMixin
from sklearn.base import OutlierMixin
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator

from sklearn.utils.validation import check_is_fitted

from sklearn.utils import (
    check_random_state,
    check_array,
    gen_batches,
    get_chunk_n_rows,
)

__all__ = ["DTM"]

INTEGER_TYPES = (numbers.Integral, np.integer)


class DTM(NeighborsBase, KNeighborsMixin, UnsupervisedMixin,
          OutlierMixin):
    def __init__(self,
                 r = 2,
                 random_state=None,
                 n_neighbors=20,
                 algorithm='auto',
                 leaf_size=30,
                 metric='minkowski',
                 p=2,
                 metric_params=None,
                 contamination="legacy",
                 n_jobs=None,
                 verbose=0):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs)

        self.contamination = contamination
        self.r = r
        self.random_state = random_state

    def fit_predict(self,X,y=None):

        return self.fit(X).predict()

    def fit(self, X, y=None):

        if self.contamination == "legacy":
            warnings.warn('default contamination parameter 0.1 will change '
                          'in version 0.22 to "auto". This will change the '
                          'predict method behavior.',
                          FutureWarning)
            self._contamination = 0.1
        else:
            self._contamination = self.contamination

        if self._contamination != 'auto':
            if not(0. < self._contamination <= .5):
                raise ValueError("contamination must be in (0, 0.5], "
                                 "got: %f" % self._contamination)

        X = check_array(X, accept_sparse=['csc'])
        if issparse(X):
            # Pre-sort indices to avoid that each individual tree of the
            # ensemble sorts the indices.
            X.sort_indices()

        super().fit(X)

        n_samples = self._fit_X.shape[0]

        if self.n_neighbors > n_samples:
            warnings.warn("n_neighbors (%s) is greater than the "
                          "total number of samples (%s). n_neighbors "
                          "will be set to (n_samples - 1) for estimation."
                          % (self.n_neighbors, n_samples))
        self.n_neighbors_ = max(1, min(self.n_neighbors, n_samples - 1))

        self.negative_dtm = self.score_samples(None)

        if self._contamination == "auto":
            # inliers score around -1 (the higher, the less abnormal).
            self.offset_ = -1.5
        else:
            self.offset_ = np.percentile(self.negative_dtm,
                                         100. * self._contamination)

        return self

    def predict(self, X=None):
        return self.score_samples(X)

    def label(self, X=None):

        check_is_fitted(self, ["offset_", "negative_dtm",
                               "n_neighbors_", "_distances_fit_X_"])

        if X is not None:
            X = check_array(X, accept_sparse='csr')
            is_inlier = np.ones(X.shape[0], dtype=int)
            is_inlier[self.decision_function(X) < 0] = -1
        else:
            is_inlier = np.ones(self._fit_X.shape[0], dtype=int)
            is_inlier[self.negative_dtm < self.offset_] = -1

        return is_inlier

    def decision_function(self, X):

        return self.score_samples(X) - self.offset_

    def score_samples(self,X):

        if X is None:
            pass
        else:
            X = check_array(X, accept_sparse='csr')

        self._distances_fit_X_, neighbors_indices_X = (
            self.kneighbors(X, n_neighbors=self.n_neighbors_))

        r = self.r
        self.dtm = (np.mean(self._distances_fit_X_ ** r, axis=1)) ** (1. / r)

        # as bigger is better:
        return -self.dtm