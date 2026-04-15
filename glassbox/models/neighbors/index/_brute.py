import numpy as np

from .._enums import DistanceMetric
from ._base import BaseIndex


class BruteForceIndex(BaseIndex):
    __slots__ = ("X_train",)

    def __init__(self, metric: DistanceMetric) -> None:
        """
        Initialize the BruteForceIndex.

        Parameters
        ----------
        metric : DistanceMetric
            The distance metric to use for querying.
        """
        super().__init__(metric)
        self.X_train: np.ndarray | None = None

    def build(self, X: np.ndarray) -> None:
        """
        Build the brute-force index by storing the training data.

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features).
        """
        self.X_train = np.asarray(X, dtype=float)

    def query(self, x: np.ndarray, k: int) -> np.ndarray:
        """
        Query the index for the k nearest neighbors using exhaustive search.

        Parameters
        ----------
        x : np.ndarray
            Query point(s), shape (n_features,) or (n_queries, n_features).
        k : int
            Number of nearest neighbors to retrieve.

        Returns
        -------
        np.ndarray
            Indices of the k nearest neighbors.
        """
        if self.X_train is None:
            raise ValueError("Index is not built yet")

        x_arr = np.asarray(x, dtype=float)
        single_query = x_arr.ndim == 1
        if single_query:
            x_arr = x_arr.reshape(1, -1)

        n_queries = x_arr.shape[0]
        nearest_idx = np.empty((n_queries, k), dtype=int)

        for i in range(n_queries):
            diff = self.X_train - x_arr[i]
            if self.metric == DistanceMetric.EUCLIDEAN:
                dist = np.sqrt(np.sum(diff**2, axis=1))
            elif self.metric == DistanceMetric.MANHATTAN:
                dist = np.sum(np.abs(diff), axis=1)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")

            nearest_idx[i] = np.argsort(dist)[:k]

        if single_query:
            return nearest_idx[0]
        return nearest_idx
