from typing import Any, Self

import numpy as np

from glassbox.core.math import calc_mode
from glassbox.models._base import BaseModel

from ._enums import DistanceMetric, SearchAlgorithm
from .index._base import BaseIndex
from .index._brute import BruteForceIndex
from .index._kdtree import KDTreeIndex


class BaseKNN(BaseModel):
    __slots__ = ("k", "metric", "algorithm", "index", "y_train")

    def __init__(
        self,
        k: int = 5,
        metric: DistanceMetric = DistanceMetric.EUCLIDEAN,
        algorithm: SearchAlgorithm = SearchAlgorithm.BRUTE_FORCE,
    ) -> None:
        """
        Initialize the BaseKNN estimator.

        Parameters
        ----------
        k : int, default=5
            Number of neighbors to use.
        metric : DistanceMetric, default=DistanceMetric.EUCLIDEAN
            Distance metric to compute distances.
        algorithm : SearchAlgorithm, default=SearchAlgorithm.BRUTE_FORCE
            Algorithm used to compute the nearest neighbors.
        """
        self.k: int = k
        self.metric: DistanceMetric = metric
        self.algorithm: SearchAlgorithm = algorithm
        self.index: BaseIndex | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fits the model to the training data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        Self
            The fitted model.
        """
        X_arr = np.asarray(X)
        self.y_train = np.asarray(y)

        if self.algorithm == SearchAlgorithm.BRUTE_FORCE:
            self.index = BruteForceIndex(metric=self.metric)
        elif self.algorithm == SearchAlgorithm.KD_TREE:
            self.index = KDTreeIndex(metric=self.metric)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        self.index.build(X_arr)
        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Predicts target values for the given data.

        Parameters
        ----------
        X : np.ndarray
            Data to predict on, of shape (n_samples, n_features) or (n_features,).
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        if self.index is None or self.y_train is None:
            raise ValueError("Model must be fitted before calling predict.")

        X_arr = np.asarray(X)
        single_query = X_arr.ndim == 1

        nearest_indices = self.index.query(X_arr, self.k)

        if single_query:
            nearest_y = self.y_train[nearest_indices].reshape(1, -1)
        else:
            nearest_y = self.y_train[nearest_indices]

        preds = self._aggregate(nearest_y)

        if single_query:
            return preds[0]
        return preds

    def _aggregate(self, nearest_y: np.ndarray) -> np.ndarray:
        """
        Aggregate the values of the nearest neighbors.

        Parameters
        ----------
        nearest_y : np.ndarray
            The target values of the nearest neighbors.

        Returns
        -------
        np.ndarray
            Aggregated predictions.
        """
        raise NotImplementedError


class KNeighborsClassifier(BaseKNN):
    __slots__ = ()

    def _aggregate(self, nearest_y: np.ndarray) -> np.ndarray:
        """
        Aggregate the values using majority voting for classification.

        Parameters
        ----------
        nearest_y : np.ndarray
            The target values of the nearest neighbors, shape (n_queries, k).

        Returns
        -------
        np.ndarray
            Predicted classes, shape (n_queries,).
        """
        n_queries = nearest_y.shape[0]
        preds = np.empty(n_queries, dtype=nearest_y.dtype)
        for i in range(n_queries):
            mode_val = calc_mode(nearest_y[i])
            preds[i] = mode_val
        return preds


class KNeighborsRegressor(BaseKNN):
    __slots__ = ()

    def _aggregate(self, nearest_y: np.ndarray) -> np.ndarray:
        """
        Aggregate the values using average for regression.

        Parameters
        ----------
        nearest_y : np.ndarray
            The target values of the nearest neighbors, shape (n_queries, k).

        Returns
        -------
        np.ndarray
            Predicted values, shape (n_queries,).
        """
        return np.mean(nearest_y, axis=1)
