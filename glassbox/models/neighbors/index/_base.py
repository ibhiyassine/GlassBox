from abc import ABC, abstractmethod

import numpy as np

from .._enums import DistanceMetric


class BaseIndex(ABC):
    __slots__ = ("metric",)

    def __init__(self, metric: DistanceMetric) -> None:
        """
        Initialize the abstract index class.

        Parameters
        ----------
        metric : DistanceMetric
            The distance metric to use.
        """
        self.metric: DistanceMetric = metric

    @abstractmethod
    def build(self, X: np.ndarray) -> None:
        """
        Build the index structure from the training data.

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features).
        """
        raise NotImplementedError

    @abstractmethod
    def query(self, x: np.ndarray, k: int) -> np.ndarray:
        """
        Query the index for the k nearest neighbors.

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
        raise NotImplementedError
