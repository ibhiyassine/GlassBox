from typing import Self

import numpy as np

from glassbox.cleaner._base import BaseTransformer


class StandardScaler(BaseTransformer):
    """
    Standardizes features by removing the mean and scaling to unit variance.
    """

    __slots__ = ["_mean", "_std"]

    def __init__(self):
        self._mean: np.ndarray = np.array([])
        self._std: np.ndarray = np.array([])

    def fit(self, X: np.ndarray) -> Self:
        """
        Compute the mean and standard deviation to be used for later scaling.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        Self
            Fitted scaler instance.
        """
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Perform standardization by centering and scaling.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed array properly scaled.
        """
        raise NotImplementedError


class MinMaxScaler(BaseTransformer):
    """
    Transforms features by scaling each feature to a given range.
    """

    __slots__ = ["_min", "_max"]

    def __init__(self):
        self._min: np.ndarray = np.array([])
        self._max: np.ndarray = np.array([])

    def fit(self, X: np.ndarray) -> Self:
        """
        Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        Self
            Fitted scaler instance.
        """
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features of X according to feature range.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed array properly scaled.
        """
        raise NotImplementedError
