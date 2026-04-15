from typing import Self

import numpy as np

from glassbox.cleaner._base import BaseTransformer
from glassbox.core.math import calc_mean, calc_std


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
        n_features = X.shape[1]
        means = []
        stds = []

        for col_idx in range(n_features):
            col = X[:, col_idx]
            if np.issubdtype(col.dtype, np.number):
                col_clean = col[~np.isnan(col)]
            else:
                col_clean = np.array(
                    [
                        x
                        for x in col
                        if x is not None and not (isinstance(x, float) and np.isnan(x))
                    ]
                )

            if len(col_clean) == 0:
                means.append(0.0)
                stds.append(1.0)
            else:
                means.append(calc_mean(col_clean))
                s = calc_std(col_clean)
                stds.append(s if s > 0 else 1.0)

        self._mean = np.array(means)
        self._std = np.array(stds)
        return self

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
        X_out = X.copy()

        # If input has NaNs, they remain NaNs. We must be careful not to subtract mean from None
        if np.issubdtype(X_out.dtype, np.number):
            # broadcast subtraction
            X_out = (X_out - self._mean) / self._std
        else:
            for col_idx in range(X_out.shape[1]):
                col = X_out[:, col_idx]
                mask = np.array([v is not None for v in col])
                X_out[mask, col_idx] = (col[mask] - self._mean[col_idx]) / self._std[
                    col_idx
                ]

        return X_out


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
        n_features = X.shape[1]
        mins = []
        maxs = []

        for col_idx in range(n_features):
            col = X[:, col_idx]
            if np.issubdtype(col.dtype, np.number):
                col_clean = col[~np.isnan(col)]
            else:
                col_clean = np.array(
                    [
                        x
                        for x in col
                        if x is not None and not (isinstance(x, float) and np.isnan(x))
                    ]
                )

            if len(col_clean) == 0:
                mins.append(0.0)
                maxs.append(1.0)
            else:
                mins.append(float(np.min(col_clean)))
                maxs.append(float(np.max(col_clean)))

        self._min = np.array(mins)
        self._max = np.array(maxs)
        return self

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
        X_out = X.copy()
        range_vals = self._max - self._min
        range_vals[range_vals == 0.0] = 1.0  # avoid division by zero

        if np.issubdtype(X_out.dtype, np.number):
            X_out = (X_out - self._min) / range_vals
        else:
            for col_idx in range(X_out.shape[1]):
                col = X_out[:, col_idx]
                mask = np.array([v is not None for v in col])
                X_out[mask, col_idx] = (col[mask] - self._min[col_idx]) / range_vals[
                    col_idx
                ]

        return X_out
