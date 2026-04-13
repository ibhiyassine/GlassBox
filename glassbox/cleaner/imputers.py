from enum import Enum
from typing import Dict, Self, Union

import numpy as np

from glassbox.cleaner._base import BaseTransformer
from glassbox.core.math import calc_mean, calc_median, calc_mode


class ImputationStrategy(Enum):
    """
    Strategies available for imputing missing values.
    """

    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    MODE = "MODE"
    CONSTANT = "CONSTANT"


class SimpleImputer(BaseTransformer):
    """
    Replaces missing values using a specified statistical strategy.

    Notes
    -----
    This imputer supports basic strategies like mean, median, mode, or a constant value.
    """

    __slots__ = ["_strategy", "_constant_value", "_fill_values"]

    def __init__(self, strategy: ImputationStrategy = ImputationStrategy.MEAN, constant_value: Union[float, str, None] = 0.0):
        """
        Parameters
        ----------
        strategy : ImputationStrategy, default=ImputationStrategy.MEAN
            The strategy used for missing value imputation.
        constant_value: Union[float, str, None], default=0.0
            The value to use when strategy is CONSTANT.
        """
        self._strategy = strategy
        self._constant_value = constant_value
        self._fill_values: Dict[str, Union[float, str]] = {}

    def fit(self, X: np.ndarray) -> Self:
        """
        Learn the imputation values from the training data.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        Self
            Fitted imputer instance.
        """
        n_features = X.shape[1]
        for col_idx in range(n_features):
            col = X[:, col_idx]

            if np.issubdtype(col.dtype, np.number):
                col_clean = col[~np.isnan(col)]
            else:
                col_clean = col[np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in col])]

            if len(col_clean) == 0:
                self._fill_values[str(col_idx)] = 0.0
                continue

            if self._strategy == ImputationStrategy.MEAN:
                self._fill_values[str(col_idx)] = calc_mean(col_clean)
            elif self._strategy == ImputationStrategy.MEDIAN:
                self._fill_values[str(col_idx)] = calc_median(col_clean)
            elif self._strategy == ImputationStrategy.MODE:
                self._fill_values[str(col_idx)] = calc_mode(col_clean)
            elif self._strategy == ImputationStrategy.CONSTANT:
                self._fill_values[str(col_idx)] = self._constant_value
            else:
                raise ValueError(f"Unknown strategy: {self._strategy}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Impute missing values in the given dataset.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed array with missing values imputed.
        """
        X_out = X.copy()
        n_features = X.shape[1]

        for col_idx in range(n_features):
            col = X_out[:, col_idx]
            fill_val = self._fill_values.get(str(col_idx), 0.0)

            if np.issubdtype(col.dtype, np.number):
                mask = np.isnan(col)
                X_out[mask, col_idx] = fill_val
            else:
                mask = np.array([v is None or (isinstance(v, float) and np.isnan(v)) for v in col])
                X_out[mask, col_idx] = fill_val

        return X_out
