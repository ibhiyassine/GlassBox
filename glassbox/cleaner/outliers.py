from typing import Dict, Self

import numpy as np

from glassbox.cleaner._base import BaseTransformer
from glassbox.core.math import calc_iqr


class OutlierCapper(BaseTransformer):
    """
    Identifies and caps numerical outliers based on specified bounds.
    """

    __slots__ = ["_bounds"]

    def __init__(self):
        self._bounds: Dict[str, Dict[str, float]] = {}

    def fit(self, X: np.ndarray) -> Self:
        """
        Detect boundaries for outlier capping from the training data.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        Self
            Fitted outlier capper instance.
        """
        n_features = X.shape[1]
        for col_idx in range(n_features):
            col = X[:, col_idx]

            if np.issubdtype(col.dtype, np.number):
                col_clean = col[~np.isnan(col)]
            else:
                col_clean = np.array([x for x in col if x is not None and not (isinstance(x, float) and np.isnan(x))])

            if len(col_clean) == 0:
                self._bounds[str(col_idx)] = {"lower": 0.0, "upper": 1.0}
            else:
                lower, upper = calc_iqr(col_clean)
                self._bounds[str(col_idx)] = {
                    "lower": lower,
                    "upper": upper
                }
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Cap outliers in the input dataset.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed array with outliers capped.
        """
        X_out = X.copy()
        n_features = X.shape[1]
        for col_idx in range(n_features):
            bounds = self._bounds.get(str(col_idx), {"lower": 0.0, "upper": 1.0})
            lower = bounds["lower"]
            upper = bounds["upper"]

            col = X_out[:, col_idx]
            if np.issubdtype(col.dtype, np.number):
                mask_clean = ~np.isnan(col)
                # Cap the values only for non-NaN elements
                X_out[mask_clean, col_idx] = np.clip(col[mask_clean], lower, upper)
            else:
                mask_clean = np.array([v is not None for v in col])
                # Objects might fail if not numbers under the hood
                try:
                    X_out[mask_clean, col_idx] = np.clip(col[mask_clean].astype(float), lower, upper)
                except ValueError:
                    pass
        return X_out
