from typing import Dict, Self

import numpy as np

from glassbox.cleaner._base import BaseTransformer


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
        raise NotImplementedError

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
        raise NotImplementedError
