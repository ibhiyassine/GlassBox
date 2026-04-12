from enum import Enum
from typing import Dict, Self, Union

import numpy as np

from glassbox.cleaner._base import BaseTransformer


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

    __slots__ = ["_strategy", "_fill_values"]

    def __init__(self, strategy: ImputationStrategy = ImputationStrategy.MEAN):
        """
        Parameters
        ----------
        strategy : ImputationStrategy, default=ImputationStrategy.MEAN
            The strategy used for missing value imputation.
        """
        self._strategy = strategy
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
        raise NotImplementedError

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
        raise NotImplementedError
