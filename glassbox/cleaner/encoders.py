from typing import Dict, List, Self

import numpy as np

from glassbox.cleaner._base import BaseTransformer


class OneHotEncoder(BaseTransformer):
    """
    Encode categorical features as a one-hot numeric array.
    """

    __slots__ = ["_categories"]

    def __init__(self):
        self._categories: Dict[int, List[str]] = {}

    def fit(self, X: np.ndarray) -> Self:
        """
        Learn the categorical levels for encoding.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        Self
            Fitted encoder instance.
        """
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the dataset into a one-hot encoded representation.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed array properly encoded.
        """
        raise NotImplementedError


class LabelEncoder(BaseTransformer):
    """
    Encode target labels with value between 0 and n_classes-1.
    """

    __slots__ = ["_mapping"]

    def __init__(self):
        self._mapping: Dict[str, int] = {}

    def fit(self, X: np.ndarray) -> Self:
        """
        Learn the vocabulary of the labels.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        Self
            Fitted encoder instance.
        """
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform labels to normalized encoding.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed array properly encoded.
        """
        raise NotImplementedError
