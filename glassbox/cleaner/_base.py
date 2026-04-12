from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class BaseTransformer(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> Self:
        """
        Calculates parameters needed for transformation.

        Parameters
        ----------
        X : np.ndarray
            Input array for learning

        Returns
        -------
        Self
            Updated state of the transformer
        """
        raise NotImplementedError

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the transformation to the dataset.

        Parameters
        ----------
        X : np.ndarray
            Input array to be transformed

        Returns
        -------
        np.ndarray
            transformed array
        """
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fits to the data, then transforms it.

        Parameters
        ----------
        X : np.ndarray
            Input array to be transformed

        Returns
        -------
        np.ndarray
            transformed array
        """
        return self.fit(X).transform(X)
