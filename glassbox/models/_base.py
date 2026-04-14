from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np


class BaseModel(ABC):
    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Predicts target values for the given data.

        Parameters
        ----------
        X : np.ndarray
            Data to predict on, of shape (n_samples, n_features).
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Predicted target values.
        """
        raise NotImplementedError
