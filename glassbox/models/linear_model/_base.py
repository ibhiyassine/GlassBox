from abc import abstractmethod
from typing import Any, Self

import numpy as np

from glassbox.models._base import BaseModel
from glassbox.models.linear_model._enums import LearningSchedule


class BaseLinearModel(BaseModel):
    """
    Abstract base class for linear models trained with gradient-based optimization.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Initial learning rate used by the optimizer.
    max_epochs : int, default=1000
        Maximum number of optimization epochs.
    tol : float, default=1e-6
        Convergence tolerance used by stopping criteria.
    schedule : LearningSchedule, default=LearningSchedule.CONSTANT
        Strategy used to update the learning rate across epochs.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        tol: float = 1e-6,
        schedule: LearningSchedule = LearningSchedule.CONSTANT,
    ) -> None:
        """
        Initialize shared linear-model hyperparameters and learned coefficients.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Initial learning rate used by the optimizer.
        max_epochs : int, default=1000
            Maximum number of optimization epochs.
        tol : float, default=1e-6
            Convergence tolerance used by stopping criteria.
        schedule : LearningSchedule, default=LearningSchedule.CONSTANT
            Strategy used to update the learning rate across epochs.
        """
        if learning_rate <= 0:
            raise ValueError("learning_rate must be strictly positive")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be strictly positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tol = tol
        self.schedule = schedule
        self.weights: np.ndarray = np.array([])
        self.bias: float = 0.0

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fit the linear model to training data.

        Parameters
        ----------
        X : np.ndarray
            Training feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Training target vector of shape (n_samples,).

        Returns
        -------
        Self
            The fitted model instance.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Predict target values for input samples.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        **kwargs : Any
            Additional keyword arguments for prediction.

        Returns
        -------
        np.ndarray
            Predicted values of shape (n_samples,).
        """
        raise NotImplementedError

    def _update_learning_rate(self, epoch: int) -> float:
        """
        Compute the learning rate to use at a specific epoch.

        Parameters
        ----------
        epoch : int
            Current optimization epoch.

        Returns
        -------
        float
            Learning rate value for the provided epoch.
        """
        if epoch < 0:
            raise ValueError("epoch must be non-negative")

        if self.schedule == LearningSchedule.CONSTANT:
            return self._calc_constant()

        if self.schedule == LearningSchedule.TIME_DECAY:
            return self._calc_time_decay(epoch)

        if self.schedule == LearningSchedule.EXPONENTIAL:
            return self._calc_exponential(epoch)

        raise ValueError("Unknown learning schedule")

    def _calc_constant(self) -> float:
        """
        Compute a constant learning rate.

        Returns
        -------
        float
            Constant learning rate value.
        """
        return float(self.learning_rate)

    def _calc_time_decay(self, epoch: int) -> float:
        """
        Compute a time-decayed learning rate.

        Parameters
        ----------
        epoch : int
            Current optimization epoch.

        Returns
        -------
        float
            Time-decayed learning rate value.
        """
        return float(self.learning_rate / (1.0 + epoch))

    def _calc_exponential(self, epoch: int) -> float:
        """
        Compute an exponentially decayed learning rate.

        Parameters
        ----------
        epoch : int
            Current optimization epoch.

        Returns
        -------
        float
            Exponentially decayed learning rate value.
        """
        exponent = -float(epoch) / float(self.max_epochs)
        return float(self.learning_rate * np.exp(exponent))
