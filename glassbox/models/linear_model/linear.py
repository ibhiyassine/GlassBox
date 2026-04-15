from typing import Any, Self

import numpy as np

from glassbox.models.linear_model._base import BaseLinearModel


class LinearRegression(BaseLinearModel):
    """
    Linear regression model.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fit the linear regression model to training data.

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
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X_arr.shape[0] == 0:
            raise ValueError("X and y cannot be empty")

        n_samples, n_features = X_arr.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        previous_loss = np.inf
        for epoch in range(self.max_epochs):
            learning_rate = self._update_learning_rate(epoch)

            predictions = X_arr @ self.weights + self.bias
            errors = predictions - y_arr

            gradient_w = (2.0 / n_samples) * (X_arr.T @ errors)
            gradient_b = 2.0 * np.mean(errors)

            self.weights -= learning_rate * gradient_w
            self.bias -= learning_rate * gradient_b

            current_loss = float(np.mean(errors**2))
            if abs(previous_loss - current_loss) <= self.tol:
                break
            previous_loss = current_loss

        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Predict continuous target values for input samples.

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
        if self.weights.size == 0:
            raise RuntimeError("Model is not fitted yet.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X_arr.shape[1] != self.weights.shape[0]:
            raise ValueError("X must have the same number of features used during fit")

        return X_arr @ self.weights + self.bias
