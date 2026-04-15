from typing import Any, Self

import numpy as np

from glassbox.models.linear_model._base import BaseLinearModel


class LogisticRegression(BaseLinearModel):
    """
    Logistic regression model for binary classification.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fit the logistic regression model to training data.

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
        y_arr = np.asarray(y)

        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y_arr.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        if X_arr.shape[0] == 0:
            raise ValueError("X and y cannot be empty")

        classes = np.unique(y_arr)
        if not np.all(np.isin(classes, np.array([0, 1]))):
            raise ValueError("y must contain binary labels encoded as 0 and 1")

        y_bin = y_arr.astype(float)

        n_samples, n_features = X_arr.shape
        self.weights = np.zeros(n_features, dtype=float)
        self.bias = 0.0

        previous_loss = np.inf
        for epoch in range(self.max_epochs):
            learning_rate = self._update_learning_rate(epoch)

            logits = X_arr @ self.weights + self.bias
            probabilities = self._sigmoid(logits)
            errors = probabilities - y_bin

            gradient_w = (X_arr.T @ errors) / n_samples
            gradient_b = float(np.mean(errors))

            self.weights -= learning_rate * gradient_w
            self.bias -= learning_rate * gradient_b

            probabilities_clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
            current_loss = float(
                -np.mean(
                    y_bin * np.log(probabilities_clipped)
                    + (1.0 - y_bin) * np.log(1.0 - probabilities_clipped)
                )
            )
            if abs(previous_loss - current_loss) <= self.tol:
                break
            previous_loss = current_loss

        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Predict class labels for input samples.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).
        **kwargs : Any
            Additional keyword arguments for prediction.

        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,).
        """
        threshold = kwargs.get("threshold", 0.5)
        if not isinstance(threshold, (int, float)):
            raise ValueError("threshold must be a numeric value")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold must be in the [0.0, 1.0] interval")

        probabilities = self.predict_proba(X)
        return (probabilities >= float(threshold)).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input samples.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted probabilities of shape (n_samples,).
        """
        if self.weights.size == 0:
            raise RuntimeError("Model is not fitted yet.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError("X must be a 2D array")
        if X_arr.shape[1] != self.weights.shape[0]:
            raise ValueError("X must have the same number of features used during fit")

        logits = X_arr @ self.weights + self.bias
        return self._sigmoid(logits)

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid function to linear logits.

        Parameters
        ----------
        z : np.ndarray
            Input logits of shape (n_samples,).

        Returns
        -------
        np.ndarray
            Sigmoid-transformed probabilities of shape (n_samples,).
        """
        positive_mask = z >= 0
        negative_mask = ~positive_mask

        result = np.empty_like(z, dtype=float)
        result[positive_mask] = 1.0 / (1.0 + np.exp(-z[positive_mask]))

        exp_z = np.exp(z[negative_mask])
        result[negative_mask] = exp_z / (1.0 + exp_z)
        return result
