from typing import Any, Self

import numpy as np

from glassbox.models._base import BaseModel


class GaussianNB(BaseModel):
    """
    Gaussian Naive Bayes classifier.

    A probabilistic classifier based on Bayes' theorem with the assumption
    that features follow a Gaussian (normal) distribution within each class.

    Parameters
    ----------
    epsilon : float, default=1e-9
        Small constant to avoid division by zero in variance calculations.

    Attributes
    ----------
    epsilon : float
        Small constant to avoid division by zero.
    classes : np.ndarray
        Unique class labels, shape (n_classes,).
    class_priors : dict
        Prior probability for each class.
    class_means : dict
        Mean of each feature per class.
    class_variances : dict
        Variance of each feature per class.
    """

    def __init__(self, epsilon: float = 1e-9) -> None:
        """
        Initialize the Gaussian Naive Bayes classifier.

        Parameters
        ----------
        epsilon : float, default=1e-9
            Small constant to avoid division by zero in variance calculations.
        """
        self.epsilon: float = epsilon
        self.classes: np.ndarray = np.array([])
        self.class_priors: dict = {}
        self.class_means: dict = {}
        self.class_variances: dict = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fit the Gaussian Naive Bayes model to training data.

        Calculates the mean, variance, and prior probability for each feature
        in each class.

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

        Raises
        ------
        ValueError
            If X and y have incompatible dimensions.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, "
                f"got {X.shape[0]} and {y.shape[0]}"
            )

        self.classes = np.unique(y)

        for cls in self.classes:
            X_cls = X[y == cls]
            self.class_means[cls] = np.mean(X_cls, axis=0)
            self.class_variances[cls] = np.var(X_cls, axis=0)
            self.class_priors[cls] = X_cls.shape[0] / X.shape[0]

        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray
            Data to predict on, of shape (n_samples, n_features).
        **kwargs : Any
            Additional keyword arguments (unused).

        Returns
        -------
        np.ndarray
            Predicted class labels of shape (n_samples,).

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if len(self.classes) == 0:
            raise ValueError("Model has not been fitted yet")

        probabilities = self.predict_proba(X)
        class_indices = np.argmax(probabilities, axis=1)
        return self.classes[class_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.

        Parameters
        ----------
        X : np.ndarray
            Data to predict on, of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class probabilities of shape (n_samples, n_classes).
            Each row sums to 1.0.

        Raises
        ------
        ValueError
            If model has not been fitted yet.
        """
        if len(self.classes) == 0:
            raise ValueError("Model has not been fitted yet")

        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_posteriors = np.zeros((n_samples, n_classes))

        for class_idx, cls in enumerate(self.classes):
            log_prior = np.log(self.class_priors[cls])
            pdf = self._calculate_pdf(class_idx, X)
            log_likelihood = np.sum(np.log(pdf), axis=1)
            log_posteriors[:, class_idx] = log_prior + log_likelihood

        # Convert from log probabilities to probabilities using softmax
        # Subtract max for numerical stability
        max_log_posteriors = np.max(log_posteriors, axis=1, keepdims=True)
        log_posteriors_stable = log_posteriors - max_log_posteriors
        probabilities = np.exp(log_posteriors_stable)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)

        return probabilities

    def _calculate_pdf(self, class_idx: int, x: np.ndarray) -> np.ndarray:
        """
        Calculate the probability density function for a given class and sample.

        Computes the Gaussian probability density function assuming independence
        between features.

        Parameters
        ----------
        class_idx : int
            Index of the class in the classes array.
        x : np.ndarray
            Sample vector of shape (n_features,).

        Returns
        -------
        np.ndarray
            Probability density values for each feature of shape (n_features,).

        Raises
        ------
        ValueError
            If class_idx is out of range.
        """
        if class_idx < 0 or class_idx >= len(self.classes):
            raise ValueError(
                f"class_idx {class_idx} is out of range [0, {len(self.classes) - 1}]"
            )

        cls = self.classes[class_idx]
        mean = self.class_means[cls]
        variance = self.class_variances[cls] + self.epsilon

        # Gaussian PDF: (1 / sqrt(2π * σ²)) * exp(-(x - μ)² / (2 * σ²))
        numerator = np.exp(-((x - mean) ** 2) / (2.0 * variance))
        denominator = np.sqrt(2.0 * np.pi * variance)

        return numerator / denominator
