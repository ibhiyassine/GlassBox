from abc import abstractmethod
from typing import Any, List, Self, Tuple

import numpy as np

from glassbox.core.math import (
    generate_bootstrap_indices,
    generate_feature_subset_indices,
)
from glassbox.models._base import BaseModel
from glassbox.models.trees._base import BaseTree


class BaseRandomForest(BaseModel):
    """
    Abstract base class for all random forest models.
    """

    def __init__(
        self, n_estimators: int = 100, max_depth: int = 100, min_samples_split: int = 2
    ) -> None:
        """
        Initialize the random forest model.

        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the forest.
        max_depth : int, default=100
            Maximum depth of individual trees.
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees: List[BaseTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fits the ensemble model to the training data.

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
        self.trees = []
        n_samples, n_features = X.shape

        for _ in range(self.n_estimators):
            tree = self._create_tree()
            tree.max_depth = self.max_depth
            tree.min_samples_split = self.min_samples_split

            X_sample, y_sample = self._bootstrap_sample(X, y)
            feature_idx = self._get_feature_subset(n_features)

            # Keep track of which features this specific tree was trained on
            tree._feature_idx = feature_idx

            X_subset = X_sample[:, feature_idx]
            tree.fit(X_subset, y_sample)

            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        """
        Predicts target values for the given data using the ensemble.

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
        if not self.trees:
            raise RuntimeError("Model is not fitted yet.")

        tree_preds = []
        for tree in self.trees:
            X_subset = X[:, tree._feature_idx]
            tree_preds.append(tree.predict(X_subset))

        # Shape of tree_preds: (n_estimators, n_samples)
        # Transpose to (n_samples, n_estimators) for sample-wise aggregation
        tree_preds = np.array(tree_preds).T

        return self._aggregate(tree_preds)

    def _bootstrap_sample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a bootstrap sample from the dataset.

        Parameters
        ----------
        X : np.ndarray
            Features to sample from.
        y : np.ndarray
            Targets to sample from.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Bootstrapped feature array and target array.
        """
        n_samples = X.shape[0]
        idxs = generate_bootstrap_indices(n_samples)
        return X[idxs], y[idxs]

    def _get_feature_subset(self, n_features: int) -> np.ndarray:
        """
        Selects a random subset of features for tree splits.

        Parameters
        ----------
        n_features : int
            Total number of available features.

        Returns
        -------
        np.ndarray
            Indices of the selected feature subset.
        """
        return generate_feature_subset_indices(n_features)

    @abstractmethod
    def _create_tree(self) -> BaseTree:
        """
        Instantiates a new bare tree corresponding to the ensemble type.

        Returns
        -------
        BaseTree
            An uninitialized base tree object.
        """
        pass

    @abstractmethod
    def _aggregate(self, predictions: np.ndarray) -> np.ndarray:
        """
        Aggregates predictions from all trees in the ensemble into a final prediction.

        Parameters
        ----------
        predictions : np.ndarray
            Predictions from individual trees.

        Returns
        -------
        np.ndarray
            The aggregated predictions.
        """
        pass
