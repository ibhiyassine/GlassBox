from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Self

import numpy as np

from glassbox.core.math import calc_split_gain
from glassbox.models._base import BaseModel


@dataclass
class _Node:
    """
    Internal node structure for tree-based models.

    Parameters
    ----------
    feature_index : int, optional
        Index of the feature used for splitting, by default None.
    threshold : float, optional
        Threshold value for the split, by default None.
    left : _Node, optional
        Left child node, by default None.
    right : _Node, optional
        Right child node, by default None.
    value : float, optional
        Value of the leaf node (if it's a leaf), by default None.
    """

    feature_index: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None
    value: Optional[float] = None

    def is_leaf_node(self) -> bool:
        """
        Determines if the current node is a leaf node.

        Returns
        -------
        bool
            True if it's a leaf node, False otherwise.
        """
        return self.value is not None


@dataclass
class _SplitResult:
    """
    Result of a data split across a feature.

    Parameters
    ----------
    feature_index : int
        Index of the feature used for splitting.
    threshold : float
        Threshold value for the split.
    left_X : np.ndarray
        Features of the left split.
    left_y : np.ndarray
        Targets of the left split.
    right_X : np.ndarray
        Features of the right split.
    right_y : np.ndarray
        Targets of the right split.
    gain : float
        Information gain or variance reduction of the split.
    """

    feature_index: int
    threshold: float
    left_X: np.ndarray
    left_y: np.ndarray
    right_X: np.ndarray
    right_y: np.ndarray
    gain: float


class BaseTree(BaseModel):
    """
    Abstract base class for all tree-based models.
    """

    def __init__(self, max_depth: int = 100, min_samples_split: int = 2) -> None:
        """
        Initialize the base tree model.

        Parameters
        ----------
        max_depth : int, default=100
            Maximum depth of the tree.
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: Optional[_Node] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fits the tree model to the training data.

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
        self.root = self._build_tree(X, y, depth=0)
        return self

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
        if self.root is None:
            raise RuntimeError("Model is not fitted yet.")
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """
        Recursively builds the tree structure.

        Parameters
        ----------
        X : np.ndarray
            Current subset of features.
        y : np.ndarray
            Current subset of targets.
        depth : int
            Current depth in the tree.

        Returns
        -------
        _Node
            The root node of the current subtree.
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_samples < self.min_samples_split or n_labels <= 1:
            leaf_value = self._create_leaf_value(y)
            return _Node(value=leaf_value)

        best_split = self._best_split(X, y)

        if best_split is None or best_split.gain <= 0:
            leaf_value = self._create_leaf_value(y)
            return _Node(value=leaf_value)

        left = self._build_tree(best_split.left_X, best_split.left_y, depth + 1)
        right = self._build_tree(best_split.right_X, best_split.right_y, depth + 1)

        return _Node(
            feature_index=best_split.feature_index,
            threshold=best_split.threshold,
            left=left,
            right=right,
        )

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Optional[_SplitResult]:
        """
        Finds the best possible split for the current training data.

        Parameters
        ----------
        X : np.ndarray
            Current subset of features.
        y : np.ndarray
            Current subset of targets.

        Returns
        -------
        Optional[_SplitResult]
            Information about the best split, or None if no split could be found.
        """
        best_split = None
        best_gain = -1.0
        n_samples, n_features = X.shape
        parent_cost = self._calculate_cost(y)

        for feat_idx in range(n_features):
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                left_idx = X_column <= threshold
                right_idx = ~left_idx

                if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
                    continue

                left_y, right_y = y[left_idx], y[right_idx]
                left_cost = self._calculate_cost(left_y)
                right_cost = self._calculate_cost(right_y)

                gain = calc_split_gain(
                    parent_cost,
                    left_cost,
                    right_cost,
                    n_samples,
                    len(left_y),
                    len(right_y),
                )

                if gain > best_gain:
                    best_gain = gain
                    best_split = _SplitResult(
                        feature_index=feat_idx,
                        threshold=threshold,
                        left_X=X[left_idx],
                        left_y=left_y,
                        right_X=X[right_idx],
                        right_y=right_y,
                        gain=gain,
                    )

        return best_split

    def _traverse_tree(self, x: np.ndarray, node: _Node) -> float:
        """
        Traverses the tree to make a prediction for a single sample.

        Parameters
        ----------
        x : np.ndarray
            A single sample of shape (n_features,).
        node : _Node
            Current node in traversal.

        Returns
        -------
        float
            Predicted value for the single sample.
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

    @abstractmethod
    def _calculate_cost(self, y: np.ndarray) -> float:
        """
        Calculates the cost function (e.g., entropy, variance) for a set of targets.

        Parameters
        ----------
        y : np.ndarray
            Target values of the current node.

        Returns
        -------
        float
            The computed cost value.
        """
        pass

    @abstractmethod
    def _create_leaf_value(self, y: np.ndarray) -> float:
        """
        Computes the final value for a leaf node.

        Parameters
        ----------
        y : np.ndarray
            Target values falling into the leaf node.

        Returns
        -------
        float
            The computed value representing the subset.
        """
        pass
