from typing import List

import numpy as np

from glassbox.core.math import calc_mode
from glassbox.models.ensemble._base import BaseRandomForest
from glassbox.models.trees.classifier import DecisionTreeClassifier


class RandomForestClassifier(BaseRandomForest):
    """
    Random Forest classifier using Decision Tree classification models.
    """

    def __init__(
        self, n_estimators: int = 100, max_depth: int = 100, min_samples_split: int = 2
    ) -> None:
        """
        Initialize the random forest classifier.

        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the forest.
        max_depth : int, default=100
            Maximum depth of individual trees.
        min_samples_split : int, default=2
            Minimum number of samples required to split an internal node.
        """
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
        )
        self.trees: List[DecisionTreeClassifier] = []

    def _create_tree(self) -> DecisionTreeClassifier:
        """
        Instantiates a new classification decision tree.

        Returns
        -------
        DecisionTreeClassifier
            A new instance of a decision tree classifier.
        """
        return DecisionTreeClassifier()

    def _aggregate(self, predictions: np.ndarray) -> np.ndarray:
        """
        Aggregates class predictions via majority vote.

        Parameters
        ----------
        predictions : np.ndarray
            Class predictions from all trees in the forest.

        Returns
        -------
        np.ndarray
            Voted class predictions.
        """
        return np.array([float(calc_mode(row)) for row in predictions])
