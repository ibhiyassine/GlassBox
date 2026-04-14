from typing import List

import numpy as np

from glassbox.core.math import calc_mean
from glassbox.models.ensemble._base import BaseRandomForest
from glassbox.models.trees.regressor import DecisionTreeRegressor


class RandomForestRegressor(BaseRandomForest):
    """
    Random Forest regressor using Decision Tree regression models.
    """

    def __init__(
        self, n_estimators: int = 100, max_depth: int = 100, min_samples_split: int = 2
    ) -> None:
        """
        Initialize the random forest regressor.

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
        self.trees: List[DecisionTreeRegressor] = []

    def _create_tree(self) -> DecisionTreeRegressor:
        """
        Instantiates a new regression decision tree.

        Returns
        -------
        DecisionTreeRegressor
            A new instance of a decision tree regressor.
        """
        return DecisionTreeRegressor()

    def _aggregate(self, predictions: np.ndarray) -> np.ndarray:
        """
        Aggregates numerical predictions via averaging.

        Parameters
        ----------
        predictions : np.ndarray
            Numerical predictions from all trees in the forest.

        Returns
        -------
        np.ndarray
            Mean aggregated predictions.
        """
        return np.array([calc_mean(row) for row in predictions])
