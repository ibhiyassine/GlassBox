import numpy as np

from glassbox.core.math import calc_gini_impurity, calc_mode
from glassbox.models.trees._base import BaseTree


class DecisionTreeClassifier(BaseTree):
    """
    A decision tree classifier.
    """

    def _calculate_cost(self, y: np.ndarray) -> float:
        """
        Calculates the impurity metric (e.g., Gini or entropy) for a classification split.

        Parameters
        ----------
        y : np.ndarray
            Target class labels of the current node.

        Returns
        -------
        float
            The computed cost (impurity) value.
        """
        return calc_gini_impurity(y)

    def _create_leaf_value(self, y: np.ndarray) -> float:
        """
        Computes the most common class label for a leaf node.

        Parameters
        ----------
        y : np.ndarray
            Target class labels falling into the leaf node.

        Returns
        -------
        float
            The predicted class label.
        """
        return float(calc_mode(y))
