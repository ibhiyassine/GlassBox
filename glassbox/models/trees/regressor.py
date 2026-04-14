import numpy as np

from glassbox.core.math import calc_mean, calc_variance
from glassbox.models.trees._base import BaseTree


class DecisionTreeRegressor(BaseTree):
    """
    A decision tree regressor.
    """

    def _calculate_cost(self, y: np.ndarray) -> float:
        """
        Calculates the variance or mean squared error for a regression split.

        Parameters
        ----------
        y : np.ndarray
            Target continuous values of the current node.

        Returns
        -------
        float
            The computed cost (variance) value.
        """
        return calc_variance(y)

    def _create_leaf_value(self, y: np.ndarray) -> float:
        """
        Computes the mean target value for a leaf node.

        Parameters
        ----------
        y : np.ndarray
            Target continuous values falling into the leaf node.

        Returns
        -------
        float
            The predicted mean value.
        """
        return calc_mean(y)
