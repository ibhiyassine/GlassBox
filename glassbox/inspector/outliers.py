from typing import Dict, List, Tuple

import numpy as np

from glassbox.frame.dataset import Dataset


class OutlierDetector:
    """
    Detects outliers within numerical columns of a dataset.
    """

    def flag_outliers(self, data: Dataset, cols: List[str]) -> Dict:
        """
        Identify outliers for specified columns.

        Parameters
        ----------
        data : Dataset
            The dataset containing the columns.
        cols : List[str]
            A list of column names to check for outliers.

        Returns
        -------
        Dict
            Mapping from column names to OutlierInfo objects.
        """
        raise NotImplementedError

    def _calc_iqr(self, col: np.ndarray) -> Tuple:
        """
        Calculate the Interquartile Range (IQR) for a column.

        Parameters
        ----------
        col : np.ndarray
            Numeric input array of shape (n_samples,).

        Returns
        -------
        Tuple
            A tuple containing (lower_bound, upper_bound).
        """
        raise NotImplementedError
