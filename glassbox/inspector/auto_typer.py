from typing import Dict

import numpy as np

from glassbox.frame.dataset import Dataset


class AutoTyper:
    """
    Infers logical data types for dataset columns.
    """

    def infer_types(self, data: Dataset) -> Dict:
        """
        Infer feature types for all columns in the dataset.

        Parameters
        ----------
        data : Dataset
            The dataset to analyze.

        Returns
        -------
        Dict
            Mapping from column names to their inferred FeatureType.
        """
        raise NotImplementedError

    def _is_boolean(self, col: np.ndarray) -> bool:
        """
        Check if a column contains only boolean-like values.

        Parameters
        ----------
        col : np.ndarray
            Input array of shape (n_samples,).

        Returns
        -------
        bool
            True if the column is boolean, False otherwise.
        """
        raise NotImplementedError

    def _is_numeric(self, col: np.ndarray) -> bool:
        """
        Check if a column contains numeric values.

        Parameters
        ----------
        col : np.ndarray
            Input array of shape (n_samples,).

        Returns
        -------
        bool
            True if the column is strictly numeric.
        """
        raise NotImplementedError
