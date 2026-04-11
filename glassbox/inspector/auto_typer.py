from typing import Dict

import numpy as np

from glassbox.frame.dataset import Dataset
from glassbox.inspector.report import FeatureType


class AutoTyper:
    """
    Infers logical data types for dataset columns.
    """

    def infer_types(self, data: Dataset) -> Dict[str, FeatureType]:
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
        results = {}
        for col_name in data.columns:
            col_data = data.get_columns(col_name).data[:, 0]
            
            if self._is_boolean(col_data):
                results[col_name] = FeatureType.BOOLEAN
            elif self._is_numeric(col_data):
                if self._is_ordinal(col_data):
                    results[col_name] = FeatureType.ORDINAL
                else:
                    results[col_name] = FeatureType.NUMERICAL
            else:
                results[col_name] = FeatureType.NOMINAL
        return results

    def _is_ordinal(self, col: np.ndarray) -> bool:
        """
        Check if a numeric column is ordinal (integers with cardinality < 20).

        Parameters
        ----------
        col : np.ndarray
            Input array of shape (n_samples,).

        Returns
        -------
        bool
            True if the column is ordinal.
        """
        col_float = col.astype(float)
        col_valid = col_float[~np.isnan(col_float)]
        
        if len(col_valid) == 0:
            return False
            
        # Check if all valid numbers are whole integers
        if not np.all(np.mod(col_valid, 1) == 0):
            return False
            
        # Check cardinality
        if len(np.unique(col_valid)) < 20:
            return True
            
        return False

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
        if self._is_numeric(col):
            col_float = col.astype(float)
            col_valid = col_float[~np.isnan(col_float)]
            return len(np.unique(col_valid)) == 2
        
        # for object/string type columns, drop None or np.nan-like strings
        try:
            uniq = np.unique(col)
            # just check len == 2 to be compliant with "exactly 2 unique values"
            return len(uniq) == 2
        except Exception:
            return False

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
        if np.issubdtype(col.dtype, np.number):
            return True
        try:
            # Array might be dtype object from mixed dataset
            col.astype(float)
            # if it passes, it strictly numeric (NaNs allowed)
            return True
        except (ValueError, TypeError):
            return False
