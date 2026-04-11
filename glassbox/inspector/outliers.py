from typing import Dict, List, Tuple

import numpy as np

from glassbox.frame.dataset import Dataset
from glassbox.inspector.report import OutlierInfo


class OutlierDetector:
    """
    Detects outliers within numerical columns of a dataset.
    """

    def flag_outliers(self, data: Dataset, cols: List[str]) -> Dict[str, OutlierInfo]:
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
        results = {}
        for col_name in cols:
            # Extract and cast to float to assure vector operations on mixed dataset object matrices
            col_data = data.get_columns(col_name).data[:, 0].astype(float)
            col_valid = col_data[~np.isnan(col_data)]
            if len(col_valid) == 0:
                results[col_name] = OutlierInfo(count=0, lower_bound=float("nan"), upper_bound=float("nan"))
                continue
            lower, upper = self._calc_iqr(col_valid)
            count = int(np.sum((col_valid < lower) | (col_valid > upper)))
            results[col_name] = OutlierInfo(count=count, lower_bound=lower, upper_bound=upper)
        return results

    def _calc_iqr(self, col: np.ndarray) -> Tuple[float, float]:
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
        sorted_col = np.sort(col)
        n = len(sorted_col)
        if n == 0:
            return 0.0, 0.0

        def get_percentile(p: float) -> float:
            idx = (n - 1) * p / 100.0
            idx_int = int(idx)
            if idx_int == n - 1:
                return float(sorted_col[idx_int])
            fraction = idx - idx_int
            return float(sorted_col[idx_int] + fraction * (sorted_col[idx_int + 1] - sorted_col[idx_int]))

        q1 = get_percentile(25.0)
        q3 = get_percentile(75.0)
        iqr = q3 - q1

        return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
