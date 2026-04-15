from typing import Dict, List

import numpy as np

from glassbox.core.math import calc_iqr
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
                results[col_name] = OutlierInfo(
                    count=0, lower_bound=float("nan"), upper_bound=float("nan")
                )
                continue
            lower, upper = calc_iqr(col_valid)

            count = int(np.sum((col_valid < lower) | (col_valid > upper)))
            results[col_name] = OutlierInfo(
                count=count, lower_bound=lower, upper_bound=upper
            )
        return results
