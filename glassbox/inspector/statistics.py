from typing import Dict, List

import numpy as np

from glassbox.frame.dataset import Dataset
from glassbox.inspector.report import CategoricalStats, CollinearityPair, NumericStats


class StatProfiler:
    """
    Calculates summary statistics for dataset columns.
    """

    def calculate_numeric_stats(self, data: Dataset, cols: List[str]) -> Dict[str, NumericStats]:
        """
        Compute statistics for numerical columns.

        Parameters
        ----------
        data : Dataset
            The dataset containing the inputs.
        cols : List[str]
            List of column names to analyze.

        Returns
        -------
        Dict
            Mapping from column names to NumericStats objects.
        """
        results = {}
        for col_name in cols:
            col_data = data.get_columns(col_name).data[:, 0].astype(float)
            col_valid = col_data[~np.isnan(col_data)]
            if len(col_valid) == 0:
                results[col_name] = NumericStats(
                    mean=float("nan"),
                    median=float("nan"),
                    std=float("nan"),
                    skew=float("nan"),
                    kurt=float("nan")
                )
                continue

            results[col_name] = NumericStats(
                mean=self._calc_mean(col_valid),
                median=self._calc_median(col_valid),
                std=self._calc_std(col_valid),
                skew=self._calc_skew(col_valid),
                kurt=self._calc_kurtosis(col_valid)
            )
        return results

    def calculate_categorical_stats(self, data: Dataset, cols: List[str]) -> Dict[str, CategoricalStats]:
        """
        Compute statistics for categorical columns.

        Parameters
        ----------
        data : Dataset
            The dataset containing the inputs.
        cols : List[str]
            List of column names to analyze.

        Returns
        -------
        Dict
            Mapping from column names to CategoricalStats objects.
        """
        results = {}
        for col_name in cols:
            col_data = data.get_columns(col_name).data[:, 0]
            valid_mask = np.array([v is not None and not (isinstance(v, float) and np.isnan(v)) for v in col_data])
            col_valid = col_data[valid_mask]
            
            if len(col_valid) == 0:
                results[col_name] = CategoricalStats(mode=float("nan"), cardinality=0)
                continue
                
            unique_vals = np.unique(col_valid)
            results[col_name] = CategoricalStats(
                mode=self._calc_mode(col_valid),
                cardinality=len(unique_vals)
            )
        return results

    def _calc_mean(self, col: np.ndarray) -> float:
        """
        Calculate the mean of a column.

        Parameters
        ----------
        col : np.ndarray
            Numeric array of shape (n_samples,).

        Returns
        -------
        float
            The calculated mean.
        """
        return float(np.sum(col) / len(col))

    def _calc_median(self, col: np.ndarray) -> float:
        """
        Calculate the median of a column.

        Parameters
        ----------
        col : np.ndarray
            Numeric array of shape (n_samples,).

        Returns
        -------
        float
            The calculated median.
        """
        sorted_col = np.sort(col)
        n = len(sorted_col)
        mid = n // 2
        if n % 2 == 0:
            return float((sorted_col[mid - 1] + sorted_col[mid]) / 2.0)
        else:
            return float(sorted_col[mid])

    def _calc_mode(self, col: np.ndarray) -> float | str:
        """
        Calculate the mode of a column.

        Parameters
        ----------
        col : np.ndarray
            Array of shape (n_samples,).

        Returns
        -------
        float | str
            The calculated mode.
        """
        vals, counts = np.unique(col, return_counts=True)
        max_idx = np.argmax(counts)
        val = vals[max_idx]
        if hasattr(val, "dtype") and np.issubdtype(val.dtype, np.number):
            return float(val)
        if isinstance(val, (int, float)):
            return float(val)
        return str(val)

    def _calc_std(self, col: np.ndarray) -> float:
        """
        Calculate the standard deviation of a column.

        Parameters
        ----------
        col : np.ndarray
            Numeric array of shape (n_samples,).

        Returns
        -------
        float
            Standard deviation.
        """
        n = len(col)
        if n <= 1:
            return 0.0
        mean_val = self._calc_mean(col)
        var = np.sum((col - mean_val) ** 2) / (n - 1)
        return float(np.sqrt(var))

    def _calc_skew(self, col: np.ndarray) -> float:
        """
        Calculate the skewness of a column.

        Parameters
        ----------
        col : np.ndarray
            Numeric array of shape (n_samples,).

        Returns
        -------
        float
            Skewness value.
        """
        n = len(col)
        if n <= 2:
            return 0.0
        std_val = self._calc_std(col)
        if std_val == 0:
            return 0.0
        mean_val = self._calc_mean(col)
        skew = np.sum(((col - mean_val) / std_val) ** 3) * (n / ((n - 1) * (n - 2)))
        return float(skew)

    def _calc_kurtosis(self, col: np.ndarray) -> float:
        """
        Calculate the kurtosis of a column.

        Parameters
        ----------
        col : np.ndarray
            Numeric array of shape (n_samples,).

        Returns
        -------
        float
            Kurtosis value.
        """
        n = len(col)
        if n <= 3:
            return 0.0
        std_val = self._calc_std(col)
        if std_val == 0:
            return 0.0
        mean_val = self._calc_mean(col)
        
        m4 = np.sum((col - mean_val) ** 4)
        term1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        term2 = (3 * ((n - 1) ** 2)) / ((n - 2) * (n - 3))
        kurtosis = term1 * (m4 / (std_val ** 4)) - term2
        return float(kurtosis)


class AssociationAnalyzer:
    """
    Analyzes pairwise correlations and associations between features.
    """

    def build_associations(self, data: Dataset, num_cols: List[str], cat_cols: List[str]) -> List[CollinearityPair]:
        """
        Compute pairwise correlation and associations across specified columns.

        Parameters
        ----------
        data : Dataset
            Input dataset.
        num_cols : List[str]
            Numerical columns to inspect with Pearson.
        cat_cols : List[str]
            Categorical columns to inspect with Cramer's V.

        Returns
        -------
        List
            A list of CollinearityPair objects containing scores.
        """
        pairs = []
        n_num = len(num_cols)
        for i in range(n_num):
            for j in range(i + 1, n_num):
                col_x_name = num_cols[i]
                col_y_name = num_cols[j]
                col_x = data.get_columns(col_x_name).data[:, 0].astype(float)
                col_y = data.get_columns(col_y_name).data[:, 0].astype(float)
                
                valid_mask = ~(np.isnan(col_x) | np.isnan(col_y))
                x_val = col_x[valid_mask]
                y_val = col_y[valid_mask]
                
                score = self._calc_pearson(x_val, y_val)
                pairs.append(CollinearityPair(feature_a=col_x_name, feature_b=col_y_name, score=score, metric="pearson"))

        n_cat = len(cat_cols)
        for i in range(n_cat):
            for j in range(i + 1, n_cat):
                col_x_name = cat_cols[i]
                col_y_name = cat_cols[j]
                col_x = data.get_columns(col_x_name).data[:, 0]
                col_y = data.get_columns(col_y_name).data[:, 0]
                
                valid_mask = np.array([
                    v_x is not None and not (isinstance(v_x, float) and np.isnan(v_x)) and
                    v_y is not None and not (isinstance(v_y, float) and np.isnan(v_y))
                    for v_x, v_y in zip(col_x, col_y)
                ])
                x_val = col_x[valid_mask]
                y_val = col_y[valid_mask]
                
                score = self._calc_cramers_v(x_val, y_val)
                pairs.append(CollinearityPair(feature_a=col_x_name, feature_b=col_y_name, score=score, metric="cramers_v"))
                
        return pairs

    def _calc_pearson(self, col_x: np.ndarray, col_y: np.ndarray) -> float:
        """
        Calculate Pearson correlation coefficient between two columns.

        Parameters
        ----------
        col_x : np.ndarray
            First numeric array of shape (n_samples,).
        col_y : np.ndarray
            Second numeric array of shape (n_samples,).

        Returns
        -------
        float
            Pearson correlation coefficient.
        """
        n = len(col_x)
        if n <= 1:
            return 0.0
            
        mean_x = np.sum(col_x) / n
        mean_y = np.sum(col_y) / n
        
        num = np.sum((col_x - mean_x) * (col_y - mean_y))
        den = np.sqrt(np.sum((col_x - mean_x) ** 2) * np.sum((col_y - mean_y) ** 2))
        
        if den == 0:
            return 0.0
            
        return float(num / den)

    def _calc_cramers_v(self, col_x: np.ndarray, col_y: np.ndarray) -> float:
        """
        Calculate Cramer's V statistic for categorical-categorical association.

        Parameters
        ----------
        col_x : np.ndarray
            First nominal array of shape (n_samples,).
        col_y : np.ndarray
            Second nominal array of shape (n_samples,).

        Returns
        -------
        float
            Cramer's V score between 0.0 and 1.0.
        """
        n = len(col_x)
        if n == 0:
            return 0.0

        x_unique, x_idx = np.unique(col_x, return_inverse=True)
        y_unique, y_idx = np.unique(col_y, return_inverse=True)

        k = len(x_unique)
        r = len(y_unique)
        if k <= 1 or r <= 1:
            return 0.0

        contingency = np.zeros((r, k))
        np.add.at(contingency, (y_idx, x_idx), 1)

        row_sums = contingency.sum(axis=1)
        col_sums = contingency.sum(axis=0)
        expected = np.outer(row_sums, col_sums) / n

        with np.errstate(divide='ignore', invalid='ignore'):
            chi2_components = ((contingency - expected) ** 2) / expected
            chi2_components[expected == 0] = 0

        chi2 = np.sum(chi2_components)

        phi2 = chi2 / n
        min_dim = min(k - 1, r - 1)
        if min_dim == 0:
            return 0.0

        v = np.sqrt(phi2 / min_dim)
        return float(v)
