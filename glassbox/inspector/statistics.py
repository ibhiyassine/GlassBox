from typing import Dict, List

import numpy as np

from glassbox.frame.dataset import Dataset


class StatProfiler:
    """
    Calculates summary statistics for dataset columns.
    """

    def calculate_stats(self, data: Dataset, cols: List[str]) -> Dict:
        """
        Compute statistics for multiple columns.

        Parameters
        ----------
        data : Dataset
            The dataset containing the inputs.
        cols : List[str]
            List of column names to analyze.

        Returns
        -------
        Dict
            Mapping from column names to FeatureStats objects.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError


class AssociationAnalyzer:
    """
    Analyzes pairwise correlations and associations between features.
    """

    def build_correlation(self, data: Dataset, cols: List[str]) -> List:
        """
        Compute pairwise correlation across specified columns.

        Parameters
        ----------
        data : Dataset
            Input dataset.
        cols : List[str]
            Columns to include in correlation output.

        Returns
        -------
        List
            A list of CollinearityPair objects containing scores.
        """
        raise NotImplementedError

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
        raise NotImplementedError
