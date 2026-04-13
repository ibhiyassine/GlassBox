from typing import Tuple

import numpy as np


# ─[ Univariate Statistics ]────────────────────────────────────────
def calc_mean(arr: np.ndarray) -> float:
    """
    Calculate the mean of a 1D array.

    Parameters
    ----------
    arr : np.ndarray
        Numeric array of shape (n_samples,).

    Returns
    -------
    float
        The calculated mean.
    """
    return float(np.sum(arr) / len(arr))


def calc_median(arr: np.ndarray) -> float:
    """
    Calculate the median of a 1D array.

    Parameters
    ----------
    arr : np.ndarray
        Numeric array of shape (n_samples,).

    Returns
    -------
    float
        The calculated median.
    """
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    mid = n // 2
    if n % 2 == 0:
        return float((sorted_arr[mid - 1] + sorted_arr[mid]) / 2.0)
    else:
        return float(sorted_arr[mid])


def calc_mode(arr: np.ndarray) -> float | str:
    """
    Calculate the mode of a 1D array.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (n_samples,).

    Returns
    -------
    float | str
        The calculated mode.
    """
    vals, counts = np.unique(arr, return_counts=True)
    max_idx = np.argmax(counts)
    val = vals[max_idx]
    if hasattr(val, "dtype") and np.issubdtype(val.dtype, np.number):
        return float(val)
    if isinstance(val, (int, float)):
        return float(val)
    return str(val)


def calc_std(arr: np.ndarray) -> float:
    """
    Calculate the standard deviation of a 1D array.

    Parameters
    ----------
    arr : np.ndarray
        Numeric array of shape (n_samples,).

    Returns
    -------
    float
        Standard deviation.
    """
    n = len(arr)
    if n <= 1:
        return 0.0
    mean_val = calc_mean(arr)
    var = np.sum((arr - mean_val) ** 2) / (n - 1)
    return float(np.sqrt(var))


def calc_skew(arr: np.ndarray) -> float:
    """
    Calculate the skewness of a 1D array.

    Parameters
    ----------
    arr : np.ndarray
        Numeric array of shape (n_samples,).

    Returns
    -------
    float
        Skewness value.
    """
    n = len(arr)
    if n <= 2:
        return 0.0
    std_val = calc_std(arr)
    if std_val == 0:
        return 0.0
    mean_val = calc_mean(arr)
    skew = np.sum(((arr - mean_val) / std_val) ** 3) * (n / ((n - 1) * (n - 2)))
    return float(skew)


def calc_kurtosis(arr: np.ndarray) -> float:
    """
    Calculate the kurtosis of a 1D array.

    Parameters
    ----------
    arr : np.ndarray
        Numeric array of shape (n_samples,).

    Returns
    -------
    float
        Kurtosis value.
    """
    n = len(arr)
    if n <= 3:
        return 0.0
    std_val = calc_std(arr)
    if std_val == 0:
        return 0.0
    mean_val = calc_mean(arr)

    m4 = np.sum((arr - mean_val) ** 4)
    term1 = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
    term2 = (3 * ((n - 1) ** 2)) / ((n - 2) * (n - 3))
    kurtosis = term1 * (m4 / (std_val**4)) - term2
    return float(kurtosis)


# ─[ Bivariate Statistics ]─────────────────────────────────────────────
def calc_pearson(arr_x: np.ndarray, arr_y: np.ndarray) -> float:
    """
    Calculate Pearson correlation coefficient between two numerical arrays.

    Parameters
    ----------
    arr_x : np.ndarray
        First numeric array of shape (n_samples,).
    arr_y : np.ndarray
        Second numeric array of shape (n_samples,).

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    n = len(arr_x)
    if n <= 1:
        return 0.0

    mean_x = np.sum(arr_x) / n
    mean_y = np.sum(arr_y) / n

    num = np.sum((arr_x - mean_x) * (arr_y - mean_y))
    den = np.sqrt(np.sum((arr_x - mean_x) ** 2) * np.sum((arr_y - mean_y) ** 2))

    if den == 0:
        return 0.0

    return float(num / den)


def calc_cramers_v(arr_x: np.ndarray, arr_y: np.ndarray) -> float:
    """
    Calculate Cramer's V statistic for categorical-categorical association between 2 arrays.

    Parameters
    ----------
    arr_x : np.ndarray
        First nominal array of shape (n_samples,).
    arr_y : np.ndarray
        Second nominal array of shape (n_samples,).

    Returns
    -------
    float
        Cramer's V score between 0.0 and 1.0.
    """
    n = len(arr_x)
    if n == 0:
        return 0.0

    x_unique, x_idx = np.unique(arr_x, return_inverse=True)
    y_unique, y_idx = np.unique(arr_y, return_inverse=True)

    k = len(x_unique)
    r = len(y_unique)
    if k <= 1 or r <= 1:
        return 0.0

    contingency = np.zeros((r, k))
    np.add.at(contingency, (y_idx, x_idx), 1)

    row_sums = contingency.sum(axis=1)
    arr_sums = contingency.sum(axis=0)
    expected = np.outer(row_sums, arr_sums) / n

    with np.errstate(divide="ignore", invalid="ignore"):
        chi2_components = ((contingency - expected) ** 2) / expected
        chi2_components[expected == 0] = 0

    chi2 = np.sum(chi2_components)

    phi2 = chi2 / n
    min_dim = min(k - 1, r - 1)
    if min_dim == 0:
        return 0.0

    v = np.sqrt(phi2 / min_dim)
    return float(v)


def calc_percentile(arr: np.ndarray, p: float) -> float:
    """
    Calculate the precise percentile of an array using interpolation.

    Parameters
    ----------
    arr : np.ndarray
        Array dimension to extract percentile from.
    p : float
        Percentile range (0-100).

    Returns
    -------
    float
        Calculated percentile.
    """
    sorted_col = np.sort(arr)
    n = len(sorted_col)
    if n == 0:
        return 0.0
    idx = (n - 1) * p / 100.0
    idx_int = int(idx)
    if idx_int == n - 1:
        return float(sorted_col[idx_int])
    fraction = idx - idx_int
    return float(sorted_col[idx_int] + fraction * (sorted_col[idx_int + 1] - sorted_col[idx_int]))


def calc_iqr(arr: np.ndarray) -> Tuple[float, float]:
    """
    Calculate the Interquartile Range (IQR) bounds.

    Parameters
    ----------
    arr : np.ndarray
        Array to bound.

    Returns
    -------
    Tuple
        Tuple containing parameters for lower and upper limits.
    """
    n = len(arr)
    if n == 0:
        return 0.0, 0.0
    q1 = calc_percentile(arr, 25.0)
    q3 = calc_percentile(arr, 75.0)
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
