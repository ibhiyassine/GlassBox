import numpy as np


def _validate_regression_inputs(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and normalize regression metric inputs.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated arrays, both of shape (n_samples,).
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    if y_pred_arr.ndim != 1:
        raise ValueError("y_pred must be a 1D array")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples")
    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true and y_pred must contain at least one sample")

    return y_true_arr, y_pred_arr


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean absolute error (MAE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).

    Returns
    -------
    float
        Mean absolute error.
    """
    y_true_arr, y_pred_arr = _validate_regression_inputs(y_true=y_true, y_pred=y_pred)
    return float(np.mean(np.abs(y_true_arr - y_pred_arr)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean squared error (MSE).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).

    Returns
    -------
    float
        Mean squared error.
    """
    y_true_arr, y_pred_arr = _validate_regression_inputs(y_true=y_true, y_pred=y_pred)
    residuals = y_true_arr - y_pred_arr
    return float(np.mean(residuals**2))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the coefficient of determination (R² score).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth target values of shape (n_samples,).
    y_pred : np.ndarray
        Predicted target values of shape (n_samples,).

    Returns
    -------
    float
        R² score.

    Notes
    -----
    If `y_true` is constant, returns 1.0 for perfect predictions and 0.0 otherwise.
    """
    y_true_arr, y_pred_arr = _validate_regression_inputs(y_true=y_true, y_pred=y_pred)

    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    y_mean = np.mean(y_true_arr)
    ss_tot = np.sum((y_true_arr - y_mean) ** 2)

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return float(1.0 - (ss_res / ss_tot))
