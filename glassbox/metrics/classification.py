import numpy as np


def _validate_classification_inputs(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and normalize classification metric inputs.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth class labels of shape (n_samples,).
    y_pred : np.ndarray
        Predicted class labels of shape (n_samples,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Validated arrays, both of shape (n_samples,).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.ndim != 1:
        raise ValueError("y_true must be a 1D array")
    if y_pred_arr.ndim != 1:
        raise ValueError("y_pred must be a 1D array")
    if y_true_arr.shape[0] != y_pred_arr.shape[0]:
        raise ValueError("y_true and y_pred must have the same number of samples")
    if y_true_arr.shape[0] == 0:
        raise ValueError("y_true and y_pred must contain at least one sample")

    return y_true_arr, y_pred_arr


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """
    Perform element-wise division with zero-denominator protection.

    Parameters
    ----------
    numerator : np.ndarray
        Numerator values.
    denominator : np.ndarray
        Denominator values.

    Returns
    -------
    np.ndarray
        Division result where zero denominators produce 0.0.
    """
    result = np.zeros_like(numerator, dtype=float)
    valid = denominator != 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth class labels of shape (n_samples,).
    y_pred : np.ndarray
        Predicted class labels of shape (n_samples,).

    Returns
    -------
    float
        Classification accuracy in the range [0.0, 1.0].
    """
    y_true_arr, y_pred_arr = _validate_classification_inputs(y_true=y_true, y_pred=y_pred)
    return float(np.mean(y_true_arr == y_pred_arr))


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the classification precision score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth class labels of shape (n_samples,).
    y_pred : np.ndarray
        Predicted class labels of shape (n_samples,).

    Returns
    -------
    float
        Precision score in the range [0.0, 1.0].

    Notes
    -----
    Uses macro averaging over all classes. Classes with zero predicted samples
    contribute a precision of 0.0.
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    true_positives = np.diag(cm).astype(float)
    predicted_positives = np.sum(cm, axis=0).astype(float)
    precision_per_class = _safe_divide(true_positives, predicted_positives)
    return float(np.mean(precision_per_class))


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the classification recall score.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth class labels of shape (n_samples,).
    y_pred : np.ndarray
        Predicted class labels of shape (n_samples,).

    Returns
    -------
    float
        Recall score in the range [0.0, 1.0].

    Notes
    -----
    Uses macro averaging over all classes. Classes with zero true samples
    contribute a recall of 0.0.
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    true_positives = np.diag(cm).astype(float)
    actual_positives = np.sum(cm, axis=1).astype(float)
    recall_per_class = _safe_divide(true_positives, actual_positives)
    return float(np.mean(recall_per_class))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the F1 score for classification.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth class labels of shape (n_samples,).
    y_pred : np.ndarray
        Predicted class labels of shape (n_samples,).

    Returns
    -------
    float
        F1 score in the range [0.0, 1.0].

    Notes
    -----
    Uses macro averaging over all classes with per-class zero-division handling.
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    true_positives = np.diag(cm).astype(float)
    predicted_positives = np.sum(cm, axis=0).astype(float)
    actual_positives = np.sum(cm, axis=1).astype(float)

    precision_per_class = _safe_divide(true_positives, predicted_positives)
    recall_per_class = _safe_divide(true_positives, actual_positives)

    f1_numerator = 2.0 * precision_per_class * recall_per_class
    f1_denominator = precision_per_class + recall_per_class
    f1_per_class = _safe_divide(f1_numerator, f1_denominator)
    return float(np.mean(f1_per_class))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute the confusion matrix for classification results.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth class labels of shape (n_samples,).
    y_pred : np.ndarray
        Predicted class labels of shape (n_samples,).

    Returns
    -------
    np.ndarray
        Confusion matrix of shape (n_classes, n_classes).

    Notes
    -----
    Rows correspond to true labels and columns correspond to predicted labels.
    Class order follows sorted unique labels from the union of `y_true` and
    `y_pred`.
    """
    y_true_arr, y_pred_arr = _validate_classification_inputs(y_true=y_true, y_pred=y_pred)

    labels = np.unique(np.concatenate((y_true_arr, y_pred_arr)))
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    matrix = np.zeros((labels.shape[0], labels.shape[0]), dtype=int)
    for true_label, pred_label in zip(y_true_arr, y_pred_arr):
        true_idx = label_to_index[true_label]
        pred_idx = label_to_index[pred_label]
        matrix[true_idx, pred_idx] += 1

    return matrix
