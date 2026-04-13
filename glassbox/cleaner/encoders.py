from typing import Dict, List, Self

import numpy as np

from glassbox.cleaner._base import BaseTransformer


class OneHotEncoder(BaseTransformer):
    """
    Encode categorical features as a one-hot numeric array.
    """

    __slots__ = ["_categories"]

    def __init__(self):
        self._categories: Dict[int, List[str]] = {}

    def fit(self, X: np.ndarray) -> Self:
        """
        Learn the categorical levels for encoding.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        Self
            Fitted encoder instance.
        """
        n_features = X.shape[1]
        for col_idx in range(n_features):
            col = X[:, col_idx]
            if np.issubdtype(col.dtype, np.number):
                col_clean = col[~np.isnan(col)]
            else:
                col_clean = np.array([x for x in col if x is not None and not (isinstance(x, float) and np.isnan(x))])

            uniques = np.unique(col_clean)
            self._categories[col_idx] = list(uniques)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the dataset into a one-hot encoded representation.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed array properly encoded.
        """
        n_features = X.shape[1]
        out_cols = []
        for col_idx in range(n_features):
            col = X[:, col_idx]
            cats = self._categories.get(col_idx, [])
            if not cats:
                continue

            for cat in cats:
                if isinstance(cat, float) and np.isnan(cat):
                    continue
                # For string objects, equal checking doesn't break, generates boolean mask
                mask = (col == cat).astype(float)
                out_cols.append(mask)

        if not out_cols:
            return np.empty((X.shape[0], 0))

        return np.column_stack(out_cols)


class LabelEncoder(BaseTransformer):
    """
    Encode target labels with value between 0 and n_classes-1.
    """

    __slots__ = ["_mapping"]

    def __init__(self):
        self._mapping: Dict[str, int] = {}

    def fit(self, X: np.ndarray) -> Self:
        """
        Learn the vocabulary of the labels.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        Self
            Fitted encoder instance.
        """
        X_flat = X.ravel()
        if np.issubdtype(X_flat.dtype, np.number):
            X_clean = X_flat[~np.isnan(X_flat)]
        else:
            X_clean = np.array([x for x in X_flat if x is not None and not (isinstance(x, float) and np.isnan(x))])

        uniques = np.unique(X_clean)
        self._mapping = {str(val): i for i, val in enumerate(uniques)}
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform labels to normalized encoding.

        Parameters
        ----------
        X : np.ndarray
            Input array of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Transformed array properly encoded.
        """
        X_flat = X.ravel()
        res = np.zeros(X_flat.shape, dtype=float)

        for i, val in enumerate(X_flat):
            if (isinstance(val, float) and np.isnan(val)) or val is None:
                res[i] = np.nan
            else:
                res[i] = self._mapping.get(str(val), -1)

        return res.reshape(X.shape)
