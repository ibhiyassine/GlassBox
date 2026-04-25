from typing import Generator, Tuple

import numpy as np

from glassbox.orchestrator.base_splitter import BaseSplitter


class KFoldSplitter(BaseSplitter):
    """
    K-fold cross-validation splitter.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    shuffle : bool
        Whether to shuffle data before splitting.
    """

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test splits for K-fold cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        Generator[Tuple[np.ndarray, np.ndarray], None, None]
            Generator yielding training and validation index tuples.
        """
        n_samples = X.shape[0]
        if self.n_splits <= 1 or self.n_splits > n_samples:
            raise ValueError("n_splits must be between 2 and n_samples")

        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            start = current
            stop = current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate(
                [indices[:start], indices[stop:]]
            ).astype(int)
            yield train_idx, test_idx
            current = stop


class StratifiedKFoldSplitter(BaseSplitter):
    """
    Stratified K-fold cross-validation splitter.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    shuffle : bool
        Whether to shuffle data before splitting.
    """

    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate stratified train/test splits for K-fold cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Data array of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        Generator[Tuple[np.ndarray, np.ndarray], None, None]
            Generator yielding training and validation index tuples.
        """
        n_samples = X.shape[0]
        if self.n_splits <= 1 or self.n_splits > n_samples:
            raise ValueError("n_splits must be between 2 and n_samples")

        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        unique_classes = np.unique(y)
        strata = {cls: [] for cls in unique_classes}
        for idx in indices:
            strata[y[idx]].append(idx)

        folds = [[] for _ in range(self.n_splits)]
        for cls in unique_classes:
            class_indices = np.array(strata[cls], dtype=int)
            if self.shuffle:
                np.random.shuffle(class_indices)
            for fold_index, sample_idx in enumerate(class_indices):
                folds[fold_index % self.n_splits].append(sample_idx)

        for fold_index in range(self.n_splits):
            test_idx = np.array(folds[fold_index], dtype=int)
            train_idx = np.array(
                [idx for i, fold in enumerate(folds) if i != fold_index for idx in fold],
                dtype=int,
            )
            yield train_idx, test_idx
