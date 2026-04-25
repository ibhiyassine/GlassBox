from abc import ABC, abstractmethod
from typing import Generator, Tuple, Self

import numpy as np


class BaseSplitter(ABC):
    """
    Abstract base class for cross-validation splitters.

    Parameters
    ----------
    n_splits : int
        Number of splits.
    shuffle : bool
        Whether to shuffle data before splitting.
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = False) -> None:
        self.n_splits: int = n_splits
        self.shuffle: bool = shuffle

    @abstractmethod
    def split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices for cross-validation.

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
        raise NotImplementedError
