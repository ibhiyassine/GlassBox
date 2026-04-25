from abc import ABC, abstractmethod
import copy
from typing import Callable, Dict, Generator, Self

import numpy as np

from glassbox.models._base import BaseModel
from glassbox.orchestrator.base_splitter import BaseSplitter


class BaseSearch(ABC):
    """
    Abstract base class for search-based model selection.

    Parameters
    ----------
    estimator : BaseModel
        The model to optimize.
    param_space : Dict
        Parameter search space.
    cv_engine : BaseSplitter
        Cross-validation splitter.
    scoring_func : Callable
        Scoring function used to evaluate candidates.

    Attributes
    ----------
    best_params_ : Dict
        Best found parameter set.
    best_score_ : float
        Best scoring value.
    best_estimator_ : BaseModel
        Best estimator instance.
    """

    def __init__(
        self,
        estimator: BaseModel,
        param_space: Dict,
        cv_engine: BaseSplitter,
        scoring_func: Callable,
    ) -> None:
        self.estimator: BaseModel = estimator
        self.param_space: Dict = param_space
        self.cv_engine: BaseSplitter = cv_engine
        self.scoring_func: Callable = scoring_func
        self.best_params_: Dict = {}
        self.best_score_: float = 0.0
        self.best_estimator_: BaseModel = estimator

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """
        Fit the search object and select the best estimator.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        Self
            The fitted search object.
        """
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must have the same number of samples"
            )

        self.best_score_ = float("-inf")
        self.best_params_ = {}
        self.best_estimator_ = copy.deepcopy(self.estimator)

        for candidate_params in self._generate_candidates():
            candidate_estimator = copy.deepcopy(self.estimator)
            for key, value in candidate_params.items():
                setattr(candidate_estimator, key, value)

            fold_scores = []
            for train_idx, val_idx in self.cv_engine.split(X, y):
                clone_estimator = copy.deepcopy(candidate_estimator)
                clone_estimator.fit(X[train_idx], y[train_idx])
                predictions = clone_estimator.predict(X[val_idx])
                fold_score = self.scoring_func(y[val_idx], predictions)
                fold_scores.append(fold_score)

            if len(fold_scores) == 0:
                continue

            mean_score = float(np.mean(fold_scores))
            if mean_score > self.best_score_:
                self.best_score_ = mean_score
                self.best_params_ = candidate_params
                self.best_estimator_ = copy.deepcopy(candidate_estimator)

        return self

    @abstractmethod
    def _generate_candidates(self) -> Generator[Dict, None, None]:
        """
        Yield candidate parameter dictionaries from the search space.

        Returns
        -------
        Generator[Dict, None, None]
            Generator of candidate parameter dictionaries.
        """
        raise NotImplementedError
