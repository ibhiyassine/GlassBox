import numpy as np
from typing import Callable, Dict, Generator

from glassbox.orchestrator.base_search import BaseSearch


class RandomizedSearchCV(BaseSearch):
    """
    Randomized search over a parameter space.

    Parameters
    ----------
    estimator : BaseModel
        The model to optimize.
    param_space : Dict
        Distribution for random search.
    cv_engine : BaseSplitter
        Cross-validation splitter.
    scoring_func : Callable
        Scoring function used to evaluate candidates.
    n_iter : int
        Number of random parameter candidates to evaluate.
    time_budget : float
        Maximum time budget for the search.
    """

    def __init__(
        self,
        estimator: "BaseModel",
        param_space: Dict,
        cv_engine: "BaseSplitter",
        scoring_func: "Callable",
        n_iter: int = 10,
        time_budget: float = 0.0,
    ) -> None:
        super().__init__(estimator, param_space, cv_engine, scoring_func)
        self.n_iter: int = n_iter
        self.time_budget: float = time_budget

    def _generate_candidates(self) -> Generator[Dict, None, None]:
        """
        Generate random parameter candidates from the search space.

        Returns
        -------
        Generator[Dict, None, None]
            Generator of candidate parameter dictionaries.
        """
        if self.n_iter <= 0:
            return

        keys = list(self.param_space.keys())
        values = [self.param_space[key] for key in keys]

        for _ in range(self.n_iter):
            candidate: Dict = {}
            for key, choices in zip(keys, values):
                if isinstance(choices, np.ndarray):
                    candidate[key] = np.random.choice(choices)
                else:
                    candidate[key] = np.random.choice(np.asarray(choices))
            yield candidate
