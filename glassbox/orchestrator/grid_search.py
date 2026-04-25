from itertools import product
from typing import Dict, Generator

from glassbox.orchestrator.base_search import BaseSearch


class GridSearchCV(BaseSearch):
    """
    Exhaustive grid search over a parameter space.

    Parameters
    ----------
    estimator : BaseModel
        The model to optimize.
    param_space : Dict
        Parameter grid for exhaustive search.
    cv_engine : BaseSplitter
        Cross-validation splitter.
    scoring_func : Callable
        Scoring function used to evaluate candidates.
    """

    def _generate_candidates(self) -> Generator[Dict, None, None]:
        """
        Generate every combination of parameters from the search space.

        Returns
        -------
        Generator[Dict, None, None]
            Generator of candidate parameter dictionaries.
        """
        keys = list(self.param_space.keys())
        values = [self.param_space[key] for key in keys]

        if len(keys) == 0:
            yield {}
            return

        for combination in product(*values):
            yield {key: value for key, value in zip(keys, combination)}
