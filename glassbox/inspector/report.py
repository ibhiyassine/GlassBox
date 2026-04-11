import dataclasses
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import numpy as np


class FeatureType(Enum):
    """
    Enumerates possible statistical data types for features.
    """

    NUMERICAL = 1
    NOMINAL = 2
    ORDINAL = 3
    BOOLEAN = 4


@dataclass
class CollinearityPair:
    """
    Represents an association or correlation between two features.
    """

    feature_a: str
    feature_b: str
    score: float
    metric: str


@dataclass
class OutlierInfo:
    """
    Stores outlier bounds and count for a single feature.
    """

    count: int
    lower_bound: float
    upper_bound: float


@dataclass
class MissingInfo:
    """
    Stores missing values count and percentage.
    """

    count: int
    percentage: float


@dataclass
class NumericStats:
    """
    Stores basic summary statistics for a numerical feature.
    """

    mean: float
    median: float
    std: float
    skew: float
    kurt: float


@dataclass
class CategoricalStats:
    """
    Stores summary statistics for a categorical feature.
    """

    mode: str | float
    cardinality: int


@dataclass
class EDAReport:
    """
    Container for the complete Exploratory Data Analysis report.
    """

    feature_types: Dict[str, FeatureType]
    missing_values: Dict[str, MissingInfo]  # name of the feature: MissingInfo
    outliers_info: Dict[str, OutlierInfo]
    summary_stats: Dict[str, Union[NumericStats, CategoricalStats]]
    collinearity_map: List[CollinearityPair]

    def to_json(self) -> str:
        """
        Serialize the entire EDA report down to a JSON string.

        Returns
        -------
        str
            JSON representation of the report.
        """
        class EnumEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Enum):
                    return obj.name
                if isinstance(obj, float) and np.isnan(obj):
                    return None
                return super().default(obj)
        
        # We need to handle nan to null if missing, but json.dumps handles nan by default to NaN.
        # But JSON standard doesn't support NaN, so let's allow it standard.
        return json.dumps(dataclasses.asdict(self), cls=EnumEncoder)
