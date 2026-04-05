from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


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


@dataclass
class OutlierInfo:
    """
    Stores outlier bounds and count for a single feature.
    """

    count: int
    lower_bound: float
    upper_bound: float


@dataclass
class FeatureStats:
    """
    Stores basic summary statistics for a feature.
    """

    mean: float
    median: float
    mode: float | str
    std: float
    skew: float
    kurt: float


@dataclass
class EDAReport:
    """
    Container for the complete Exploratory Data Analysis report.
    """

    feature_types: Dict[str, FeatureType]
    missing_values: Dict[str, int]  # name of the feature: number of missing values
    outliers_info: Dict[str, OutlierInfo]
    summary_stats: Dict[str, FeatureStats]
    collinearity_map: List[CollinearityPair]

    def to_json(self) -> str:
        """
        Serialize the entire EDA report down to a JSON string.

        Returns
        -------
        str
            JSON representation of the report.
        """
        raise NotImplementedError
