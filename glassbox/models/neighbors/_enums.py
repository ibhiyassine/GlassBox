from enum import Enum


class DistanceMetric(Enum):
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"


class SearchAlgorithm(Enum):
    BRUTE_FORCE = "brute_force"
    KD_TREE = "kd_tree"
