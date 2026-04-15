from dataclasses import dataclass

import numpy as np

from .._enums import DistanceMetric
from ._base import BaseIndex


@dataclass
class KDNode:
    __slots__ = ("point_idx", "axis", "left", "right")
    point_idx: int | list[int]
    axis: int
    left: "KDNode | None"
    right: "KDNode | None"


class KDTreeIndex(BaseIndex):
    __slots__ = ("root", "leaf_size", "X_train")

    def __init__(self, metric: DistanceMetric, leaf_size: int = 30) -> None:
        """
        Initialize the KDTreeIndex.

        Parameters
        ----------
        metric : DistanceMetric
            The distance metric to use.
        leaf_size : int, default=30
            Number of points at which to switch to brute-force.
        """
        super().__init__(metric)
        self.root: KDNode | None = None
        self.leaf_size: int = leaf_size
        self.X_train: np.ndarray | None = None

    def build(self, X: np.ndarray) -> None:
        """
        Build the KD-Tree structure from the training data.

        Parameters
        ----------
        X : np.ndarray
            Training data, shape (n_samples, n_features).
        """
        self.X_train = np.asarray(X, dtype=float)
        indices = np.arange(len(self.X_train))
        self.root = self._build_tree(indices, depth=0)

    def query(self, x: np.ndarray, k: int) -> np.ndarray:
        """
        Query the KD-Tree for the k nearest neighbors.

        Parameters
        ----------
        x : np.ndarray
            Query point(s), shape (n_features,) or (n_queries, n_features).
        k : int
            Number of nearest neighbors to retrieve.

        Returns
        -------
        np.ndarray
            Indices of the k nearest neighbors.
        """
        if self.X_train is None or self.root is None:
            raise ValueError("Index is not built yet")

        x_arr = np.asarray(x, dtype=float)
        single_query = x_arr.ndim == 1
        if single_query:
            x_arr = x_arr.reshape(1, -1)

        n_queries = x_arr.shape[0]
        nearest_idx = np.empty((n_queries, k), dtype=int)

        for i in range(n_queries):
            best_k = []
            self._search(self.root, x_arr[i], k, best_k)
            nearest_idx[i] = [idx for _, idx in best_k]

        if single_query:
            return nearest_idx[0]
        return nearest_idx

    def _build_tree(self, indices: np.ndarray, depth: int) -> KDNode | None:
        """
        Recursively build the KD-Tree.

        Parameters
        ----------
        indices : np.ndarray
            Indices of the dataset.
        depth : int
            Current depth in the tree.

        Returns
        -------
        KDNode
            Root node of the constructed subtree.
        """
        if len(indices) == 0:
            return None

        if len(indices) <= self.leaf_size:
            return KDNode(point_idx=indices.tolist(), axis=-1, left=None, right=None)

        assert self.X_train is not None
        axis = depth % self.X_train.shape[1]
        sorted_idx = indices[np.argsort(self.X_train[indices, axis])]

        mid = len(sorted_idx) // 2
        median_idx = sorted_idx[mid]

        left_indices = sorted_idx[:mid]
        right_indices = sorted_idx[mid + 1 :]

        return KDNode(
            point_idx=median_idx,
            axis=axis,
            left=self._build_tree(left_indices, depth + 1),
            right=self._build_tree(right_indices, depth + 1),
        )

    def _distance(self, p1: np.ndarray, p2: np.ndarray) -> float:
        diff = p1 - p2
        if self.metric == DistanceMetric.EUCLIDEAN:
            return float(np.sqrt(np.sum(diff**2)))
        elif self.metric == DistanceMetric.MANHATTAN:
            return float(np.sum(np.abs(diff)))
        raise ValueError(f"Unsupported metric: {self.metric}")

    def _search(self, node: KDNode | None, x: np.ndarray, k: int, best_k: list) -> None:
        """
        Recursive search helper targeting localized subsets in bounding dimensions.
        """
        if node is None or self.X_train is None:
            return

        if isinstance(node.point_idx, list):
            for idx in node.point_idx:
                dist = self._distance(x, self.X_train[idx])
                self._update_best_k(best_k, k, dist, idx)
            return

        idx = node.point_idx
        axis = node.axis

        dist = self._distance(x, self.X_train[idx])
        self._update_best_k(best_k, k, dist, idx)

        diff_axis = x[axis] - self.X_train[idx, axis]
        axis_dist = abs(diff_axis)

        first, second = node.left, node.right
        if diff_axis > 0:
            first, second = node.right, node.left

        self._search(first, x, k, best_k)

        if len(best_k) < k or axis_dist < best_k[-1][0]:
            self._search(second, x, k, best_k)

    def _update_best_k(self, best_k: list, k: int, dist: float, idx: int) -> None:
        best_k.append((dist, idx))
        best_k.sort(key=lambda item: item[0])
        if len(best_k) > k:
            best_k.pop()
