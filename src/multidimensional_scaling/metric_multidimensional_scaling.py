from typing import Callable

import numpy as np
from sklearn.metrics import pairwise

from .errors import InvalidMetricError


def get_distance_metric_function(
    metric: str,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    metrics_ = pairwise.PAIRED_DISTANCES
    if metric not in metrics_:
        raise InvalidMetricError(metric)
    return metrics_[metric]


class MultidimensionalMetricScaling:
    def __init__(self, n_coordinates=2, metric="euclidean"):
        """
        Starting from a set of n vectors with dimension p or pointwise distances between
        every pair of the set of n points, metric multidimensional scaling aims to find
        a lower-dimensional representation of the n vectors such that the distances in
        the lower-dimensional representation reasonably approximates the pointwise
        distances in the original dimension.

        :param n_coordinates: Integer number of dimensions of the lower-dimensional
        representation of the data.
        :param metric: Distance metric used to compute interpoint distances between
        points. Allowed metrics are those pairwise metrics as in `sklearn.metrics.pairwise`.
        """
        self._n_coordinates = n_coordinates
        self._metric = metric
