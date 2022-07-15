import functools as ft
from typing import Callable, Union

import numpy as np
import pandas as pd
from sklearn import metrics

from .errors import InvalidMetricError


class MetricMultidimensionalScaling:
    """
    Metric Multidimensional Scaling class.

    Starting from a set of `n` vectors with dimension `p` or a matrix of interpoint
    distances between every pair of the set of n points, metric multidimensional
    scaling aims to find a lower-dimensional representation of the n vectors such
    that the distances in the lower-dimensional representation reasonably
    approximates the pointwise distances in the original dimension.

    :param data:
        Can be either a pandas.DataFrame or a numpy.ndarray object containing the `n`
        datapoints in their original space or a `n x n` matrix of distances between each
        pair of points. Is the number of rows and columns of the object do not match
        then the distance matrix for each pair of rows will be computed, otherwise, it
        will implied that the data passed is a matrix of distances.
    :param n_coordinates:
        Integer number of dimensions of the lower-dimensional representation of the
        data.
    :param metric:
        Distance metric used to compute interpoint distances between points. Allowed
        metrics are those pairwise metrics as in `sklearn.metrics.pairwise`.
    """

    VALID_METRICS = [
        "euclidean",
        "manhattan",
        "l1",
        "l2",
        "cosine",
        "chebyshev",
        "hamming",
        "canberra",
        "braycurtis",
    ]

    def __init__(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        n_coordinates: int = 2,
        metric: str = "euclidean",
    ):
        self._original_data = data
        self.data = data.values if isinstance(data, pd.DataFrame) else data
        self._n_coordinates = n_coordinates
        self.fn_metric = self._get_distance_metric_function(metric)

        if self._check_if_matrix_is_square(self.data):
            self.distances = data.values
        else:
            self.distances = self._compute_interpoint_distances(
                self.fn_metric, self.data
            )

    def _check_if_matrix_is_square(self, data: np.ndarray) -> bool:
        """Checks if the number of rows and columns of the data matrix match."""
        nrows, ncols = data.shape
        return nrows == ncols

    def _compute_interpoint_distances(
        self, fn_: Callable[[np.ndarray], np.ndarray], data: np.ndarray
    ):
        """
        Compute pairwise distances for each of the data points according to the chosen
        metric.
        """
        distances = fn_(data)
        return distances

    def _get_distance_metric_function(
        self, metric: str
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        if metric not in self.VALID_METRICS:
            raise InvalidMetricError(metric, self.VALID_METRICS)
        else:
            fn_distance_metric = ft.partial(metrics.pairwise_distances, metric=metric)

        return fn_distance_metric
