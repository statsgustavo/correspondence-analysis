from __future__ import annotations

import functools as ft
from typing import Callable, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import metrics

from ..visualization import multidimensional_scaling_2d_plot
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
            self.distances = self.data
        else:
            self.distances = self._compute_interpoint_distances(
                self.fn_metric, self.data
            )

        self._eigenvalues, self._eigenvectors = self._fit()

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

    def _spectral_decomposition(
        self, matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        return eigenvalues, eigenvectors

    def _lower_dimensional_coordinates(self, eigenvalues, eigenvectors):
        is_positive = np.argwhere(eigenvalues > 1e-10).ravel()
        eigenvalues, eigenvectors = (
            eigenvalues[is_positive],
            eigenvectors[:, is_positive],
        )
        lower_dimensional_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        return (
            eigenvalues[: self._n_coordinates],
            lower_dimensional_matrix[:, : self._n_coordinates],
        )

    def _fit(self):
        nobs = self.distances.shape[0]
        proximity_matrix = -0.5 * np.square(self.distances)

        centering_matrix = np.eye(nobs) - (1 / nobs) * np.ones_like(proximity_matrix)
        double_centered_matrix = centering_matrix @ proximity_matrix @ centering_matrix

        eigenvalues, eigenvectors = self._lower_dimensional_coordinates(
            *self._spectral_decomposition(double_centered_matrix)
        )
        return eigenvalues, eigenvectors

    @property
    def explained_variance(self):
        """
        Returns eigenvalues from spectral decomposition of the double centered distance
        matrix.
        """
        return self._eigenvalues

    @property
    def principal_coordinates(self):
        """
        Returns the first `n_cooordinates` eigenvectors corresponding to the greatest
        positive-valued eigen values from the spectral decoposition of the double
        centered distance matrix.
        """
        return self._eigenvectors

    def plot2d(self, annotate=False):
        """Two-dimensioal graphic representation of the data."""
        x, y = self.principal_coordinates[:, 0], self.principal_coordinates[:, 1]
        annotations = self._original_data.index if annotate else None
        splot = multidimensional_scaling_2d_plot(x, y, annotations)
        return splot
