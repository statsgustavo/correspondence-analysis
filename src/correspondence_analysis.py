from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.contingency_table import ContingencyTable
from src.types import TableType


@dataclass
class OneDimensionResults:
    """
    Container class for storing correspondence analysis statistics of one dimension
    of a contingency table for correspondence analysis.

    :param mass:
    :param distance:
    :param inertia:
    :param factor_scores:
    :param cor:
    :param ctr:
    :param angle:
    """

    mass: np.ndarray
    distance: np.ndarray
    inertia: np.ndarray
    factor_scores: np.ndarray
    cor: np.ndarray
    ctr: np.ndarray
    angle: np.ndarray


class CorrespondenceAnalysis(ContingencyTable):
    """Correspondence analisys class."""

    def __init__(self, table: TableType):
        """
        Loads contingency table and runs corresponce analysis on it.

        :param table:
            Contingency table for which correspondence analysis will be performed.
        """
        super(CorrespondenceAnalysis, self).__init__(table)
        self._fit()

    def _fit(self):
        row_weights, column_weights = self._weights()
        row_distances, column_distances = self._distance()
        row_inertia, column_inertia = self._inertia((row_distances, column_distances))

        self.rows, self.columns = (
            OneDimensionResults(row_weights, row_distances, row_inertia, *([None] * 4)),
            OneDimensionResults(
                column_weights, column_distances, column_inertia, *([None] * 4)
            ),
        )

    def _weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Diagonal matrix of the inverse square roots of the row/column proportions
        relative to the whole table.
        """
        return tuple(
            map(
                lambda p: np.diag(1 / np.sqrt(p)),
                [self.rows.proportions.ravel(), self.columns.proportions.ravel()],
            )
        )

    def _mass(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Proportion of the whole table in the category represented by the row/column.
        """
        return self.rows.proportions, self.columns.proportions

    def _distance(self) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted distance from row/column profile to the average axis' profile."""
        row_squared_differences = np.square(
            (self.table_proportions / self.rows.proportions)
            - self.columns.proportions.T
        )

        column_square_differences = np.square(
            (self.table_proportions / self.columns.proportions.T)
            - self.rows.proportions
        )

        row_distances = (row_squared_differences / self.columns.proportions.T).sum(
            1, keepdims=True
        )
        column_distances = (
            (column_square_differences / self.rows.proportions).sum(0, keepdims=True).T
        )

        return row_distances, column_distances

    def _inertia(self, distances: Tuple[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Weighted average of chi-squared distance between row/column profile and their
        average profile.

        :param distances: a tuple of np.ndarray containing the distances from each
        row/column profiles to their axis' average profile.
        """
        row_distances, column_distances = distances

        row_inertia = (
            row_distances
            * self.rows.proportions
            / np.sum(row_distances * self.rows.proportions)
        )

        column_inertia = (
            column_distances
            * self.columns.proportions
            / np.sum(column_distances * self.columns.proportions)
        )
        return row_inertia, column_inertia

    def _profile_correlation(self):
        """
        Correlation between row/column profile and their axis. Morover it is the
        proportion of variance in a profile explained by the axis.
        """

    def _profile_contribution(self):
        """
        The contribution of the row/column profile to the inertia of its axis.
        """

    def _angle(self):
        """Angle between the axis and the profile."""
