from dataclasses import dataclass
from typing import Tuple

import numpy as np
from src.correspondence.contingency_table import ContingencyTable
from src.types import TableType


@dataclass
class OneDimensionResults:
    """
    Container class for storing correspondence analysis statistics of one dimension
    of a contingency table for correspondence analysis.

    :param names:
    :param mass:
    :param weights:
    :param distance:
    :param inertia:
    :param factor_scores:
    :param cor:
    :param ctr:
    """

    name: np.ndarray
    mass: np.ndarray
    weight: np.ndarray
    distance: np.ndarray
    inertia: np.ndarray
    factor_score: np.ndarray
    cor: np.ndarray
    ctr: np.ndarray


@dataclass
class CorrespondenceAnalysisResults:
    """Container class for correspondence analysis results."""

    row: OneDimensionResults
    column: OneDimensionResults


class BaseCorrespondenceAnalysis(ContingencyTable):
    """Correspondence analisys base class."""

    def __init__(self, table: TableType):
        """
        Loads contingency table and computes some of its statistics of interest for
        correspondence analysis.

        :param table:
            Contingency table for which correspondence analysis will be performed.
        """
        super(BaseCorrespondenceAnalysis, self).__init__(table)

    def weights(self) -> Tuple[np.ndarray, np.ndarray]:
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

    def mass(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Proportion of the whole table in the category represented by the row/column.
        """
        return self.rows.proportions, self.columns.proportions

    def distance(self) -> Tuple[np.ndarray, np.ndarray]:
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

    def inertia(self, distances: Tuple[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
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
