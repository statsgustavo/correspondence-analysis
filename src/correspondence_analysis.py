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
    :param inertia:
    :param distance:
    :param factor_scores:
    :param cor:
    :param ctr:
    :param angle:
    """

    mass: np.ndarray
    inertia: np.ndarray
    distance: np.ndarray
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
        self.rows, self.columns = (
            OneDimensionResults(row_weights, *([None] * 6)),
            OneDimensionResults(column_weights, *([None] * 6)),
        )

    def _weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Proportion of the whole table in the category represented by the row/column.
        """
        return tuple(
            map(
                lambda p: np.diag(1 / np.sqrt(p)),
                [self.rows.proportions, self.columns.proportions],
            )
        )

    def _inertia(self):
        """
        Weighted average of chi-squared distance between row/column profile and their
        average profile.
        """

    def _distance(self):
        """Weighted distance from row/column profile to the average axis' profile."""

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
