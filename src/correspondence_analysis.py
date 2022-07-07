from dataclasses import dataclass
from typing import Tuple

import numpy as np

from src.base import BaseCorrespondenceAnalysis


@dataclass
class OneDimensionResults:
    """
    Container class for storing correspondence analysis statistics of one dimension
    of a contingency table for correspondence analysis.

    :param weights:
    :param mass:
    :param distance:
    :param inertia:
    :param factor_scores:
    :param cor:
    :param ctr:
    :param angle:
    """

    mass: np.ndarray
    weights: np.ndarray
    distance: np.ndarray
    inertia: np.ndarray
    factor_scores: np.ndarray
    cor: np.ndarray
    ctr: np.ndarray
    angle: np.ndarray


@dataclass
class CorrespondenceAnalysisResults:
    """Container class for correspondence analysis results."""

    row: OneDimensionResults
    column: OneDimensionResults


class CorrespondenceAnalysis(BaseCorrespondenceAnalysis):
    """Correspondence analysis class."""

    def __init__(self, table):
        """
        Loads contingency table and computes correspondence analysis.

        :param table:
            Contingency table for which correspondence analysis will be performed.
        """
        super(CorrespondenceAnalysis, self).__init__(table)
        self._fit()

    def _fit(self):
        """
        Runs correspondence analysis for the inputed contingency table.
        """
        row_mass, column_mass = self.mass()
        row_weights, column_weights = weights = self.weights()
        row_distance, column_distance = distances = self.distance()
        row_inertia, column_inertia = self.inertia(distances)

        self.standardized_residuals_matrix = self._standardized_residuals_matrix(
            weights
        )

        (
            left_singular_vectors,
            singular_values,
            right_singular_vectors,
        ) = np.linalg.svd(self.standardized_residuals_matrix, full_matrices=False)

        row_scores, column_scores = (
            self._factor_scores(row_weights, singular_values, left_singular_vectors),
            self._factor_scores(
                column_weights, singular_values, right_singular_vectors.T
            ),
        )

        row_cor, column_cor = (
            self._profile_correlation(row_scores, row_distance),
            self._profile_correlation(column_scores, column_distance),
        )

        row_ctr, column_ctr = (
            self._profile_contribution(row_mass, row_scores, singular_values),
            self._profile_contribution(column_mass, column_scores, singular_values),
        )

        self.profiles = CorrespondenceAnalysisResults(
            OneDimensionResults(
                row_mass,
                row_weights,
                row_distance,
                row_inertia,
                row_scores,
                row_cor,
                row_ctr,
                *([None] * 1)
            ),
            OneDimensionResults(
                column_mass,
                column_weights,
                column_distance,
                column_inertia,
                column_scores,
                column_cor,
                column_ctr,
                *([None] * 1)
            ),
        )

    def _standardized_residuals_matrix(
        self, weights: Tuple[np.ndarray, np.ndarray]
    ) -> np.ndarray:
        """
        Computes the matrix to be decomposed using singular value decomposition
        such that the factor scores for row/column profiles can be obtained with
        its resulting projection matrices.
        """
        row, column = weights
        expected_proportions = np.outer(self.rows.proportions, self.columns.proportions)
        return row @ (self.table_proportions - expected_proportions) @ column

    def _factor_scores(self, weights, singular_values, singular_vectors):
        """
        Computes the row/column factos scores (aka principal coodinates).

        :param weights: diagonal matrix of row/colulmn profile weights.
        :param singular_values: vector of singular values obtained from svd
        decomposition.
        :param singular_vectors: matrix of singular vectors (either left or right) from
        svd decomposition.

        :return factor_scores: matrix of row/column factor scores.
        """
        factor_scores = weights @ singular_vectors @ np.diag(singular_values)

        return factor_scores

    def _profile_correlation(self, factors, distances):
        """
        Correlation between row/column profile and their axis. Morover it is the
        proportion of variance in a profile explained by the axis.
        """
        return np.square(factors) / distances

    def _profile_contribution(self, mass, factors, singular_values):
        """
        The contribution of the row/column profile to the inertia of its axis.
        """
        eigenvalues = np.square(singular_values).reshape(-1, 1)
        return mass * np.square(factors) / eigenvalues.T

    def _angle(self):
        """Angle between the axis and the profile."""
