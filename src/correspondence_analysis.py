from dataclasses import dataclass
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt

from src.base import BaseCorrespondenceAnalysis


@dataclass
class OneDimensionResults:
    """
    Container class for storing correspondence analysis statistics of one dimension
    of a contingency table for correspondence analysis.

    :param mass:
    :param weights:
    :param distance:
    :param inertia:
    :param factor_scores:
    :param cor:
    :param ctr:
    """

    mass: np.ndarray
    weights: np.ndarray
    distance: np.ndarray
    inertia: np.ndarray
    factor_scores: np.ndarray
    cor: np.ndarray
    ctr: np.ndarray


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

    def _generalized_singular_value_decomposition(
        self, standardized_residuals_matrix: np.array
    ) -> Tuple[np.array, np.array, np.array]:
        """Computes standardized residuals matrix SVD decomposition."""
        U, S, Vt = np.linalg.svd(standardized_residuals_matrix, full_matrices=False)
        return U, S, Vt

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
        ) = self._generalized_singular_value_decomposition(
            self.standardized_residuals_matrix
        )

        self.eigenvalues = self._eigenvalues(singular_values)

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
            self._profile_contribution(row_mass, row_scores, self.eigenvalues),
            self._profile_contribution(column_mass, column_scores, self.eigenvalues),
        )

        self.profiles = CorrespondenceAnalysisResults(
            self._set_profile(
                row_mass,
                row_weights,
                row_distance,
                row_inertia,
                row_scores,
                row_cor,
                row_ctr,
            ),
            self._set_profile(
                column_mass,
                column_weights,
                column_distance,
                column_inertia,
                column_scores,
                column_cor,
                column_ctr,
            ),
        )

    def _set_profile(
        self, mass, weights, distances, inertia, scores, correlation, contribution
    ):
        """
        Creates instance of OneDimentionResults object with the corresponding row/column
        profiles statistics.
        """
        return OneDimensionResults(
            mass, weights, distances, inertia, scores, correlation, contribution
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

    def _eigenvalues(self, singular_values):
        """Computes factors eigenvalues."""
        return np.square(singular_values).reshape(-1, 1)

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

    def _profile_contribution(self, mass, factors, eigenvalues):
        """
        The contribution of the row/column profile to the inertia of its axis.
        """
        return mass * np.square(factors) / eigenvalues.T

    def _angle(self):
        """Angle between the axis and the profile."""

    def plot_factors(self):
        """
        Plots the first two coordinates for rows and colum profiles. Row and column
        coordinates are superimposed.
        """
        _ = plt.figure(figsize=(10, 10))
        splot = plt.subplot(111)

        splot.scatter(
            self.profiles.row.factor_scores[:0],
            self.profiles.row.factor_scores[:1],
            s=1000 * self.profiles.row.inertia,
            label="Row profiles",
            color="C0",
        )
        for i, profile_name in enumerate(self.rows.levels):
            f1, f2 = self.profiles.row.factor_scores[i, :2]
            splot.annotate(
                profile_name(f1, f2),
                (f1 + 0.01, f2 + 0.01),
                xycoords="data",
                textcoords="offset points",
                verticalalignment="bottom",
            )

        splot.scatter(
            self.profiles.column.factor_scores[:0],
            self.profiles.column.factor_scores[:1],
            s=1000 * self.profiles.column.inertia,
            label="Column profiles",
            color="C1",
        )

        for i, profile_name in enumerate(self.columns.levels):
            g1, g2 = self.profiles.column.factor_scores[i, :2]
            splot.annotate(
                profile_name,
                (g1, g2),
                xycoords="data",
                textcoords="offset points",
                verticalalignment="bottom",
            )

        splot.spines["left"].set_position("center")
        splot.spines["bottom"].set_position("center")
        splot.spines["right"].set_visible(False)
        splot.spines["top"].set_visible(False)
        splot.get_xaxis().set_ticks([])
        splot.get_yaxis().set_ticks([])

        return splot
