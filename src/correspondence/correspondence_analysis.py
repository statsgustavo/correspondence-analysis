from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src import reports
from src.correspondence.base import (
    BaseCorrespondenceAnalysis,
    CorrespondenceAnalysisResults,
    OneDimensionResults,
)
from src.visualization import plot_profile_coordiantes


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
        self.total_inertia = self.eigenvalues.sum()

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
                self.rows.levels,
                row_mass,
                row_weights,
                row_distance,
                row_inertia,
                row_scores,
                row_cor,
                row_ctr,
            ),
            self._set_profile(
                self.columns.levels,
                column_mass,
                column_weights,
                column_distance,
                column_inertia,
                column_scores,
                column_cor,
                column_ctr,
            ),
        )

        (
            self.inertia_summary,
            self.rows_summary,
            self.columns_summary,
        ) = self._summary_tables()

    def _set_profile(
        self, name, mass, weight, distance, inertia, score, correlation, contribution
    ):
        """
        Creates instance of OneDimentionResults object with the corresponding row/column
        profiles statistics.
        """
        return OneDimensionResults(
            name, mass, weight, distance, inertia, score, correlation, contribution
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

    def _profile_summary(self, profiles: OneDimensionResults) -> pd.DataFrame:
        """
        Build row/column profile summary as a pandas.DataFrame.

        :param profiles: OneDimensionResults object for either rows or columns.
        :returns table: pandas.DataFrame summary with mass, inertia, score, contribution
        and correlation for each row/column profile.
        """
        column_names = ["Mass", "Inertia", "F1", "F2", "CTR1", "CTR2", "COR1", "COR2"]
        table = pd.DataFrame(
            np.column_stack(
                [
                    profiles.mass,
                    profiles.inertia,
                    profiles.factor_score[:, :2],
                    profiles.ctr[:, :2],
                    profiles.cor[:, :2],
                ]
            ),
            index=profiles.name.ravel(),
            columns=column_names,
        )
        return table

    def _inertia_summary(self):
        """
        Summary table of inertia for the first two dimensions of row and column
        profiles. Since rows and columns share the same eigenvalues, the results are
        exactly the same for the dimensions of rows or columns.

        :returns table: pandas.DataFrame summary with inertia (eigenvalue), chi-square,
        percentage of explained variance and its cumulative value.
        """
        column_names = ["Inertia", "Chi-square", "Percent"]

        n = self.table.sum()
        eigenvalues = self.eigenvalues.ravel()[:2]

        table = pd.DataFrame(
            np.column_stack(
                [
                    eigenvalues,
                    n * eigenvalues,
                    eigenvalues / self.total_inertia,
                ]
            ),
            index=["Dimension 1", "Dimension 2"],
            columns=column_names,
        )
        totals = pd.DataFrame(
            table.sum(0).values, columns=["Total"], index=column_names
        )
        return pd.concat([table, totals.T], axis=0)

    def _summary_tables(self):
        eigenvalues, rows, columns = (
            self._inertia_summary(),
            self._profile_summary(self.profiles.row),
            self._profile_summary(self.profiles.column),
        )
        return eigenvalues, rows, columns

    def summary(self, precision=3):
        """Shows summary tables of the correspondence analysis."""
        report = reports.Report(
            [
                reports.Table("Inertia", self.inertia_summary.round(precision)),
                reports.Table("Row profiles", self.rows_summary.round(precision)),
                reports.Table("Column profiles", self.columns_summary.round(precision)),
            ]
        )
        return report.render()

    def plot_factors(self):
        """
        Plots the first two coordinates for rows and colum profiles. Row and column
        coordinates are superimposed.
        """
        _ = plt.figure(figsize=(10, 10))
        splot = plt.subplot(111)
        splot = plot_profile_coordiantes(self.profiles.row, self.profiles.column, splot)
        return splot
