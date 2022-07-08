import numpy as np
from src.correspondence_analysis import CorrespondenceAnalysis


class TestCorrespondenceAnalysis:
    """Tests for correspondence analysis implementation."""

    def test_weights_matrix(self, array, weights):
        """Tests calculation of diagonal weights matrix."""
        expected_row, expected_column = weights
        ca = CorrespondenceAnalysis(array.data)
        row, column = ca.profiles.row.weights, ca.profiles.column.weights

        assert row.shape == expected_row.shape
        assert column.shape == expected_column.shape

        assert np.allclose(row, expected_row, atol=1e-3)
        assert np.allclose(column, expected_column, atol=1e-3)

    def test_distances(self, array, distances):
        """
        Tests calculation of row/column profiles distances to their axis' average
        profile.
        """
        expected_row, expected_column = distances
        ca = CorrespondenceAnalysis(array.data)
        row, column = ca.profiles.row.distance, ca.profiles.column.distance

        assert row.shape == expected_row.shape
        assert column.shape == expected_column.shape

        assert np.allclose(row, expected_row, atol=1e-4)
        assert np.allclose(column, expected_column, atol=1e-4)

    def test_inertia(self, array, inertia):
        """Tests calculation of row/column profiles inertia."""
        expected_row, expected_column = inertia
        ca = CorrespondenceAnalysis(array.data)
        row, column = ca.profiles.row.inertia, ca.profiles.column.inertia

        assert row.shape == expected_row.shape
        assert column.shape == expected_column.shape

        assert np.allclose(row, expected_row / expected_row.sum(), atol=1e-2)
        assert np.allclose(column, expected_column / expected_column.sum(), atol=1e-2)

    def test_standardized_residuals_matrix_shape(
        self, array, standardized_residuals_matrix
    ):
        """Tests shape of standardized residuals matrix."""
        ca = CorrespondenceAnalysis(array.data)
        assert (
            ca.standardized_residuals_matrix.shape
            == standardized_residuals_matrix.shape
        )

    def test_standardized_residuals_matrix_values(
        self, array, standardized_residuals_matrix
    ):
        """Tests correctness of standardized residuals matrix calculations."""
        ca = CorrespondenceAnalysis(array.data)

        assert np.allclose(
            ca.standardized_residuals_matrix, standardized_residuals_matrix, atol=1e-4
        )

    def test_factor_scores_shape(self, array, factor_scores):
        """Test factor scores matrices shapes."""
        row_expected, column_expected = factor_scores
        ca = CorrespondenceAnalysis(array.data)
        assert ca.profiles.row.factor_scores.shape[0] == row_expected.shape[0]
        assert ca.profiles.column.factor_scores.shape[0] == column_expected.shape[0]

    def test_factor_scores_values(self, array, factor_scores):
        """Test correctness of factor scores matrices values."""
        row_expected, column_expected = factor_scores
        ca = CorrespondenceAnalysis(array.data)
        assert np.allclose(
            ca.profiles.row.factor_scores[:, :2], row_expected, atol=1e-3
        )
        assert np.allclose(
            ca.profiles.column.factor_scores[:, :2], column_expected, atol=1e-3
        )

    def test_profile_correlation_shape(self, array, profile_correlation):
        """Test profile correlation matrices shapes."""
        row_expected, column_expected = profile_correlation
        ca = CorrespondenceAnalysis(array.data)
        assert ca.profiles.row.cor.shape[0] == row_expected.shape[0]
        assert ca.profiles.column.cor.shape[0] == column_expected.shape[0]

    def test_profile_correlation_values(self, array, profile_correlation):
        """Test correctness of profile correlation values."""
        row_expected, column_expected = profile_correlation
        ca = CorrespondenceAnalysis(array.data)
        assert np.allclose(ca.profiles.row.cor[:, :2], row_expected, atol=1e-3)
        assert np.allclose(ca.profiles.column.cor[:, :2], column_expected, atol=1e-3)

    def test_profile_contribution_shape(self, array, profile_contribution):
        """Test profile contribution matrices shapes."""
        row_expected, column_expected = profile_contribution
        ca = CorrespondenceAnalysis(array.data)
        assert ca.profiles.row.ctr.shape[0] == row_expected.shape[0]
        assert ca.profiles.column.ctr.shape[0] == column_expected.shape[0]

    def test_profile_contribution_values(self, array, profile_contribution):
        """Test correctness of profile contribution values."""
        row_expected, column_expected = profile_contribution
        ca = CorrespondenceAnalysis(array.data)
        assert np.allclose(ca.profiles.row.ctr[:, :2], row_expected, atol=1e-3)
        assert np.allclose(ca.profiles.column.ctr[:, :2], column_expected, atol=1e-3)
