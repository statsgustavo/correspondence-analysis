import numpy as np
from src.correspondence_analysis import CorrespondenceAnalysis


class TestCorrespondenceAnalysis:
    """Tests for correspondence analysis implementation."""

    def test_weights_matrix(self, array, weights):
        """Tests calculation of diagonal weights matrix."""
        expected_row, expected_column = weights
        ca = CorrespondenceAnalysis(array.data)
        row, column = ca.profiles.row.mass, ca.profiles.column.mass

        assert row.shape == expected_row.shape
        assert column.shape == expected_column.shape

        assert np.allclose(row, expected_row)
        assert np.allclose(column, expected_column)

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

        assert np.allclose(row, expected_row)
        assert np.allclose(column, expected_column)

    def test_inertia(self, array, inertia):
        """Tests calculation of row/column profiles inertia."""
        expected_row, expected_column = inertia
        ca = CorrespondenceAnalysis(array.data)
        row, column = ca.profiles.row.inertia, ca.profiles.column.inertia

        assert row.shape == expected_row.shape
        assert column.shape == expected_column.shape

        assert np.allclose(row, expected_row)
        assert np.allclose(column, expected_column)

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
            ca.standardized_residuals_matrix, standardized_residuals_matrix
        )
