import numpy as np
from src.correspondence_analysis import CorrespondenceAnalysis


class TestCorrespondenceAnalysis:
    """Tests for correspondence analysis implementation."""

    def test_weights_matrix(self, array, weights):
        """Tests calculation of diagonal weights matrix."""
        expected_row, expected_column = weights
        ca = CorrespondenceAnalysis(array.data)
        row, column = ca.rows.mass, ca.columns.mass

        assert row.shape == expected_row.shape
        assert column.shape == expected_column.shape

        assert np.allclose(row, expected_row)
        assert np.allclose(column, expected_column)

    def test_distances(self, array, distances):
        """Tests calculation of diagonal weights matrix."""
        expected_row, expected_column = distances
        ca = CorrespondenceAnalysis(array.data)
        row, column = ca.rows.distance, ca.columns.distance

        assert row.shape == expected_row.shape
        assert column.shape == expected_column.shape

        assert np.allclose(row, expected_row)
        assert np.allclose(column, expected_column)
