import numpy as np
from src.correspondence_analysis import CorrespondenceAnalysis, destructure_dictionary


def test_destructure_dictionary(dictionary):
    seq1 = destructure_dictionary(dictionary, ["pi", "phi", "fibonacci"])
    assert seq1 == [
        [3, 1, 4, 1, 5, 9, 2],
        [1, 6, 1, 8, 0, 3, 3],
        [1, 1, 2, 3, 5, 8, 11],
    ]

    seq2 = destructure_dictionary(dictionary, ["phi", "pi", "fibonacci"])
    assert seq2 == [
        [1, 6, 1, 8, 0, 3, 3],
        [3, 1, 4, 1, 5, 9, 2],
        [1, 1, 2, 3, 5, 8, 11],
    ]

    seq3 = destructure_dictionary(dictionary)
    assert seq3 == [
        [1, 1, 2, 3, 5, 8, 11],
        [1, 6, 1, 8, 0, 3, 3],
        [3, 1, 4, 1, 5, 9, 2],
    ]


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
