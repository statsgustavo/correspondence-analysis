import numpy as np
import pytest
from src.multidimensional_scaling import metric_multidimensional_scaling as mms
from src.multidimensional_scaling.errors import InvalidMetricError


class TestMetricMultidimensionalScaling:
    """Tests for metric multidimensional scaling implementation."""

    def test_get_distance_metric_function_raises_error(self, array):
        """
        Tests if metric validation raises error when invalid metric is passed.
        """
        data = array.data
        scaling = mms.MetricMultidimensionalScaling(data)
        with pytest.raises(InvalidMetricError):
            _ = scaling._get_distance_metric_function("speed")  # pylint: disable=W0212

    def test_get_distance_metric_function_returns_function(self, array):
        """
        Tests if object returned by metric validation is a function.
        """
        data = array.data
        scaling = mms.MetricMultidimensionalScaling(data)
        for m in ["euclidean", "l1", "l2", "cosine"]:
            fn = scaling._get_distance_metric_function(m)  # pylint: disable=W0212
            assert callable(fn)

    def test_distance_matrix_symmetric(self, array):
        """Tests if resulting distances matrix is a square matrix."""
        data = array.data
        nrows, ncols = mms.MetricMultidimensionalScaling(data).distances.shape
        assert nrows == ncols

    def test_distance_matrix_diagonal_is_zero(self, array):
        """Tests if resulting distances matrix is a square matrix."""

        data = array.data
        diagonal = np.diag(mms.MetricMultidimensionalScaling(data).distances)
        assert (diagonal == 0).all()

    def test_spectral_decomposition(self, metric_mds):
        """Tests spectral decomposition of the double centered distance matrix."""
        data, centered = metric_mds.distances, metric_mds.centered
        expected_eigenvalues, _ = (
            metric_mds.eigenvalues,
            metric_mds.eigenvectors,
        )
        mmds = mms.MetricMultidimensionalScaling(data)
        eigenvalues, _ = mmds._spectral_decomposition(centered)  # pylint: disable=W0212
        assert np.allclose(eigenvalues[eigenvalues > 1e-10], expected_eigenvalues)
        # assert np.allclose(eigenvectors[:, eigenvalues > 1e-10], expected_eigenvectors)

    def test_lower_dimensional_coordinates(self, metric_mds):
        data = metric_mds.distances
        expected_eigenvalues, expected_eigenvectors = (
            metric_mds.eigenvalues,
            metric_mds.eigenvectors,
        )
        mmds = mms.MetricMultidimensionalScaling(data, n_coordinates=1)
        (
            eigenvalues,
            eigenvectors,
        ) = mmds._lower_dimensional_coordinates(  # pylint: disable=W0212
            expected_eigenvalues, expected_eigenvectors
        )

        assert eigenvalues.shape == (1,)
        assert eigenvectors.shape == (expected_eigenvectors.shape[0], 1)

    # def test_eigenvalues(self, metric_mds):
    #     data = metric_mds.distances
    #     expected_eigenvalues = metric_mds.eigenvalues
    #     mmds = mms.MetricMultidimensionalScaling(data)
    #     assert mmds.explained_variance == expected_eigenvalues
