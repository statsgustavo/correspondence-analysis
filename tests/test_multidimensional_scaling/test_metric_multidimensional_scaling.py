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
        scaling = mms.MetricMultidimensionalScaling(array.data)
        with pytest.raises(InvalidMetricError):
            _ = scaling._get_distance_metric_function("speed")  # pylint: disable=W0212

    def test_get_distance_metric_function_returns_function(self, array):
        """
        Tests if object returned by metric validation is a function.
        """
        scaling = mms.MetricMultidimensionalScaling(array.data)
        for m in ["euclidean", "l1", "l2", "cosine"]:
            fn = scaling._get_distance_metric_function(m)  # pylint: disable=W0212
            assert callable(fn)

    def test_distance_matrix_symmetric(self, dataframe):
        """Tests if resulting distances matrix is a square matrix."""
        data = dataframe.data
        nrows, ncols = mms.MetricMultidimensionalScaling(data).distances.shape
        assert nrows == ncols

    def test_distance_matrix_diagonal_is_zero(self, dataframe):
        """Tests if resulting distances matrix is a square matrix."""

        data = dataframe.data
        diagonal = np.diag(mms.MetricMultidimensionalScaling(data).distances)
        assert (diagonal == 0).all()
