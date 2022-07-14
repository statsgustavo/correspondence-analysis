import pytest
from src.multidimensional_scaling import metric_multidimensional_scaling as mms
from src.multidimensional_scaling.errors import InvalidMetricError


class TestMetricMultidimensionalScaling:
    """Tests for metric multidimensional scaling implementation."""

    def test_get_distance_metric_function_raises_error(self):
        """
        Tests if metric validation raises error when invalid metric is passed.
        """
        with pytest.raises(InvalidMetricError):
            _ = mms.get_distance_metric_function("speed")

    def test_get_distance_metric_function_returns_function(self):
        """
        Tests if object returned by metric validation is a function.
        """
        for m in ["euclidean", "l1", "l2", "cosine"]:
            fn = mms.get_distance_metric_function(m)
            assert callable(fn)
