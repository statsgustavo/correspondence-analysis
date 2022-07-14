from typing import Any

from sklearn.metrics import pairwise


class InvalidMetricError(Exception):
    """Exception for invalid distance metrics."""

    VALID_METRICS = list(pairwise.PAIRED_DISTANCES.keys())

    def __init__(self, value: Any):
        self.value = value
        self.message = (
            "Invalid distance metric. Expected one of {VALID_METRICS}"
            + "by got `{self.value}`."
        )
        super().__init__(self.message)
