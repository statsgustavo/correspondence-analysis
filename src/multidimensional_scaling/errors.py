from typing import Any, List


class InvalidMetricError(Exception):
    """Exception for invalid distance metrics."""

    def __init__(self, value: Any, expected: List[str]):
        self.value = value
        self._expected = expected
        self.message = (
            "Invalid distance metric. Expected one of {self._expected}"
            + "by got `{self.value}`."
        )
        super().__init__(self.message)
