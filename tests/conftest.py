from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
import pytest


@dataclass
class DatasetFixture:
    """Container class for data fixtures."""

    row_idx: np.ndarray
    column_idx: np.ndarray
    data: Union[pd.DataFrame, np.ndarray]


@pytest.fixture(scope="session")
def dataframe() -> DatasetFixture:
    """Fixture for contingency tables as pandas.DataFrames object."""
    table = pd.DataFrame(
        {
            "None": [4, 4, 25, 18, 10],
            "Light": [2, 3, 10, 24, 6],
            "Medium": [3, 7, 12, 33, 7],
            "Heavy": [2, 4, 4, 13, 2],
        },
        index=["SM", "JM", "SE", "JE", "SC"],
    )
    return DatasetFixture(table.index.values, table.columns.values, table)


@pytest.fixture(scope="session")
def array() -> DatasetFixture:
    """Fixture for contingency tables as numpy.ndarray object."""
    table = np.array(
        [
            [4, 4, 25, 18, 10],
            [2, 3, 10, 24, 6],
            [3, 7, 12, 33, 7],
            [2, 4, 4, 13, 2],
        ]
    )
    row_idx = np.array(["R1", "R2", "R3", "R4"])
    column_idx = np.array(["C1", "C2", "C3", "C4", "C5"])
    return DatasetFixture(row_idx, column_idx, table)
