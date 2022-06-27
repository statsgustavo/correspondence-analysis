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
        [[4, 2, 3, 2], [4, 3, 7, 4], [25, 10, 12, 4], [18, 24, 33, 13], [10, 6, 7, 2]]
    )
    row_idx = np.array(["R1", "R2", "R3", "R4", "R5"])
    column_idx = np.array(["C1", "C2", "C3", "C4"])
    return DatasetFixture(row_idx, column_idx, table)


@pytest.fixture(scope="session")
def weights():
    """Correspondence analysis weights fixture."""
    row, column = (
        np.diag(np.array([4.18872947, 3.27448045, 1.94533126, 1.48093951, 2.7784888])),
        np.diag(np.array([1.77874518, 2.07096328, 1.76434215, 2.7784888])),
    )
    return row, column


@pytest.fixture(scope="session")
def mass():
    """Correspondence analysis mass fixture."""
    row, column = (
        np.array([[0.05699482], [0.09326425], [0.2642487], [0.45595855], [0.12953368]]),
        np.array([[0.31606218], [0.23316062], [0.32124352], [0.12953368]]),
    )
    return row, column


@pytest.fixture(scope="session")
def distances():
    """Correspondence analysis distances fixture."""
    row, column = (
        np.array(
            [[0.04689781], [0.12739262], [0.14499268], [0.05761188], [0.04672912]]
        ),
        np.array([[0.1556221], [0.03027453], [0.03925446], [0.12610259]]),
    )
    return row, column


@pytest.fixture(scope="session")
def inertia():
    """Correspondence analysis distances fixture."""
    row, column = (
        np.array([[0.03137618], [0.13946703], [0.44974987], [0.30835392], [0.071053]]),
        np.array([[0.57737221], [0.08285995], [0.14802515], [0.19174269]]),
    )
    return row, column
