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


@pytest.fixture(scope="session")
def standardized_residuals_matrix():
    return np.array(
        [
            [0.02020239, -0.02538438, -0.02043562, 0.03468162],
            [-0.05097522, -0.04205447, 0.0364484, 0.07864884],
            [0.15922216, -0.03947701, -0.07795287, -0.07298869],
            [-0.13394189, 0.05533047, 0.06404368, 0.03413421],
            [0.05373569, 0.00509777, -0.02618966, -0.04953368],
        ]
    )


@pytest.fixture(scope="session")
def factor_scores():
    row_factors = np.array(
        [
            [-0.06576838, -0.19373707, 0.07098096, 0.0],
            [0.25895841, -0.24330466, -0.03370516, 0.0],
            [-0.38059487, -0.01065991, -0.00515575, 0.0],
            [0.2329519, 0.05774393, 0.00330537, 0.0],
            [-0.20108911, 0.07891126, -0.00808107, 0.0],
        ]
    )

    column_factors = np.array(
        [
            [-0.3933084, 0.0312689, 0.0147212, 0.0],
            [-0.0969851, 0.1410643, 0.0017552, 0.0],
            [-0.0118754, 0.0922339, -0.0256591, 0.0],
            [0.4270974, 0.1342793, 0.032026, 0.0],
        ]
    )

    return row_factors, column_factors


@pytest.fixture(scope="session")
def profile_correlation():
    row = np.array(
        [
            [0.092232, 0.8003364, 0.1074316, 0.0],
            [0.5263999, 0.4646825, 0.0089176, 0.0],
            [0.9990329, 0.0007837, 0.0001833, 0.0],
            [0.9419341, 0.0578762, 0.0001896, 0.0],
            [0.8653455, 0.133257, 0.0013975, 0.0],
        ]
    )

    column = np.array(
        [
            [0.9940204, 0.0062828, 0.0013926, 0.0],
            [0.3106937, 0.6572897, 0.0001018, 0.0],
            [0.0035926, 0.2167164, 0.0167723, 0.0],
            [1.4465382, 0.1429862, 0.0081336, 0.0],
        ]
    )

    return row, column
