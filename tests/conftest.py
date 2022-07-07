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
            [-0.3933084, -0.0304921, -0.0008905, 0.0],
            [0.0994559, 0.1410643, 0.021998, 0.0],
            [0.196321, 0.0073591, -0.0256591, 0.0],
            [0.293776, -0.1977657, 0.0262108, 0.0],
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
            [0.9940204, 0.0059745, 0.0000051, 0.0],
            [0.3267262, 0.6572897, 0.0159842, 0.0],
            [0.981848, 0.0013796, 0.0167723, 0.0],
            [0.6843977, 0.3101542, 0.005448, 0.0],
        ]
    )

    return row, column


@pytest.fixture(scope="session")
def profile_contribution():
    row = np.array(
        [
            [0.0032977, 0.2135576, 0.6943311, 0.0242081],
            [0.0836587, 0.5511506, 0.256186, 0.0562231],
            [0.5120055, 0.0029976, 0.0169842, 0.4157313],
            [0.3309739, 0.1517722, 0.0120452, 0.5032519],
            [0.0700641, 0.0805221, 0.0204535, 0.0005856],
        ]
    )

    column = np.array(
        [
            [0.6539958, 0.029336, 0.000606, 0.3160622],
            [0.0308498, 0.4631737, 0.2728159, 0.2331606],
            [0.1656165, 0.0017368, 0.5114032, 0.3212435],
            [0.1495379, 0.5057536, 0.2151749, 0.1295337],
        ]
    )

    return row, column
