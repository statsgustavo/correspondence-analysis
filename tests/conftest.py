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
        [
            [4, 2, 4, 4, 1, 2, 2, 4, 1],
            [3, 4, 2, 2, 1, 1, 0, 3, 2],
            [6, 4, 5, 2, 3, 1, 1, 3, 0],
            [2, 0, 5, 1, 3, 3, 3, 1, 5],
            [2, 5, 0, 1, 4, 1, 2, 1, 3],
            [3, 3, 1, 0, 0, 3, 0, 2, 1],
            [0, 0, 0, 0, 1, 4, 1, 5, 3],
            [0, 2, 0, 11, 1, 3, 10, 1, 1],
            [2, 1, 1, 0, 2, 4, 0, 2, 0],
            [0, 1, 4, 1, 6, 0, 3, 0, 6],
        ],
        index=[
            "red",
            "orange",
            "yellow",
            "green",
            "blue",
            "purple",
            "white",
            "black",
            "pink",
            "brown",
        ],
        columns=[
            "Video",
            "Jazz",
            "Country",
            "Rap",
            "Pop",
            "Opera",
            "Low F",
            "High F",
            "Middle F",
        ],
    )
    zero_shift = 0.0001
    return DatasetFixture(table.index.values, table.columns.values, table + zero_shift)


@pytest.fixture(scope="session")
def array() -> DatasetFixture:
    """Fixture for contingency tables as numpy.ndarray object."""
    table = np.array(
        [
            [4, 2, 4, 4, 1, 2, 2, 4, 1],
            [3, 4, 2, 2, 1, 1, 0, 3, 2],
            [6, 4, 5, 2, 3, 1, 1, 3, 0],
            [2, 0, 5, 1, 3, 3, 3, 1, 5],
            [2, 5, 0, 1, 4, 1, 2, 1, 3],
            [3, 3, 1, 0, 0, 3, 0, 2, 1],
            [0, 0, 0, 0, 1, 4, 1, 5, 3],
            [0, 2, 0, 11, 1, 3, 10, 1, 1],
            [2, 1, 1, 0, 2, 4, 0, 2, 0],
            [0, 1, 4, 1, 6, 0, 3, 0, 6],
        ],
    )
    zero_shift = 0.0001
    row_idx = np.array(["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"])
    column_idx = np.array(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"])
    return DatasetFixture(row_idx, column_idx, table + zero_shift)


@pytest.fixture(scope="session")
def weights():
    """Correspondence analysis weights fixture."""
    row, column = (
        np.diag(
            np.array(
                [
                    2.8723,
                    3.3166,
                    2.8142,
                    2.9341,
                    3.2282,
                    3.9027,
                    3.7607,
                    2.613,
                    4.062,
                    3.0706,
                ]
            )
        ),
        np.diag(np.array([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0])),
    )
    return row, column


@pytest.fixture(scope="session")
def mass():
    """Correspondence analysis mass fixture."""
    row, column = (
        np.array(
            [
                [0.121],
                [0.091],
                [0.126],
                [0.116],
                [0.096],
                [0.066],
                [0.071],
                [0.146],
                [0.061],
                [0.106],
            ]
        ),
        np.array(
            [
                [0.111],
                [0.111],
                [0.111],
                [0.111],
                [0.111],
                [0.111],
                [0.111],
                [0.111],
                [0.111],
            ]
        ),
    )
    return row, column


@pytest.fixture(scope="session")
def distances():
    """Correspondence analysis distances fixture."""
    row, column = (
        np.array(
            [
                [0.2187],
                [0.3333],
                [0.4544],
                [0.4121],
                [0.5207],
                [0.7573],
                [1.3876],
                [1.5362],
                [0.8749],
                [1.0203],
            ]
        ),
        np.array(
            [
                [0.6432],
                [0.6251],
                [0.5946],
                [1.1947],
                [0.5726],
                [0.712],
                [0.9459],
                [0.6707],
                [0.7562],
            ]
        ),
    )
    return row, column


@pytest.fixture(scope="session")
def inertia():
    """Correspondence analysis distances fixture."""
    row, column = (
        np.array(
            [
                [0.026],
                [0.030],
                [0.057],
                [0.048],
                [0.050],
                [0.050],
                [0.099],
                [0.224],
                [0.053],
                [0.108],
            ]
        ),
        np.array(
            [
                [0.071],
                [0.069],
                [0.066],
                [0.133],
                [0.064],
                [0.079],
                [0.105],
                [0.074],
                [0.084],
            ]
        ),
    )
    return row, column


@pytest.fixture(scope="session")
def standardized_residuals_matrix():
    return np.array(
        [
            [0.058, -0.029, 0.058, 0.058, -0.0725, -0.029, -0.029, 0.058, -0.0725],
            [0.0503, 0.1005, 0.0, 0.0, -0.0503, -0.0503, -0.1005, 0.0503, 0.0],
            [
                0.1374,
                0.0521,
                0.0948,
                -0.0332,
                0.0095,
                -0.0758,
                -0.0758,
                0.0095,
                -0.1184,
            ],
            [
                -0.0247,
                -0.1136,
                0.1087,
                -0.0692,
                0.0198,
                0.0198,
                0.0198,
                -0.0692,
                0.1087,
            ],
            [
                -0.0054,
                0.1413,
                -0.1033,
                -0.0543,
                0.0924,
                -0.0543,
                -0.0054,
                -0.0543,
                0.0435,
            ],
            [0.092, 0.092, -0.0263, -0.0854, -0.0854, 0.092, -0.0854, 0.0329, -0.0263],
            [
                -0.0886,
                -0.0886,
                -0.0886,
                -0.0886,
                -0.0317,
                0.1393,
                -0.0317,
                0.1963,
                0.0823,
            ],
            [
                -0.1276,
                -0.0484,
                -0.1276,
                0.3079,
                -0.088,
                -0.0088,
                0.2683,
                -0.088,
                -0.088,
            ],
            [0.041, -0.0205, -0.0205, -0.0821, 0.041, 0.1641, -0.0821, 0.041, -0.0821],
            [-0.1086, -0.062, 0.0775, -0.062, 0.1706, -0.1086, 0.031, -0.1086, 0.1706],
        ]
    )


@pytest.fixture(scope="session")
def factor_scores():
    row_factors = np.array(
        [
            [-0.026, 0.299],
            [-0.314, 0.232],
            [-0.348, 0.202],
            [-0.044, -0.490],
            [-0.082, -0.206],
            [-0.619, 0.475],
            [-0.328, 0.057],
            [1.195, 0.315],
            [-0.570, 0.300],
            [0.113, -0.997],
        ]
    )

    column_factors = np.array(
        [
            [-0.541, 0.386],
            [-0.257, 0.275],
            [-0.291, -0.309],
            [0.991, 0.397],
            [-0.122, -0.637],
            [-0.236, 0.326],
            [0.954, -0.089],
            [-0.427, 0.408],
            [-0.072, -0.757],
        ]
    )

    return row_factors, column_factors


@pytest.fixture(scope="session")
def profile_correlation():
    row = np.array(
        [
            [0.003, 0.41],
            [0.295, 0.161],
            [0.267, 0.089],
            [0.005, 0.583],
            [0.013, 0.081],
            [0.505, 0.298],
            [0.077, 0.002],
            [0.929, 0.065],
            [0.371, 0.103],
            [0.012, 0.973],
        ]
    )

    column = np.array(
        [
            [0.454, 0.232],
            [0.105, 0.121],
            [0.142, 0.161],
            [0.822, 0.132],
            [0.026, 0.709],
            [0.078, 0.149],
            [0.962, 0.008],
            [0.271, 0.249],
            [0.007, 0.759],
        ]
    )

    return row, column


@pytest.fixture(scope="session")
def profile_contribution():
    row = np.array(
        [
            [0.0, 0.056],
            [0.031, 0.025],
            [0.053, 0.027],
            [0.001, 0.144],
            [0.002, 0.021],
            [0.087, 0.077],
            [0.026, 0.001],
            [0.726, 0.075],
            [0.068, 0.028],
            [0.005, 0.545],
        ]
    )

    column = np.array(
        [
            [0.113, 0.086],
            [0.025, 0.044],
            [0.033, 0.055],
            [0.379, 0.091],
            [0.006, 0.234],
            [0.022, 0.061],
            [0.351, 0.005],
            [0.070, 0.096],
            [0.002, 0.330],
        ]
    )

    return row, column


@pytest.fixture(scope="session")
def eigenvalues():
    return np.array(
        [
            [0.2880168],
            [0.1932599],
            [0.1382847],
            [0.0721578],
            [0.0339265],
            [0.0172454],
            [0.0030265],
            [0.0001659],
            [0.0],
        ]
    )
