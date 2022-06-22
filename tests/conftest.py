from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
import pytest


@dataclass
class DatasetFixture:
    row_idx: np.ndarray
    column_idx: np.ndarray
    data: Union[pd.DataFrame, np.ndarray]


@pytest.fixture(scope="session")
def dataframe():
    data = pd.DataFrame(
        {
            "None": [4, 4, 25, 18, 10],
            "Light": [2, 3, 10, 24, 6],
            "Medium": [3, 7, 12, 33, 7],
            "Heavy": [2, 4, 4, 13, 2],
        },
        index=["SM", "JM", "SE", "JE", "SC"],
    )

    return DatasetFixture(data.index.values, data.columns.values, data)


@pytest.fixture(scope="session")
def array(dataframe):
    data = dataframe.data.values
    row_idx = np.char.add("R", np.arange(1, data.shape[0] + 1).astype(str))
    column_idx = np.char.add("C", np.arange(1, data.shape[1] + 1).astype(str))
    return DatasetFixture(row_idx, column_idx, data)
