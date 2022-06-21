import pandas as pd
import pytest


@pytest.fixture(scope="session")
def table_dataframe():
    return pd.DataFrame(
        {
            "None": [4, 4, 25, 18, 10],
            "Light": [2, 3, 10, 24, 6],
            "Medium": [3, 7, 12, 33, 7],
            "Heavy": [2, 4, 4, 13, 2],
        },
        index = ["SM", "JM", "SE", "JE", "SC"]
    )

@pytest.fixture
def table_array(table_dataframe):
    return table_dataframe.values
