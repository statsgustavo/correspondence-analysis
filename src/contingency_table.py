from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.types import TableType


class ContingencyTable(sm.stats.Table):
    """Contingency table class."""

    def __init__(self, table: TableType, shift_zeros: Optional[bool] = True):
        """
        Loads table with frequencies of two categorical random variables with multiple
        levels each and computes appropiate statistics.

        :params table:
            Contingency table of frequencies.
        """
        super(ContingencyTable, self).__init__(table, shift_zeros)

    def _category_levels(self):
        """
        Gets row and columns names in case a table dataframe otherwise sets appropriate
        names for them.
        """
        nrows, ncols = self.table.shape
        if isinstance(self.table_orig, pd.DataFrame):
            row, column = self.table_orig.index.values, self.table_orig.columns.values
        else:
            row, column = (
                np.char.add("R", np.arange(1, nrows + 1).astype(str)),
                np.char.add("C", np.arange(1, ncols + 1).astype(str)),
            )
        return row, column
