from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.types import TableType


@dataclass
class Dimension:
    """Container class for dimensional summary statistics of a contingency object."""

    name: str
    levels: np.ndarray
    counts: np.ndarray
    proportions: np.ndarray


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
        self._parse()

    def _parse(self):
        """Parses table and computes dimension attributes."""
        self.table_proportions = self.table / self.table.sum()
        row_levels, column_levels = self._category_levels()

        (
            row_sample_proportions,
            column_sample_proportions,
        ) = self.sample_marginal_proportions()

        row_totals, column_totals = self.sample_marginal_totals()

        self.rows = Dimension("rows", row_levels, row_totals, row_sample_proportions)
        self.columns = Dimension(
            "columns", column_levels, column_totals, column_sample_proportions
        )

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

    def sample_marginal_totals(self):
        """Sample totals of rows and columns of the table."""
        return self.table.sum(1), self.table.sum(0)

    def sample_marginal_proportions(self):
        """Sample proportions of rows and columns of the table."""
        proportions = self.table / self.table.sum()
        return proportions.sum(1), proportions.sum(0)
