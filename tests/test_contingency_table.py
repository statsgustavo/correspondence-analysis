import pytest
from src.contingency_table import ContingencyTable


class TestContingencyTable:
    def test_category_levels_dataframe(self, dataframe):
        table = ContingencyTable(dataframe.data)
        row, column = table._category_levels()
        assert (row == dataframe.row_idx).all()
        assert (column == dataframe.column_idx).all()

    def test_category_levels_array(self, array):
        table = ContingencyTable(array.data)
        row, column = table._category_levels()
        assert (row == array.row_idx).all()
        assert (column == array.column_idx).all()

    def test_sample_marginal_totals(self, array):
        table = ContingencyTable(array.data)
        row, column = table.sample_marginal_totals()
        assert (row == array.data.sum(1)).all()
        assert (column == array.data.sum(0)).all()

    def test_sample_marginal_proportions(self, array):
        table = ContingencyTable(array.data)
        proportions = array.data / array.data.sum()
        row, column = table.sample_marginal_proportions()
        assert (row == proportions.sum(1)).all()
        assert (column == proportions.sum(0)).all()
