from src.contingency_table import ContingencyTable


class TestContingencyTable:
    """Tests for src.contingency.ContingencyTable."""

    def test_category_levels_dataframe(self, dataframe):
        """
        Tests retrival of pandas.DataFrame contingency table index and column names.
        """
        table = ContingencyTable(dataframe.data)
        assert (table.rows.levels == dataframe.row_idx).all()
        assert (table.columns.levels == dataframe.column_idx).all()

    def test_category_levels_array(self, array):
        """
        Tests handling of index and column names when a numpy.array is passed as data.
        """
        table = ContingencyTable(array.data)
        assert (table.rows.levels == array.row_idx).all()
        assert (table.columns.levels == array.column_idx).all()

    def test_sample_marginal_totals(self, array):
        """Tests calculation of sample marginal totals."""
        table = ContingencyTable(array.data)
        row, column = table.sample_marginal_totals()
        assert (row == array.data.sum(1)).all()
        assert (column == array.data.sum(0)).all()

    def test_sample_marginal_proportions(self, array):
        """Tests calculation of sample marginal proportions."""
        table = ContingencyTable(array.data)
        proportions = array.data / array.data.sum()
        row, column = table.sample_marginal_proportions()
        assert (row == proportions.sum(1)).all()
        assert (column == proportions.sum(0)).all()
