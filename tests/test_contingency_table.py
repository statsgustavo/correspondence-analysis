from src.correspondence.contingency_table import ContingencyTable


class TestContingencyTable:
    """Tests for src.contingency.ContingencyTable."""

    def test_category_levels_dataframe(self, dataframe):
        """
        Tests retrival of pandas.DataFrame contingency table index and column names.
        """
        table = ContingencyTable(dataframe.data)
        assert (table.rows.levels.ravel() == dataframe.row_idx).all()
        assert (table.columns.levels.ravel() == dataframe.column_idx).all()

    def test_category_levels_array(self, array):
        """
        Tests handling of index and column names when a numpy.array is passed as data.
        """
        table = ContingencyTable(array.data)
        assert (table.rows.levels.ravel() == array.row_idx).all()
        assert (table.columns.levels.ravel() == array.column_idx).all()

    def test_sample_marginal_totals(self, array):
        """Tests calculation of sample marginal totals."""
        table = ContingencyTable(array.data)
        row, column = table.sample_marginal_totals()
        assert (row.ravel() == array.data.sum(1)).all()
        assert (column.ravel() == array.data.sum(0)).all()

    def test_sample_marginal_proportions(self, array):
        """Tests calculation of sample marginal proportions."""
        table = ContingencyTable(array.data)
        proportions = array.data / array.data.sum()
        row, column = table.sample_marginal_proportions()
        assert (row.ravel() == proportions.sum(1)).all()
        assert (column.ravel() == proportions.sum(0)).all()
