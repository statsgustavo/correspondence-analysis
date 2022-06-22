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
