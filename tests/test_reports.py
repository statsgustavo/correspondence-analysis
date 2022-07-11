import bs4
from src.reports import Report, Table


class TestReport:
    def test_loading_css_style(self, dataframe):
        tables = [Table("Contingency table fixture", dataframe.data)]
        report = Report(tables)
        css = (
            bs4.BeautifulSoup(report.template(), features="html.parser")
            .find("style")
            .contents[0]
        )
        expected_css = (
            "div {\n    padding: 0 0;\n}\n\n"
            + "div.tables-container {\n    max-width: 50%;\n}\n\n"
            + "h3.table-name {\n    border-style: hidden hidden solid hidden;\n"
            + "    max-width: 20%;\n}\n\n"
            + "h3.report-title {\n    border-style: hidden hidden solid hidden;\n}"
        )
        assert css == expected_css

    def test_table_content(self, dataframe):
        tables = [Table("Contingency table fixture", dataframe.data)]
        report = Report(tables)
        html = bs4.BeautifulSoup(report.template(), features="html.parser").find(
            "div", {"class": "tables-container"}
        )

        expected_html = bs4.BeautifulSoup(
            '<div class="tables-container">\n'
            + '<h3 id="contingency-table-fixture" class="table-name">Contingency table fixture</h3>\n'
            + f'<div id="contingency-table-fixture" class="table-content">{dataframe.data.to_html()}\n</div>\n'
            + "</div>",
            features="html.parser",
        )
        assert list(html.stripped_strings) == list(expected_html.stripped_strings)
