import re

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
            "div .table {\n    padding: 0 0;\n}\n\nh3 .table "
            + "{\n    border-style: hidden hidden solid hidden;\n}"
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
