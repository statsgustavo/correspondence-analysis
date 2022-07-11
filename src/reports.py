import pathlib
from typing import List

import pandas as pd
from IPython.display import HTML


class Style:
    def __init__(self, component: str):
        css_file = (
            pathlib.Path(__file__).parents[1].joinpath("style", f"{component}.css")
        )

        with open(css_file, "r", encoding="utf-8") as f:
            self._css = f.read()

    @property
    def css(self):
        return self._css


class Table:
    def __init__(self, title: str, table: pd.DataFrame):
        self._title = title
        self._table = table.to_html()
        self._style = Style("table")

    @property
    def style(self) -> Style:
        return self._style

    @property
    def template(self):
        id = "-".join(self._title.lower().split())
        return f"""
            <h3 id="{id}" class="table-name">{self._title}</h3>
            <div id="{id}" class="table-content">{self._table}</div>
        """


class Report:
    def __init__(self, tables: List[Table]):
        self._tables = tables

    def _header(self):
        return f"""
            <header>
                <style>{self._tables[0].style.css}</style>
            </header>
        """

    def _body(self):
        content = '<div class="tables-container">\n'
        for table in self._tables:
            content += f"{table.template}\n"

        return f"\t<body>{content}</div></body>"

    def template(self):
        return f"""
        <html>
        {self._header()}
        {self._body()}
        </html>
        """

    def render(self):
        return HTML(self.template())
