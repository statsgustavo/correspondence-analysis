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
        id_ = "-".join(self._title.lower().split())
        return f"""
        <div class="g-col-6">
        <h3 id="{id_}" class="table-name">{self._title}</h3>
        <div id="{id_}" class="table-content">{self._table}</div>
        </div>
        """


class Report:
    def __init__(self, tables: List[Table]):
        self._tables = tables

    def _head(self):
        html = (
            f"<style>{self._tables[0].style.css}</style>\n"
            + '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0-beta1/dist/css/'
            + 'bootstrap.min.css" rel="stylesheet" integrity="sha384-0evHe/X+R7YkIZDRvuz'
            + 'KMRqM+OrBnVFBL6DOitfPri4tjfHxaWutUpFmBp4vmVor" crossorigin="anonymous">'
        )
        return html

    def _body(self):
        content = '<div class="tables-container grid">\n'
        for table in self._tables:
            content += f"{table.template}\n"

        return f"{content}</div>"

    def template(self):
        return f"""
        <html>
        <head>
        {self._head()}
        <head>
        <body>
        <h3 class="report-title">Correspondence analysis results</h3>
        {self._body()}
        </body>
        <html>
        """

    def render(self):
        return HTML(self.template())
