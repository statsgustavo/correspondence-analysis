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
        <div class="table">
        <h3 id="{id_}" class="table-name">{self._title}</h3>
        <div id="{id_}" class="table-content">{self._table}</div>
        </div>
        """


class Report:
    def __init__(self, title: str, tables: List[Table]):
        self._tables = tables
        self._title = title

    def _head(self):
        html = f"<style>{self._tables[0].style.css}</style>\n"
        return html

    def _body(self):
        content = '<div class="container">\n'
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
        <h3 class="report-title">{self._title}</h3>
        {self._body()}
        </body>
        <html>
        """

    def render(self):
        return HTML(self.template())
