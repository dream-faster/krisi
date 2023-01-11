import base64
import pkgutil
from typing import List, Optional

import plotly.graph_objects as go
from weasyprint import CSS, HTML

from krisi.report.type import PathConst


def figure_to_html(figure: go.Figure) -> str:
    image = str(base64.b64encode(figure.to_image(format="png", scale=2)))[2:-1]
    return f'<img style="width: 100%;" src="data:image/png;base64,{image}"><br>'


def convert_figures(figures: List[go.Figure]) -> str:
    return "<br>".join([figure_to_html(figure) for figure in figures])


def create_html_report(template_file: str, images_html: str, title: str) -> str:
    template_html = pkgutil.get_data(__name__, template_file)
    template_html = template_html.decode()
    report_html = template_html.replace("{{ FIGURES }}", images_html).replace(
        "{{ TITLE }}", title
    )
    return report_html


def convert_html_to_pdf(
    source_html: str, output_path: str, report_name: str, css_template_url: str
) -> None:
    import os

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    css_string = pkgutil.get_data(__name__, css_template_url)
    css_string = css_string.decode()

    css = CSS(string=css_string)
    pdf = HTML(string=source_html).write_pdf(stylesheets=[css])

    open(f"{output_path}/{report_name}", "wb").write(pdf)


def create_pdf_report(
    figures: List,
    path: str = PathConst.default_save_path,
    title: str = "Time Series Report",
    html_template_url: str = PathConst.html_template_url,
    css_template_url: str = PathConst.css_template_url,
):
    images_html = convert_figures(figures)
    report_html = create_html_report(html_template_url, images_html, title)
    convert_html_to_pdf(report_html, path, f"Report-{title}.pdf", css_template_url)
