import base64
import pkgutil
from typing import List, Optional

import plotly.graph_objects as go
from weasyprint import CSS, HTML


def figure_to_base64(figures: List[go.Figure]):
    images_html = ""
    for figure in figures:
        image = str(base64.b64encode(figure.to_image(format="png", scale=2)))[2:-1]
        images_html += (
            f'<img style="width: 100%;" src="data:image/png;base64,{image}"><br>'
        )
    return images_html


def create_html_report(template_file: str, images_html: str, title: str) -> str:

    template_html = pkgutil.get_data(__name__, template_file)
    template_html = template_html.decode()
    report_html = template_html.replace("{{ FIGURES }}", images_html).replace(
        "{{ TITLE }}", title
    )
    return report_html


def convert_html_to_pdf(
    source_html: str, output_path: str, report_name: str, html_template_css: str
) -> None:
    import os

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    css_string = pkgutil.get_data(__name__, html_template_css)
    css_string = css_string.decode()

    css = CSS(string=css_string)
    pdf = HTML(string=source_html).write_pdf(stylesheets=[css])

    open(f"{output_path}/{report_name}", "wb").write(pdf)


def create_pdf_report(
    figures: List,
    path: str = "output",
    title: str = "Time Series Report",
    html_template_url: str = "template.html",
    html_template_css: str = "template.css",
):
    images_html = figure_to_base64(figures)
    report_html = create_html_report(html_template_url, images_html, title)
    convert_html_to_pdf(report_html, path, f"Report-{title}.pdf", html_template_css)
