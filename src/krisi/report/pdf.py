import base64
from typing import List

from weasyprint import CSS, HTML


def figure_to_base64(figures):
    images_html = ""
    for figure in figures:
        image = str(base64.b64encode(figure.to_image(format="png", scale=2)))[2:-1]
        images_html += f'<img width="30px" style="width: 100%;" src="data:image/png;base64,{image}"><br>'
    return images_html


import pkgutil


def create_html_report(
    template_file: str, images_html: str, title: str = "Time Series Report"
) -> str:

    template_html = pkgutil.get_data(__name__, template_file)
    template_html = template_html.decode()
    report_html = template_html.replace("{{ FIGURES }}", images_html).replace(
        "{{ TITLE }}", title
    )
    return report_html


def convert_html_to_pdf(source_html: str, output_path: str, report_name: str) -> None:
    import os

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    css_string = pkgutil.get_data(__name__, "template.css")
    css_string = css_string.decode()

    css = CSS(string=css_string)
    pdf = HTML(string=source_html).write_pdf(stylesheets=[css])

    open(f"{output_path}/{report_name}", "wb").write(pdf)


def create_pdf_report(figures: List, path: str = "output", title: str = ""):
    # [fig.update_layout(width=900.0) for fig in figures]
    images_html = figure_to_base64(figures)
    report_html = create_html_report(f"template.html", images_html, title)
    convert_html_to_pdf(report_html, path, report_name=f"Report-{title}.pdf")
