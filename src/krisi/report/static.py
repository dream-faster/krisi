import base64
from typing import List

from xhtml2pdf import pisa


def figure_to_base64(figures):
    images_html = ""
    for figure in figures:
        image = str(base64.b64encode(figure.to_image(format="png", scale=2)))[2:-1]
        images_html += f'<img src="data:image/png;base64,{image}"><br>'
    return images_html


def create_html_report(
    template_file: str, images_html: str, title: str = "Time Series Report"
) -> str:
    import pkgutil

    template_html = pkgutil.get_data(__name__, template_file)
    template_html = str(template_html)
    # with open(template_file, "r") as f:
    #     template_html = f.read()
    report_html = template_html.replace("{{ FIGURES }}", images_html).replace(
        "{{ TITLE }}", title
    )
    # report_html = template_html.replace("{{ TITLE }}", title)
    return report_html


def convert_html_to_pdf(source_html: str, output_path: str, report_name: str):
    import os

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f"{output_path}/{report_name}", "w+b") as f:
        pisa_status = pisa.CreatePDF(source_html, dest=f)
    return pisa_status.err


def create_pdf_report(figures: List, path: str = "output", title: str = ""):
    [fig.update_layout(width=900.0) for fig in figures]
    images_html = figure_to_base64(figures)
    report_html = create_html_report(f"template.html", images_html, title)
    convert_html_to_pdf(report_html, path, report_name="report.pdf")
