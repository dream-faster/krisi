import base64

from xhtml2pdf import pisa
from typing import List


def figure_to_base64(figures):
    images_html = ""
    for figure in figures:
        image = str(base64.b64encode(figure.to_image(format="png", scale=2)))[2:-1]
        images_html += f'<img src="data:image/png;base64,{image}"><br>'
    return images_html


def create_html_report(template_file, images_html):
    with open(template_file, "r") as f:
        template_html = f.read()
    report_html = template_html.replace("{{ FIGURES }}", images_html)
    return report_html


def convert_html_to_pdf(source_html, output_filename):
    with open(f"{output_filename}", "w+b") as f:
        pisa_status = pisa.CreatePDF(source_html, dest=f)
    return pisa_status.err


def create_pdf_report(figures: List, path: str = "reporting"):
    [fig.update_layout(width=900.0) for fig in figures]
    images_html = figure_to_base64(figures)
    report_html = create_html_report(f"{path}/template.html", images_html)
    convert_html_to_pdf(report_html, f"{path}/report.pdf")
