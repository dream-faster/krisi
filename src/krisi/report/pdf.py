from typing import TYPE_CHECKING, Dict, List

if TYPE_CHECKING:
    import plotly.graph_objects as go

from krisi.report.type import InteractiveFigure, PathConst


def __figure_to_html(figure: "go.Figure") -> str:
    import base64

    image = str(base64.b64encode(figure.to_image(format="png", scale=5)))[2:-1]
    return f'<img style="max-width:100%; max-height:100%;" src="data:image/png;base64,{image}"/>'


def __create_html_report(
    template_file: str, html_elements_to_inject: dict[str, str]
) -> str:
    import pkgutil

    template_html = pkgutil.get_data(__name__, template_file)
    template_html = template_html.decode()

    for key, value in html_elements_to_inject.items():
        template_html = template_html.replace(f"{{ {key} }}", value)
    return template_html


def __convert_html_to_pdf(
    source_html: str, output_path: str, report_name: str, css_template_url: str
) -> None:
    import os
    import pkgutil

    from weasyprint import CSS, HTML

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    css_string = pkgutil.get_data(__name__, css_template_url)
    css_string = css_string.decode()

    css = CSS(string=css_string)
    pdf = HTML(string=source_html).write_pdf(stylesheets=[css])

    open(f"{output_path}/{report_name}", "wb").write(pdf)


def convert_figures_to_html(figures: List["go.Figure"]) -> str:
    return "<br>".join([__figure_to_html(figure) for figure in figures])


def create_pdf_report(
    path: str = PathConst.default_save_path,
    title: str = "Time Series Report",
    html_template_url: str = PathConst.html_template_url,
    css_template_url: str = PathConst.css_template_url,
    html_elements_to_inject: dict[str, str] = dict(),
):
    html_elements_to_inject["title"] = title
    report_html = __create_html_report(html_template_url, html_elements_to_inject)
    __convert_html_to_pdf(report_html, path, f"{title}.pdf", css_template_url)


def append_sizes(
    diagram_dict: Dict[str, InteractiveFigure]
) -> Dict[str, InteractiveFigure]:
    size_dict = dict(
        residuals_display_acf_plot=(750.0, 750.0),
        residuals_display_density_plot=(350.0, 350.0),
        residuals_display_time_series=(1500.0, 750.0),
    )

    for key, value in size_dict.items():
        if key in diagram_dict.keys():
            diagram_dict[key].width = value[0]
            diagram_dict[key].height = value[1]

    return diagram_dict
