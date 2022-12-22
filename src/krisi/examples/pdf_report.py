from krisi.examples.basic_report_rolling import basic_report_rolling
from krisi.report.report import Report
from krisi.report.types_ import DisplayModes

insample_report, outsample_report = basic_report_rolling()

for metric in insample_report.get_default_metrics():
    metric.get_diagram_over_time()

report = Report(modes=[DisplayModes.pdf])

report.add(
    [
        diagram
        for diagram in [
            metric.get_diagram_over_time()
            for metric in insample_report.get_default_metrics()
        ]
        if diagram is not None
    ]
)
report.generate_launch()
