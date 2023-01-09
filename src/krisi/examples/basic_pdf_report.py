from krisi.examples.basic_report_rolling import basic_report_rolling
from krisi.report.report import Report
from krisi.report.type import DisplayModes

insample_report, outsample_report = basic_report_rolling()

report = Report(
    title=f"Scorecard on {insample_report.dataset_name}, model: {insample_report.model_name}",
    modes=[DisplayModes.pdf],
)

report.add(insample_report.get_rolling_diagrams())
report.generate_launch()
