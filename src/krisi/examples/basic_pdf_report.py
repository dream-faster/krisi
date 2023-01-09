from krisi.examples.basic_report_rolling import basic_report_rolling
from krisi.report.report import Report
from krisi.report.type import DisplayModes

insample_report, outsample_report = basic_report_rolling()


insample_report.generate_report()
