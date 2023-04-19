"""
Basic PDF report
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'

from examples.basic_report_rolling import basic_report_rolling

insample_report, outsample_report = basic_report_rolling()
outsample_report.generate_report(display_modes="pdf")
