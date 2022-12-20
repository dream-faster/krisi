def test_scorecard():
    from krisi.evaluate.type import SampleTypes
    from krisi.examples.basic_report import basic_report

    report_insample, report_outsample = basic_report()

    assert (
        report_insample.sample_type == SampleTypes.insample
    ), f"Wrong sample type {report_insample.sample_type} =/= SampleTypes.insample"
    assert (
        report_outsample.sample_type == SampleTypes.outsample
    ), f"Wrong sample type {report_outsample.sample_type} =/= SampleTypes.outsample"

    assert any(
        [metric.result is not None for metric in report_insample.get_default_metrics()]
    ), f"No score stored in {report_insample}."
    assert any(
        [metric.result is not None for metric in report_outsample.get_default_metrics()]
    ), f"No score stored in {report_outsample}."
