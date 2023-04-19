def test_scorecard():
    from examples.training_arima_report import training_arima_report

    from krisi.evaluate.type import SampleTypes

    report_insample, report_outsample = training_arima_report()

    assert (
        report_insample.sample_type == SampleTypes.insample
    ), f"Wrong sample type {report_insample.sample_type} =/= SampleTypes.insample"
    assert (
        report_outsample.sample_type == SampleTypes.outofsample
    ), f"Wrong sample type {report_outsample.sample_type} =/= SampleTypes.outofsample"

    assert any(
        [metric.result is not None for metric in report_insample.get_default_metrics()]
    ), f"No score stored in {report_insample}."
    assert any(
        [metric.result is not None for metric in report_outsample.get_default_metrics()]
    ), f"No score stored in {report_outsample}."
