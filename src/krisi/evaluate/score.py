from typing import Any, Dict, List, Optional, Union

from krisi.evaluate.library.benchmarking import Model
from krisi.evaluate.metric import Metric
from krisi.evaluate.scorecard import ScoreCard
from krisi.evaluate.type import (
    Calculation,
    DatasetType,
    Predictions,
    Probabilities,
    SampleTypes,
    Targets,
    Weights,
)
from krisi.utils.iterable_helpers import wrap_in_list


def score(
    y: Targets,
    predictions: Predictions,
    probabilities: Optional[Probabilities] = None,
    sample_weight: Optional[Weights] = None,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    project_name: Optional[str] = None,
    default_metrics: Optional[Union[List[Metric], Metric]] = None,
    custom_metrics: Optional[Union[List[Metric], Metric]] = None,
    dataset_type: Optional[Union[DatasetType, str]] = None,
    sample_type: Union[str, SampleTypes] = SampleTypes.outofsample,
    calculation: Union[
        List[Union[Calculation, str]], Union[Calculation, str]
    ] = Calculation.single,
    rolling_args: Optional[Dict[str, Any]] = None,
    raise_exceptions: bool = False,
    benchmark_model: Optional[Model] = None,
    **kwargs,
) -> ScoreCard:
    """
    Creates a ScoreCard based on the passed in arguments, evaluates and then returns the ScoreCard.

    Parameters
    ----------
    y: Targets
        True Targets to which the metrics are evaluated to.
    predictions: Predictions
        The single point predictions to which the metrics are evaluated to.
    model_name: Optional[str]
        The name of the model that the predictions were generated by. Used for identifying scorecards.
    dataset_name: Optional[str]
        The name of the dataset from which the `y` (targets) orginate from. Used for reporting.
    project_name: Optional[str]
        The name of the project. Used for reporting and saving to a directory (eg.: multiple scorecards)
    default_metrics: Optional[List[Metric]]
        Default metrics that get evaluated. See `library`.
    custom_metrics: Optional[List[Metric]]
        Custom metrics that get evaluated. If specified it will evaluate these after `default_metric`
        See `library`.
    dataset_type: Optional[Union[DatasetType, str]]
        Whether the task was a binar/multi-label classifiction of regression. If set to `None` it will infer from the target.
    sample_type: SampleTypes = SampleTypes.outofsample
        Whether we should evaluate it on insample or out of sample.

            - `SampleTypes.outofsample`
            - `SampleTypes.insample`
    calculation: Union[ List[Union[Calculation, str]], Union[Calculation, str] ], optional
        Whether it should evaluate `Metrics` on a rolling basis or on the whole prediction or both, by default Calculation.single

            - `Calculation.single`
            - `Calculation.rolling`
            - `Calculation.benchmark`
    rolling_args : Dict[str, Any], optional
        Arguments to be passed onto `pd.DataFrame.rolling`.
        Default:

        - The window size of the rolling metric evaluation. If `None` evaluation over time will be on expanding window basis, by default `len(dataset)//100`.
        - The step size of the rolling metric evaluation, by default `len(dataset)//100`.

    Returns
    -------
    ScoreCard
        The ScoreCard Evaluated

    Raises
    ------
    ValueError
        If Calculation type is incorrectly specified.
    """

    calculations = [
        Calculation.from_str(calculation) for calculation in wrap_in_list(calculation)
    ]

    sc = ScoreCard(
        y=y,
        predictions=predictions,
        probabilities=probabilities,
        sample_weight=sample_weight,
        model_name=model_name,
        dataset_name=dataset_name,
        project_name=project_name,
        sample_type=sample_type,
        dataset_type=dataset_type,
        default_metrics=default_metrics,
        custom_metrics=custom_metrics,
        rolling_args=rolling_args,
        raise_exceptions=raise_exceptions,
        **kwargs,
    )

    assert any(
        [
            calc
            in [
                Calculation.single,
                Calculation.benchmark,
                Calculation.rolling,
            ]
            for calc in calculation
        ]
    ), f"Calculation type {calculation} not recognized."

    if Calculation.single in calculations:
        sc.evaluate()
    if Calculation.rolling in calculations:
        sc.evaluate_over_time()
    if Calculation.benchmark in calculations:
        assert benchmark_model is not None, "You need to define a benchmark model!"
        sc.evaluate_benchmark(benchmark_model)

    sc.cleanup_group()
    return sc
