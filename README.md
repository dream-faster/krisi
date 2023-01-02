

<p align="center">
  <a href="https://img.shields.io/github/actions/workflow/status/dream-faster/krisi/sphinx.yml"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/dream-faster/krisi/sphinx.yml?logo=readthedocs"></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://dream-faster.github.io/krisi/">
    <img src="docs/logo.svg" alt="Logo" width="90" >
  </a>

<h3 align="center"> <i>(/creesee/)</i></h3>
  <p align="center">
    Testing and Reporting Framework for Time Series Analysis
    <br />
    <a href="https://github.com/dream-faster/krisi">View Demo</a>  ~
    <a href="https://github.com/dream-faster/krisi/tree/main/src/krisi/examples">Check Examples</a> ~
    <a href="https://dream-faster.github.io/krisi/"><strong>Explore the docs »</strong></a>
  </p>
</div>
<br />

Krisi is a Scoring library for Time-Series Forecasting. It calculates, stores and vizualises the performance of your predictions!

Krisi is from the ground-up extensible and lightweight and comes with the fundmental metrics for regression and classification (wip).

It can generate reports in:
- static pdf
- interactive html 
- pretty formatted for console.
  
<br/>

## Krisi tries to solve

- Reporting attached to modelling (Darts, Statsmodel)
- Extendability
- Rolling window based evaluation
- Lightweight (few dependcies)

<br/>

## Installation


The project was entire built in ``python``. 

Prerequisites

* ``python >= 3.7`` and ``pip``


Install from git directly

```
pip install https://github.com/dream-faster/krisi/archive/main.zip 
```

OR

1. Clone the project by running
    ```
    git clone https://github.com/dream-faster/krisi.git
    ```

2. Navigate to the project root directory

3. Install krisi by executing 
    ```
    pip install -e .
    ```
<br/>

## Quickstart

```python
from krisi.evaluate import ScoreCard
import numpy as np

sc = ScoreCard("<your_model_name>")

# Random targets and predictions for Demo
target, predictions = np.random.rand(1000), np.random.rand(1000)

# Calculate predefined metrics
sc.evaluate(target, predictions, defaults=True)

# Add a new metric
sc["own_metric"] = (target - predictions).mean()

# Print the result
sc.print_summary()
```
Outputs:
```
+- Result of <your_model_name> on <your_dataset_name> tested on-+
|                                                               |
|                     Residual Diagnostics                      |
| +--------------+-------------------------------+------------+ |
| |  Mean of the | 0.012                         | {}         | |
| |    Residuals |                               |            | |
| | (residuals_… |                               |            | |
| |     Standard | 0.393                         | {}         | |
| | Deviation of |                               |            | |
| |          the |                               |            | |
| |    Residuals |                               |            | |
| | (residuals_… |                               |            | |
| |    Ljung Box |       lb_stat  lb_pvalue      | {}         | |
| |   Statistics | 1    2.068069   0.150412      |            | |
| | (ljung_box_… | 2    5.813866   0.054643      |            | |
| |              | 3    5.819907   0.120709      |            | |
| |              | 4    7.725251   0.102177      |            | |
| |              | 5   11.398585   0.044026      |            | |
| |              | 6   13.214052   0.039760      |            | |
| |              | 7   13.508170   0.060653      |            | |
| |              | 8   16.192656   0.039704      |            | |
| |              | 9   16.300064   0.060874      |            | |
| |              | 10  16.390463   0.088987      |            | |
| +--------------+-------------------------------+------------+ |
|                                                               |
|                 Forecast Errors - Regression                  |
| +--------------+-------------------------------+------------+ |
| |         Mean | 0.318                         | {}         | |
| |     Absolute |                               |            | |
| |  Error (mae) |                               |            | |
| |         Mean | 2.599                         | {}         | |
| |     Absolute |                               |            | |
| |   Percentage |                               |            | |
| | Error (mape) |                               |            | |
| | Mean Squared | 0.155                         | {          | |
| |  Error (mse) |                               |     'squa… | |
| |              |                               | True       | |
| |              |                               | }          | |
| |    Root Mean | 0.393                         | {          | |
| |      Squared |                               |     'squa… | |
| | Error (rmse) |                               | False      | |
| |              |                               | }          | |
| |    Root Mean | 0.269                         | {          | |
| |  Squared Log |                               |     'squa… | |
| |        Error |                               | False      | |
| |      (rmsle) |                               | }          | |
| +--------------+-------------------------------+------------+ |
|                                                               |
|                                Unknown                        |
| +--------------+-------------------------------+------------+ |
| |   own_metric | 0.012                         | {}         | |
| | (own_metric) |                               |            | |
| +--------------+-------------------------------+------------+ |
|                                                               |
+---------------------------------------------------------------+
```

Creating more sophisticated ``Metric``s with metadata. 
```python
import numpy as np
from krisi.evaluate import Metric, MetricCategories, ScoreCard

sc = ScoreCard("<your_model_name>")

# Random targets and predictions for Demo
target, predictions = np.random.rand(100), np.random.rand(100)
# Calculate a random metric for Demo
calculated_metric_example = (target - predictions).mean()

# Adding a simple new metric (a float)
# As a Dictionary:
sc["metric_barebones"] = calculated_metric_example

# As an Object assignment:

sc.another_metric_barebones = calculated_metric_example * 2.0


sc["metric_with_metadata"] = Metric(
    name="A new, own Metric",
    category=MetricCategories.residual,
    result=calculated_metric_example * 3.0,
    parameters={"hyper_1": 5.0},
)

# Updating the metadata of an existing metric
sc.metric_barebones = dict(info="Giving description to a metric")

# Print a pretty summary to the console
sc.print_summary(with_info=True)
```



<br/>


## Default Metrics

See ``evaluate/library/default_metrics.py`` for source.
Contributors are continously adding new default metrics, press watch to keep track of the project and see in issues planned default metrics.

<b> Residual Diagnostics </b>
- Mean of the Residuals
- Standard Deviation of the Residuals
- Ljung Box Statistics
- (wip) Autocorrelation of Residuals


<b> Regression Errors</b>
- Mean Absolute Error
- Mean Absolute Percentage Error
- Mean Squared Error
- Root Mean Squared Error
- Root Mean Squared Log Error





## Contribution


The project uses ``isort`` and ``black`` for formatting.

Submit an issue or reach out to us on info at dream-faster.ai for any inquiries.
