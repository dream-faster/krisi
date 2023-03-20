

<p align="center">
  <a href="https://dream-faster.github.io/krisi/"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/dream-faster/krisi/docs.yaml?logo=readthedocs"></a>
  <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a href="https://github.com/dream-faster/fold/actions/workflows/tests.yaml"><img alt="Tests" src="https://github.com/dream-faster/fold/actions/workflows/tests.yaml/badge.svg"/></a>
  <a href="https://discord.gg/EKJQgfuBpE"><img alt="Discord Community" src="https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white"></a>
</p>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://dream-faster.github.io/krisi/">
    <img src="https://raw.githubusercontent.com/dream-faster/krisi/main/docs/images/logo.svg" alt="Logo" width="90" >
  </a>

<h3 align="center"> <b>KRISI</b><br> <i>(/creesee/)</i></h3>
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

Krisi is from the ground-up extensible and lightweight and comes with the fundamental metrics for regression and classification.

It can generate reports in:
- static **PDF** (with ``plotly``)
- interactive **HTML** (with ``plotly``)
- pretty formatted for **console** (with ``rich`` and ``plotext``)

<br/>

<div>
  <img src="https://raw.githubusercontent.com/dream-faster/krisi/main/docs/images/output_examples.png" alt="Output Examples: HTML, Console, PDF" width="100%" >

</div>
  
<br/>

## Krisi solves the following problems

- Most TS libraries attach reporting to modelling (eg.: Darts, Statsmodel).<br/> **→ Krisi is independent of any modelling method or library.**
- Extendability is tedious: only works by subclassing objects.<br/>
**→ Krisi supports easy configuration of custom metrics along with an extensive library of predefined metrics.**
- Missing Rolling window based evaluation.<br/>
**→ Krisi supports evaluating metrics over time.**
- Too many dependencies.<br/>
**→ Krisi has few hard dependencies (only core libarries, eg.: sklearn and plotting libraries).**
- Visualisation results are too basic.<br/>
**→ With Krisi you can decide to share and interactive HTML, a static PDF or quickly look at results pretty printed to the console.**

<br/>

## Installation


The project was entirely built in ``python``. 

Prerequisites

* ``python >= 3.7`` and ``pip``


Then run:

*  ``pip install krisi``

If you'd like to also use interactive plotting (html) and pdf generation then run:

*  ``pip install krisi "krisi[plotting]"``

<br/>

## Quickstart

You can quickly evaluate your predictions by running:

```python
import numpy as np
from krisi.evaluate import score

score(y=np.random.rand(1000), predictions=np.random.rand(1000)).print_summary()
```

Krisi's main object is the ``ScoreCard`` that contains predefined ``Metric``s and which you can add further ``Metric``s to.


```python
from krisi.evaluate import ScoreCard
import numpy as np

# Random targets and predictions for Demo
target, predictions = np.random.rand(1000), np.random.rand(1000)

sc = ScoreCard(target, predictions)

# Calculate predefined metrics
sc.evaluate(defaults=True)

# Add a new metric
sc["own_metric"] = (target - predictions).mean()

# Print the result
sc.print_summary()
```
Outputs:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━ Result of <your_model_name> on <your_dataset_name> tested on insample ━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                                                        ┃
┃                                                  Residual Diagnostics                                                  ┃
┃ ╭───────────────────────────┬─────────────────────────────────────────────────────────────┬──────────────────────────╮ ┃
┃ │               Metric Name │ Result                                                      │ Parameters               │ ┃
┃ ├───────────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────┤ ┃
┃ │     Mean of the Residuals │ 0.007                                                       │ {}                       │ ┃
┃ │          (residuals_mean) │                                                             │                          │ ┃
┃ ├───────────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────┤ ┃
┃ │ Standard Deviation of the │ 0.409                                                       │ {}                       │ ┃
┃ │ Residuals (residuals_std) │                                                             │                          │ ┃
┃ ╰───────────────────────────┴─────────────────────────────────────────────────────────────┴──────────────────────────╯ ┃
┃                                              Forecast Errors - Regression                                              ┃
┃ ╭───────────────────────────┬─────────────────────────────────────────────────────────────┬──────────────────────────╮ ┃
┃ │ Mean Absolute Error (mae) │ 0.332                                                       │ {}                       │ ┃
┃ ├───────────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────┤ ┃
┃ │  Mean Absolute Percentage │ 2.85                                                        │ {}                       │ ┃
┃ │              Error (mape) │                                                             │                          │ ┃
┃ ├───────────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────┤ ┃
┃ │  Mean Squared Error (mse) │ 0.168                                                       │ {'squared': True}        │ ┃
┃ ├───────────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────┤ ┃
┃ │   Root Mean Squared Error │ 0.41                                                        │ {'squared': False}       │ ┃
┃ │                    (rmse) │                                                             │                          │ ┃
┃ ├───────────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────┤ ┃
┃ │     Root Mean Squared Log │ 0.281                                                       │ {'squared': False}       │ ┃
┃ │             Error (rmsle) │                                                             │                          │ ┃
┃ ├───────────────────────────┼─────────────────────────────────────────────────────────────┼──────────────────────────┤ ┃
┃ │            R-squared (r2) │ -0.923                                                      │ {}                       │ ┃
┃ ╰───────────────────────────┴─────────────────────────────────────────────────────────────┴──────────────────────────╯ ┃
┃                                                            Unknown                                                     ┃
┃ ╭───────────────────────────┬─────────────────────────────────────────────────────────────┬──────────────────────────╮ ┃
┃ │   own_metric (own_metric) │ 0.007                                                       │ {}                       │ ┃
┃ ╰───────────────────────────┴─────────────────────────────────────────────────────────────┴──────────────────────────╯ ┃
┃                                                                                                                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

Creating more sophisticated ``Metric``s with metadata. 
```python
import numpy as np
from krisi.evaluate import Metric, MetricCategories, ScoreCard


# Random targets and predictions for Demo
target, predictions = np.random.rand(100), np.random.rand(100)

# Create ScoreCard
sc = ScoreCard(target, predictions)

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
Outputs:
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ Result of <your_model_name> on <your_dataset_name> tested on insample ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃                                                                                                                                 ┃
┃                                                      Residual Diagnostics                                                       ┃
┃ ╭─────────────────────┬────────────────────────────────────────────────┬────────────────────┬─────────────────────────────────╮ ┃
┃ │         Metric Name │ Result                                         │ Parameters         │ Info                            │ ┃
┃ ├─────────────────────┼────────────────────────────────────────────────┼────────────────────┼─────────────────────────────────┤ ┃
┃ │         Mean of the │ 0.035                                          │ {}                 │ ''                              │ ┃
┃ │           Residuals │                                                │                    │                                 │ ┃
┃ │    (residuals_mean) │                                                │                    │                                 │ ┃
┃ ├─────────────────────┼────────────────────────────────────────────────┼────────────────────┼─────────────────────────────────┤ ┃
┃ │  Standard Deviation │ 0.42                                           │ {}                 │ ''                              │ ┃
┃ │    of the Residuals │                                                │                    │                                 │ ┃
┃ │     (residuals_std) │                                                │                    │                                 │ ┃
┃ ├─────────────────────┼────────────────────────────────────────────────┼────────────────────┼─────────────────────────────────┤ ┃
┃ │   A new, own Metric │ 0.105                                          │ {'hyper_1': 5.0}   │ 'Giving description to a        │ ┃
┃ │ (yet_another_metri… │                                                │                    │ metric'                         │ ┃
┃ ╰─────────────────────┴────────────────────────────────────────────────┴────────────────────┴─────────────────────────────────╯ ┃
┃                                                  Forecast Errors - Regression                                                   ┃
┃ ╭─────────────────────┬────────────────────────────────────────────────┬────────────────────┬─────────────────────────────────╮ ┃
┃ │ Mean Absolute Error │ 0.35                                           │ {}                 │ '(Mean absolute error)          │ ┃
┃ │               (mae) │                                                │                    │ represents the difference       │ ┃
┃ │                     │                                                │                    │ between the original and        │ ┃
┃ │                     │                                                │                    │ predicted values extracted by   │ ┃
┃ │                     │                                                │                    │ averaged the absolute           │ ┃
┃ │                     │                                                │                    │ difference over the data set.'  │ ┃
┃ ├─────────────────────┼────────────────────────────────────────────────┼────────────────────┼─────────────────────────────────┤ ┃
┃ │       Mean Absolute │ 2.543                                          │ {}                 │ ''                              │ ┃
┃ │    Percentage Error │                                                │                    │                                 │ ┃
┃ │              (mape) │                                                │                    │                                 │ ┃
┃ ├─────────────────────┼────────────────────────────────────────────────┼────────────────────┼─────────────────────────────────┤ ┃
┃ │  Mean Squared Error │ 0.178                                          │ {'squared': True}  │ '(Mean Squared Error)           │ ┃
┃ │               (mse) │                                                │                    │ represents the difference       │ ┃
┃ │                     │                                                │                    │ between the original and        │ ┃
┃ │                     │                                                │                    │ predicted values extracted by   │ ┃
┃ │                     │                                                │                    │ squared the average difference  │ ┃
┃ │                     │                                                │                    │ over the data set.'             │ ┃
┃ ├─────────────────────┼────────────────────────────────────────────────┼────────────────────┼─────────────────────────────────┤ ┃
┃ │   Root Mean Squared │ 0.421                                          │ {'squared': False} │ '(Root Mean Squared Error) is   │ ┃
┃ │        Error (rmse) │                                                │                    │ the error rate by the square    │ ┃
┃ │                     │                                                │                    │ root of Mean Squared Error.'    │ ┃
┃ ├─────────────────────┼────────────────────────────────────────────────┼────────────────────┼─────────────────────────────────┤ ┃
┃ │   Root Mean Squared │ 0.29                                           │ {'squared': False} │ ''                              │ ┃
┃ │   Log Error (rmsle) │                                                │                    │                                 │ ┃
┃ ├─────────────────────┼────────────────────────────────────────────────┼────────────────────┼─────────────────────────────────┤ ┃
┃ │      R-squared (r2) │ -1.28                                          │ {}                 │ '(Coefficient of determination) │ ┃
┃ │                     │                                                │                    │ represents the coefficient of   │ ┃
┃ │                     │                                                │                    │ how well the values fit         │ ┃
┃ │                     │                                                │                    │ compared to the original        │ ┃
┃ │                     │                                                │                    │ values. The value from 0 to 1   │ ┃
┃ │                     │                                                │                    │ interpreted as percentages. The │ ┃
┃ │                     │                                                │                    │ higher the value is, the better │ ┃
┃ │                     │                                                │                    │ the model is.'                  │ ┃
┃ ╰─────────────────────┴────────────────────────────────────────────────┴────────────────────┴─────────────────────────────────╯ ┃
┃                                                                 Unknown                                                         ┃
┃ ╭─────────────────────┬────────────────────────────────────────────────┬────────────────────┬─────────────────────────────────╮ ┃
┃ │          own_metric │ 0.035                                          │ {}                 │ ''                              │ ┃
┃ │        (own_metric) │                                                │                    │                                 │ ┃
┃ ├─────────────────────┼────────────────────────────────────────────────┼────────────────────┼─────────────────────────────────┤ ┃
┃ │      another_metric │ 0.07                                           │ {}                 │ ''                              │ ┃
┃ │    (another_metric) │                                                │                    │                                 │ ┃
┃ ╰─────────────────────┴────────────────────────────────────────────────┴────────────────────┴─────────────────────────────────╯ ┃
┃                                                                                                                                 ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
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

Submit an issue or reach out to us on info at dreamfaster.ai for any inquiries.

[Join our Discord community!](https://discord.gg/EKJQgfuBpE)
