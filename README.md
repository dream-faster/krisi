<p align="center" style="display:flex; width:100%; align-items:center; justify-content:center;">
  <a style="margin:2px" href="https://dream-faster.github.io/krisi/"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/dream-faster/krisi/docs.yaml?logo=readthedocs"></a>
  <a style="margin:2px" href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
  <a style="margin:2px" href="https://github.com/dream-faster/krisi/actions/workflows/tests.yaml"><img alt="Tests" src="https://github.com/dream-faster/krisi/actions/workflows/tests.yaml/badge.svg"/></a>
  <a style="margin:2px" href="https://discord.gg/EKJQgfuBpE"><img alt="Discord Community" src="https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white"></a>
</p>


<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://dream-faster.github.io/krisi/">
    <img src="https://raw.githubusercontent.com/dream-faster/krisi/main/docs/images/logo.svg" alt="Logo" width="90" >
  </a>

<h3 align="center"> <b>KRISI</b><br> <i>(/creesee/)</i></h3>
  <p align="center">
    Evaluation and Reporting Framework for Time Series Forecasting
    <br />
    <!-- <a href="https://github.com/dream-faster/krisi">View Demo</a>  ~ -->
    <a href="https://dream-faster.github.io/krisi/generated/gallery/">Check Examples</a> ~
    <a href="https://dream-faster.github.io/krisi/"><strong>Explore the docs »</strong></a>
  </p>
</div>
<br />

Krisi is a Scoring library for Time-Series Forecasting. It calculates, stores and vizualises the performance of your predictions!

Krisi is tailored to measure performance over time (metrics over time). It is from the ground-up extensible and lightweight and comes with fundamental metrics for regression, classification and residual diagnostics.

It can generate reports in:
- static **PDF** (with ``plotly``)
- interactive **web (HTML)** (with ``plotly``)
- pretty formatted for **console** (with ``rich`` and ``plotext``)
- each figure displayed or saved as an **svg**

<br/>

<div>
  <img src="https://raw.githubusercontent.com/dream-faster/krisi/main/docs/images/output_examples.png" alt="Output Examples: HTML, Console, PDF" width="100%" >
</div>
  
<br/>

## Krisi solves the following problems
- Missing Rolling window based evaluation.<br/>
**→ Krisi supports evaluating metrics over time.**
- Most TS libraries attach reporting to modelling (eg.: Darts, Statsmodel).<br/> **→ Krisi is independent of any modeling method or library.**
- Extendability is tedious: only works by subclassing objects.<br/>
**→ Krisi supports easy configuration of custom metrics along with an extensive library of predefined metrics.**
- Too many dependencies.<br/>
**→ Krisi has few hard dependencies (only core libaries, eg.: sklearn and plotting libraries).**
- Visualisation results are too basic.<br/>
**→ With Krisi you can decide to share and interactive web, a static PDF, each metric diagram displayed inline or quickly look at the ScoreCard pretty printed to the console.**

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

score(y=np.random.rand(1000), predictions=np.random.rand(1000)).print()
```

Krisi's main object is the ``ScoreCard`` that contains predefined ``Metric``s and which you can add further ``Metric``s to.


```python
import numpy as np
from krisi import score

sc = score(
    y=np.random.normal(0, 0.1, 1000),
    predictions=np.random.normal(0, 0.1, 1000),
).print()
```
<details>
<summary>
Outputs 👇
</summary>

```
┏━━━━━━━━━━━ Result of <your_model_name> on <your_dataset_name> tested on outofsample ━━━━━━━━━━━┓
┃                                                                                                ┃
┃                                Targets and Predictions Analysis                                ┃
┃ ╭─────────────────┬────────────────────────────────────────────────┬────────────┬────────────╮ ┃
┃ │     Series Type │ Histogram                                      │      Types │   Indicies │ ┃
┃ ├─────────────────┼────────────────────────────────────────────────┼────────────┼────────────┤ ┃
┃ │         Targets │     ┌────────────────────────────────────────┐ │    NaNs: 0 │   Start: 0 │ ┃
┃ │                 │ 77.0┤               █                        │ │     dtype: │   End: 999 │ ┃
┃ │                 │ 64.2┤             █████████                  │ │    float64 │            │ ┃
┃ │                 │ 51.3┤            ███████████                 │ │            │            │ ┃
┃ │                 │ 38.5┤          ██████████████                │ │            │            │ ┃
┃ │                 │ 25.7┤        █████████████████               │ │            │            │ ┃
┃ │                 │ 12.8┤    ██████████████████████████          │ │            │            │ ┃
┃ │                 │  0.0┤██████████████████████████████████   ███│ │            │            │ ┃
┃ │                 │     └┬─────────┬─────────┬────────┬─────────┬┘ │            │            │ ┃
┃ │                 │    -0.31     -0.13     0.06     0.24     0.42  │            │            │ ┃
┃ ├─────────────────┼────────────────────────────────────────────────┼────────────┼────────────┤ ┃
┃ │     Predictions │     ┌────────────────────────────────────────┐ │    NaNs: 0 │   Start: 0 │ ┃
┃ │                 │ 83.0┤               █ ██                     │ │     dtype: │   End: 999 │ ┃
┃ │                 │ 69.2┤              █████                     │ │    float64 │            │ ┃
┃ │                 │ 55.3┤           ███████████                  │ │            │            │ ┃
┃ │                 │ 41.5┤         ██████████████                 │ │            │            │ ┃
┃ │                 │ 27.7┤        █████████████████               │ │            │            │ ┃
┃ │                 │ 13.8┤     ███████████████████████            │ │            │            │ ┃
┃ │                 │  0.0┤█ ██████████████████████████████  █   ██│ │            │            │ ┃
┃ │                 │     └┬─────────┬─────────┬────────┬─────────┬┘ │            │            │ ┃
┃ │                 │    -0.32     -0.13     0.06     0.26     0.45  │            │            │ ┃
┃ ╰─────────────────┴────────────────────────────────────────────────┴────────────┴────────────╯ ┃
┃                                      Residual Diagnostics                                      ┃
┃ ╭─────────────────────┬─────────────────────────────────────────────────┬────────────────────╮ ┃
┃ │         Metric Name │ Result                                          │ Parameters         │ ┃
┃ ├─────────────────────┼─────────────────────────────────────────────────┼────────────────────┤ ┃
┃ │           Residuals │ 0      0.048193                                 │ {}                 │ ┃
┃ │         (residuals) │ 1      0.086293                                 │                    │ ┃
┃ │                     │ 2      0.076336                                 │                    │ ┃
┃ │                     │ 3      0.120280                                 │                    │ ┃
┃ │                     │ 4      0.119357                                 │                    │ ┃
┃ │                     │          ...                                    │                    │ ┃
┃ │                     │ 995   -0.205864                                 │                    │ ┃
┃ │                     │ 996   -0.047068                                 │                    │ ┃
┃ │                     │ 997    0.107008                                 │                    │ ┃
┃ │                     │ 998    0.180816                                 │                    │ ┃
┃ │                     │ 999   -0.171919                                 │                    │ ┃
┃ │                     │ Length: 1000, dtype: float64                    │                    │ ┃
┃ ├─────────────────────┼─────────────────────────────────────────────────┼────────────────────┤ ┃
┃ │           Ljung Box │       lb_stat  lb_pvalue                        │ {}                 │ ┃
┃ │          Statistics │ 1    0.133822   0.714501                        │                    │ ┃
┃ │ (ljung_box_statist… │ 2    0.136426   0.934062                        │                    │ ┃
┃ │                     │ 3    0.287036   0.962448                        │                    │ ┃
┃ │                     │ 4    0.313093   0.988953                        │                    │ ┃
┃ │                     │ 5    0.384962   0.995734                        │                    │ ┃
┃ │                     │ 6    2.577713   0.859671                        │                    │ ┃
┃ │                     │ 7    4.538755   0.716046                        │                    │ ┃
┃ │                     │ 8    6.237988   0.620593                        │                    │ ┃
┃ │                     │ 9    7.639680   0.570825                        │                    │ ┃
┃ │                     │ 10  12.217638   0.270755                        │                    │ ┃
┃ ├─────────────────────┼─────────────────────────────────────────────────┼────────────────────┤ ┃
┃ │         Mean of the │ 0.007                                           │ {}                 │ ┃
┃ │           Residuals │                                                 │                    │ ┃
┃ │    (residuals_mean) │                                                 │                    │ ┃
┃ ├─────────────────────┼─────────────────────────────────────────────────┼────────────────────┤ ┃
┃ │  Standard Deviation │ 0.137                                           │ {}                 │ ┃
┃ │    of the Residuals │                                                 │                    │ ┃
┃ │     (residuals_std) │                                                 │                    │ ┃
┃ ╰─────────────────────┴─────────────────────────────────────────────────┴────────────────────╯ ┃
┃                                  Forecast Errors - Regression                                  ┃
┃ ╭─────────────────────┬─────────────────────────────────────────────────┬────────────────────╮ ┃
┃ │ Mean Absolute Error │ 0.11                                            │ {}                 │ ┃
┃ │               (mae) │                                                 │                    │ ┃
┃ ├─────────────────────┼─────────────────────────────────────────────────┼────────────────────┤ ┃
┃ │       Mean Absolute │ 6.45                                            │ {}                 │ ┃
┃ │    Percentage Error │                                                 │                    │ ┃
┃ │              (mape) │                                                 │                    │ ┃
┃ ├─────────────────────┼─────────────────────────────────────────────────┼────────────────────┤ ┃
┃ │      Symmetric Mean │ 0.714                                           │ {}                 │ ┃
┃ │ Absolute Percentage │                                                 │                    │ ┃
┃ │       Error (smape) │                                                 │                    │ ┃
┃ ├─────────────────────┼─────────────────────────────────────────────────┼────────────────────┤ ┃
┃ │  Mean Squared Error │ 0.019                                           │ {'squared': True}  │ ┃
┃ │               (mse) │                                                 │                    │ ┃
┃ ├─────────────────────┼─────────────────────────────────────────────────┼────────────────────┤ ┃
┃ │   Root Mean Squared │ 0.137                                           │ {'squared': False} │ ┃
┃ │        Error (rmse) │                                                 │                    │ ┃
┃ ├─────────────────────┼─────────────────────────────────────────────────┼────────────────────┤ ┃
┃ │   Root Mean Squared │ 'Mean Squared Logarithmic Error cannot be used  │ {'squared': False} │ ┃
┃ │   Log Error (rmsle) │ when targets contain negative values.'          │                    │ ┃
┃ ├─────────────────────┼─────────────────────────────────────────────────┼────────────────────┤ ┃
┃ │   R-squared (r_two) │ -0.868                                          │ {}                 │ ┃
┃ ╰─────────────────────┴─────────────────────────────────────────────────┴────────────────────╯ ┃
┃                                                                                                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

</details>


Evaluating `Metric`s over time (on a rolling basis). 
```python
import numpy as np
from krisi import score

score(
    y=np.random.rand(10000),
    predictions=np.random.rand(10000),
    calculation="rolling",
).print()
```
<details>
<summary>
Outputs 👇
</summary>

```
┏━ Result of Model_20230505-092459e9197d1c on Dataset_20230505-0924594280871━┓
┃                                                                            ┃
┃                      Targets and Predictions Analysis                      ┃
┃ ╭─────────────┬──────────────────────────────────────┬─────────┬─────────╮ ┃
┃ │ Series Type │ Histogram                            │   Types │ Indici… │ ┃
┃ ├─────────────┼──────────────────────────────────────┼─────────┼─────────┤ ┃
┃ │     Targets │      ┌─────────────────────────────┐ │ NaNs: 0 │  Start: │ ┃
┃ │             │ 280.0┤ ██ █   ████  ██ ██  ██ █    │ │  dtype: │       0 │ ┃
┃ │             │ 233.3┤█████████████████████████████│ │ float64 │    End: │ ┃
┃ │             │ 186.7┤█████████████████████████████│ │         │    9999 │ ┃
┃ │             │ 140.0┤█████████████████████████████│ │         │         │ ┃
┃ │             │  93.3┤█████████████████████████████│ │         │         │ ┃
┃ │             │  46.7┤█████████████████████████████│ │         │         │ ┃
┃ │             │   0.0┤█████████████████████████████│ │         │         │ ┃
┃ │             │      └┬──────┬──────┬──────┬──────┬┘ │         │         │ ┃
┃ │             │     -0.01  0.24   0.50   0.76  1.01  │         │         │ ┃
┃ ├─────────────┼──────────────────────────────────────┼─────────┼─────────┤ ┃
┃ │ Predictions │      ┌─────────────────────────────┐ │ NaNs: 0 │  Start: │ ┃
┃ │             │ 278.0┤ ██ ████████ █████    ██ ████│ │  dtype: │       0 │ ┃
┃ │             │ 231.7┤█████████████████████████████│ │ float64 │    End: │ ┃
┃ │             │ 185.3┤█████████████████████████████│ │         │    9999 │ ┃
┃ │             │ 139.0┤█████████████████████████████│ │         │         │ ┃
┃ │             │  92.7┤█████████████████████████████│ │         │         │ ┃
┃ │             │  46.3┤█████████████████████████████│ │         │         │ ┃
┃ │             │   0.0┤█████████████████████████████│ │         │         │ ┃
┃ │             │      └┬──────┬──────┬──────┬──────┬┘ │         │         │ ┃
┃ │             │     -0.01  0.24   0.50   0.76  1.01  │         │         │ ┃
┃ ╰─────────────┴──────────────────────────────────────┴─────────┴─────────╯ ┃
┃                            Residual Diagnostics                            ┃
┃ ╭─────────────────┬──────────────────────────────────────┬───────────────╮ ┃
┃ │     Metric Name │ Result                               │ Parameters    │ ┃
┃ ├─────────────────┼──────────────────────────────────────┼───────────────┤ ┃
┃ │       Residuals │ 0      -0.000481                     │ {}            │ ┃
┃ │     (residuals) │ 1       0.385508                     │               │ ┃
┃ │                 │ 2      -0.119003                     │               │ ┃
┃ │                 │ 3       0.308599                     │               │ ┃
┃ │                 │ 4      -0.194476                     │               │ ┃
┃ │                 │           ...                        │               │ ┃
┃ │                 │ 9995   -0.157386                     │               │ ┃
┃ │                 │ 9996   -0.399353                     │               │ ┃
┃ │                 │ 9997    0.495072                     │               │ ┃
┃ │                 │ 9998    0.006810                     │               │ ┃
┃ │                 │ 9999   -0.131237                     │               │ ┃
┃ │                 │ Length: 10000, dtype: float64        │               │ ┃
┃ ├─────────────────┼──────────────────────────────────────┼───────────────┤ ┃
┃ │       Ljung Box │ 'zero-size array to reduction        │ {}            │ ┃
┃ │      Statistics │ operation maximum which has no       │               │ ┃
┃ │ (ljung_box_sta… │ identity'                            │               │ ┃
┃ ├─────────────────┼──────────────────────────────────────┼───────────────┤ ┃
┃ │     Mean of the │       ┌────────────────────────────┐ │ {}            │ ┃
┃ │       Residuals │  0.138┤ ▖  ▌        ▗              │ │               │ ┃
┃ │ (residuals_mea… │  0.092┤▐▌  ▌▗▌     ▗█▄ ▖           │ │               │ ┃
┃ │                 │  0.047┤▐▙▌ ▙█▌▖▄▗▗ ▐▝█▖█▌ ▄    ▗  ▖│ │               │ ┃
┃ │                 │  0.001┤▟█▙▛██▙▌██▟▟▟ █▚█▌▟▐▐▖▗▟█▟█▛│ │               │ ┃
┃ │                 │ -0.044┤▐██▘█▐▐█▝█▝▌▘ ▝▐▛▝█▝▛▙█▌▜▌▌ │ │               │ ┃
┃ │                 │ -0.090┤▝▜▜ █  ▀ ▝             ▘ ▘▌ │ │               │ ┃
┃ │                 │ -0.135┤    ▜                     ▘ │ │               │ ┃
┃ │                 │       └┬──────┬──────┬─────┬──────┬┘ │               │ ┃
┃ │                 │       1.0   25.8   50.5  75.2 100.0  │               │ ┃
┃ ├─────────────────┼──────────────────────────────────────┼───────────────┤ ┃
┃ │        Standard │      ┌─────────────────────────────┐ │ {}            │ ┃
┃ │    Deviation of │ 0.454┤▗▟▄▙▚▖▄▖▄▙▄▄▄▌▄▜▄▄█▄▟▟▄▄▄▙▖▄▄│ │               │ ┃
┃ │   the Residuals │ 0.379┤▛▘▝▀▝▀▀▚▀▘▘▘▀▝▝  ▝▘ ▘▀▘ ▀▜▀▀▘│ │               │ ┃
┃ │ (residuals_std) │ 0.303┤▌                            │ │               │ ┃
┃ │                 │ 0.227┤▌                            │ │               │ ┃
┃ │                 │ 0.151┤▌                            │ │               │ ┃
┃ │                 │ 0.076┤▌                            │ │               │ ┃
┃ │                 │ 0.000┤▌                            │ │               │ ┃
┃ │                 │      └┬──────┬──────┬──────┬──────┬┘ │               │ ┃
┃ │                 │      1.0   25.8   50.5   75.2 100.0  │               │ ┃
┃ ╰─────────────────┴──────────────────────────────────────┴───────────────╯ ┃
┃                        Forecast Errors - Regression                        ┃
┃ ╭─────────────────┬──────────────────────────────────────┬───────────────╮ ┃
┃ │   Mean Absolute │      ┌─────────────────────────────┐ │ {}            │ ┃
┃ │     Error (mae) │ 0.388┤ ▟▗▙▚▖▗ ▗▌ ▄▗▌▄▄▄▗▙▗▐▟▗▗ ▄ ▗▗│ │               │ ┃
┃ │                 │ 0.323┤▞▘▜█▝▛▜▚▜▛▀▘▜▀▀▝ ▀▘▀▀▛▛▀▀▜▙▀▘│ │               │ ┃
┃ │                 │ 0.259┤▌                            │ │               │ ┃
┃ │                 │ 0.194┤▌                            │ │               │ ┃
┃ │                 │ 0.130┤▌                            │ │               │ ┃
┃ │                 │ 0.065┤▌                            │ │               │ ┃
┃ │                 │ 0.000┤▌                            │ │               │ ┃
┃ │                 │      └┬──────┬──────┬──────┬──────┬┘ │               │ ┃
┃ │                 │      1.0   25.8   50.5   75.2 100.0  │               │ ┃
┃ ├─────────────────┼──────────────────────────────────────┼───────────────┤ ┃
┃ │   Mean Absolute │     ┌──────────────────────────────┐ │ {}            │ ┃
┃ │      Percentage │ 55.6┤                  ▟           │ │               │ ┃
┃ │    Error (mape) │ 46.3┤▟                 █           │ │               │ ┃
┃ │                 │ 37.1┤█                 █           │ │               │ ┃
┃ │                 │ 27.8┤█                 █           │ │               │ ┃
┃ │                 │ 18.5┤█             ▟   █           │ │               │ ┃
┃ │                 │  9.3┤█ ▖     ▗ ▗▌ ▟█  ▐█ ▐  ▖ ▗ ▗  │ │               │ ┃
┃ │                 │  0.0┤▛▀▜▚▚█▞█▀▀▛▝▀█▛▞█▟▀▛▀▙▀▀▜▟▛▀▚▛│ │               │ ┃
┃ │                 │     └┬──────┬───────┬──────┬──────┬┘ │               │ ┃
┃ │                 │     1.0   25.8    50.5   75.2 100.0  │               │ ┃
┃ ├─────────────────┼──────────────────────────────────────┼───────────────┤ ┃
┃ │  Symmetric Mean │      ┌─────────────────────────────┐ │ {}            │ ┃
┃ │        Absolute │ 0.444┤ ▞▖▄▌ ▗ ▟▌▗▄▄▌▙▄▟▗▙▗▄▗▗▄▗▙ ▖▗│ │               │ ┃
┃ │      Percentage │ 0.370┤▞ ▜▀▝▛▀█▀▀▛▘▜▜▝ ▘▀▀▘▀▀▌ ▀▝█▀▌│ │               │ ┃
┃ │   Error (smape) │ 0.296┤▌                            │ │               │ ┃
┃ │                 │ 0.222┤▌                            │ │               │ ┃
┃ │                 │ 0.149┤▌                            │ │               │ ┃
┃ │                 │ 0.075┤▌                            │ │               │ ┃
┃ │                 │ 0.001┤▌                            │ │               │ ┃
┃ │                 │      └┬──────┬──────┬──────┬──────┬┘ │               │ ┃
┃ │                 │      1.0   25.8   50.5   75.2 100.0  │               │ ┃
┃ ├─────────────────┼──────────────────────────────────────┼───────────────┤ ┃
┃ │    Mean Squared │      ┌─────────────────────────────┐ │ {             │ ┃
┃ │     Error (mse) │ 0.209┤ ▟ ▄▚   ▗▖ ▗▗▌▄▄▄▗▙ ▐▟   ▖   │ │     'squared… │ ┃
┃ │                 │ 0.174┤▗▘▜█▐▙█▌█▙█▛▟▚▜▝▀▟▛▜▟█▛▜▟█▖▟▞│ │ True          │ ┃
┃ │                 │ 0.139┤▌  ▝ ▘▝▜▝▘  ▝    ▝▘ ▘ ▘  ▜▀▘ │ │ }             │ ┃
┃ │                 │ 0.104┤▌                            │ │               │ ┃
┃ │                 │ 0.070┤▌                            │ │               │ ┃
┃ │                 │ 0.035┤▌                            │ │               │ ┃
┃ │                 │ 0.000┤▌                            │ │               │ ┃
┃ │                 │      └┬──────┬──────┬──────┬──────┬┘ │               │ ┃
┃ │                 │      1.0   25.8   50.5   75.2 100.0  │               │ ┃
┃ ├─────────────────┼──────────────────────────────────────┼───────────────┤ ┃
┃ │       Root Mean │      ┌─────────────────────────────┐ │ {             │ ┃
┃ │   Squared Error │ 0.457┤ ▞▄▙▚▖▄▖▄▙▄▄▗▌▟▜▄▄█▄▟▟▄▄▄▙ ▗▄│ │     'squared… │ ┃
┃ │          (rmse) │ 0.381┤▞ ▝▀ ▀▀▚▀▘▘▘▀▝▝  ▝▘ ▘▀▘ ▀▜▀▀▘│ │ False         │ ┃
┃ │                 │ 0.305┤▌                            │ │ }             │ ┃
┃ │                 │ 0.229┤▌                            │ │               │ ┃
┃ │                 │ 0.153┤▌                            │ │               │ ┃
┃ │                 │ 0.077┤▌                            │ │               │ ┃
┃ │                 │ 0.000┤▌                            │ │               │ ┃
┃ │                 │      └┬──────┬──────┬──────┬──────┬┘ │               │ ┃
┃ │                 │      1.0   25.8   50.5   75.2 100.0  │               │ ┃
┃ ├─────────────────┼──────────────────────────────────────┼───────────────┤ ┃
┃ │       Root Mean │      ┌─────────────────────────────┐ │ {             │ ┃
┃ │     Squared Log │ 0.314┤ ▞▖▄▚▖▄▖▄▙▄▄▗▌▄▜▄▄█▄▐▟▄▄▗▙▖▄▗│ │     'squared… │ ┃
┃ │   Error (rmsle) │ 0.261┤▞ ▝▀ ▀▀▚▀▘▘▘▜▝▝  ▝▘ ▀▀▘ ▀▜▀▀▘│ │ False         │ ┃
┃ │                 │ 0.209┤▌                            │ │ }             │ ┃
┃ │                 │ 0.157┤▌                            │ │               │ ┃
┃ │                 │ 0.105┤▌                            │ │               │ ┃
┃ │                 │ 0.053┤▌                            │ │               │ ┃
┃ │                 │ 0.000┤▌                            │ │               │ ┃
┃ │                 │      └┬──────┬──────┬──────┬──────┬┘ │               │ ┃
┃ │                 │      1.0   25.8   50.5   75.2 100.0  │               │ ┃
┃ ├─────────────────┼──────────────────────────────────────┼───────────────┤ ┃
┃ │       R-squared │      ┌─────────────────────────────┐ │ {}            │ ┃
┃ │         (r_two) │ -0.49┤                   ▗▌    ▟   │ │               │ ┃
┃ │                 │ -0.71┤▐    ▟▐ ▗▖▗▖▄  ▗▖  ▐▌▖   █ ▗▗│ │               │ ┃
┃ │                 │ -0.93┤▝▖▟▌ █▐▙█▌▐▙▜▗▄█▌▐▖▐██▗▙██▙▟▌│ │               │ ┃
┃ │                 │ -1.15┤ ▙▜▙▜▀█▛█▛█▌▐▌▛▜█▟▙▞▜▜█▜▛▛▘▜▌│ │               │ ┃
┃ │                 │ -1.37┤ ▜▐▜   ▘▝▌▜ ▝▌▌▐▝▝█ ▐        │ │               │ ┃
┃ │                 │ -1.59┤              ▘     ▐        │ │               │ ┃
┃ │                 │ -1.81┤                    ▐        │ │               │ ┃
┃ │                 │      └┬──────┬──────┬──────┬──────┬┘ │               │ ┃
┃ │                 │      1.0   25.8   50.5   75.2 100.0  │               │ ┃
┃ ╰─────────────────┴──────────────────────────────────────┴───────────────╯ ┃
┃                                                                            ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

</details>

You can also show the results in a Report easily:

```python
import numpy as np
from krisi import score

score(
    y=np.random.rand(10000),
    predictions=np.random.rand(10000),
    calculation="rolling",
).generate_report('pdf')
```


Generates:

<div>
  <img src="https://raw.githubusercontent.com/dream-faster/krisi/main/docs/images/pdf_example.svg" alt="PDF report on Metrics over time" width="100%" >
</div>

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


## Our Open-core Time Series Toolkit

[![Krisi](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_krisi.svg)](https://github.com/dream-faster/krisi)
[![Fold](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_fold.svg)](https://github.com/dream-faster/fold)
[![Fold/Models](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_fold_models.svg)](https://github.com/dream-faster/fold-models)
[![Fold/Wrapper](https://raw.githubusercontent.com/dream-faster/fold/main/docs/images/overview_diagrams/dream_faster_suite_fold_wrappers.svg)](https://github.com/dream-faster/fold-wrappers)

If you want to try them out, we'd love to hear about your use case and help, [please book a free 30-min call with us](https://calendly.com/nowcasting/consultation)!

## Advanced usage

Creating ``Metric``s with metadata.

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
sc.print(with_info=True)
```
<details>
<summary>
Outputs 👇
</summary>

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

</details>

## Contribution

Join our [Discord](https://discord.gg/EKJQgfuBpE) for live discussion!

Submit an issue or reach out to us on info at dream-faster.ai for any inquiries.

