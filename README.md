

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

<h3 align="center"> <b>(/creesee/)</b></h3>
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
<br/>
  


## Installation
---

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
<br/>

## Quickstart
---
```python
from krisi.evaluate import ScoreCard

sc = ScoreCard("<your_model_name>")

# Generating random target and predictions for demonstration purposes
target, predictions = np.random.rand(1000), np.random.rand(1000)

# Calculate predefined metrics
for metric in sc.get_default_metrics():
    metric.evaluate(target, predictions)

# Add a new metric
sc["own_metric"] = (target - predictions).mean()

# Print the result
sc.print_summary()

```




<br/>
<br/>

## Contribution
---

The project uses ``isort`` and ``black`` for formatting.

Submit an issue or reach out to us on info at dream-faster.ai for any inquiries.
