from typing import List

import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess


def make_it_stationary(ds: pd.Series) -> pd.Series:
    """If the ds, is a random walk with drift, take first differences to make it stationary."""
    print(f"Making the Series {ds.name} stationary.")
    return ds.diff().dropna()


def generating_arima_synthetic_data(
    target_col: str,
    nsample: int = 1000,
    ar: List[float] = [1.0, -0.9],
    ma: List[float] = [1.0],
) -> pd.Series:
    """Generate Synthetic data with ARIMA process."""
    print(f"Generating Synthetic ARIMA process with nsamples:{nsample}")

    # Plot 1: AR parameter = +0.9,
    # exponent of PHI is opposite due to original signal processing literature convention
    ar = np.array(ar)
    ma = np.array(ma)
    AR_object = ArmaProcess(ar, ma)
    simulated_data = AR_object.generate_sample(nsample=nsample)

    return pd.Series(
        simulated_data,
        name=target_col,
        index=pd.date_range(end="2022", periods=nsample, freq="D"),
    )
