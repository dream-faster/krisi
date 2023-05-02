import numpy as np
import pandas as pd


def corr_without_symmetry(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr()
    return corr.mask(np.tril(np.ones(corr.shape)).astype(np.bool_))


def unroll(df: pd.DataFrame, datetime: pd.DatetimeIndex) -> pd.Series:
    df = df.melt(ignore_index=False).dropna()

    df.index = df.index + "_" + df.variable
    df = df.drop(columns=["variable"])
    df = df.rename(columns={"value": datetime})
    return df.squeeze()
