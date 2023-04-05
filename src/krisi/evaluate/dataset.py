import pandas as pd


def __check_gaps(
    df: pd.DataFrame, time_delta: pd.Timedelta = pd.Timedelta("00:60:00")
) -> pd.Series:
    mask = df.index.to_series().diff() == time_delta
    return mask[mask == False]


def __check_frequency(df: pd.DataFrame) -> str:
    return pd.infer_freq(df.index)


def __check_nans(df: pd.DataFrame):
    return df[df.isna()]


def check_consistency(df: pd.DataFrame, name: str):
    print(f"-- Consistency for {name} --")
    print(f"Gaps should be 1: {len(__check_gaps(df))}")
    print(f"Frequency should be 'H': {__check_frequency(df)}")
    print(f"These are the NaN values: {__check_nans(df)}")
    print("----------------------------")
