import pandas as pd


def check_gaps(
    df: pd.DataFrame, time_delta: pd.Timedelta = pd.Timedelta("00:60:00")
) -> pd.Series:
    mask = df.index.to_series().diff() == time_delta
    return mask[mask == False]


def check_frequency(df):
    return pd.infer_freq(df.index)


def check_consistency(df, name):
    print(f"-- Consistency for {name} --")
    print(f"Gaps should be 1: {len(check_gaps(df))}")
    print(f"Frequency should be 'H': {check_frequency(df)}")
    print("----------------------------")
