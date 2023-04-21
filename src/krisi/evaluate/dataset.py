import pandas as pd


def __check_gaps(df: pd.DataFrame, time_delta: pd.Timedelta) -> pd.Series:
    mask = df.index.to_series().diff() == time_delta
    return mask[mask == False]


def __check_frequency(df: pd.DataFrame) -> str:
    return pd.infer_freq(df.index)


def __check_nans(df: pd.DataFrame):
    return df[df.isna()]


def check_consistency(
    df: pd.DataFrame,
    title: str = "",
    time_delta: pd.Timedelta = pd.Timedelta("00:60:00"),
):
    """
    Prints a summary on:

    * the number of NaNs
    * the frequency of the dataset
    * if there are irregular gaps within the indexes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that we would like to check the inconsistency of.
    title : str
        Title of the report
    time_delta: pd.Timedelta
        The size of gaps within indexes, to check irregular indexes.

    """
    print(f"-- Consistency for {title} --")
    print(f"Gaps should be 1: {len(__check_gaps(df, time_delta))}")
    print(f"Frequency should be 'H': {__check_frequency(df)}")
    print(f"These are the NaN values: {__check_nans(df)}")
    print("----------------------------")
