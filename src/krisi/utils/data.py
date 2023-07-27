import itertools
from random import sample
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from krisi.sharedtypes import Task


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
    from statsmodels.tsa.arima_process import ArmaProcess

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


def create_probabilities(num_labels: int = 3, num_samples: int = 1000):
    probabilities = np.random.rand(num_samples, num_labels)
    probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
    probabilities = list(zip(*probabilities))
    probabilities = pd.DataFrame({i: probabilities[i] for i in range(num_labels)})
    return probabilities


def generate_random_classification(
    num_labels: int = 3, num_samples: int = 1000
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.Series]:
    y = pd.Series(np.random.randint(0, num_labels, num_samples))
    predictions = pd.Series(np.random.randint(0, num_labels, num_samples))
    sample_weight = pd.Series(np.random.random(num_samples))

    probabilities = create_probabilities(num_labels=num_labels, num_samples=num_samples)
    return y, predictions, probabilities, sample_weight


def generate_synthetic_data(
    num_features_increasingly_important: int = 1,  # component, increasing importance, added to `y`
    num_features_decreasingly_important: int = 1,  # component, decreasing importance, added to `y`
    num_features_stationary: int = 1,  # component + autoregressive transformation, static importance, added to `y`
    num_features_correlated_noise: int = 1,  # component, static importance, added to `y`
    num_features_uncorrelated_noise: int = 1,  # pure noise, added as features ONLY to `X`
    num_features_oscillates_importance: int = 1,  # component, oscilating importance, added to `y`
    num_obs: int = 200,
    index: Optional[pd.Index] = None,
    task: Union[str, Task] = Task.regression,
):
    task = Task.from_str(task)
    # Based on https://github.com/convergenceIM/alpha-scientist/blob/master/content/04_Walk_Forward_Modeling.ipynb
    if index is None:
        index = pd.RangeIndex(0, num_obs)

    def add_memory(s, n_days=50, memory_strength=0.1):
        """adds autoregressive behavior to series of data"""
        out = (1 - memory_strength) * s + memory_strength * s.ewm(n_days).mean()
        return out

    class Relevance:
        increasingly_important = np.linspace(0.5, 1.5, num_obs)
        decreasingly_important = np.linspace(1.5, 0.5, num_obs)
        oscillates_importance = pd.Series(
            np.sin(2 * np.pi * np.linspace(0, 1, num_obs) * 2) + 1, index=index
        )
        stationary = 1.0

    features = [
        (
            "increasingly_important",
            num_features_increasingly_important,
            Relevance.increasingly_important,
        ),
        (
            "decreasingly_important",
            num_features_decreasingly_important,
            Relevance.decreasingly_important,
        ),
        (
            "oscillates_importance",
            num_features_oscillates_importance,
            Relevance.oscillates_importance,
        ),
        ("stationary", num_features_stationary, Relevance.stationary),
    ]

    # generate feature data
    def get_feature(name: str, relevance) -> pd.Series:
        feature = pd.Series(np.random.randn(num_obs), index=index, name=name)
        feature = add_memory(feature, 10, 0.1)
        feature *= relevance
        return feature

    def get_features(
        feature_type: str, num_feature: int, relevance: Relevance
    ) -> pd.DataFrame:
        df = pd.concat(
            [get_feature(f"{feature_type}_{i}", relevance) for i in range(num_feature)],
            axis="columns",
        )
        return df

    features_correlated_noise = pd.concat(
        [
            pd.Series(
                np.random.randn(num_obs) * 3, index=index, name=f"correlated_noise_{i}"
            )
            for i in range(num_features_correlated_noise)
        ],
        axis="columns",
    )
    X = pd.concat(
        [
            get_features(feature_type, num_feature, relevance)
            for feature_type, num_feature, relevance in features
        ]
        + [features_correlated_noise],
        axis="columns",
    )

    y = (
        X.sum(axis="columns") + np.random.randn(num_obs) * 3
    )  # We add an uncorrelated noise to the target
    y.name = "y"

    X = pd.concat(
        [X]
        + [
            pd.Series(
                np.random.randn(num_obs) * 3,
                index=index,
                name=f"uncorrelated_noise_{i}",
            )
            for i in range(num_features_uncorrelated_noise)
        ],
        axis="columns",
    )

    if task == Task.classification:
        y = (y > y.median()).astype("int")
    elif task == Task.multi_classification:
        label = y.copy()
        label[y <= y.quantile(0.33)] = 0
        label[y >= y.quantile(0.66)] = 2
        label[(y > y.quantile(0.33)) & (y < y.quantile(0.66))] = 1

        y = label

    return X, y


def generate_synthetic_predictions_binary(
    target: pd.Series,
    sample_weights: Optional[pd.Series] = None,
    index: Optional[pd.Index] = None,
    smoothing_window: Optional[int] = None,
) -> pd.DataFrame:
    if index is None:
        index = target.index
    if sample_weights is None:
        sample_weights = pd.Series(np.ones(len(target)), index=index)
    target = target.copy()

    weighted_target = target * sample_weights
    expanding_target_mean = weighted_target.expanding(min_periods=1).mean()
    prob_mean_class_1 = expanding_target_mean.mean()
    prob_std_class_1 = expanding_target_mean.std()
    prob_class_1 = pd.Series(
        np.random.normal(prob_mean_class_1, prob_std_class_1, len(index)).clip(0, 1)
    )

    if smoothing_window is not None:
        prob_class_1 = prob_class_1.rolling(
            window=smoothing_window, min_periods=1
        ).mean()

    prob_class_0 = 1 - prob_class_1
    return pd.DataFrame(
        {
            "predictions_RandomClassifier": (prob_class_1 > prob_class_1.mean())
            .astype("int")
            .values,
            "probabilities_RandomClassifier_0": prob_class_0.values,
            "probabilities_RandomClassifier_1": prob_class_1.values,
        },
        index=index,
    )


def shuffle_df_in_chunks(
    df: pd.DataFrame, chunk_size: Union[int, float]
) -> pd.DataFrame:
    """Shuffles a dataframe by rows in chunks.

    Parameters
    ----------
    df : pd.DataFrame
        Data, whose rows should be shuffled in chunks
    chunk_size : Union[int, float]
        Either a fixed chunk size or a fraction of the total number of rows

    Returns
    -------
    pd.DataFrame
        Shuffled DataFrame by rows in chunks.
    """
    original_index = df.index
    df = df.reset_index(drop=True)
    chunk_size = (
        int(chunk_size * len(df)) if isinstance(chunk_size, float) else chunk_size
    )
    num_rows = df.shape[0]
    num_chunks = num_rows // chunk_size
    shuffled_idx = []
    for i in sample(list(range(num_chunks)), num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
        shuffled_idx.append(df.index[start_idx:end_idx])

    # If there are any remaining rows, shuffle and append them as well
    if num_rows % chunk_size != 0:
        shuffled_idx.append(df.index[num_chunks * chunk_size :])

    df = df.copy()
    df.index = df.index[np.concatenate(shuffled_idx)]
    df.sort_index(inplace=True)
    df.index = original_index

    return df


def combinatorial_shuffling_in_chunks(
    df: pd.DataFrame, chunk_size: int, select_index: int
) -> pd.DataFrame:
    original_index = df.index
    df = df.reset_index(drop=True)
    chunks = [(i, i + chunk_size) for i in range(0, len(df), chunk_size)]
    if chunks[-1][1] > len(df):
        chunks[-2] = (chunks[-2][0], len(df))
        chunks = chunks[:-1]

    premuted_chunks = list(itertools.permutations(chunks))[
        1:
    ]  # skip the first one, which is the original order

    flatten_ranges = [
        id for start, stop in premuted_chunks[select_index] for id in range(start, stop)
    ]
    df = df.copy()
    df.index = df.index[flatten_ranges]
    df.sort_index(inplace=True)
    df.index = original_index
    return df
