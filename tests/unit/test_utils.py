import pandas as pd

from krisi.utils.data import shuffle_df_in_chunks


def test_shuffle_df_in_chunks():
    df = pd.DataFrame(
        [[i, i + 1, i + 2] for i in range(10) for _ in range(10)],
        columns=["a", "b", "c"],
    )
    chunk_size = 3
    df_copy = shuffle_df_in_chunks(df.copy(), chunk_size)
    assert not df_copy.equals(df)
    assert df_copy.shape == df.shape
    assert df_copy.index.equals(df.index)
    assert df_copy.columns.equals(df.columns)
    assert df_copy["a"][:chunk_size].nunique() == 1
