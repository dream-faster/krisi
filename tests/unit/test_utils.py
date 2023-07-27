import pandas as pd

from krisi.utils.data import combinatorial_shuffling_in_chunks, shuffle_df_in_chunks


def test_shuffle_df_in_chunks():
    chunk_size = 10
    df = pd.DataFrame(
        [[i, i + 1, i + 2] for i in range(chunk_size) for _ in range(chunk_size)],
        columns=["a", "b", "c"],
    )
    df_copy = shuffle_df_in_chunks(df.copy(), chunk_size)
    assert not df_copy.equals(df)
    assert df_copy.shape == df.shape
    assert df_copy.index.equals(df.index)
    assert df_copy.columns.equals(df.columns)
    assert df_copy["a"][:chunk_size].nunique() == 1
    assert df_copy["b"][:chunk_size].nunique() == 1
    assert df_copy["c"][:chunk_size].nunique() == 1


def test_combinatorical_shuffling_in_chunks():
    chunk_size = 10
    df = pd.DataFrame(
        [[i, i + 1, i + 2] for i in range(chunk_size) for _ in range(chunk_size)],
        columns=["a", "b", "c"],
    )
    for i in range(10):
        df_copy = combinatorial_shuffling_in_chunks(df.copy(), chunk_size, i)
        assert not df_copy.equals(df)
        assert df_copy.shape == df.shape
        assert df_copy.index.equals(df.index)
        assert df_copy.columns.equals(df.columns)
        assert df_copy["a"][:chunk_size].nunique() == 1
        assert df_copy["b"][:chunk_size].nunique() == 1
        assert df_copy["c"][:chunk_size].nunique() == 1
