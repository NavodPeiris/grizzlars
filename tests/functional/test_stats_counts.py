import numpy as np

import grizzlars as gl


def test_quantile_corr_nunique_value_counts_and_describe():
    df = gl.DataFrame({
        "price": [1.0, 2.0, 3.0],
        "volume": [10.0, 20.0, 30.0],
        "symbol": ["a", "a", "b"],
    })

    # quantile
    q = df.quantile("price", 0.5)
    assert abs(q - 2.0) < 1e-12

    # correlation (perfect linear relationship)
    corr = df.corr("price", "volume")
    assert abs(corr - 1.0) < 1e-12

    # nunique / unique
    assert df.nunique("symbol") == 2
    uniq = df.unique("symbol")
    assert set(uniq) == {"a", "b"}

    # n_missing (none)
    assert df.n_missing("price") == 0

    # value_counts returns a small DataFrame-ish result; sum of counts == rows
    vc = df.value_counts("symbol")
    assert sum(vc["count"]) == len(df)

    # describe returns a DataFrame with numeric stats
    stats = df.describe()
    assert "price" in stats.columns
