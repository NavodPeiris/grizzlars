import numpy as np

import grizzlars as gl


def test_with_column_and_assign_select():
    df = gl.DataFrame({"price": [10.0, 20.0, 30.0], "symbol": ["A", "B", "A"]})

    # non-mutating with_column and assign
    df2 = df.with_column("log_price", np.log(df["price"]))
    assert "log_price" in df2.columns

    df3 = df.assign(rank=[1, 2, 3])
    assert "rank" in df3.columns

    # select subset
    sel = df3.select(["symbol", "price"]) 
    assert list(sel.columns) == ["symbol", "price"]
