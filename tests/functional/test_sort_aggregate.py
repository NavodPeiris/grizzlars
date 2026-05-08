import numpy as np

import grizzlars as gl


def test_sort_and_aggregate():
    df = gl.DataFrame({"k": [3, 1, 2], "v": [10.0, 20.0, 30.0]})
    sorted_df = df.sort("k")
    assert list(sorted_df["k"]) == [1, 2, 3]

    mean_v = df.mean("v")
    sum_v = df.sum("v")
    assert abs(mean_v - 20.0) < 1e-12
    assert abs(sum_v - 60.0) < 1e-12
