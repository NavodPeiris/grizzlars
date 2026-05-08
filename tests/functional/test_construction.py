import numpy as np

import grizzlars as gl


def test_construct_from_dict():
    data = {"x": [10, 20, 30], "y": [1.5, 2.5, 3.5]}
    df = gl.DataFrame(data)
    assert len(df) == 3
    assert set(df.columns) == {"x", "y"}


def test_construct_from_numpy():
    x = np.array([1, 2, 3], dtype=np.int64)
    y = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    df = gl.DataFrame({"x": x, "y": y})
    assert df.dtypes()["x"] in ("int64",)
    assert df.dtypes()["y"] in ("double",)
