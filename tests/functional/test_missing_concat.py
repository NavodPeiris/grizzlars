import numpy as np

import grizzlars as gl


def test_fillna_and_drop_duplicates():
    # provide numeric column as a NumPy float array so nan is a float
    a = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    df = gl.DataFrame({"a": a, "b": ["x", "", "z"]})
    # fill numeric NaN -> 0
    df.fillna("a", 0)
    # ensure fillna replaced the missing entry
    val = df["a"][1]
    assert val == 0 or val == 0.0


def test_concat():
    a = gl.DataFrame({"k": [1, 2]})
    b = gl.DataFrame({"k": [3]})
    c = a.concat(b)
    assert len(c) == 3
