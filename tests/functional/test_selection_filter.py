import numpy as np

import grizzlars as gl


def test_column_access_and_len():
    df = gl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 3


def test_iloc_and_head_tail():
    df = gl.DataFrame({"v": [0, 1, 2, 3, 4]})
    head = df.head(2)
    tail = df.tail(2)
    assert len(head) == 2
    assert len(tail) == 2


def test_filter_mask():
    df = gl.DataFrame({"n": [1, 2, 3, 4, 5]})
    mask = df["n"] > 2
    filtered = df.filter(mask)
    assert len(filtered) == 3
    assert all(x >= 3 for x in filtered["n"])
