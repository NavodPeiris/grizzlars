import csv
import os

import pytest
import numpy as np

import grizzlars as gl


def test_read_write_csv(tmp_path):
    data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
    df = gl.DataFrame(data)

    out = tmp_path / "test_io.csv"
    df.to_csv(str(out))

    df2 = gl.read_csv(str(out))

    # Check columns and lengths — read_csv may add an `index` column, allow that
    assert set(df.columns).issubset(set(df2.columns))
    assert len(df2) == 3

    # Numeric equality for column `a`
    np.testing.assert_array_equal(np.asarray(df2["a"]), np.array([1, 2, 3]))

