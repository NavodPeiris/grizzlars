import numpy as np

import grizzlars as gl


def test_rolling_mean_simple():
    df = gl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})
    rm = df.rolling_mean("v", window=2)
    # Expect length 4 and rm[1] == mean(1,2) == 1.5
    assert len(rm) == 4
    assert abs(rm[1] - 1.5) < 1e-12
