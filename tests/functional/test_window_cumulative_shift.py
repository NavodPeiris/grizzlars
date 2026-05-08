import math

import numpy as np

import grizzlars as gl


def _arr(x):
    return list(x)


def test_cumulative_and_shift_pct_change():
    df = gl.DataFrame({"v": [1.0, 2.0, 3.0, 4.0]})

    csum = _arr(df.cumsum("v"))
    assert csum == [1.0, 3.0, 6.0, 10.0]

    cprod = _arr(df.cumprod("v"))
    assert cprod == [1.0, 2.0, 6.0, 24.0]

    cmin = _arr(df.cummin("v"))
    assert cmin == [1.0, 1.0, 1.0, 1.0]

    # shift and pct_change produce arrays with same length; first element is NaN
    s = df.shift("v", n=1)
    assert len(s) == 4
    assert math.isnan(s[0])

    pct = df.pct_change("v")
    assert len(pct) == 4
    assert math.isnan(pct[0])
