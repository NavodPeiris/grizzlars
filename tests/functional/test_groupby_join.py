import grizzlars as gl


def test_groupby_sum():
    df = gl.DataFrame({"cat": ["a", "a", "b"], "val": [1, 2, 3]})
    gb = df.groupby("cat").agg({"val": "sum"})
    # Expect two groups: a -> 3, b -> 3
    vals = {row[0]: row[1] for row in zip(gb["cat"], gb["val"]) }
    assert vals.get("a") == 3
    assert vals.get("b") == 3


def test_join_inner_and_left():
    left = gl.DataFrame({"id": [1, 2, 3], "x": [10, 20, 30]})
    right = gl.DataFrame({"id": [2, 3], "y": [200, 300]})
    # make id the index for both
    left_i = gl.DataFrame({"x": left["x"]}, index=left["id"])
    right_i = gl.DataFrame({"y": right["y"]}, index=right["id"])

    inner = left_i.join(right_i, how="inner")
    left_join = left_i.join(right_i, how="left")

    assert len(inner) == 2
    assert len(left_join) == 3
