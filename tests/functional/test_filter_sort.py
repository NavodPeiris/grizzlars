import grizzlars as gl


def test_filter_string_mode_and_sort_aliases():
    df = gl.DataFrame({"price": [100, 50, 200], "idx": [0, 1, 2]})

    # string operator mode
    cheap = df.filter("price", "<", 150)
    assert len(cheap) == 2

    # combined mask
    mask = (df["price"] > 50) & (df["price"] < 250)
    filtered = df.filter(mask)
    assert len(filtered) == 2

    # sort_values alias for sort
    s1 = df.sort("price")
    s2 = df.sort_values("price")
    assert list(s1["price"]) == list(s2["price"]) 

    # sort_index (should preserve index-based sort)
    si = df.sort_index()
    assert len(si) == len(df)
