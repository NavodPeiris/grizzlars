import grizzlars as gl


def main() -> None:
    # Build a DataFrame from a dict — index auto-assigned 0..4
    df = gl.DataFrame(
        {
            "price":  [100.5, 200.0, 150.75, 300.0, 250.5],
            "volume": [1000,  2000,   1500,  3000,  2500],
            "symbol": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
            "active": [True,   True,    False,  True,   True],
        }
    )

    print("── DataFrame ──────────────────────────────")
    print(df)

    print("\n── Statistics ─────────────────────────────")
    print(f"  mean price : {df.mean('price'):.2f}")
    print(f"  std  price : {df.std('price'):.2f}")
    print(f"  min  price : {df.min('price'):.2f}")
    print(f"  max  price : {df.max('price'):.2f}")
    print(f"  sum  volume: {df.sum('volume'):.0f}")

    print("\n── describe() ─────────────────────────────")
    print(df.describe())

    print("\n── sorted by price descending ─────────────")
    print(df.sort_values("price", ascending=False))

    print("\n── head(3) ────────────────────────────────")
    print(df.sort_index().head(3))

    print("\n── dtypes ─────────────────────────────────")
    for col, t in df.dtypes().items():
        print(f"  {col}: {t}")

    # Round-trip to CSV
    df.to_csv("output.csv")
    df2 = gl.read_csv("output.csv", index_col="index")
    print(f"\n── read back from CSV: {df2.shape[0]} rows, {df2.shape[1]} cols ──")
    print(df2.head())


if __name__ == "__main__":
    main()
