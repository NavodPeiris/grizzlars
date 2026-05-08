<p align="center">
  <img src="https://github.com/NavodPeiris/grizzlars/blob/main/banner.png?raw=true" width="700"/>
</p>

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/github/license/NavodPeiris/grizzlars"></a>
    <a href="https://github.com/NavodPeiris/grizzlars/releases"><img src="https://img.shields.io/github/v/release/NavodPeiris/grizzlars?color=ffa"></a>
    <a href="support os"><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.10+-aff.svg"></a>
    <a href="https://github.com/NavodPeiris/grizzlars/issues"><img src="https://img.shields.io/github/issues/NavodPeiris/grizzlars?color=9cc"></a>
    <a href="https://github.com/NavodPeiris/grizzlars/stargazers"><img src="https://img.shields.io/github/stars/NavodPeiris/grizzlars?color=ccf"></a>
    <a href="https://pypi.org/project/grizzlars/"><img src="https://static.pepy.tech/badge/grizzlars"></a>
</p>

# Grizzlars

> A Python DataFrame library backed by a multithreaded C++ engine — built for speed.

grizzlars wraps [DataFrame](https://github.com/hosseinmoein/DataFrame), a high-performance C++ DataFrame, with a clean Python API. Columns are stored as typed `std::vector<T>` buffers — no GIL-bound Python object overhead. Sort, filter, groupby, join, and aggregate operations run in parallel across all CPU cores automatically.

---

## Installation

Requires Python 3.10 or higher

```bash
pip install grizzlars
```

---

## Quick Start

```python
import grizzlars as gl

df = gl.DataFrame({
  "symbol": ["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
  "price":  [189.3,  175.1,   415.2,  185.0,  502.7],
  "volume": [52_000_000, 18_000_000, 22_000_000, 31_000_000, 14_000_000],
  "active": [True, True, True, False, True],
})

print(df)
# Load from CSV
df = gl.read_csv("prices.csv")
```

---

## Column Types

| Python / NumPy type | grizzlars type | C++ storage                |
| ------------------- | -------------- | -------------------------- |
| `float` / `float64` | `"double"`     | `std::vector<double>`      |
| `int` / `int64`     | `"int64"`      | `std::vector<int64_t>`     |
| `bool`              | `"bool"`       | `std::vector<bool>`        |
| `str`               | `"string"`     | `std::vector<std::string>` |

The index is always `uint64` and defaults to `0..N-1`.

---

## API Reference

### I/O

#### `grizzlars.read_csv(path, index_col=None, dtype=None)`

Read a CSV file into a DataFrame. Uses a multithreaded native C++ reader by default.

```python
df = gl.read_csv("data.csv")

# Promote a column to the index
df = gl.read_csv("data.csv", index_col="Id")

# Force a column to a specific type (triggers slower Python fallback)
df = gl.read_csv("data.csv", dtype={"code": str})
```

#### `df.to_csv(path, index=True)`

Write the DataFrame to a CSV file.

```python
df.to_csv("output.csv")
df.to_csv("output.csv", index=False)  # omit index column
```

---

### Construction

#### `grizzlars.DataFrame(data=None, index=None)`

Build a DataFrame from a dict of lists or NumPy arrays.

```python
df = gl.DataFrame({
  "x": [1, 2, 3],
  "y": [4.0, 5.0, 6.0],
})

# Custom index
df = gl.DataFrame({"x": [10, 20, 30]}, index=[100, 200, 300])
```

---

### Inspection

```python
df.shape          # (rows, cols) — tuple
len(df)           # row count
df.columns        # list of column names
df.index          # numpy uint64 array of index values
df.dtypes()       # {"col": "double" | "int64" | "bool" | "string", ...}
```

---

### Column Access & Mutation

```python
# Read a column — returns numpy array (numeric/bool) or list (string)
prices = df["price"]

# Add or overwrite a column in-place
df["log_price"] = np.log(df["price"])
df["label"] = ["cheap", "expensive", "mid"]

# Check membership
"price" in df   # True / False

# Non-mutating variants
df2 = df.with_column("log_price", np.log(df["price"]))
df2 = df.assign(log_price=np.log(df["price"]), rank=[1, 2, 3])

# Select a subset of columns
df2 = df.select(["symbol", "price"])

# Rename columns in-place
df.rename({"symbol": "ticker", "price": "close"})

# Drop a column in-place
df.drop("log_price")
```

---

### Slicing

```python
df.head(10)          # first 10 rows
df.tail(10)          # last 10 rows

df.iloc[0]           # single row as DataFrame
df.iloc[10:50]       # slice (step=1 only)
df.iloc[-1]          # last row
```

---

### Filtering

`filter()` is lazy — the boolean mask is stored and data is only copied when a materialising operation is called. `len()` and `.shape` are always O(1).

```python
# Mask mode (recommended — compose with numpy operators)
cheap = df.filter(df["price"] < 200)
active = df.filter(df["active"] == True)

# String operator mode
cheap = df.filter("price", "<", 200)
# Operators: ">" ">=" "<" "<=" "==" "!="

# Combine conditions
mask = (df["price"] < 200) & (df["volume"] > 10_000_000)
df.filter(mask)

# len() and shape are free (no materialisation)
print(len(cheap))     # instant
print(cheap.shape)    # instant

# Materialises on first real operation
print(cheap["symbol"])
cheap.sort("price")
```

---

### Sorting

All sort operations are non-mutating and return a new DataFrame.

```python
df.sort("price")                       # ascending
df.sort("price", ascending=False)      # descending
df.sort_values("volume", ascending=False)  # alias for sort()
df.sort_index()                        # sort by index ascending
df.sort_index(ascending=False)         # sort by index descending
```

---

### Statistics

All scalar stats operate on a single column and return a Python float or int.

```python
df.mean("price")         # arithmetic mean
df.std("price")          # sample standard deviation (n-1)
df.sum("price")          # total
df.min("price")          # minimum value
df.max("price")          # maximum value
df.count("price")        # non-null count

df.quantile("price", 0.5)    # median (q in [0, 1])
df.corr("price", "volume")   # Pearson correlation
df.cov("price", "volume")    # sample covariance

df.nunique("symbol")         # number of distinct values
df.unique("symbol")          # sorted array of distinct values
df.n_missing("price")        # count of NaN / empty-string values

# Frequency table — returns DataFrame with ["value", "count"]
df.value_counts("symbol")
```

#### `df.describe()`

Returns a DataFrame with `count / mean / std / min / max / sum` for every numeric column.

```python
stats = df.describe()
# statistic  |  price  |  volume
# -----------+---------+---------
# count      |  5.0    |  5.0
# mean       |  ...    |  ...
# std        |  ...    |  ...
# min        |  ...    |  ...
# max        |  ...    |  ...
# sum        |  ...    |  ...
```

---

### GroupBy

`groupby()` returns a `_GroupBy` object. Chain `.agg()` or a shorthand method.

```python
# agg() accepts a dict of {column: function}
# Functions: "mean", "sum", "min", "max", "count", "std"
result = df.groupby("sector").agg({"price": "mean", "volume": "sum"})

# Shorthand methods
df.groupby("sector").mean("price")
df.groupby("sector").sum("volume")
df.groupby("sector").min("price")
df.groupby("sector").max("price")
df.groupby("sector").count("price")
df.groupby("sector").std("price")
```

GroupBy uses `string_view` keys internally — zero string copies during bucketing.

---

### Join

Joins operate on the DataFrame index. Load CSVs with `index_col=` to set the join key.

```python
left  = gl.read_csv("orders.csv",   index_col="order_id")
right = gl.read_csv("products.csv", index_col="order_id")

inner  = left.join(right, how="inner")   # default
left_j = left.join(right, how="left")    # unmatched right → NaN / ""
right_j = left.join(right, how="right")
outer  = left.join(right, how="outer")
```

The join uses a hash table probe — O(n + m) with parallel column scatter.

---

### Concat

Vertically stack two DataFrames (append rows). The index resets to `0..N-1`.

```python
combined = df_a.concat(df_b)

# Stack many frames
from functools import reduce
all_data = reduce(lambda a, b: a.concat(b), frames)
```

Only columns present in both frames with the same type are kept.

---

### Window Functions

All window functions return a NumPy array (not a new DataFrame).

```python
df.rolling_mean("price", window=20)   # 20-period moving average
df.rolling_sum("volume", window=5)
df.rolling_std("price", window=20)
df.rolling_min("price", window=10)
df.rolling_max("price", window=10)

# Generic form
df.rolling("price", window=20, func="mean")
# func: "mean" | "sum" | "std" | "min" | "max"
```

---

### Cumulative Functions

```python
df.cumsum("volume")    # cumulative sum
df.cumprod("factor")   # cumulative product
df.cummin("price")     # running minimum
df.cummax("price")     # running maximum
```

---

### Shift & Percent Change

```python
df.shift("price", n=1)    # lag by 1 period; NaN at boundary
df.shift("price", n=-1)   # lead by 1 period
df.pct_change("price")    # (price[i] - price[i-1]) / price[i-1]; first element NaN
```

---

### Data Cleaning

```python
# Remove rows with duplicate values in a column (keep first)
df.drop_duplicates("symbol")

# Remove rows where a column is NaN or empty string
df.drop_na("price")

# Fill NaN / empty values in-place (returns self)
df.fillna("price", 0.0)
df.fillna("label", "unknown")
```

---

### Threading

grizzlars automatically enables multithreading on import using all logical CPU cores.
You can adjust it at runtime.

```python
import grizzlars as gl

gl.set_optimum_thread_level()   # auto-detect (called on import)
gl.set_thread_level(4)          # pin to 4 threads
gl.get_thread_level()           # returns current thread count
```

---

## Performance

grizzlars is built for analytical workloads on large datasets:

- **CSV load** — memory-mapped file read, multithreaded chunk parsing, move semantics for string columns
- **Filter** — lazy evaluation; boolean mask stored until a materialising operation; `len()` is always O(1) via SIMD `count_nonzero`
- **Sort** — `string_view` comparison keys (zero heap allocation per comparison); parallel permutation scatter
- **GroupBy** — `unordered_map<string_view>` bucketing (zero string copies); parallel aggregation
- **Join** — hash table probe O(n + m); parallel column scatter across all cores
- **Aggregate / describe** — direct C++ vector reduction, no Python loop overhead

Benchmark against polars on a 100 000-row customer dataset (12 columns, mixed string/numeric):

| Operation  | grizzlars     |
| ---------- | ------------- |
| read_csv   | 2.41× faster  |
| sort       | 1.33× faster  |
| filter     | 24.93× faster |
| groupby    | 2.69× faster  |
| aggregate  | 20.39× faster |
| describe   | 61.97× faster |
| join inner | 2.15× slower  |
| join left  | 4.07× slower  |

Full test result:

```
===============================================================================
  Customer data benchmark  —  grizzlars vs polars
  Dataset: customers-100000.csv  (16912 KiB)
===============================================================================

  Rows: 100,000    Columns: 12

  ── Load ──────────────────────────────────────────────────────────────
  read_csv (customers)                       polars    76.13 ms   grizzlars    31.53 ms    → grizzlars is 2.41× faster

  ── Memory ────────────────────────────────────────────────────────────
  RSS delta after load                       polars      N/A   grizzlars      N/A
  In-process data size                       polars    15.5 MiB   grizzlars    15.9 MiB

  ── Operations ────────────────────────────────────────────────────────
  sort(Last Name asc)                        polars    45.97 ms   grizzlars    34.67 ms    → grizzlars is 1.33× faster
  filter(Index > 50) → 99,950 rows           polars    26.17 ms   grizzlars     1.05 ms    → grizzlars is 24.93× faster
  groupby Country → 243 groups               polars    13.79 ms   grizzlars     5.13 ms    → grizzlars is 2.69× faster
  agg(mean/sum/std/min/max)                  polars     5.72 ms   grizzlars    280.4 µs    → grizzlars is 20.39× faster
  describe                                   polars    23.60 ms   grizzlars    380.9 µs    → grizzlars is 61.97× faster

  ── Joins  (customers ⋈ people-100000.csv) ───────────────────────────
  join inner → 100,000 rows                  polars    20.54 ms   grizzlars    44.12 ms    → polars is 2.15× faster
  join left  → 100,000 rows (~50 000 unmatched) polars     9.90 ms   grizzlars    40.30 ms    → polars is 4.07× faster

===============================================================================
```

---

## Project Structure

```
grizzlars/
├── DataFrame/             core C++ library
├── grizzlars/             Python package
│   └── __init__.py        DataFrame class + read_csv
├── src/
│   └── grizzlars_bindings.cpp   pybind11 C++ extension
├── tests/
│   ├── data               data for tests
│   ├── functional         functional tests
│   └── performance        performance tests
├── CMakeLists.txt
└── pyproject.toml
```
