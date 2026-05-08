"""
Benchmark: grizzlars vs polars — stock data (numeric-heavy).

Dataset
-------
  All CSVs in tests/data/stock_data/ stacked into one DataFrame.
  A "Symbol" column (ticker name) is added before stacking so
  GroupBy has a meaningful categorical key.

  Columns after stack:
    Symbol      str   (11 unique tickers)
    Date        str
    Open        f64
    High        f64
    Low         f64
    Close       f64
    Adj Close   f64
    Volume      i64

Operations compared
-------------------
  1. CSV load + stack   load every stock CSV, tag with Symbol, concat
  2. Memory usage       RSS delta + in-process size estimate
  3. Sort               sort by Close ascending (numeric)
  4. Filter             rows where Volume > 100 000
  5. GroupBy            mean Close per Symbol  (11 groups)
  6. Aggregate          mean / sum / std / min / max on Close
  7. Describe           full stats DataFrame for all numeric columns

Run
---
  uv run python tests/numeric_benchmark_test.py
  uv run pytest  tests/numeric_benchmark_test.py -s
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np

import grizzlars as gl

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    print("WARNING: polars not installed — run `uv add polars`")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "stock_data"
CSV_FILES = sorted(DATA_DIR.glob("*.csv"))

VOLUME_THRESHOLD = 100_000

# ── helpers ───────────────────────────────────────────────────────────────────

def rss_mb() -> float:
    if not HAS_PSUTIL:
        return float("nan")
    return psutil.Process().memory_info().rss / 1024 / 1024


def elapsed(fn):
    t0 = time.perf_counter()
    r  = fn()
    return r, time.perf_counter() - t0


def fmt_time(s: float) -> str:
    if s < 1e-3:
        return f"{s * 1e6:>8.1f} µs"
    if s < 1.0:
        return f"{s * 1e3:>8.2f} ms"
    return f"{s:>8.3f}  s"


def fmt_mb(mb: float) -> str:
    return f"{mb:>7.1f} MiB" if mb == mb else "     N/A"


def _df_size_mb(df: gl.DataFrame) -> float:
    total = 0
    for col in df.columns:
        raw = df[col]
        if isinstance(raw, np.ndarray):
            total += raw.nbytes
        else:
            total += sum(len(s) for s in raw)
    total += df.index.nbytes
    return total / 1024 / 1024


def result_row(label: str, t_polars: float, t_grizzlars: float) -> None:
    if t_grizzlars > 0:
        ratio = t_polars / t_grizzlars
        faster, rx = ("grizzlars", ratio) if ratio >= 1 else ("polars", 1 / ratio)
        note = f"→ {faster} is {rx:.2f}x faster"
    else:
        note = ""
    print(
        f"  {label:<34} polars {fmt_time(t_polars)}   "
        f"grizzlars {fmt_time(t_grizzlars)}    {note}"
    )


# ── load + stack ──────────────────────────────────────────────────────────────

def load_stack_polars() -> pl.DataFrame:
    frames = []
    for path in CSV_FILES:
        ticker = path.stem
        df = pl.read_csv(path).with_columns(pl.lit(ticker).alias("Symbol"))
        frames.append(df)
    return pl.concat(frames)


def load_stack_grizzlars() -> gl.DataFrame:
    frames = []
    for path in CSV_FILES:
        ticker = path.stem
        df = gl.read_csv(str(path))
        df["Symbol"] = [ticker] * len(df)
        frames.append(df)
    combined = frames[0]
    for df in frames[1:]:
        combined = combined.concat(df)
    return combined


# ── individual benchmarks ─────────────────────────────────────────────────────

def bench_sort(df_p: pl.DataFrame, df_g: gl.DataFrame):
    tp = elapsed(lambda: df_p.sort("Close"))[1]
    tg = elapsed(lambda: df_g.sort("Close"))[1]
    result_row("sort(Close asc)", tp, tg)
    return tp, tg


def bench_filter(df_p: pl.DataFrame, df_g: gl.DataFrame):
    def polars_filter():
        return df_p.filter(pl.col("Volume") > VOLUME_THRESHOLD)

    def grizzlars_filter():
        return df_g.filter(df_g["Volume"] > VOLUME_THRESHOLD)

    rp, tp = elapsed(polars_filter)
    rg, tg = elapsed(grizzlars_filter)

    n_p = len(rp)
    n_g = len(rg)
    assert n_p == n_g, f"row count mismatch: polars={n_p}, grizzlars={n_g}"

    result_row(f"filter(Volume > {VOLUME_THRESHOLD:,}) → {n_p:,} rows", tp, tg)
    return tp, tg


def bench_groupby(df_p: pl.DataFrame, df_g: gl.DataFrame):
    def polars_groupby():
        return (
            df_p.group_by("Symbol")
                .agg(pl.col("Close").mean().alias("Close"))
                .sort("Symbol")
        )

    def grizzlars_groupby():
        return df_g.groupby("Symbol").agg({"Close": "mean"})

    rp, tp = elapsed(polars_groupby)
    rg, tg = elapsed(grizzlars_groupby)

    n_p = len(rp)
    n_g = len(rg)
    assert n_p == n_g, f"group count mismatch: polars={n_p}, grizzlars={n_g}"

    result_row(f"groupby Symbol → {n_p} groups (mean Close)", tp, tg)
    return tp, tg


def bench_aggregate(df_p: pl.DataFrame, df_g: gl.DataFrame):
    def polars_agg():
        return df_p.select([
            pl.col("Close").mean().alias("mean"),
            pl.col("Close").sum().alias("sum"),
            pl.col("Close").std().alias("std"),
            pl.col("Close").min().alias("min"),
            pl.col("Close").max().alias("max"),
        ])

    def grizzlars_agg():
        return {
            "mean": df_g.mean("Close"),
            "sum":  df_g.sum("Close"),
            "std":  df_g.std("Close"),
            "min":  df_g.min("Close"),
            "max":  df_g.max("Close"),
        }

    tp = elapsed(polars_agg)[1]
    tg = elapsed(grizzlars_agg)[1]
    result_row("agg(mean/sum/std/min/max on Close)", tp, tg)
    return tp, tg


def bench_describe(df_p: pl.DataFrame, df_g: gl.DataFrame):
    tp = elapsed(lambda: df_p.describe())[1]
    tg = elapsed(lambda: df_g.describe())[1]
    result_row("describe", tp, tg)
    return tp, tg


# ── main entry ────────────────────────────────────────────────────────────────

def run_benchmark():
    if not HAS_POLARS:
        print("polars not available — skipping benchmark")
        return

    total_size = sum(p.stat().st_size for p in CSV_FILES)
    print()
    print("=" * 79)
    print("  Stock data benchmark  —  grizzlars vs polars  (numeric-heavy)")
    print(f"  Dataset: {len(CSV_FILES)} CSVs from {DATA_DIR.name}/  "
          f"({total_size / 1024:.0f} KiB total)")
    print("=" * 79)

    # ── Load + stack ──────────────────────────────────────────────────────────

    gc.collect()
    mem0 = rss_mb()
    df_p, tp_load = elapsed(load_stack_polars)
    mem1 = rss_mb()
    polars_rss = mem1 - mem0

    gc.collect()
    mem2 = rss_mb()
    df_g, tg_load = elapsed(load_stack_grizzlars)
    mem3 = rss_mb()
    grizzlars_rss = mem3 - mem2

    print()
    print(f"  Rows: {len(df_p):,}    Columns: {len(df_p.columns)}    "
          f"Tickers: {df_p['Symbol'].n_unique()}")
    print()
    print("  ── Load + stack ──────────────────────────────────────────────────────")
    result_row("read_csv x all + concat", tp_load, tg_load)

    print()
    print("  ── Memory ────────────────────────────────────────────────────────────")
    polars_size  = df_p.estimated_size("mb")
    grizzlars_size = _df_size_mb(df_g)
    print(f"  {'RSS delta after load':<34} "
          f"polars {fmt_mb(polars_rss)}   grizzlars {fmt_mb(grizzlars_rss)}")
    print(f"  {'In-process data size':<34} "
          f"polars {fmt_mb(polars_size)}   grizzlars {fmt_mb(grizzlars_size)}")

    print()
    print("  ── Operations ────────────────────────────────────────────────────────")

    bench_sort(df_p, df_g)
    bench_filter(df_p, df_g)
    bench_groupby(df_p, df_g)
    bench_aggregate(df_p, df_g)
    bench_describe(df_p, df_g)

    print()
    print("=" * 79)
    print()


# ── pytest entry point ────────────────────────────────────────────────────────

def test_stock_benchmark(capsys):
    run_benchmark()
    out = capsys.readouterr().out
    assert "grizzlars" in out or not HAS_POLARS


if __name__ == "__main__":
    run_benchmark()
