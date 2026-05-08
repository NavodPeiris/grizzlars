"""
Benchmark: grizzlars vs polars — Warehouse and Retail Sales (~307 K rows).

Mixed workload: string columns (SUPPLIER, ITEM DESCRIPTION, ITEM TYPE)
alongside numeric columns (YEAR, MONTH, ITEM CODE, RETAIL SALES,
RETAIL TRANSFERS, WAREHOUSE SALES).

Operations compared
-------------------
  1. CSV load          read Warehouse_and_Retail_Sales.csv
  2. Memory usage      RSS delta + in-process size estimate
  3. Sort              sort by WAREHOUSE SALES descending (numeric)
  4. Filter            rows where WAREHOUSE SALES > 0  (non-zero sales)
  5. GroupBy (low)     sum WAREHOUSE SALES per ITEM TYPE  (~5 groups)
  6. GroupBy (high)    sum WAREHOUSE SALES per SUPPLIER  (high cardinality)
  7. Aggregate         mean / sum / std / min / max on WAREHOUSE SALES
  8. Describe          full stats for all numeric columns

Run
---
  uv run python tests/mixed_benchmark_test.py
  uv run pytest  tests/mixed_benchmark_test.py -s
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import numpy as np

import grizzlars

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

CSV_PATH = Path(__file__).parent / "data" / "Warehouse_and_Retail_Sales.csv"

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


def _df_size_mb(df: "grizzlars.DataFrame") -> float:
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
        note = f"→ {faster} is {rx:.2f}× faster"
    else:
        note = ""
    print(
        f"  {label:<42} polars {fmt_time(t_polars)}   "
        f"grizzlars {fmt_time(t_grizzlars)}    {note}"
    )


# ── loaders ───────────────────────────────────────────────────────────────────

def load_polars() -> "pl.DataFrame":
    return pl.read_csv(CSV_PATH, schema_overrides={"ITEM CODE": pl.String})


def load_grizzlars() -> "grizzlars.DataFrame":
    return grizzlars.read_csv(str(CSV_PATH))


# ── individual benchmarks ─────────────────────────────────────────────────────

def bench_sort(df_p: "pl.DataFrame", df_g: "grizzlars.DataFrame"):
    tp = elapsed(lambda: df_p.sort("WAREHOUSE SALES", descending=True))[1]
    tg = elapsed(lambda: df_g.sort("WAREHOUSE SALES", ascending=False))[1]
    result_row("sort(WAREHOUSE SALES desc)", tp, tg)
    return tp, tg


def bench_filter(df_p: "pl.DataFrame", df_g: "grizzlars.DataFrame"):
    def polars_filter():
        return df_p.filter(pl.col("WAREHOUSE SALES") > 0)

    def grizzlars_filter():
        return df_g.filter(df_g["WAREHOUSE SALES"] > 0)

    rp, tp = elapsed(polars_filter)
    rg, tg = elapsed(grizzlars_filter)

    n_p = len(rp)
    n_g = len(rg)
    assert n_p == n_g, f"row count mismatch: polars={n_p}, grizzlars={n_g}"

    result_row(f"filter(WAREHOUSE SALES > 0) → {n_p:,} rows", tp, tg)
    return tp, tg


def bench_groupby_low(df_p: "pl.DataFrame", df_g: "grizzlars.DataFrame"):
    """Low-cardinality string key: ITEM TYPE (~5 groups)."""

    def polars_gb():
        return (
            df_p.group_by("ITEM TYPE")
                .agg(pl.col("WAREHOUSE SALES").sum().alias("WAREHOUSE SALES"))
                .sort("ITEM TYPE")
        )

    def grizzlars_gb():
        return df_g.groupby("ITEM TYPE").agg({"WAREHOUSE SALES": "sum"})

    rp, tp = elapsed(polars_gb)
    rg, tg = elapsed(grizzlars_gb)

    n_p, n_g = len(rp), len(rg)
    assert n_p == n_g, f"group count mismatch: polars={n_p}, grizzlars={n_g}"

    result_row(f"groupby ITEM TYPE → {n_p} groups (sum)", tp, tg)
    return tp, tg


def bench_groupby_high(df_p: "pl.DataFrame", df_g: "grizzlars.DataFrame"):
    """High-cardinality string key: SUPPLIER (hundreds of groups)."""

    def polars_gb():
        return (
            df_p.group_by("SUPPLIER")
                .agg(pl.col("WAREHOUSE SALES").sum().alias("WAREHOUSE SALES"))
                .sort("SUPPLIER")
        )

    def grizzlars_gb():
        return df_g.groupby("SUPPLIER").agg({"WAREHOUSE SALES": "sum"})

    rp, tp = elapsed(polars_gb)
    rg, tg = elapsed(grizzlars_gb)

    n_p, n_g = len(rp), len(rg)
    assert n_p == n_g, f"group count mismatch: polars={n_p}, grizzlars={n_g}"

    result_row(f"groupby SUPPLIER → {n_p} groups (sum)", tp, tg)
    return tp, tg


def bench_aggregate(df_p: "pl.DataFrame", df_g: "grizzlars.DataFrame"):
    def polars_agg():
        return df_p.select([
            pl.col("WAREHOUSE SALES").mean().alias("mean"),
            pl.col("WAREHOUSE SALES").sum().alias("sum"),
            pl.col("WAREHOUSE SALES").std().alias("std"),
            pl.col("WAREHOUSE SALES").min().alias("min"),
            pl.col("WAREHOUSE SALES").max().alias("max"),
        ])

    def grizzlars_agg():
        return {
            "mean": df_g.mean("WAREHOUSE SALES"),
            "sum":  df_g.sum("WAREHOUSE SALES"),
            "std":  df_g.std("WAREHOUSE SALES"),
            "min":  df_g.min("WAREHOUSE SALES"),
            "max":  df_g.max("WAREHOUSE SALES"),
        }

    tp = elapsed(polars_agg)[1]
    tg = elapsed(grizzlars_agg)[1]
    result_row("agg(mean/sum/std/min/max)", tp, tg)
    return tp, tg


def bench_describe(df_p: "pl.DataFrame", df_g: "grizzlars.DataFrame"):
    tp = elapsed(lambda: df_p.describe())[1]
    tg = elapsed(lambda: df_g.describe())[1]
    result_row("describe", tp, tg)
    return tp, tg


# ── main entry ────────────────────────────────────────────────────────────────

def run_benchmark():
    if not HAS_POLARS:
        print("polars not available — skipping benchmark")
        return

    print()
    print("=" * 83)
    print("  Warehouse & Retail Sales benchmark  —  grizzlars vs polars  (mixed)")
    print(f"  Dataset: {CSV_PATH.name}  ({CSV_PATH.stat().st_size / 1024:.0f} KiB)")
    print("=" * 83)

    # ── Load ──────────────────────────────────────────────────────────────────

    gc.collect()
    mem0 = rss_mb()
    df_p, tp_load = elapsed(load_polars)
    mem1 = rss_mb()
    polars_rss = mem1 - mem0

    gc.collect()
    mem2 = rss_mb()
    df_g, tg_load = elapsed(load_grizzlars)
    mem3 = rss_mb()
    grizzlars_rss = mem3 - mem2

    print()
    print(f"  Rows: {len(df_p):,}    Columns: {len(df_p.columns)}")
    print()
    print("  ── Load ──────────────────────────────────────────────────────────────")
    result_row("read_csv", tp_load, tg_load)

    print()
    print("  ── Memory ────────────────────────────────────────────────────────────")
    polars_size   = df_p.estimated_size("mb")
    grizzlars_size = _df_size_mb(df_g)
    print(f"  {'RSS delta after load':<42} "
          f"polars {fmt_mb(polars_rss)}   grizzlars {fmt_mb(grizzlars_rss)}")
    print(f"  {'In-process data size':<42} "
          f"polars {fmt_mb(polars_size)}   grizzlars {fmt_mb(grizzlars_size)}")

    print()
    print("  ── Operations ────────────────────────────────────────────────────────")

    bench_sort(df_p, df_g)
    bench_filter(df_p, df_g)
    bench_groupby_low(df_p, df_g)
    bench_groupby_high(df_p, df_g)
    bench_aggregate(df_p, df_g)
    bench_describe(df_p, df_g)

    print()
    print("=" * 83)
    print()


# ── pytest entry point ────────────────────────────────────────────────────────

def test_mixed_benchmark(capsys):
    run_benchmark()
    out = capsys.readouterr().out
    assert "grizzlars" in out or not HAS_POLARS


if __name__ == "__main__":
    run_benchmark()
