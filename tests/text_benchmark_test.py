"""
Benchmark: grizzlars vs polars — customer data (100 000 rows × 12 cols).

Operations compared
-------------------
  1. CSV load          read customers-100000.csv into a DataFrame
  2. Memory usage      RSS delta + in-process size estimate
  3. Sort              sort by Last Name ascending (string sort)
  4. Filter            rows where Index > 50  (~99 950 rows)
  5. GroupBy           count customers per Country
  6. Aggregate         mean / sum / std / min / max on Index column
  7. Describe          full stats DataFrame for numeric column (Index)
  8. Join (inner)      customers ⋈ people on Index (100 000 × 100 000)
  9. Join (left)       customers LEFT JOIN people[50 000] on Index

Prerequisites
-------------
  uv add polars psutil

Run
---
  uv run python tests/text_benchmark_test.py
  uv run pytest  tests/text_benchmark_test.py -s
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

CSV_PATH    = Path(__file__).parent / "data" / "customers-100000.csv"
PEOPLE_PATH = Path(__file__).parent / "data" / "people-100000.csv"

# Columns pulled from people that don't clash with customers columns.
# Both files share First Name / Last Name / Email — those are dropped.
PEOPLE_JOIN_COLS = ["Index", "User Id", "Sex", "Job Title"]

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


# ── CSV loaders ───────────────────────────────────────────────────────────────

def load_polars() -> "pl.DataFrame":
    return pl.read_csv(CSV_PATH)


def load_grizzlars() -> "grizzlars.DataFrame":
    return grizzlars.read_csv(str(CSV_PATH))


def load_people_polars() -> "pl.DataFrame":
    """People CSV — non-clashing columns only, Index kept as join key."""
    return pl.read_csv(PEOPLE_PATH).select(PEOPLE_JOIN_COLS)


def load_people_grizzlars() -> "grizzlars.DataFrame":
    """People CSV with Index as frame index; non-clashing columns only."""
    df = grizzlars.read_csv(str(PEOPLE_PATH), index_col="Index")
    keep = [c for c in df.columns if c in ("User Id", "Sex", "Job Title")]
    return df.select(keep)


# ── individual benchmarks ─────────────────────────────────────────────────────

def bench_sort(df_p: "pl.DataFrame", df_g: "grizzlars.DataFrame"):
    tp = elapsed(lambda: df_p.sort("Last Name"))[1]
    tg = elapsed(lambda: df_g.sort("Last Name"))[1]
    result_row("sort(Last Name asc)", tp, tg)
    return tp, tg


def bench_filter(df_p: "pl.DataFrame", df_g: "grizzlars.DataFrame"):
    def polars_filter():
        return df_p.filter(pl.col("Index") > 50)

    def grizzlars_filter():
        return df_g.filter(df_g["Index"] > 50)

    rp, tp = elapsed(polars_filter)
    rg, tg = elapsed(grizzlars_filter)

    n_p = len(rp)
    n_g = len(rg)
    assert n_p == n_g, f"row count mismatch: polars={n_p}, grizzlars={n_g}"

    result_row(f"filter(Index > 50) → {n_p:,} rows", tp, tg)
    return tp, tg


def bench_groupby(df_p: "pl.DataFrame", df_g: "grizzlars.DataFrame"):
    def polars_groupby():
        return (
            df_p.group_by("Country")
                .agg(pl.col("Index").count().alias("Index"))
                .sort("Country")
        )

    def grizzlars_groupby():
        return df_g.groupby("Country").agg({"Index": "count"})

    rp, tp = elapsed(polars_groupby)
    rg, tg = elapsed(grizzlars_groupby)

    n_p = len(rp)
    n_g = len(rg)
    assert n_p == n_g, f"group count mismatch: polars={n_p}, grizzlars={n_g}"

    result_row(f"groupby Country → {n_p} groups", tp, tg)
    return tp, tg


def bench_aggregate(df_p: "pl.DataFrame", df_g: "grizzlars.DataFrame"):
    def polars_agg():
        return df_p.select([
            pl.col("Index").mean().alias("mean"),
            pl.col("Index").sum().alias("sum"),
            pl.col("Index").std().alias("std"),
            pl.col("Index").min().alias("min"),
            pl.col("Index").max().alias("max"),
        ])

    def grizzlars_agg():
        return {
            "mean": df_g.mean("Index"),
            "sum":  df_g.sum("Index"),
            "std":  df_g.std("Index"),
            "min":  df_g.min("Index"),
            "max":  df_g.max("Index"),
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


def bench_join_inner(
    df_p_c: "pl.DataFrame",     df_p_p: "pl.DataFrame",
    df_g_c: "grizzlars.DataFrame", df_g_p: "grizzlars.DataFrame",
):
    """Inner join on Index — every customer row matches a people row."""

    def polars_join():
        return df_p_c.join(df_p_p, on="Index", how="inner")

    def grizzlars_join():
        return df_g_c.join(df_g_p, how="inner")

    rp, tp = elapsed(polars_join)
    rg, tg = elapsed(grizzlars_join)

    n_p = len(rp)
    n_g = len(rg)
    assert n_p == n_g, f"row count mismatch: polars={n_p}, grizzlars={n_g}"

    result_row(f"join inner → {n_p:,} rows", tp, tg)
    return tp, tg


def bench_join_left(
    df_p_c: "pl.DataFrame",     df_p_p: "pl.DataFrame",
    df_g_c: "grizzlars.DataFrame", df_g_p: "grizzlars.DataFrame",
):
    """Left join: customers (100 000) LEFT JOIN people[first 50 000].
    The upper half of customers gets no match — exercises the null-fill path."""

    # Restrict people to Index <= 50 000 before timing
    df_p_p_half = df_p_p.filter(pl.col("Index") <= 50_000)
    df_g_p_half = df_g_p.filter(df_g_p.index <= 50_000)

    def polars_join():
        return df_p_c.join(df_p_p_half, on="Index", how="left")

    def grizzlars_join():
        return df_g_c.join(df_g_p_half, how="left")

    rp, tp = elapsed(polars_join)
    rg, tg = elapsed(grizzlars_join)

    n_p = len(rp)
    n_g = len(rg)
    assert n_p == n_g, f"row count mismatch: polars={n_p}, grizzlars={n_g}"

    result_row(f"join left  → {n_p:,} rows (~50 000 unmatched)", tp, tg)
    return tp, tg


# ── main entry ────────────────────────────────────────────────────────────────

def run_benchmark():
    if not HAS_POLARS:
        print("polars not available — skipping benchmark")
        return

    print()
    print("=" * 79)
    print("  Customer data benchmark  —  grizzlars vs polars")
    print(f"  Dataset: {CSV_PATH.name}  ({CSV_PATH.stat().st_size / 1024:.0f} KiB)")
    print("=" * 79)

    # ── Load customers ────────────────────────────────────────────────────────

    gc.collect()
    mem0 = rss_mb()
    df_p, tp_load = elapsed(load_polars)
    mem1 = rss_mb()
    polars_rss_delta = mem1 - mem0

    gc.collect()
    mem2 = rss_mb()
    df_g, tg_load = elapsed(load_grizzlars)
    mem3 = rss_mb()
    grizzlars_rss_delta = mem3 - mem2

    print()
    print(f"  Rows: {len(df_p):,}    Columns: {len(df_p.columns)}")
    print()
    print("  ── Load ──────────────────────────────────────────────────────────────")
    result_row("read_csv (customers)", tp_load, tg_load)

    print()
    print("  ── Memory ────────────────────────────────────────────────────────────")
    polars_size   = df_p.estimated_size("mb")
    grizzlars_size = _df_size_mb(df_g)
    print(f"  {'RSS delta after load':<42} "
          f"polars {fmt_mb(polars_rss_delta)}   "
          f"grizzlars {fmt_mb(grizzlars_rss_delta)}")
    print(f"  {'In-process data size':<42} "
          f"polars {fmt_mb(polars_size)}   "
          f"grizzlars {fmt_mb(grizzlars_size)}")

    print()
    print("  ── Operations ────────────────────────────────────────────────────────")

    bench_sort(df_p, df_g)
    bench_filter(df_p, df_g)
    bench_groupby(df_p, df_g)
    bench_aggregate(df_p, df_g)
    bench_describe(df_p, df_g)

    # ── Load people then benchmark joins ─────────────────────────────────────
    # grizzlars: Index column becomes the frame index so join_by_index works.
    # polars:   Index stays as a regular column; join is on="Index".

    print()
    print("  ── Joins  (customers ⋈ people-100000.csv) ───────────────────────────")

    df_p_people = load_people_polars()
    df_g_people = load_people_grizzlars()

    # Customers also need Index as frame index for grizzlars join_by_index
    df_g_c_idx = grizzlars.read_csv(str(CSV_PATH), index_col="Index")

    bench_join_inner(df_p, df_p_people, df_g_c_idx, df_g_people)
    bench_join_left(df_p, df_p_people, df_g_c_idx, df_g_people)

    print()
    print("=" * 79)
    print()


# ── pytest entry point ────────────────────────────────────────────────────────

def test_customer_benchmark(capsys):
    run_benchmark()
    out = capsys.readouterr().out
    assert "grizzlars" in out or not HAS_POLARS


if __name__ == "__main__":
    run_benchmark()
