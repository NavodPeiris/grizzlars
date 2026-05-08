"""
grizzlars — Python wrapper for the ultrafast hmdf C++ DataFrame library.

Supported column types: float64 (double), int64, bool, str.
Index type: unsigned 64-bit integer (auto-assigned 0..N-1 if omitted).
"""

from __future__ import annotations

import csv
import operator as _op
import re
from typing import Optional, Union

import numpy as np

from ._grizzlars import (
    GrizzlarFrame as _GrizzlarFrame,
    set_thread_level,
    set_optimum_thread_level,
    get_thread_level,
)

__version__ = "0.1.0"
__all__ = ["DataFrame", "read_csv",
           "set_thread_level", "set_optimum_thread_level", "get_thread_level"]

# Enable multithreading automatically on import using all logical CPU cores.
set_optimum_thread_level()

# Matches hmdf column-name annotations like  :12265:<double>  or  :12265:<unsigned long>
_HMDF_ANNOTATION = re.compile(r":\d+:<[^>]+>$")

_OPS = {
    ">":  _op.gt,
    ">=": _op.ge,
    "<":  _op.lt,
    "<=": _op.le,
    "==": _op.eq,
    "!=": _op.ne,
}


class _LazyFilterFrame:
    """
    Lazy filter result — stores the boolean mask without materialising the data.

    Inspired by polars (filters are zero-copy Arrow bitmap views) and vaex
    (expression trees deferred until terminal operation).

    • __len__ and .shape return the pre-counted row count instantly.
    • Every other DataFrame method triggers one-time materialisation.
    """

    def __init__(self, source: "DataFrame", mask: np.ndarray) -> None:
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_mask",   mask)
        # np.count_nonzero is SIMD-vectorised — ~1-2 ms for 2 M bools
        object.__setattr__(self, "_cached_len", int(np.count_nonzero(mask)))
        object.__setattr__(self, "_realized",   None)

    # ── materialization ───────────────────────────────────────────────────────

    def _realize(self) -> "DataFrame":
        r = object.__getattribute__(self, "_realized")
        if r is None:
            source = object.__getattribute__(self, "_source")
            mask   = object.__getattribute__(self, "_mask")
            r = DataFrame._from_frame(source._frame.filter_by_mask(mask))
            object.__setattr__(self, "_realized", r)
        return r

    # ── cheap operations (no materialization) ─────────────────────────────────

    def __len__(self) -> int:
        return object.__getattribute__(self, "_cached_len")

    @property
    def shape(self) -> tuple:
        n   = object.__getattribute__(self, "_cached_len")
        src = object.__getattribute__(self, "_source")
        return (n, len(src.columns))

    # ── all other operations proxy to the materialised frame ──────────────────

    def __getattr__(self, name: str):
        return getattr(self._realize(), name)

    def __getitem__(self, key):
        return self._realize()[key]

    def __setitem__(self, key, value):
        self._realize()[key] = value

    def __contains__(self, item) -> bool:
        return item in self._realize()

    def __repr__(self) -> str:
        n   = object.__getattribute__(self, "_cached_len")
        src = object.__getattribute__(self, "_source")
        return (f"<LazyFilterFrame {n:,} rows × {len(src.columns)} cols "
                f"[not yet materialised]>")


class _ILocIndexer:
    """Supports df.iloc[start:stop] and df.iloc[i] syntax."""

    def __init__(self, df: "DataFrame") -> None:
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, slice):
            n = len(self._df)
            start, stop, step = key.indices(n)
            if step != 1:
                raise ValueError("iloc only supports step=1 slices")
            return self._df._from_frame(self._df._frame.iloc(start, stop))
        if isinstance(key, int):
            n = len(self._df)
            i = key if key >= 0 else n + key
            return self._df._from_frame(self._df._frame.iloc(i, i + 1))
        raise TypeError(f"iloc key must be int or slice, got {type(key)}")


class _GroupBy:
    """Returned by DataFrame.groupby(col). Supports .agg(dict) and shorthand methods."""

    def __init__(self, df: "DataFrame", by: str) -> None:
        self._df = df
        self._by = by

    def agg(self, spec: dict) -> "DataFrame":
        """
        Aggregate grouped columns.

        Parameters
        ----------
        spec : dict
            Mapping of column_name -> aggregation_function string.
            Supported: "mean", "sum", "min", "max", "count", "std".

        Example
        -------
        >>> df.groupby("year").agg({"close": "mean", "volume": "sum"})
        """
        pairs = list(spec.items())
        return DataFrame._from_frame(self._df._frame.groupby_agg(self._by, pairs))

    def mean(self, col: str) -> "DataFrame":
        return self.agg({col: "mean"})

    def sum(self, col: str) -> "DataFrame":
        return self.agg({col: "sum"})

    def min(self, col: str) -> "DataFrame":
        return self.agg({col: "min"})

    def max(self, col: str) -> "DataFrame":
        return self.agg({col: "max"})

    def count(self, col: str) -> "DataFrame":
        return self.agg({col: "count"})

    def std(self, col: str) -> "DataFrame":
        return self.agg({col: "std"})


class DataFrame:
    """
    High-performance DataFrame backed by the C++ hmdf library.

    Parameters
    ----------
    data : dict, optional
        Mapping of column name -> list or numpy array.
    index : list or array-like, optional
        Unsigned integer index.  Defaults to 0..N-1.

    Examples
    --------
    >>> df = grizzlars.DataFrame({"price": [100.5, 200.0], "volume": [1000, 2000]})
    >>> df.filter(df["price"] > 150).sort("price")
    """

    def __init__(
        self,
        data: Optional[dict] = None,
        index: Optional[Union[list, np.ndarray]] = None,
    ) -> None:
        self._frame = _GrizzlarFrame()
        if data:
            n = max(len(v) for v in data.values())
            idx = list(range(n)) if index is None else list(index)
            self._frame.load_index(idx)
            for name, values in data.items():
                v = values if isinstance(values, (list, np.ndarray)) else list(values)
                self._frame.load_column(name, v)

    @classmethod
    def _from_frame(cls, frame: _GrizzlarFrame) -> "DataFrame":
        obj = cls.__new__(cls)
        obj._frame = frame
        return obj

    def _copy(self) -> "DataFrame":
        return DataFrame._from_frame(self._frame.deep_copy())

    # ── column access ─────────────────────────────────────────────────────────

    def __getitem__(self, col: str) -> Union[np.ndarray, list]:
        return self._frame.get_column(col)

    def __setitem__(self, col: str, values) -> None:
        v = values if isinstance(values, (list, np.ndarray)) else list(values)
        self._frame.load_column(col, v)

    def __contains__(self, col: str) -> bool:
        return self._frame.has_column(col)

    @property
    def index(self) -> np.ndarray:
        return self._frame.get_index()

    @property
    def columns(self) -> list[str]:
        return self._frame.columns()

    @property
    def shape(self) -> tuple[int, int]:
        return self._frame.shape()

    def __len__(self) -> int:
        return self.shape[0]

    def dtypes(self) -> dict[str, str]:
        """Return a mapping of column name -> type string."""
        return {c: self._frame.col_type(c) for c in self.columns}

    @property
    def iloc(self) -> _ILocIndexer:
        """Integer-location based indexing: df.iloc[0:100], df.iloc[-1]."""
        return _ILocIndexer(self)

    # ── statistics ────────────────────────────────────────────────────────────

    def mean(self, col: str) -> float:
        return self._frame.mean(col)

    def std(self, col: str) -> float:
        return self._frame.std(col)

    def sum(self, col: str) -> float:
        return self._frame.sum(col)

    def min(self, col: str) -> float:
        return self._frame.min(col)

    def max(self, col: str) -> float:
        return self._frame.max(col)

    def count(self, col: str) -> int:
        return self._frame.count(col)

    def describe(self) -> "DataFrame":
        """
        Return count / mean / std / min / max / sum for every numeric column
        as a DataFrame (one row per statistic, one column per numeric column).
        """
        stats = self._frame.describe()
        if not stats:
            return DataFrame()
        stat_names = ["count", "mean", "std", "min", "max", "sum"]
        data: dict = {"statistic": stat_names}
        for col, d in stats.items():
            data[col] = [float(d[s]) for s in stat_names]
        return DataFrame(data, index=list(range(len(stat_names))))

    def quantile(self, col: str, q: float) -> float:
        """Return the q-th quantile of a column (q in [0, 1])."""
        return self._frame.quantile(col, q)

    def corr(self, col1: str, col2: str) -> float:
        """Pearson correlation between two numeric columns."""
        return self._frame.corr(col1, col2)

    def cov(self, col1: str, col2: str) -> float:
        """Sample covariance (n-1 denominator) between two numeric columns."""
        return self._frame.cov(col1, col2)

    def nunique(self, col: str) -> int:
        """Number of distinct values in *col*."""
        return self._frame.nunique(col)

    def unique(self, col: str) -> Union[np.ndarray, list]:
        """Sorted unique values in *col*."""
        return self._frame.unique_values(col)

    def n_missing(self, col: str) -> int:
        """Count of NaN / empty-string values in *col*."""
        return self._frame.n_missing(col)

    def value_counts(self, col: str) -> "DataFrame":
        """Return a DataFrame with ["value", "count"] sorted by count descending."""
        return DataFrame._from_frame(self._frame.value_counts(col))

    # ── sorting (non-mutating — always return a new DataFrame) ───────────────

    def sort(self, by: str, ascending: bool = True) -> "DataFrame":
        """Return a new DataFrame sorted by *by* column (non-mutating)."""
        return DataFrame._from_frame(self._frame.sort_by(by, ascending))

    def sort_values(self, by: str, ascending: bool = True) -> "DataFrame":
        """Alias for sort()."""
        return self.sort(by, ascending)

    def sort_index(self, ascending: bool = True) -> "DataFrame":
        """Return a new DataFrame sorted by index (non-mutating)."""
        return DataFrame._from_frame(self._frame.sort_index(ascending))

    # ── slicing / filtering ───────────────────────────────────────────────────

    def head(self, n: int = 5) -> "DataFrame":
        return DataFrame._from_frame(self._frame.iloc(0, min(n, len(self))))

    def tail(self, n: int = 5) -> "DataFrame":
        total = len(self)
        return DataFrame._from_frame(self._frame.iloc(max(0, total - n), total))

    def filter(self, col_or_mask, op: Optional[str] = None,
               value=None) -> "_LazyFilterFrame":
        """
        Return rows matching a condition — lazy (polars/vaex style).

        The boolean mask is stored but data is NOT copied until a materialising
        operation is called (sort, groupby, column access, etc.).
        ``len()`` and ``.shape`` are always O(1) / O(n SIMD).

        Mask mode (polars-style):
            df.filter(df["volume"] > 5_000_000)
            df.filter(np.array([True, False, True, ...]))

        Column + operator mode:
            df.filter("volume", ">", 5_000_000)
            Operators: ">", ">=", "<", "<=", "==", "!="
        """
        if op is None:
            mask = col_or_mask
            if not isinstance(mask, np.ndarray):
                mask = np.asarray(mask, dtype=bool)
        else:
            if op not in _OPS:
                raise ValueError(f"Unknown operator {op!r}. Choose from {list(_OPS)}")
            raw = self[col_or_mask]
            arr = np.asarray(raw) if not isinstance(raw, np.ndarray) else raw
            mask = _OPS[op](arr, value)
        return _LazyFilterFrame(self, mask)

    def select(self, columns: list[str]) -> "DataFrame":
        """Return a new DataFrame containing only the specified columns."""
        return DataFrame._from_frame(self._frame.select_columns(columns))

    def with_column(self, name: str, values) -> "DataFrame":
        """
        Return a new DataFrame with *name* column added / replaced (non-mutating).

        Example
        -------
        >>> df.with_column("year", [int(d[:4]) for d in df["date"]])
        """
        result = self._copy()
        v = values if isinstance(values, (list, np.ndarray)) else list(values)
        result._frame.load_column(name, v)
        return result

    def assign(self, **kwargs) -> "DataFrame":
        """
        Return a new DataFrame with extra columns added (non-mutating).

        Example
        -------
        >>> df.assign(year=[int(d[:4]) for d in df["date"]], quarter=[1]*len(df))
        """
        result = self._copy()
        for name, values in kwargs.items():
            v = values if isinstance(values, (list, np.ndarray)) else list(values)
            result._frame.load_column(name, v)
        return result

    # ── groupby ───────────────────────────────────────────────────────────────

    def groupby(self, by: str) -> _GroupBy:
        """
        Group by *by* column.

        Example
        -------
        >>> df.groupby("year").agg({"close": "mean", "volume": "sum"})
        """
        return _GroupBy(self, by)

    # ── join / concat ─────────────────────────────────────────────────────────

    def join(self, other: "DataFrame", how: str = "inner") -> "DataFrame":
        """Join two DataFrames on their shared index. how: inner/left/right/outer."""
        return DataFrame._from_frame(self._frame.join_by_index(other._frame, how))

    def concat(self, other: "DataFrame") -> "DataFrame":
        """Vertically concatenate two DataFrames (stack rows). Index resets to 0..N-1."""
        return DataFrame._from_frame(self._frame.concat_frame(other._frame))

    # ── window functions ──────────────────────────────────────────────────────

    def rolling_mean(self, col: str, window: int) -> np.ndarray:
        return self._frame.rolling(col, window, "mean")

    def rolling_sum(self, col: str, window: int) -> np.ndarray:
        return self._frame.rolling(col, window, "sum")

    def rolling_std(self, col: str, window: int) -> np.ndarray:
        return self._frame.rolling(col, window, "std")

    def rolling_min(self, col: str, window: int) -> np.ndarray:
        return self._frame.rolling(col, window, "min")

    def rolling_max(self, col: str, window: int) -> np.ndarray:
        return self._frame.rolling(col, window, "max")

    def rolling(self, col: str, window: int, func: str = "mean") -> np.ndarray:
        """Generic rolling window. func: "mean" | "sum" | "std" | "min" | "max"."""
        return self._frame.rolling(col, window, func)

    # ── cumulative functions ──────────────────────────────────────────────────

    def cumsum(self, col: str) -> np.ndarray:
        return self._frame.cumulative(col, "sum")

    def cumprod(self, col: str) -> np.ndarray:
        return self._frame.cumulative(col, "prod")

    def cummin(self, col: str) -> np.ndarray:
        return self._frame.cumulative(col, "min")

    def cummax(self, col: str) -> np.ndarray:
        return self._frame.cumulative(col, "max")

    # ── shift / pct_change ────────────────────────────────────────────────────

    def shift(self, col: str, n: int = 1) -> np.ndarray:
        """Shift column values by *n* periods (NaN fill at boundary)."""
        return self._frame.shift_col(col, n)

    def pct_change(self, col: str) -> np.ndarray:
        """Percent change between consecutive elements. First element is NaN."""
        return self._frame.pct_change(col)

    # ── data cleaning ─────────────────────────────────────────────────────────

    def drop_duplicates(self, col: str) -> "DataFrame":
        """Return a new DataFrame with duplicate values in *col* removed (keep first)."""
        return DataFrame._from_frame(self._frame.drop_duplicates(col))

    def drop_na(self, col: str) -> "DataFrame":
        """Return a new DataFrame with NaN / empty-string rows in *col* removed."""
        return DataFrame._from_frame(self._frame.drop_na(col))

    def fillna(self, col: str, value) -> "DataFrame":
        """Fill NaN / empty values in *col* with *value* (in-place; returns self)."""
        self._frame.fillna(col, value)
        return self

    def rename(self, mapping: dict[str, str]) -> "DataFrame":
        """Rename columns in-place and return self."""
        for old, new in mapping.items():
            self._frame.rename_col(old, new)
        return self

    def drop(self, col: str) -> "DataFrame":
        """Drop a column in-place and return self."""
        self._frame.drop_column(col)
        return self

    # ── I/O ───────────────────────────────────────────────────────────────────

    def to_csv(self, path: str, index: bool = True) -> None:
        self._frame.to_csv(path, index)

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        rows, ncols = self.shape
        cols = self.columns
        col_w = 13
        idx_w = 8

        lines: list[str] = []
        header = f"{'':>{idx_w}}" + "".join(f"  {c:>{col_w}}" for c in cols)
        sep = "─" * len(header)
        lines += [header, sep]

        show = min(rows, 10)
        idx = self.index
        for i in range(show):
            row = f"{idx[i]:>{idx_w}}"
            for col in cols:
                raw = self[col]
                val = raw[i] if hasattr(raw, "__getitem__") else "?"
                row += f"  {str(val):>{col_w}}"
            lines.append(row)

        if rows > 10:
            lines.append(f"  ... ({rows - 10} more rows)")

        lines.append(f"\n[{rows} rows × {ncols} columns]")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()


# ── module-level helpers ──────────────────────────────────────────────────────

def read_csv(
    path: str,
    index_col: Optional[str] = None,
    dtype: Optional[dict] = None,
) -> DataFrame:
    """
    Read a CSV file into a DataFrame.

    Uses a native C++ reader by default (dramatically faster than Python's
    csv.DictReader for large files).  Automatically strips hmdf column-name
    annotations (e.g. ``:12265:<double>``) so hmdf-written files load cleanly.

    Falls back to a pure-Python path only when *dtype* overrides are supplied.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    index_col : str, optional
        Column to use as the index (must contain unsigned integers).
        Matched after annotation stripping, so pass the clean name
        (e.g. ``"INDEX"``, not ``"INDEX:12265:<unsigned long>"``).
    dtype : dict, optional
        Mapping of clean column name -> callable for explicit type conversion.
        When set, forces the slower Python fallback path.
    """
    # ── fast path: native C++ reader ──────────────────────────────────────────
    if dtype is None:
        frame = _GrizzlarFrame.read_csv_native(str(path), index_col or "")
        return DataFrame._from_frame(frame)

    # ── fallback: Python reader (supports custom dtype converters) ────────────
    raw: dict[str, list] = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for col, val in row.items():
                raw.setdefault(col, []).append(val)

    raw = {_HMDF_ANNOTATION.sub("", k): v for k, v in raw.items()}

    data: dict = {}
    for col, vals in raw.items():
        if dtype and col in dtype:
            data[col] = [dtype[col](v) for v in vals]
            continue
        try:
            data[col] = [int(v) for v in vals]
        except ValueError:
            try:
                data[col] = [float(v) for v in vals]
            except ValueError:
                data[col] = vals

    index = None
    if index_col and index_col in data:
        raw_idx = data.pop(index_col)
        index = [int(v) for v in raw_idx]

    return DataFrame(data, index=index)
