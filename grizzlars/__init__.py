"""
grizzlars — Python wrapper for the ultrafast hmdf C++ DataFrame library.

Supported column types: float64 (double), int64, bool, str.
Index type: unsigned 64-bit integer (auto-assigned 0..N-1 if omitted).

numpy and pandas are optional — they are only imported inside functions
that explicitly need them (to_pandas, plotting, values property, etc.).
All core data operations go through the C++ backend.
"""

from __future__ import annotations

import csv
import io as _io
import json as _json
import math as _math
import operator as _op
import random as _random
import re
import sys as _sys
from typing import Any, Callable, Optional, Union

from ._grizzlars import (
    GrizzlarFrame as _GrizzlarFrame,
    set_thread_level,
    set_optimum_thread_level,
    get_thread_level,
)

__version__ = "0.1.0"
__all__ = ["DataFrame", "Series", "read_csv",
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

# ── native column dtype dispatch ──────────────────────────────────────────────
#
# The native GrizzlarFrame (src/grizzlars_shim.h) exposes typed load_column_*/
# get_column_* methods rather than one dynamically-typed pair — litgen needs
# concrete, non-template signatures to generate bindings, so the runtime
# type dispatch that used to live in C++ (detect_type()) now lives here.

def _load_col(frame, name: str, values) -> None:
    """Dispatch to the correctly-typed native load_column_* for *values*."""
    _np = _sys.modules.get("numpy")
    if _np is not None and isinstance(values, _np.ndarray):
        kind = values.dtype.kind
        if kind == "f":
            frame.load_column_double(name, values.tolist())
        elif kind in ("i", "u"):
            frame.load_column_int64(name, values.tolist())
        elif kind == "b":
            frame.load_column_bool(name, values.astype("uint8").tolist())
        else:
            frame.load_column_string(name, [str(v) for v in values.tolist()])
        return

    vals = values if isinstance(values, list) else list(values)
    if not vals:
        frame.load_column_double(name, [])
        return
    first = vals[0]
    if isinstance(first, bool):
        frame.load_column_bool(name, [1 if v else 0 for v in vals])
    elif isinstance(first, float):
        frame.load_column_double(name, [float(v) for v in vals])
    elif isinstance(first, int):
        frame.load_column_int64(name, [int(v) for v in vals])
    else:
        frame.load_column_string(name, [str(v) for v in vals])


def _mask_to_list(mask) -> list:
    """Coerce a boolean mask (numpy array, pandas Series, or list-like) into
    a plain list of 0/1 ints for filter_by_mask_list (nanobind's vector<T>
    caster wants a Sequence, not an ndarray)."""
    _np = _sys.modules.get("numpy")
    if _np is not None and isinstance(mask, _np.ndarray):
        return mask.astype("uint8").tolist()
    return [1 if v else 0 for v in mask]


def _get_col(frame, name: str):
    """Dispatch to the correctly-typed native get_column_* for *name*."""
    t = frame.col_type(name)
    if t == "double":
        import numpy as _np
        return _np.asarray(frame.get_column_double(name))
    if t == "int64":
        import numpy as _np
        return _np.asarray(frame.get_column_int64(name))
    if t == "bool":
        return list(frame.get_column_bool(name))
    return list(frame.get_column_string(name))


# ── display helpers ───────────────────────────────────────────────────────────

def _is_nan(v) -> bool:
    return isinstance(v, float) and v != v

def _display(v) -> str:
    return "NaN" if _is_nan(v) or v == "" else str(v)


# ── _ColComparison ─────────────────────────────────────────────────────────────

class _ColComparison:
    """
    Lazy column comparison returned by Series.__gt__ / __lt__ etc. when the
    Series has a tracked source DataFrame and column name.

    Passed to DataFrame.filter() it triggers the fast C++ filter_col_scalar()
    path — no Python per-row loop needed.

    Combining two _ColComparisons with & or | falls back to materialized masks.
    """

    __slots__ = ("_df", "_col", "_op", "_scalar")

    def __init__(self, df, col: str, op: str, scalar) -> None:
        self._df     = df
        self._col    = col
        self._op     = op   # ">", ">=", "<", "<=", "==", "!="
        self._scalar = scalar

    def to_mask(self) -> list:
        """Materialize to a bool list via C++ compare_col_scalar (if available)."""
        try:
            return self._df._frame.compare_col_scalar_double(self._col, self._op, float(self._scalar))
        except (AttributeError, TypeError, ValueError, RuntimeError):
            pass
        fn = _OPS[self._op]
        raw = _get_col(self._df._frame, self._col)
        return [fn(v, self._scalar) if not (_is_nan(v) if isinstance(v, float) else False)
                else False for v in raw]

    def __and__(self, other):
        if isinstance(other, _ColComparison):
            return _CombinedComparison(self, other, "and")
        return NotImplemented

    def __or__(self, other):
        if isinstance(other, _ColComparison):
            return _CombinedComparison(self, other, "or")
        return NotImplemented

    def __invert__(self):
        inv = {">" : "<=", ">=" : "<", "<" : ">=", "<=" : ">", "==" : "!=", "!=" : "=="}
        return _ColComparison(self._df, self._col, inv[self._op], self._scalar)

    # bool() forces materialization (needed when used as a bool mask list)
    def __iter__(self):
        return iter(self.to_mask())

    def __len__(self):
        return len(self._df)


class _CombinedComparison:
    """Lazy AND / OR of two _ColComparison objects."""

    __slots__ = ("_left", "_right", "_logic")

    def __init__(self, left, right, logic: str) -> None:
        self._left  = left
        self._right = right
        self._logic = logic  # "and" | "or"

    def to_mask(self) -> list:
        lm = self._left.to_mask()
        rm = self._right.to_mask()
        if self._logic == "and":
            return [a and b for a, b in zip(lm, rm)]
        return [a or b for a, b in zip(lm, rm)]

    def __and__(self, other):
        return _CombinedComparison(self, other, "and")

    def __or__(self, other):
        return _CombinedComparison(self, other, "or")

    def __iter__(self):
        return iter(self.to_mask())

    def __len__(self):
        return len(self._left)


# ── Series ────────────────────────────────────────────────────────────────────

class Series:
    """Lightweight 1-D column returned by DataFrame.__getitem__. No numpy/pandas dependency."""

    def __init__(self, data, name=None, dtype=None, _src_frame=None, _src_col=None):
        # Lazy mode: data is None, loaded on first access via _get_data()
        if data is None:
            self._data    = None
            self._src_frame = _src_frame
            self._src_col   = _src_col
        elif isinstance(data, list):
            self._data = data
            self._src_frame = None
            self._src_col   = None
        else:
            try:
                self._data = list(data)
            except TypeError:
                self._data = [data]
            self._src_frame = None
            self._src_col   = None
        self.name = name
        self._dtype_str = dtype  # "double"/"int64"/"bool"/"string"

    def _get_data(self) -> list:
        if self._data is None:
            self._data = _get_col(self._src_frame._frame, self._src_col)
            if not isinstance(self._data, list):
                self._data = list(self._data)
        return self._data

    # Support len(), iteration, indexing
    def __len__(self):
        if self._data is None and self._src_frame is not None:
            return len(self._src_frame)
        return len(self._data)
    def __iter__(self): return iter(self._get_data())
    def __getitem__(self, i): return self._get_data()[i]

    # Element-wise comparisons — fast path when we have a source frame
    def _cmp(self, other, op: str):
        if self._src_frame is not None and self._src_col is not None and not isinstance(other, Series):
            return _ColComparison(self._src_frame, self._src_col, op, other)
        data = self._get_data()
        fn = _OPS[op]
        return Series([fn(v, other) if not (_is_nan(v) if isinstance(v, float) else False)
                       else False for v in data], dtype="bool")

    def __eq__(self, other):
        if self._src_frame is not None and self._src_col is not None and not isinstance(other, Series):
            return _ColComparison(self._src_frame, self._src_col, "==", other)
        data = self._get_data()
        return Series([v == other for v in data], dtype="bool")
    def __ne__(self, other):
        if self._src_frame is not None and self._src_col is not None and not isinstance(other, Series):
            return _ColComparison(self._src_frame, self._src_col, "!=", other)
        data = self._get_data()
        return Series([v != other for v in data], dtype="bool")
    def __lt__(self, other): return self._cmp(other, "<")
    def __le__(self, other): return self._cmp(other, "<=")
    def __gt__(self, other): return self._cmp(other, ">")
    def __ge__(self, other): return self._cmp(other, ">=")

    # Logical operators on bool Series
    def __and__(self, other):
        if isinstance(other, (_ColComparison, _CombinedComparison)):
            return _CombinedComparison(self, other, "and")
        return Series([a and b for a, b in zip(self._get_data(), other._get_data())], dtype="bool")
    def __or__(self, other):
        if isinstance(other, (_ColComparison, _CombinedComparison)):
            return _CombinedComparison(self, other, "or")
        return Series([a or b for a, b in zip(self._get_data(), other._get_data())], dtype="bool")
    def __invert__(self):
        return Series([not v for v in self._get_data()], dtype="bool")

    @property
    def dtype(self): return self._dtype_str

    def unique(self):
        seen = set(); result = []
        for v in self._get_data():
            if v not in seen:
                seen.add(v); result.append(v)
        return result

    def value_counts(self):
        from collections import Counter
        c = Counter(v for v in self._get_data() if not (_is_nan(v) if isinstance(v, float) else False))
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    def to_list(self): return list(self._get_data())

    def to_numpy(self):
        import numpy as np
        return np.asarray(self._get_data())

    def __array__(self, dtype=None):
        import numpy as np
        return np.asarray(self._get_data(), dtype=dtype)

    def isnull(self):
        return Series([_is_nan(v) or v == "" for v in self._get_data()], dtype="bool")

    def notnull(self):
        return Series([not (_is_nan(v) or v == "") for v in self._get_data()], dtype="bool")

    def nunique(self):
        return len(set(v for v in self._get_data()
                       if not (_is_nan(v) if isinstance(v, float) else v == "")))

    def head(self, n: int = 5) -> "Series":
        return Series(self._get_data()[:n], name=self.name, dtype=self._dtype_str)

    def tail(self, n: int = 5) -> "Series":
        d = self._get_data()
        return Series(d[-n:] if n else [], name=self.name, dtype=self._dtype_str)

    def take(self, indices, axis=0) -> "Series":
        d = self._get_data()
        sz = len(d)
        norm = [int(i) if i >= 0 else sz + int(i) for i in indices]
        return Series([d[i] for i in norm], name=self.name, dtype=self._dtype_str)

    def __repr__(self):
        d = self._get_data()
        lines = [f"{i}    {v}" for i, v in enumerate(d[:10])]
        if len(d) > 10:
            lines.append(f"... {len(d) - 10} more")
        if self.name:
            lines.append(f"Name: {self.name}")
        return "\n".join(lines)


# ── lazy filter ───────────────────────────────────────────────────────────────

class _LazyFilterFrame:
    """
    Lazy filter result — stores the boolean mask without materialising the data.

    Inspired by polars (filters are zero-copy Arrow bitmap views) and vaex
    (expression trees deferred until terminal operation).

    • __len__ and .shape return the pre-counted row count instantly.
    • Every other DataFrame method triggers one-time materialisation.
    """

    def __init__(self, source: "DataFrame", mask) -> None:
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_mask",   mask)
        # count True values without numpy; defer for lazy comparisons
        if isinstance(mask, (_ColComparison, _CombinedComparison)):
            cached_len = -1  # unknown until materialized
        elif isinstance(mask, (list, Series)):
            cached_len = sum(1 for v in mask if v)
        else:
            try:
                import numpy as np
                cached_len = int(np.count_nonzero(mask))
            except ImportError:
                cached_len = sum(1 for v in mask if v)
        object.__setattr__(self, "_cached_len", cached_len)
        object.__setattr__(self, "_realized",   None)

    # ── materialization ───────────────────────────────────────────────────────

    def _realize(self) -> "DataFrame":
        r = object.__getattribute__(self, "_realized")
        if r is None:
            source = object.__getattribute__(self, "_source")
            mask   = object.__getattribute__(self, "_mask")
            if isinstance(mask, (_ColComparison, _CombinedComparison)):
                mask_list = mask.to_mask()
                r = DataFrame._from_frame(source._frame.filter_by_mask_list(mask_list))
            elif isinstance(mask, (list, Series)):
                mask_list = list(mask)
                r = DataFrame._from_frame(source._frame.filter_by_mask_list(mask_list))
            else:
                r = DataFrame._from_frame(source._frame.filter_by_mask_list(_mask_to_list(mask)))
            object.__setattr__(self, "_realized", r)
        return r

    # ── cheap operations (no materialization) ─────────────────────────────────

    def __len__(self) -> int:
        n = object.__getattribute__(self, "_cached_len")
        if n < 0:
            n = len(self._realize())
            object.__setattr__(self, "_cached_len", n)
        return n

    @property
    def shape(self) -> tuple:
        n   = len(self)
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


# ── indexers ──────────────────────────────────────────────────────────────────

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
        # numpy integer array (sklearn CV fold indices) — order-preserving via numpy fancy-index
        _np = _sys.modules.get("numpy")
        if _np is not None and isinstance(key, _np.ndarray) and key.dtype.kind in ("i", "u"):
            return _np.asarray(self._df)[key]
        if isinstance(key, list) and key and isinstance(key[0], int):
            import numpy as _np2
            return _np2.asarray(self._df)[_np2.asarray(key, dtype=_np2.intp)]
        raise TypeError(f"iloc key must be int, slice, or int array, got {type(key)}")


class _LocIndexer:
    """Label-based indexer: df.loc[label], df.loc[start:stop], df.loc[bool_mask]."""

    def __init__(self, df: "DataFrame") -> None:
        self._df = df

    def __getitem__(self, key):
        idx = list(self._df.index)
        # Two-axis: df.loc[row_key, col_key]
        if isinstance(key, tuple):
            row_key, col_key = key[0], key[1]
            row_df = _LocIndexer(self._df)[row_key]
            if isinstance(col_key, list):
                return row_df.select(col_key)
            return row_df[col_key]
        # Boolean mask (list, Series, or numpy array)
        if isinstance(key, (list, Series)):
            mask_list = [bool(v) for v in key]
            return self._df._from_frame(self._df._frame.filter_by_mask_list(mask_list))
        # Try numpy array
        try:
            import numpy as np
            if isinstance(key, np.ndarray):
                if key.dtype == bool or key.dtype == np.bool_:
                    return self._df._from_frame(self._df._frame.filter_by_mask_list(_mask_to_list(key)))
                # Integer label array — O(n) dict lookup instead of O(n²) list.index
                idx_map = {int(v): i for i, v in enumerate(idx)}
                n = len(idx)
                mask = np.zeros(n, dtype=np.bool_)
                for k in key:
                    p = idx_map.get(int(k))
                    if p is not None:
                        mask[p] = True
                return self._df._from_frame(self._df._frame.filter_by_mask_list(_mask_to_list(mask)))
        except ImportError:
            pass
        # Slice by label
        if isinstance(key, slice):
            start_i = idx.index(key.start) if key.start is not None else 0
            stop_i = (idx.index(key.stop) + 1) if key.stop is not None else len(idx)
            return self._df.iloc[start_i:stop_i]
        # Single label
        try:
            i = idx.index(int(key))
        except ValueError:
            raise KeyError(key)
        return self._df.iloc[i]

    def __setitem__(self, key, value):
        raise NotImplementedError("loc assignment not yet supported")


class _AtIndexer:
    """Scalar label-based accessor: df.at[row_label, col_name]."""

    def __init__(self, df: "DataFrame") -> None:
        self._df = df

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("at requires (row_label, col_name)")
        row_label, col = key
        idx = list(self._df.index)
        try:
            i = idx.index(int(row_label))
        except ValueError:
            raise KeyError(row_label)
        return self._df[col][i]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("at requires (row_label, col_name)")
        row_label, col = key
        idx = list(self._df.index)
        try:
            i = idx.index(int(row_label))
        except ValueError:
            raise KeyError(row_label)
        arr = list(self._df[col])
        arr[i] = value
        self._df[col] = arr


class _IAtIndexer:
    """Scalar integer-position accessor: df.iat[row_pos, col_pos]."""

    def __init__(self, df: "DataFrame") -> None:
        self._df = df

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("iat requires (row_pos, col_pos)")
        r, c = key
        col = self._df.columns[c]
        return self._df[col][r]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError("iat requires (row_pos, col_pos)")
        r, c = key
        col = self._df.columns[c]
        arr = list(self._df[col])
        arr[r] = value
        self._df[col] = arr


# ── groupby ───────────────────────────────────────────────────────────────────

class _GroupBy:
    """Returned by DataFrame.groupby(col). Supports .agg(dict/str/callable) and shorthand methods."""

    def __init__(self, df: "DataFrame", by: str) -> None:
        self._df = df
        self._by = by

    def agg(self, spec) -> "DataFrame":
        """
        Aggregate grouped columns.

        spec can be:
          - dict: {col: func_str}  e.g. {"close": "mean", "volume": "sum"}
          - str: apply the same function to all numeric columns
          - callable: apply to all numeric columns
        """
        if isinstance(spec, str):
            func = spec
            numeric_cols = [c for c in self._df.columns
                            if c != self._by and
                            self._df._frame.col_type(c) in ("double", "int64")]
            spec = {c: func for c in numeric_cols}
        elif callable(spec):
            numeric_cols = [c for c in self._df.columns
                            if c != self._by and
                            self._df._frame.col_type(c) in ("double", "int64")]
            spec = {c: spec for c in numeric_cols}
        if isinstance(spec, dict):
            resolved = {}
            for col, fn in spec.items():
                resolved[col] = fn
            has_callable = any(callable(v) for v in resolved.values())
            if not has_callable:
                agg_cols = list(resolved.keys())
                agg_funcs = list(resolved.values())
                return DataFrame._from_frame(
                    self._df._frame.groupby_agg(self._by, agg_cols, agg_funcs))
            return self._python_agg(resolved)
        raise TypeError(f"agg spec must be dict, str, or callable, got {type(spec)}")

    def _python_agg(self, spec: dict) -> "DataFrame":
        """Python-level groupby aggregation for callable functions."""
        by_col = list(self._df[self._by])
        groups: dict = {}
        for i, key in enumerate(by_col):
            groups.setdefault(key, []).append(i)
        result: dict = {self._by: [], **{col: [] for col in spec}}
        for key, positions in sorted(groups.items()):
            result[self._by].append(key)
            for col, fn in spec.items():
                vals = [self._df[col][p] for p in positions]
                if callable(fn):
                    result[col].append(fn(vals))
                elif fn == "mean":
                    result[col].append(sum(vals) / len(vals) if vals else float("nan"))
                elif fn == "sum":
                    result[col].append(sum(vals))
                elif fn == "min":
                    result[col].append(min(vals))
                elif fn == "max":
                    result[col].append(max(vals))
                elif fn == "count":
                    result[col].append(len(vals))
                elif fn == "std":
                    n = len(vals)
                    if n < 2:
                        result[col].append(0.0)
                    else:
                        m = sum(vals) / n
                        result[col].append(_math.sqrt(sum((v - m)**2 for v in vals) / (n - 1)))
                elif fn == "median":
                    sv = sorted(vals)
                    n = len(sv)
                    result[col].append((sv[n//2] + sv[(n-1)//2]) / 2.0 if n else float("nan"))
                elif fn == "first":
                    result[col].append(vals[0])
                elif fn == "last":
                    result[col].append(vals[-1])
                else:
                    raise ValueError(f"Unknown aggregation function: {fn!r}")
        return DataFrame(result)

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

    def median(self, col: str) -> "DataFrame":
        return self.agg({col: "median"})

    def first(self, col: str) -> "DataFrame":
        return self.agg({col: "first"})

    def last(self, col: str) -> "DataFrame":
        return self.agg({col: "last"})


# ── DataFrame ─────────────────────────────────────────────────────────────────

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
        index=None,
    ) -> None:
        self._frame = _GrizzlarFrame()
        if data:
            n = max(len(v) for v in data.values())
            idx = list(range(n)) if index is None else list(index)
            self._frame.load_index(idx)
            for name, values in data.items():
                _np = _sys.modules.get("numpy")
                if _np is not None and isinstance(values, _np.ndarray):
                    _load_col(self._frame, name, values)
                elif isinstance(values, list):
                    _load_col(self._frame, name, values)
                else:
                    try:
                        v = list(values)
                    except TypeError:
                        v = [values]
                    _load_col(self._frame, name, v)

    @classmethod
    def _from_frame(cls, frame: _GrizzlarFrame) -> "DataFrame":
        obj = cls.__new__(cls)
        obj._frame = frame
        return obj

    def _copy(self) -> "DataFrame":
        return DataFrame._from_frame(self._frame.deep_copy())

    def copy(self) -> "DataFrame":
        return DataFrame._from_frame(self._frame.deep_copy())

    # ── column access ─────────────────────────────────────────────────────────

    def __getitem__(self, key):
        # _ColComparison / _CombinedComparison → fast C++ filter
        if isinstance(key, _ColComparison):
            try:
                return DataFrame._from_frame(
                    self._frame.filter_col_scalar_double(key._col, key._op, float(key._scalar))
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass
            return DataFrame._from_frame(self._frame.filter_by_mask_list(key.to_mask()))
        if isinstance(key, _CombinedComparison):
            return DataFrame._from_frame(self._frame.filter_by_mask_list(key.to_mask()))
        # Boolean Series → row filter (no numpy needed)
        if isinstance(key, Series) and key.dtype == "bool":
            return DataFrame._from_frame(self._frame.filter_by_mask_list(key.to_list()))
        # Plain bool list → row filter
        if isinstance(key, list) and key and isinstance(key[0], bool):
            return DataFrame._from_frame(self._frame.filter_by_mask_list(key))
        # numpy array support (optional) — use sys.modules to avoid slow cold import
        _np = _sys.modules.get("numpy")
        if _np is not None and isinstance(key, _np.ndarray) and (key.dtype == bool or key.dtype == _np.bool_):
            return DataFrame._from_frame(self._frame.filter_by_mask_list(_mask_to_list(key)))
        # pandas Series support (optional) — use sys.modules to avoid slow cold import
        _pd = _sys.modules.get("pandas")
        if _pd is not None and isinstance(key, _pd.Series) and (str(key.dtype) in ("bool", "boolean")):
            return DataFrame._from_frame(self._frame.filter_by_mask_list(_mask_to_list(key.to_numpy())))
        # Slice → row range
        if isinstance(key, slice):
            total = len(self)
            start, stop, step = key.indices(total)
            if step == 1:
                return DataFrame._from_frame(self._frame.iloc(start, stop))
            indices = range(start, stop, step)
            return self.take(indices)
        # List of column names → sub-DataFrame
        if isinstance(key, list) and key and isinstance(key[0], str):
            return DataFrame._from_frame(self._frame.select_columns(key))
        # String column name → lazy Series (data loaded on first access)
        if isinstance(key, str):
            dtype = self._frame.col_type(key)
            return Series(None, name=key, dtype=dtype,
                          _src_frame=self, _src_col=key)
        raise TypeError(
            f"DataFrame key must be a column name (str) or boolean array/Series, "
            f"got {type(key).__name__}"
        )

    def __setitem__(self, col, values) -> None:
        # List of column names + 2D array (e.g. from sklearn transform)
        if isinstance(col, list):
            try:
                import numpy as _np
                arr = _np.asarray(values)
                for i, c in enumerate(col):
                    _load_col(self._frame, c, arr[:, i].tolist())
                return
            except ImportError:
                pass
            # fallback: values is a list-of-lists / iterable of rows
            rows = list(values)
            for i, c in enumerate(col):
                _load_col(self._frame, c, [row[i] for row in rows])
            return
        if isinstance(values, list):
            v = values
        else:
            try:
                v = list(values)
            except TypeError:
                v = [values]
        _load_col(self._frame, col, v)

    def __contains__(self, col: str) -> bool:
        return self._frame.has_column(col)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        try:
            frame = object.__getattribute__(self, "_frame")
            if frame.has_column(name):
                self[name] = value
                return
        except AttributeError:
            pass
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        # Dot-notation column access: df.Age, df.Gender, …
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            cols = self.columns
        except AttributeError:
            raise AttributeError(name)
        if name in cols:
            return self[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __iter__(self):
        """Iterate over column names (pandas-compatible)."""
        return iter(self.columns)

    def __array__(self, dtype=None):
        """Allow numpy to convert this DataFrame to a 2-D array (used by sklearn etc.)."""
        import numpy as _np
        arr = self.values
        if dtype is not None:
            # Give a useful error listing which columns still contain strings
            string_cols = [c for c in self.columns if self._frame.col_type(c) == "string"]
            if string_cols:
                raise ValueError(
                    f"Cannot convert DataFrame to {dtype}: the following columns are still "
                    f"string type and must be encoded first: {string_cols}"
                )
            return arr.astype(dtype)
        return arr

    def __bool__(self):
        raise ValueError(
            "The truth value of a DataFrame is ambiguous. "
            "Use df.empty, df.any() or df.all()."
        )

    @property
    def index(self):
        import numpy as _np
        return _np.asarray(self._frame.get_index(), dtype=_np.uint64)

    @property
    def columns(self) -> list:
        return self._frame.columns()

    @property
    def shape(self) -> tuple:
        return self._frame.shape()

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def dtypes(self) -> dict:
        """Return a mapping of column name -> type string."""
        return {c: self._frame.col_type(c) for c in self.columns}

    @property
    def iloc(self) -> _ILocIndexer:
        """Integer-location based indexing: df.iloc[0:100], df.iloc[-1]."""
        return _ILocIndexer(self)

    @property
    def loc(self) -> _LocIndexer:
        """Label-based indexing: df.loc[label], df.loc[start:stop], df.loc[mask]."""
        return _LocIndexer(self)

    @property
    def at(self) -> _AtIndexer:
        """Fast scalar label-based access: df.at[row_label, col_name]."""
        return _AtIndexer(self)

    @property
    def iat(self) -> _IAtIndexer:
        """Fast scalar integer-position access: df.iat[row_pos, col_pos]."""
        return _IAtIndexer(self)

    @property
    def T(self) -> "DataFrame":
        """Transpose the DataFrame."""
        return self.transpose()

    @property
    def axes(self) -> list:
        """Return [index, columns]."""
        return [self.index, self.columns]

    @property
    def empty(self) -> bool:
        """True if the DataFrame has no rows."""
        return self.shape[0] == 0

    @property
    def ndim(self) -> int:
        """Always 2 for a DataFrame."""
        return 2

    @property
    def size(self) -> int:
        """Total number of elements (rows × columns)."""
        r, c = self.shape
        return r * c

    @property
    def values(self):
        """Return the DataFrame as a 2-D numpy array (object dtype if mixed types)."""
        import numpy as np
        cols = self.columns
        if not cols:
            return np.empty((len(self), 0))
        # get_column returns a numpy array for numeric types — no list round-trip needed
        arrays = [np.asarray(_get_col(self._frame, c)) for c in cols]
        dtypes_set = {a.dtype.kind for a in arrays}
        if dtypes_set <= {"f", "i", "u"}:  # all numeric — np.column_stack upcasts to float64
            return np.column_stack(arrays)
        out = np.empty((len(self), len(cols)), dtype=object)
        for j, a in enumerate(arrays):
            out[:, j] = a
        return out

    # ── info / metadata ───────────────────────────────────────────────────────

    def keys(self) -> list:
        """Return column names (pandas-compatible)."""
        return self.columns

    def info(self, verbose: bool = True) -> None:
        """Print a concise summary of the DataFrame."""
        rows, ncols = self.shape
        print(f"<grizzlars.DataFrame>")
        print(f"RangeIndex: {rows} entries")
        print(f"Data columns ({ncols} total):")
        if verbose:
            for col in self.columns:
                typ = self._frame.col_type(col)
                missing = self._frame.n_missing(col)
                non_null = rows - missing
                print(f"  {col}: {non_null} non-null {typ}")
        total_mem = self._memory_bytes()
        print(f"memory usage: {total_mem / 1024:.1f}+ KB")

    def _memory_bytes(self) -> int:
        total = 0
        rows = self.shape[0]
        for col in self.columns:
            typ = self._frame.col_type(col)
            if typ == "double":
                total += rows * 8
            elif typ == "int64":
                total += rows * 8
            elif typ == "bool":
                total += rows
            else:
                total += sum(len(str(v)) for v in self[col])
        return total

    def memory_usage(self, deep: bool = False) -> dict:
        """Return estimated memory usage per column in bytes."""
        rows = self.shape[0]
        result = {"Index": rows * 8}
        for col in self.columns:
            typ = self._frame.col_type(col)
            if typ in ("double", "int64"):
                result[col] = rows * 8
            elif typ == "bool":
                result[col] = rows
            else:
                if deep:
                    result[col] = sum(len(str(v)) for v in self[col])
                else:
                    result[col] = rows * 8
        return result

    # ── statistics ────────────────────────────────────────────────────────────

    def _numeric_cols(self) -> list:
        """Columns that support numeric reduction (double, int64, bool)."""
        return [c for c in self.columns
                if self._frame.col_type(c) in ("double", "int64", "bool")]

    def multi_stats(self, col: str) -> dict:
        """Compute count/mean/std/min/max/sum in a single C++ pass (fastest)."""
        try:
            return dict(self._frame.multi_stat_col(col))
        except AttributeError:
            return {
                "count": self._frame.count(col),
                "mean":  self._frame.mean(col),
                "std":   self._frame.std_dev(col),
                "min":   self._frame.col_min(col),
                "max":   self._frame.col_max(col),
                "sum":   self._frame.sum(col),
            }

    def sum(self, col: Optional[str] = None):
        """Sum of column values. No col → one-row DataFrame of all numeric columns."""
        if col is not None:
            if self._frame.col_type(col) == "bool":
                return sum(1 for v in self[col] if v)
            return self._frame.sum(col)
        return DataFrame._from_frame(self._frame.reduce_all("sum"))

    def mean(self, col: Optional[str] = None):
        """Mean of column values. No col → one-row DataFrame of all numeric columns."""
        if col is not None:
            if self._frame.col_type(col) == "bool":
                raw = list(self[col])
                n = len(raw)
                return sum(1 for v in raw if v) / n if n else float("nan")
            return self._frame.mean(col)
        return DataFrame._from_frame(self._frame.reduce_all("mean"))

    def std(self, col: Optional[str] = None):
        """Standard deviation. No col → one-row DataFrame."""
        if col is not None:
            if self._frame.col_type(col) == "bool":
                raw = [1.0 if v else 0.0 for v in self[col]]
                n = len(raw)
                if n < 2: return 0.0
                m = sum(raw) / n
                return _math.sqrt(sum((v - m)**2 for v in raw) / (n - 1))
            return self._frame.std_dev(col)
        return DataFrame._from_frame(self._frame.reduce_all("std"))

    def min(self, col: Optional[str] = None):
        """Minimum value. No col → one-row DataFrame."""
        if col is not None:
            if self._frame.col_type(col) == "bool":
                raw = list(self[col])
                return 0.0 if any(not v for v in raw) else 1.0
            return self._frame.col_min(col)
        return DataFrame._from_frame(self._frame.reduce_all("min"))

    def max(self, col: Optional[str] = None):
        """Maximum value. No col → one-row DataFrame."""
        if col is not None:
            if self._frame.col_type(col) == "bool":
                raw = list(self[col])
                return 1.0 if any(v for v in raw) else 0.0
            return self._frame.col_max(col)
        return DataFrame._from_frame(self._frame.reduce_all("max"))

    def count(self, col: Optional[str] = None):
        """Count non-null values. No col → one-row DataFrame."""
        if col is not None:
            typ = self._frame.col_type(col)
            if typ == "double":
                return sum(1 for v in self[col] if not _is_nan(v))
            elif typ == "string":
                return sum(1 for v in self[col] if v != "")
            return int(self._frame.count(col))
        return DataFrame._from_frame(self._frame.reduce_all("count"))

    def median(self, col: Optional[str] = None):
        """Median value. No col → one-row DataFrame."""
        if col is not None:
            raw = [v for v in self[col] if not (_is_nan(v) if isinstance(v, float) else False)]
            if not raw: return float("nan")
            sv = sorted(raw)
            n = len(sv)
            return (sv[n//2] + sv[(n-1)//2]) / 2.0
        return DataFrame._from_frame(self._frame.reduce_all("median"))

    def var(self, col: Optional[str] = None, ddof: int = 1):
        """Variance. No col → one-row DataFrame."""
        if col is not None:
            raw = [float(v) for v in self[col] if not (_is_nan(v) if isinstance(v, float) else False)]
            n = len(raw)
            if n <= ddof: return float("nan")
            m = sum(raw) / n
            return sum((v - m)**2 for v in raw) / (n - ddof)
        return DataFrame._from_frame(self._frame.reduce_all("var"))

    def sem(self, col: Optional[str] = None, ddof: int = 1):
        """Standard error of the mean, ignoring NaN."""
        def _sem(c):
            raw = [float(v) for v in self[c] if not (_is_nan(v) if isinstance(v, float) else False)]
            n = len(raw)
            if n <= ddof: return float("nan")
            m = sum(raw) / n
            s = _math.sqrt(sum((v - m)**2 for v in raw) / (n - ddof))
            return s / _math.sqrt(n)
        if col is not None:
            return _sem(col)
        return DataFrame({c: [_sem(c)] for c in self._numeric_cols()})

    def skew(self, col: Optional[str] = None):
        """Sample skewness — delegates to C++."""
        if col is not None:
            return self._frame.skew_col(col)
        return DataFrame({c: [self._frame.skew_col(c)] for c in self._numeric_cols()})

    def kurt(self, col: Optional[str] = None):
        """Excess kurtosis — delegates to C++."""
        if col is not None:
            return self._frame.kurt_col(col)
        return DataFrame({c: [self._frame.kurt_col(c)] for c in self._numeric_cols()})

    def kurtosis(self, col: Optional[str] = None):
        """Alias for kurt()."""
        return self.kurt(col)

    def prod(self, col: Optional[str] = None):
        """Product of all values, ignoring NaN."""
        def _prod(c):
            raw = [float(v) for v in self[c] if not (_is_nan(v) if isinstance(v, float) else False)]
            result = 1.0
            for v in raw: result *= v
            return result
        if col is not None:
            return _prod(col)
        return DataFrame({c: [_prod(c)] for c in self._numeric_cols()})

    def product(self, col: Optional[str] = None):
        """Alias for prod()."""
        return self.prod(col)

    def abs(self, col: Optional[str] = None):
        """Absolute value in-place per column; returns new DataFrame. If col given, modifies that col."""
        result = self._copy()
        if col is not None:
            result._frame.abs_col(col)
            return result
        for c in self.columns:
            if self._frame.col_type(c) in ("double", "int64"):
                result._frame.abs_col(c)
        return result

    def idxmax(self, col: str) -> int:
        """Return the index label of the maximum value in *col*."""
        raw = list(self[col])
        best_i, best_v = 0, float("-inf")
        for i, v in enumerate(raw):
            if not _is_nan(v) and v > best_v:
                best_v = v; best_i = i
        return int(list(self.index)[best_i])

    def idxmin(self, col: str) -> int:
        """Return the index label of the minimum value in *col*."""
        raw = list(self[col])
        best_i, best_v = 0, float("inf")
        for i, v in enumerate(raw):
            if not _is_nan(v) and v < best_v:
                best_v = v; best_i = i
        return int(list(self.index)[best_i])

    def rank(self, col: str, method: str = "average", ascending: bool = True):
        """Return rank of each element in *col*."""
        try:
            from scipy.stats import rankdata
            import numpy as np
            arr = [float(v) for v in self[col]]
            if not ascending:
                arr = [-v for v in arr]
            return rankdata(arr, method=method)
        except ImportError:
            raw = [float(v) for v in self[col]]
            if not ascending:
                raw = [-v for v in raw]
            order = sorted(range(len(raw)), key=lambda i: raw[i])
            rank_arr = [0.0] * len(raw)
            for r, i in enumerate(order):
                rank_arr[i] = r + 1.0
            return rank_arr

    def mode(self, col: str):
        """Return the most frequent value(s) in *col* — delegates to C++."""
        t = self._frame.col_type(col)
        if t == "double":
            return list(self._frame.mode_col_double(col))
        if t == "int64":
            return list(self._frame.mode_col_int64(col))
        return list(self._frame.mode_col_string(col))

    def describe(self) -> "DataFrame":
        """
        Return count / mean / std / min / 25% / 50% / 75% / max for every
        numeric column as a DataFrame (one row per statistic).
        """
        stats = self._frame.describe()
        if not stats:
            return DataFrame()
        stat_names = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        data: dict = {"statistic": stat_names}
        for col, d in stats.items():
            data[col] = [
                float(d["count"]),
                float(d["mean"]),
                float(d["std"]),
                float(d["min"]),
                float(d["25%"]),
                float(d["50%"]),
                float(d["75%"]),
                float(d["max"]),
            ]
        return DataFrame(data, index=list(range(len(stat_names))))

    def quantile(self, col: str, q: float) -> float:
        """Return the q-th quantile of a column (q in [0, 1])."""
        return self._frame.quantile(col, q)

    def corr(self, col1: Optional[str] = None, col2: Optional[str] = None,
             method: str = "pearson", numeric_only: bool = True):
        """
        Pearson correlation.

        corr()              — full correlation matrix (as pandas DataFrame for seaborn compat)
        corr(col1, col2)    — scalar correlation between two columns
        """
        if col1 is not None and col2 is not None:
            return self._frame.corr(col1, col2)
        # Full correlation matrix — return as pandas DataFrame for seaborn compatibility
        num_cols = [c for c in self.columns
                    if self._frame.col_type(c) in ("double", "int64")]
        matrix = {c: [] for c in num_cols}
        for i, ci in enumerate(num_cols):
            for j, cj in enumerate(num_cols):
                if i == j:
                    matrix[ci].append(1.0)
                elif j < i:
                    matrix[ci].append(matrix[cj][i])
                else:
                    matrix[ci].append(self._frame.corr(ci, cj))
        try:
            import pandas as _pd
            return _pd.DataFrame(matrix, index=num_cols)
        except ImportError:
            return DataFrame(matrix)

    def cov(self, col1: Optional[str] = None, col2: Optional[str] = None,
            numeric_only: bool = True):
        """
        Sample covariance (n-1 denominator).

        cov()              — full covariance matrix
        cov(col1, col2)    — scalar covariance between two columns
        """
        if col1 is not None and col2 is not None:
            return self._frame.cov(col1, col2)
        num_cols = [c for c in self.columns
                    if self._frame.col_type(c) in ("double", "int64")]
        matrix = {c: [] for c in num_cols}
        for i, ci in enumerate(num_cols):
            for j, cj in enumerate(num_cols):
                if i == j:
                    raw = [float(v) for v in self[ci] if not (_is_nan(v) if isinstance(v, float) else False)]
                    n = len(raw)
                    if n < 2:
                        matrix[ci].append(0.0)
                    else:
                        m = sum(raw) / n
                        matrix[ci].append(sum((v - m)**2 for v in raw) / (n - 1))
                elif j < i:
                    matrix[ci].append(matrix[cj][i])
                else:
                    matrix[ci].append(self._frame.cov(ci, cj))
        try:
            import pandas as _pd
            return _pd.DataFrame(matrix, index=num_cols)
        except ImportError:
            return DataFrame(matrix)

    def corrwith(self, other: "DataFrame") -> dict:
        """Pairwise Pearson correlation with another DataFrame (matching columns)."""
        result = {}
        for col in self.columns:
            if col in other:
                result[col] = self._frame.corr(col, col) if False else (
                    self._frame.corr(col, col)
                )
        return result

    def nunique(self, col: str) -> int:
        """Number of distinct values in *col*."""
        return self._frame.nunique(col)

    def unique(self, col: str):
        """Sorted unique values in *col*."""
        t = self._frame.col_type(col)
        if t == "double":
            return self._frame.unique_double(col)
        if t == "int64":
            return self._frame.unique_int64(col)
        return self._frame.unique_string(col)

    def n_missing(self, col: str) -> int:
        """Count of NaN / empty-string values in *col*."""
        return self._frame.n_missing(col)

    def value_counts(self, col: str) -> "DataFrame":
        """Return a DataFrame with ["value", "count"] sorted by count descending."""
        t = self._frame.col_type(col)
        if t == "double":
            return DataFrame._from_frame(self._frame.value_counts_double(col))
        if t == "int64":
            return DataFrame._from_frame(self._frame.value_counts_int64(col))
        return DataFrame._from_frame(self._frame.value_counts_string(col))

    # ── boolean reductions ────────────────────────────────────────────────────

    def any(self, col: str) -> bool:
        """Return True if any value in *col* is truthy."""
        raw = self[col]
        return any(bool(v) and not _is_nan(v) for v in raw)

    def all(self, col: str) -> bool:
        """Return True if all values in *col* are truthy."""
        raw = self[col]
        return all(bool(v) and not _is_nan(v) for v in raw)

    # ── sorting ───────────────────────────────────────────────────────────────

    def sort(self, by: str, ascending: bool = True) -> "DataFrame":
        """Return a new DataFrame sorted by *by* column (non-mutating)."""
        return DataFrame._from_frame(self._frame.sort_by(by, ascending))

    def sort_values(
        self,
        by,
        ascending=True,
    ) -> "DataFrame":
        """Return a new DataFrame sorted by one or more columns."""
        if isinstance(by, list):
            asc_list = [ascending] * len(by) if isinstance(ascending, bool) else ascending
            result = self
            for col, asc in zip(reversed(by), reversed(asc_list)):
                result = DataFrame._from_frame(result._frame.sort_by(col, asc))
            return result
        return self.sort(by, bool(ascending))

    def sort_index(self, ascending: bool = True) -> "DataFrame":
        """Return a new DataFrame sorted by index (non-mutating)."""
        return DataFrame._from_frame(self._frame.sort_index(ascending))

    # ── slicing / filtering ───────────────────────────────────────────────────

    def head(self, n: int = 5) -> "DataFrame":
        return DataFrame._from_frame(self._frame.iloc(0, min(n, len(self))))

    def tail(self, n: int = 5) -> "DataFrame":
        total = len(self)
        return DataFrame._from_frame(self._frame.iloc(max(0, total - n), total))

    def take(self, indices, axis=0) -> "DataFrame":
        """Select rows by integer position (used by sklearn, numpy, etc.)."""
        _ = axis  # only row-axis supported
        idx_list = [int(i) for i in indices]
        # Fast C++ path (available after rebuild)
        try:
            return DataFrame._from_frame(self._frame.take_rows(idx_list))
        except AttributeError:
            pass
        # Pre-rebuild fallback: if indices are sorted use C++ mask filter (fast)
        sz = len(self)
        norm = [i if i >= 0 else sz + i for i in idx_list]
        if norm == sorted(norm):
            mask = [False] * sz
            for i in norm:
                mask[i] = True
            return DataFrame._from_frame(self._frame.filter_by_mask_list(mask))
        # Unsorted fallback: sort → mask → reorder (still cheaper than per-element Python)
        order = sorted(range(len(norm)), key=lambda k: norm[k])
        sorted_norm = [norm[k] for k in order]
        mask = [False] * sz
        for i in sorted_norm:
            mask[i] = True
        filtered = DataFrame._from_frame(self._frame.filter_by_mask_list(mask))
        # filtered rows are in original order; reorder to match requested indices
        filtered_raw = filtered._frame.get_index()
        src_raw = self._frame.get_index()
        filtered_pos = {src_raw[sorted_norm[k]]: k for k in range(len(sorted_norm))}
        reorder = [filtered_pos[filtered_raw[i]] for i in range(len(filtered_raw))]
        inv = [0] * len(reorder)
        for new, old in enumerate(reorder):
            inv[old] = new
        # apply inv permutation to unsorted request
        final_order = [inv[order[k]] for k in range(len(order))]
        cols = filtered.columns
        col_data = [list(filtered[c]) for c in cols]
        data = {cols[j]: [col_data[j][final_order[i]] for i in range(len(final_order))]
                for j in range(len(cols))}
        return DataFrame(data)

    def filter(self, col_or_mask, op: Optional[str] = None,
               value=None) -> "_LazyFilterFrame":
        """
        Return rows matching a condition — lazy (polars/vaex style).

        Mask mode (polars-style):
            df.filter(df["volume"] > 5_000_000)
            df.filter([True, False, True, ...])

        Column + operator mode:
            df.filter("volume", ">", 5_000_000)
            Operators: ">", ">=", "<", "<=", "==", "!="
        """
        if op is None:
            mask = col_or_mask
            # Fast path: _ColComparison → single C++ call (no Python loop)
            if isinstance(mask, _ColComparison):
                try:
                    return DataFrame._from_frame(
                        self._frame.filter_col_scalar_double(mask._col, mask._op, float(mask._scalar))
                    )
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    pass
                return _LazyFilterFrame(self, mask.to_mask())
            # Combined comparison: materialize via to_mask()
            if isinstance(mask, _CombinedComparison):
                return _LazyFilterFrame(self, mask.to_mask())
            # Series or list of bools: use as-is
            if isinstance(mask, (Series, list)):
                return _LazyFilterFrame(self, mask)
            # numpy array: keep as numpy for filter_by_mask
            try:
                import numpy as np
                if isinstance(mask, np.ndarray):
                    return _LazyFilterFrame(self, mask)
            except ImportError:
                pass
            return _LazyFilterFrame(self, mask)
        else:
            if op not in _OPS:
                raise ValueError(f"Unknown operator {op!r}. Choose from {list(_OPS)}")
            # Fast C++ path for col+op+scalar
            try:
                return DataFrame._from_frame(
                    self._frame.filter_col_scalar_double(col_or_mask, op, float(value))
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass
            raw = list(self[col_or_mask])
            fn = _OPS[op]
            mask = [fn(v, value) for v in raw]
            return _LazyFilterFrame(self, mask)

    def query(self, expr: str) -> "DataFrame":
        """
        Filter rows using a query string.
        Supports simple expressions: "col > value", "col == 'str'", "col != value".
        """
        m = re.match(r"^\s*(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)\s*$", expr)
        if not m:
            raise ValueError(f"Cannot parse query expression: {expr!r}")
        col, op_str, val_str = m.group(1), m.group(2), m.group(3).strip()
        val_str = val_str.strip("'\"")
        try:
            val: Any = int(val_str)
        except ValueError:
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
        return self.filter(col, op_str, val)

    def select(self, columns: list) -> "DataFrame":
        """Return a new DataFrame containing only the specified columns."""
        return DataFrame._from_frame(self._frame.select_columns(columns))

    def nlargest(self, n: int, col: str) -> "DataFrame":
        """Return the *n* rows with the largest values in *col*."""
        return self.sort(col, ascending=False).head(n)

    def nsmallest(self, n: int, col: str) -> "DataFrame":
        """Return the *n* rows with the smallest values in *col*."""
        return self.sort(col, ascending=True).head(n)

    def sample(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> "DataFrame":
        """Return a random sample of rows."""
        total = len(self)
        if frac is not None:
            n = max(1, int(total * frac))
        if n is None:
            n = 1
        rng = _random.Random(random_state)
        positions = sorted(rng.sample(range(total), min(n, total)))
        mask = [False] * total
        for p in positions:
            mask[p] = True
        return DataFrame._from_frame(self._frame.filter_by_mask_list(mask))

    def isin(self, col: str, values) -> list:
        """Return a bool list: True where *col* value is in *values*."""
        t = self._frame.col_type(col)
        if t == "string":
            return list(self._frame.isin_col_string(col, [str(v) for v in values]))
        return list(self._frame.isin_col_double(col, [float(v) for v in values]))

    def duplicated(self, subset=None, keep: str = "first") -> list:
        """Return a bool list marking duplicate rows."""
        cols = self.columns if subset is None else ([subset] if isinstance(subset, str) else subset)
        return self._frame.duplicated_rows(cols, keep)

    def with_column(self, name: str, values) -> "DataFrame":
        """Return a new DataFrame with *name* column added / replaced (non-mutating)."""
        result = self._copy()
        v = values if isinstance(values, list) else list(values)
        _load_col(result._frame, name, v)
        return result

    def assign(self, **kwargs) -> "DataFrame":
        """Return a new DataFrame with extra columns added (non-mutating)."""
        result = self._copy()
        for name, values in kwargs.items():
            v = values if isinstance(values, list) else list(values)
            _load_col(result._frame, name, v)
        return result

    def insert(self, loc: int, column: str, value) -> None:
        """Insert a column at integer position *loc* (in-place)."""
        v = value if isinstance(value, list) else list(value)
        _load_col(self._frame, column, v)
        cols = self.columns
        if column in cols:
            cols.remove(column)
            cols.insert(loc, column)
            reordered = self._frame.select_columns(cols)
            self._frame = reordered

    def pop(self, col: str):
        """Remove and return a column."""
        data = self[col]
        self._frame.drop_column(col)
        return data

    # ── iteration ─────────────────────────────────────────────────────────────

    def items(self):
        """Iterate over (column_name, values) pairs."""
        for col in self.columns:
            yield col, self[col]

    def iterrows(self):
        """Iterate over (index_label, row_dict) pairs."""
        idx = list(self.index)
        cols = self.columns
        col_data = {col: _get_col(self._frame, col) for col in cols}
        for i in range(len(self)):
            row = {col: col_data[col][i] for col in cols}
            yield int(idx[i]), row

    def itertuples(self, index: bool = True, name: str = "Pandas"):
        """Iterate over rows as namedtuples."""
        from collections import namedtuple
        cols = self.columns
        safe_cols = [re.sub(r"[^a-zA-Z0-9_]", "_", c) for c in cols]
        fields = (["Index"] + safe_cols) if index else safe_cols
        Row = namedtuple(name or "Row", fields)
        idx = list(self.index)
        col_data = {col: _get_col(self._frame, col) for col in cols}
        for i in range(len(self)):
            vals = [col_data[col][i] for col in cols]
            if index:
                yield Row(int(idx[i]), *vals)
            else:
                yield Row(*vals)

    def equals(self, other: "DataFrame") -> bool:
        """Return True if two DataFrames have the same shape, columns, and values."""
        if self.shape != other.shape:
            return False
        if self.columns != other.columns:
            return False
        for col in self.columns:
            a = list(self[col])
            b = list(other[col])
            for va, vb in zip(a, b):
                if _is_nan(va) and _is_nan(vb):
                    continue
                if va != vb:
                    return False
        return True

    # ── groupby ───────────────────────────────────────────────────────────────

    def groupby(self, by: str) -> _GroupBy:
        """Group by *by* column."""
        return _GroupBy(self, by)

    def agg(self, func) -> "DataFrame":
        """
        Aggregate all numeric columns with *func*.
        func can be a string ("mean", "sum", …) or a callable.
        Returns a one-row DataFrame with the result.
        """
        numeric_cols = [c for c in self.columns
                        if self._frame.col_type(c) in ("double", "int64")]
        if isinstance(func, str):
            # Use C++ reduce_all for supported funcs
            supported = {"mean", "sum", "min", "max", "std", "count", "median", "var"}
            if func in supported:
                return DataFrame._from_frame(self._frame.reduce_all(func))
            raise ValueError(f"Unknown aggregation: {func!r}")
        if isinstance(func, dict):
            results = {}
            for col, fn in func.items():
                raw = [float(v) for v in self[col]
                       if not (_is_nan(v) if isinstance(v, float) else False)]
                if callable(fn):
                    results[col] = [fn(raw)]
                elif fn == "mean":
                    results[col] = [sum(raw)/len(raw) if raw else float("nan")]
                elif fn == "sum":
                    results[col] = [sum(raw)]
                elif fn == "min":
                    results[col] = [min(raw) if raw else float("nan")]
                elif fn == "max":
                    results[col] = [max(raw) if raw else float("nan")]
                elif fn == "count":
                    results[col] = [len(raw)]
                elif fn == "std":
                    n = len(raw)
                    if n < 2: results[col] = [0.0]
                    else:
                        m = sum(raw)/n
                        results[col] = [_math.sqrt(sum((v-m)**2 for v in raw)/(n-1))]
                elif fn == "median":
                    sv = sorted(raw); n = len(sv)
                    results[col] = [(sv[n//2]+sv[(n-1)//2])/2.0 if n else float("nan")]
                else:
                    raise ValueError(f"Unknown aggregation: {fn!r}")
            return DataFrame(results)
        if callable(func):
            results = {}
            for col in numeric_cols:
                raw = [float(v) for v in self[col]
                       if not (_is_nan(v) if isinstance(v, float) else False)]
                results[col] = [func(raw)]
            return DataFrame(results)
        raise TypeError(f"agg func must be str, dict, or callable")

    def aggregate(self, func) -> "DataFrame":
        """Alias for agg()."""
        return self.agg(func)

    def apply(self, func: Callable, axis: int = 0) -> "DataFrame":
        """
        Apply *func* along an axis.
        axis=0: apply to each column (func receives a list of values).
        axis=1: apply to each row (func receives a dict of {col: value}).
        """
        if axis == 0:
            result = {}
            for col in self.columns:
                result[col] = func(list(self[col]))
            first = next(iter(result.values()))
            if isinstance(first, list):
                return DataFrame(result)
            return DataFrame({k: [v] for k, v in result.items()})
        else:
            cols = self.columns
            col_data = {col: _get_col(self._frame, col) for col in cols}
            rows = []
            for i in range(len(self)):
                row = {col: col_data[col][i] for col in cols}
                rows.append(func(row))
            if rows and isinstance(rows[0], dict):
                out: dict = {}
                for row in rows:
                    for k, v in row.items():
                        out.setdefault(k, []).append(v)
                return DataFrame(out)
            return DataFrame({"result": rows})

    def transform(self, func: Callable, col: Optional[str] = None) -> "DataFrame":
        """Apply *func* element-wise; returns a DataFrame of the same shape."""
        result = self._copy()
        target_cols = [col] if col else self.columns
        for c in target_cols:
            result[c] = [func(v) for v in self[c]]
        return result

    # ── join / concat / merge ─────────────────────────────────────────────────

    def join(self, other: "DataFrame", how: str = "inner") -> "DataFrame":
        """Join two DataFrames on their shared index. how: inner/left/right/outer."""
        return DataFrame._from_frame(self._frame.join_by_index(other._frame, how))

    def concat(self, other: "DataFrame") -> "DataFrame":
        """Vertically concatenate two DataFrames (stack rows). Index resets to 0..N-1."""
        return DataFrame._from_frame(self._frame.concat_frame(other._frame))

    def merge(
        self,
        other: "DataFrame",
        on=None,
        left_on=None,
        right_on=None,
        how: str = "inner",
        suffixes: tuple = ("_x", "_y"),
    ) -> "DataFrame":
        """
        Merge two DataFrames on a key column (Python-level implementation).
        how: "inner" | "left" | "right" | "outer"
        """
        left_key = on or left_on
        right_key = on or right_on
        if left_key is None:
            raise ValueError("merge requires 'on', 'left_on', or 'right_on'")
        lk = [left_key] if isinstance(left_key, str) else left_key
        rk = [right_key] if isinstance(right_key, str) else right_key

        right_rows: dict = {}
        rk_data = {c: _get_col(other._frame, c) for c in rk}
        for i in range(len(other)):
            key = tuple(rk_data[c][i] for c in rk)
            right_rows.setdefault(key, []).append(i)

        lcols = self.columns
        rcols = [c for c in other.columns if c not in rk]

        out: dict = {c: [] for c in lcols}
        for c in rcols:
            out_col = c if c not in out else c + suffixes[1]
            out[out_col] = []

        lk_data = {c: _get_col(self._frame, c) for c in lk}
        lcols_data = {c: _get_col(self._frame, c) for c in lcols}
        rcols_data = {c: _get_col(other._frame, c) for c in rcols}
        matched_right: set = set()
        for i in range(len(self)):
            key = tuple(lk_data[c][i] for c in lk)
            matches = right_rows.get(key, [])
            if matches:
                for j in matches:
                    for c in lcols:
                        out[c].append(lcols_data[c][i])
                    for c in rcols:
                        out_col = c if c not in {*lcols} else c + suffixes[1]
                        out[out_col].append(rcols_data[c][j])
                    matched_right.add((key, j))
            elif how in ("left", "outer"):
                for c in lcols:
                    out[c].append(lcols_data[c][i])
                for c in rcols:
                    out_col = c if c not in {*lcols} else c + suffixes[1]
                    out[out_col].append(float("nan"))

        if how in ("right", "outer"):
            for i in range(len(other)):
                key = tuple(rk_data[c][i] for c in rk)
                if (key, i) not in matched_right:
                    for c in lcols:
                        out[c].append(float("nan") if c not in lk else rk_data[rk[lk.index(c)]][i])
                    for c in rcols:
                        out_col = c if c not in {*lcols} else c + suffixes[1]
                        out[out_col].append(rcols_data[c][i])

        return DataFrame(out)

    # ── window functions ──────────────────────────────────────────────────────

    def rolling_mean(self, col: str, window: int):
        return self._frame.rolling(col, window, "mean")

    def rolling_sum(self, col: str, window: int):
        return self._frame.rolling(col, window, "sum")

    def rolling_std(self, col: str, window: int):
        return self._frame.rolling(col, window, "std")

    def rolling_min(self, col: str, window: int):
        return self._frame.rolling(col, window, "min")

    def rolling_max(self, col: str, window: int):
        return self._frame.rolling(col, window, "max")

    def rolling(self, col: str, window: int, func: str = "mean"):
        """Generic rolling window. func: "mean" | "sum" | "std" | "min" | "max"."""
        return self._frame.rolling(col, window, func)

    # ── cumulative functions ──────────────────────────────────────────────────

    def cumsum(self, col: str):
        return self._frame.cumulative(col, "sum")

    def cumprod(self, col: str):
        return self._frame.cumulative(col, "prod")

    def cummin(self, col: str):
        return self._frame.cumulative(col, "min")

    def cummax(self, col: str):
        return self._frame.cumulative(col, "max")

    # ── shift / pct_change ────────────────────────────────────────────────────

    def shift(self, col: str, n: int = 1):
        """Shift column values by *n* periods (NaN fill at boundary)."""
        return self._frame.shift_col(col, n)

    def pct_change(self, col: str):
        """Percent change between consecutive elements. First element is NaN."""
        return self._frame.pct_change(col)

    def diff(self, col: str, periods: int = 1) -> list:
        """First discrete difference of *col* values. First *periods* elements are NaN."""
        return self._frame.diff_col(col, periods)

    # ── data cleaning ─────────────────────────────────────────────────────────

    def drop_duplicates(
        self,
        subset=None,
        keep: str = "first",
    ) -> "DataFrame":
        """Return a new DataFrame with duplicate rows removed."""
        if subset is None or isinstance(subset, str):
            col = subset or (self.columns[0] if self.columns else None)
            if col:
                return DataFrame._from_frame(self._frame.drop_duplicates(col))
            return self._copy()
        mask = self._frame.duplicated_rows(
            [subset] if isinstance(subset, str) else list(subset), keep)
        not_mask = [not v for v in mask]
        return DataFrame._from_frame(self._frame.filter_by_mask_list(not_mask))

    def drop_na(self, col: str) -> "DataFrame":
        """Return a new DataFrame with NaN / empty-string rows in *col* removed."""
        return DataFrame._from_frame(self._frame.drop_na(col))

    def dropna(
        self,
        subset=None,
    ) -> "DataFrame":
        """
        Return a new DataFrame with rows containing NaN removed.
        subset: column name or list of column names to check (default: all columns).
        """
        cols = self.columns if subset is None else (
            [subset] if isinstance(subset, str) else list(subset))
        result = self
        for col in cols:
            result = DataFrame._from_frame(result._frame.drop_na(col))
        return result

    def fillna(self, col: str, value) -> "DataFrame":
        """Fill NaN / empty values in *col* with *value* (in-place; returns self)."""
        t = self._frame.col_type(col)
        if t == "string":
            self._frame.fillna_string(col, str(value))
        else:
            self._frame.fillna_double(col, float(value))
        return self

    def ffill(self, col: str) -> "DataFrame":
        """Forward-fill NaN values in *col* (in-place; returns self) — delegates to C++."""
        self._frame.ffill_col(col)
        return self

    def bfill(self, col: str) -> "DataFrame":
        """Backward-fill NaN values in *col* (in-place; returns self) — delegates to C++."""
        self._frame.bfill_col(col)
        return self

    def _apply_replace_col(self, col_name: str, mapping: dict) -> None:
        col_type = self._frame.col_type(col_name)
        if not mapping:
            return
        val_sample = next(iter(mapping.values()))
        # Cross-type: string column → numeric values; replace then astype
        if col_type == "string" and isinstance(val_sample, (int, float)):
            str_mapping = {str(k): str(v) for k, v in mapping.items()}
            self._frame.replace_col_string(col_name, list(str_mapping.keys()), list(str_mapping.values()))
            target_type = "int64" if all(isinstance(v, int) for v in mapping.values()) else "double"
            self._frame.astype_col(col_name, target_type)
        elif col_type == "string":
            self._frame.replace_col_string(col_name, [str(k) for k in mapping.keys()], [str(v) for v in mapping.values()])
        else:
            self._frame.replace_col_double(
                col_name,
                [float(k) for k in mapping.keys()],
                [float(v) for v in mapping.values()])

    def replace(
        self,
        to_replace,
        value=None,
        col: Optional[str] = None,
        inplace: bool = False,
    ) -> "DataFrame":
        """
        Replace values.

        replace(to_replace, value)              — replace scalar/list in all columns
        replace({old: new, …})                  — replace via dict in all columns
        replace({"ColName": {old: new, …}})     — pandas-style per-column nested dict
        replace(to_replace, value, col="col")   — restrict to one column
        inplace=True modifies in-place and returns None.
        """
        target = self if inplace else self._copy()

        # pandas nested-dict form: {"ColName": {old: new, ...}}
        if (
            isinstance(to_replace, dict)
            and to_replace
            and isinstance(next(iter(to_replace.values())), dict)
        ):
            for col_name, mapping in to_replace.items():
                target._apply_replace_col(col_name, mapping)
        else:
            if isinstance(to_replace, dict):
                mapping = to_replace
            else:
                vals = to_replace if isinstance(to_replace, list) else [to_replace]
                mapping = {v: value for v in vals}
            if col:
                target._apply_replace_col(col, mapping)
            else:
                for c in target.columns:
                    try:
                        target._apply_replace_col(c, mapping)
                    except Exception:
                        pass

        return None if inplace else target

    def isna(self) -> "DataFrame":
        """Return a boolean DataFrame: True where values are NaN / empty — delegates to C++."""
        return DataFrame._from_frame(self._frame.isna_frame())

    def isnull(self) -> "DataFrame":
        """Alias for isna()."""
        return self.isna()

    def notna(self) -> "DataFrame":
        """Return a boolean DataFrame: True where values are NOT NaN / empty — delegates to C++."""
        return DataFrame._from_frame(self._frame.notna_frame())

    def notnull(self) -> "DataFrame":
        """Alias for notna()."""
        return self.notna()

    def where(self, cond, other=float("nan")) -> "DataFrame":
        """
        Return a copy where values NOT satisfying *cond* are replaced by *other*.
        cond can be a boolean DataFrame or a boolean array/list.
        """
        if isinstance(cond, DataFrame) and isinstance(other, (int, float)):
            return DataFrame._from_frame(self._frame.where_frame(cond._frame, float(other)))
        # Fallback: Python-level
        result = self._copy()
        cols = self.columns
        for j, col in enumerate(cols):
            raw = list(self[col])
            if isinstance(cond, DataFrame):
                mask_col = list(cond[col])
                result[col] = [v if mask_col[i] else other for i, v in enumerate(raw)]
            elif isinstance(cond, list):
                result[col] = [v if cond[i] else other for i, v in enumerate(raw)]
            else:
                try:
                    import numpy as np
                    arr = np.asarray(cond)
                    if arr.ndim == 2:
                        result[col] = [v if arr[i, j] else other for i, v in enumerate(raw)]
                    else:
                        result[col] = [v if arr[i] else other for i, v in enumerate(raw)]
                except ImportError:
                    result[col] = [v if cond[i] else other for i, v in enumerate(raw)]
        return result

    def mask(self, cond, other=float("nan")) -> "DataFrame":
        """Inverse of where() — replace values WHERE *cond* is True."""
        result = self._copy()
        cols = self.columns
        for j, col in enumerate(cols):
            raw = list(self[col])
            if isinstance(cond, DataFrame):
                mask_col = list(cond[col])
                result[col] = [other if mask_col[i] else v for i, v in enumerate(raw)]
            elif isinstance(cond, list):
                result[col] = [other if cond[i] else v for i, v in enumerate(raw)]
            else:
                try:
                    import numpy as np
                    arr = np.asarray(cond)
                    if arr.ndim == 2:
                        result[col] = [other if arr[i, j] else v for i, v in enumerate(raw)]
                    else:
                        result[col] = [other if arr[i] else v for i, v in enumerate(raw)]
                except ImportError:
                    result[col] = [other if cond[i] else v for i, v in enumerate(raw)]
        return result

    def clip(
        self,
        lower=None,
        upper=None,
        col: Optional[str] = None,
    ) -> "DataFrame":
        """Clip numeric values to [lower, upper] bounds — delegates to C++."""
        result = self._copy()
        lo = float(lower) if lower is not None else float("-inf")
        hi = float(upper) if upper is not None else float("inf")
        target_cols = [col] if col else [
            c for c in self.columns
            if self._frame.col_type(c) in ("double", "int64")
        ]
        for c in target_cols:
            result._frame.clip_col(c, lo, hi)
        return result

    def round(self, decimals: int = 0) -> "DataFrame":
        """Round all numeric columns to *decimals* decimal places — delegates to C++."""
        result = self._copy()
        for col in self.columns:
            if self._frame.col_type(col) in ("double",):
                result._frame.round_col(col, decimals)
        return result

    def rename(self, mapping: dict) -> "DataFrame":
        """Rename columns in-place and return self."""
        for old, new in mapping.items():
            self._frame.rename_col(old, new)
        return self

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        inplace: bool = False,
        errors: str = "raise",
    ) -> Optional["DataFrame"]:
        """
        Drop rows or columns.
        labels: str or list.  axis: 0/'index' to drop rows, 1/'columns' to drop columns.
        """
        if columns is not None:
            labels = columns
            axis = 1
        elif index is not None:
            labels = index
            axis = 0

        if labels is None:
            raise ValueError("must specify labels, columns, or index")

        drop_list = [labels] if isinstance(labels, str) else list(labels)
        is_col_axis = axis in (1, "columns")

        if inplace:
            target = self
        else:
            target = self._copy()

        if is_col_axis:
            for c in drop_list:
                if c not in target:
                    if errors == "raise":
                        raise KeyError(c)
                    continue
                target._frame.drop_column(c)
        else:
            idx = list(target.index)
            drop_set = set(drop_list)
            mask = [int(v) not in drop_set for v in idx]
            new_frame = target._frame.filter_by_mask_list(mask)
            target._frame = new_frame

        if inplace:
            return None
        return target

    def astype(self, dtype) -> "DataFrame":
        """
        Cast columns to specified types — delegates to C++ astype_col.
        dtype can be a single type or a dict {col: type}.
        """
        result = self._copy()

        def _map_type(t):
            if t in (float, "float", "float64"): return "double"
            if t in (int, "int", "int64"): return "int64"
            if t in (str, "str", "string", "object"): return "string"
            if t in (bool, "bool"): return "bool"
            return None

        if isinstance(dtype, dict):
            for col, t in dtype.items():
                mapped = _map_type(t)
                if mapped:
                    result._frame.astype_col(col, mapped)
        else:
            mapped = _map_type(dtype)
            if mapped:
                for col in self.columns:
                    try:
                        result._frame.astype_col(col, mapped)
                    except Exception:
                        pass
        return result

    # ── reshaping ─────────────────────────────────────────────────────────────

    def transpose(self) -> "DataFrame":
        """Return the transpose — delegates to C++."""
        return DataFrame._from_frame(self._frame.transpose_frame())

    def set_index(self, col: str, drop: bool = True) -> "DataFrame":
        """Set *col* as the index — delegates to C++."""
        return DataFrame._from_frame(self._frame.set_index_col(col, drop))

    def reset_index(self, drop: bool = False) -> "DataFrame":
        """Reset the index to 0..N-1 — delegates to C++."""
        return DataFrame._from_frame(self._frame.reset_index_frame(drop))

    def melt(
        self,
        id_vars=None,
        value_vars=None,
        var_name: str = "variable",
        value_name: str = "value",
    ) -> "DataFrame":
        """Unpivot from wide to long format — delegates to C++."""
        id_cols = ([id_vars] if isinstance(id_vars, str) else list(id_vars or []))
        val_cols = ([value_vars] if isinstance(value_vars, str)
                    else list(value_vars or [c for c in self.columns if c not in id_cols]))
        return DataFrame._from_frame(
            self._frame.melt_frame(id_cols, val_cols, var_name, value_name))

    def pivot_table(
        self,
        values: str,
        index: str,
        columns: Optional[str] = None,
        aggfunc=None,
    ) -> "DataFrame":
        """Create a pivot table."""
        if aggfunc is None:
            aggfunc = "mean"
        if columns is None:
            return self.groupby(index).agg({values: aggfunc})
        row_keys = sorted(set(self[index]))
        col_keys = sorted(set(self[columns]))
        from collections import defaultdict
        groups: dict = defaultdict(list)
        idx_col = _get_col(self._frame, index)
        col_col = _get_col(self._frame, columns)
        val_col = _get_col(self._frame, values)
        for i in range(len(self)):
            rk = idx_col[i]
            ck = col_col[i]
            groups[(rk, ck)].append(val_col[i])

        def _apply_agg(vals):
            if callable(aggfunc):
                return aggfunc(vals)
            if aggfunc == "mean":
                return sum(vals) / len(vals) if vals else float("nan")
            if aggfunc == "sum":
                return sum(vals)
            if aggfunc == "min":
                return min(vals) if vals else float("nan")
            if aggfunc == "max":
                return max(vals) if vals else float("nan")
            if aggfunc == "count":
                return len(vals)
            sv = sorted(vals); n = len(sv)
            return (sv[n//2] + sv[(n-1)//2]) / 2.0 if n else float("nan")

        data: dict = {index: row_keys}
        for ck in col_keys:
            data[str(ck)] = [_apply_agg(groups.get((rk, ck), [float("nan")]))
                             for rk in row_keys]
        return DataFrame(data)

    # ── arithmetic operators ──────────────────────────────────────────────────

    def _arith(self, other, op_str, op_fn) -> "DataFrame":
        if isinstance(other, DataFrame):
            return DataFrame._from_frame(self._frame.arith_frame_op(op_str, other._frame))
        try:
            scalar = float(other)
            return DataFrame._from_frame(self._frame.arith_scalar(op_str, scalar))
        except (TypeError, ValueError):
            pass
        # Fallback for non-numeric scalars
        result = self._copy()
        for col in self.columns:
            if self._frame.col_type(col) not in ("double", "int64"):
                continue
            raw = list(self[col])
            result[col] = [op_fn(v, other) for v in raw]
        return result

    def __add__(self, other):  return self._arith(other, "+", _op.add)
    def __radd__(self, other): return self._arith(other, "+", lambda a, b: _op.add(b, a))
    def __sub__(self, other):  return self._arith(other, "-", _op.sub)
    def __rsub__(self, other): return self._arith(other, "-", lambda a, b: _op.sub(b, a))
    def __mul__(self, other):  return self._arith(other, "*", _op.mul)
    def __rmul__(self, other): return self._arith(other, "*", lambda a, b: _op.mul(b, a))
    def __truediv__(self, other):   return self._arith(other, "/", _op.truediv)
    def __rtruediv__(self, other):  return self._arith(other, "/", lambda a, b: _op.truediv(b, a))
    def __floordiv__(self, other):  return self._arith(other, "//", _op.floordiv)
    def __rfloordiv__(self, other): return self._arith(other, "//", lambda a, b: _op.floordiv(b, a))
    def __mod__(self, other):  return self._arith(other, "%", _op.mod)
    def __rmod__(self, other): return self._arith(other, "%", lambda a, b: _op.mod(b, a))
    def __pow__(self, other):  return self._arith(other, "**", _op.pow)
    def __rpow__(self, other): return self._arith(other, "**", lambda a, b: _op.pow(b, a))
    def __neg__(self):         return DataFrame._from_frame(self._frame.arith_scalar("*", -1.0))
    def __abs__(self):         return self.abs()

    # ── comparison operators (return bool DataFrame) ──────────────────────────

    def _compare(self, other, op_str, op_fn) -> "DataFrame":
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return DataFrame._from_frame(self._frame.compare_scalar(op_str, float(other)))
        # Fallback
        result_data = {}
        for col in self.columns:
            raw = list(self[col])
            if isinstance(other, DataFrame):
                oth = list(other[col]) if col in other else [False] * len(raw)
                try:
                    result_data[col] = [op_fn(a, b) for a, b in zip(raw, oth)]
                except TypeError:
                    result_data[col] = [False] * len(raw)
            else:
                try:
                    result_data[col] = [op_fn(v, other) for v in raw]
                except TypeError:
                    result_data[col] = [False] * len(raw)
        return DataFrame(result_data, index=list(self.index))

    def __eq__(self, other): return self._compare(other, "==", _op.eq)
    def __ne__(self, other): return self._compare(other, "!=", _op.ne)
    def __lt__(self, other): return self._compare(other, "<",  _op.lt)
    def __le__(self, other): return self._compare(other, "<=", _op.le)
    def __gt__(self, other): return self._compare(other, ">",  _op.gt)
    def __ge__(self, other): return self._compare(other, ">=", _op.ge)

    # ── I/O ───────────────────────────────────────────────────────────────────

    def to_csv(self, path: str, index: bool = True) -> None:
        self._frame.to_csv(path, index)

    def to_dict(self, orient: str = "dict") -> dict:
        """Convert DataFrame to a dictionary."""
        cols = self.columns
        idx = list(self.index)
        if orient == "list":
            return {col: list(self[col]) for col in cols}
        if orient == "records":
            return [{col: self[col][i] for col in cols} for i in range(len(self))]
        if orient == "index":
            return {int(idx[i]): {col: self[col][i] for col in cols}
                    for i in range(len(self))}
        if orient == "series":
            return {col: self[col] for col in cols}
        return {col: {int(idx[i]): self[col][i] for i in range(len(self))}
                for col in cols}

    def to_numpy(self):
        """Return the DataFrame as a 2-D numpy array."""
        return self.values

    def to_pandas(self):
        """Convert to a pandas DataFrame (enables plotly, seaborn, sklearn, etc.)."""
        import pandas as pd
        data = {col: list(self[col]) for col in self.columns}
        return pd.DataFrame(data, index=list(self.index))

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):
        """Implement the Python DataFrame Interchange Protocol."""
        return self.to_pandas().__dataframe__(
            nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    def to_json(
        self,
        path_or_buf=None,
        orient: str = "records",
        indent: Optional[int] = None,
    ) -> Optional[str]:
        """Serialize to JSON string or file."""
        data = self.to_dict(orient=orient)
        text = _json.dumps(data, indent=indent, default=str)
        if path_or_buf is None:
            return text
        with open(path_or_buf, "w") as f:
            f.write(text)
        return None

    def to_string(
        self,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
    ) -> str:
        """Return string representation."""
        return self.__repr__()

    def to_html(self, index: bool = True) -> str:
        """Return HTML table representation."""
        return self._repr_html_()

    # ── plotting ──────────────────────────────────────────────────────────────

    def hist(self, column=None, by=None, grid=True, xlabelsize=None, xrot=None,
             ylabelsize=None, yrot=None, ax=None, sharex=False, sharey=False,
             figsize=None, layout=None, bins=10, legend=False, **kwargs):
        """Plot histograms of numeric columns (delegates to pandas)."""
        return self.to_pandas().hist(
            column=column, by=by, grid=grid, xlabelsize=xlabelsize, xrot=xrot,
            ylabelsize=ylabelsize, yrot=yrot, ax=ax, sharex=sharex, sharey=sharey,
            figsize=figsize, layout=layout, bins=bins, legend=legend, **kwargs,
        )

    def plot(self, *args, **kwargs):
        """Access matplotlib plotting (delegates to pandas)."""
        return self.to_pandas().plot(*args, **kwargs)

    def boxplot(self, column=None, by=None, ax=None, fontsize=None, rot=0,
                grid=True, figsize=None, layout=None, **kwargs):
        """Draw a boxplot (delegates to pandas)."""
        return self.to_pandas().boxplot(
            column=column, by=by, ax=ax, fontsize=fontsize, rot=rot,
            grid=grid, figsize=figsize, layout=layout, **kwargs,
        )

    # ── repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        rows, ncols = self.shape
        cols = self.columns
        col_w = 13
        idx_w = 8

        lines: list = []
        header = f"{'':>{idx_w}}" + "".join(f"  {c:>{col_w}}" for c in cols)
        sep = "─" * len(header)
        lines += [header, sep]

        show = min(rows, 10)
        idx = list(self.index)
        # Pre-load all column data once (avoid O(show×ncols) lazy loads)
        col_data = {col: _get_col(self._frame, col) for col in cols}
        for i in range(show):
            row = f"{idx[i]:>{idx_w}}"
            for col in cols:
                v = col_data[col][i]
                val = "NaN" if (isinstance(v, float) and v != v) or v == "" else str(v)
                row += f"  {val:>{col_w}}"
            lines.append(row)

        if rows > 10:
            lines.append(f"  ... ({rows - 10} more rows)")

        lines.append(f"\n[{rows} rows × {ncols} columns]")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows, ncols = self.shape
        cols = self.columns
        idx = list(self.index)

        max_rows = 10
        show = min(rows, max_rows)
        truncated = rows > max_rows

        th_style = "padding:4px 10px;border:1px solid #ddd;background:#c3c3c3;text-align:right;white-space:nowrap;color:black;"
        td_style = "padding:4px 10px;border:1px solid #ddd;text-align:right;white-space:nowrap;color:black;"

        html = [
            '<div style="max-width:100%;overflow-x:auto;font-family:monospace;font-size:13px;">',
            '<table style="border-collapse:collapse;border:1px solid #ddd;">',
            "<thead>",
            '<tr style="background:#dcdcdc;">',
            f'<th style="{th_style}"></th>',
        ]
        for col in cols:
            html.append(f'<th style="{th_style}">{col}</th>')
        html += ["</tr>", "</thead>", "<tbody>"]

        # Pre-load all column data once
        col_data = {col: _get_col(self._frame, col) for col in cols}
        for i in range(show):
            row_bg = "#fff" if i % 2 == 0 else "#fafafa"
            html.append(f'<tr style="background:{row_bg};">')
            html.append(f'<th style="{th_style}">{idx[i]}</th>')
            for col in cols:
                v = col_data[col][i]
                val = "NaN" if (isinstance(v, float) and v != v) or v == "" else v
                html.append(f'<td style="{td_style}">{val}</td>')
            html.append("</tr>")

        if truncated:
            html.append(
                f'<tr><td colspan="{ncols + 1}" style="text-align:center;padding:4px 10px;'
                f'color:#999;border:1px solid #ddd;">... {rows - max_rows} more rows</td></tr>'
            )

        html += [
            "</tbody>",
            "</table>",
            f'<p style="font-size:11px;color:#888;margin:4px 0 0;">{rows} rows × {ncols} columns</p>',
            "</div>",
        ]
        return "".join(html)

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
    dtype : dict, optional
        Mapping of clean column name -> callable for explicit type conversion.
        When set, forces the slower Python fallback path.
    """
    # ── fast path: native C++ reader ──────────────────────────────────────────
    if dtype is None:
        frame = _GrizzlarFrame.read_csv_native(str(path), index_col or "")
        return DataFrame._from_frame(frame)

    # ── fallback: Python reader (supports custom dtype converters) ────────────
    raw: dict = {}
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


def get_dummies(
    data: "DataFrame",
    columns=None,
    prefix=None,
    prefix_sep: str = "_",
    dummy_na: bool = False,
    drop_first: bool = False,
    dtype=None,
) -> "DataFrame":
    """
    One-hot encode categorical columns (equivalent to pandas.get_dummies).

    Parameters
    ----------
    data : DataFrame
    columns : list of str, optional
        Columns to encode. Defaults to all string columns.
    prefix : str or list, optional
        Prefix for new column names. Defaults to the original column name.
    prefix_sep : str
        Separator between prefix and category value. Default "_".
    dummy_na : bool
        If True, add a column for NaN/empty values. Default False.
    drop_first : bool
        Drop the first category level to avoid multicollinearity. Default False.
    dtype : ignored
        Accepted for API compatibility; new columns are always int64 (0/1).
    """
    _ = dtype  # accepted for API compatibility only
    result = data._copy()

    if columns is None:
        columns = [c for c in data.columns if data._frame.col_type(c) == "string"]

    prefixes = {}
    if prefix is None:
        prefixes = {c: c for c in columns}
    elif isinstance(prefix, str):
        prefixes = {c: prefix for c in columns}
    else:
        prefixes = {c: p for c, p in zip(columns, prefix)}

    for col in columns:
        col_type = data._frame.col_type(col)
        if col_type != "string":
            raise TypeError(
                f"get_dummies: column '{col}' has type '{col_type}', expected 'string'. "
                f"Only string/categorical columns can be one-hot encoded."
            )
        col_vals = list(data[col])  # list of raw str values
        unique_vals = []
        seen = set()
        for v in col_vals:
            is_empty = (v == "" or v is None)
            if is_empty:
                if dummy_na and "<NA>" not in seen:
                    seen.add("<NA>")
                    unique_vals.append("<NA>")
            else:
                if v not in seen:
                    seen.add(v)
                    unique_vals.append(v)

        if drop_first and unique_vals:
            unique_vals = unique_vals[1:]

        pfx = prefixes.get(col, col)
        for uv in unique_vals:
            new_col = f"{pfx}{prefix_sep}{uv}"
            if uv == "<NA>":
                _load_col(result._frame, new_col, [1 if (v == "" or v is None) else 0 for v in col_vals])
            else:
                _load_col(result._frame, new_col, [1 if v == uv else 0 for v in col_vals])

        result._frame.drop_column(col)

    return result
