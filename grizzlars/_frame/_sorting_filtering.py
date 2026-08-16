"""Sorting (sort/sort_values/sort_index) and row selection/filtering
(head/tail/take/filter/query/select/nlargest/nsmallest/sample/isin/
duplicated/with_column/assign/insert/pop).
"""

from __future__ import annotations

import random as _random
import re
from typing import Any, Optional

from .._comparison import _ColComparison, _CombinedComparison
from .._helpers import _OPS, _load_col
from .._lazy_filter import _LazyFilterFrame
from .._series import Series


class _SortingFilteringMixin:

    # ── sorting ───────────────────────────────────────────────────────────────

    def sort(self, by: str, ascending: bool = True) -> "DataFrame":
        """Return a new DataFrame sorted by *by* column (non-mutating)."""
        return type(self)._from_frame(self._frame.sort_by(by, ascending))

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
                result = type(self)._from_frame(result._frame.sort_by(col, asc))
            return result
        return self.sort(by, bool(ascending))

    def sort_index(self, ascending: bool = True) -> "DataFrame":
        """Return a new DataFrame sorted by index (non-mutating)."""
        return type(self)._from_frame(self._frame.sort_index(ascending))

    # ── slicing / filtering ───────────────────────────────────────────────────

    def head(self, n: int = 5) -> "DataFrame":
        return type(self)._from_frame(self._frame.iloc(0, min(n, len(self))))

    def tail(self, n: int = 5) -> "DataFrame":
        total = len(self)
        return type(self)._from_frame(self._frame.iloc(max(0, total - n), total))

    def take(self, indices, axis=0) -> "DataFrame":
        """Select rows by integer position (used by sklearn, numpy, etc.)."""
        _ = axis  # only row-axis supported
        idx_list = [int(i) for i in indices]
        # Fast C++ path (available after rebuild)
        try:
            return type(self)._from_frame(self._frame.take_rows(idx_list))
        except AttributeError:
            pass
        # Pre-rebuild fallback: if indices are sorted use C++ mask filter (fast)
        sz = len(self)
        norm = [i if i >= 0 else sz + i for i in idx_list]
        if norm == sorted(norm):
            mask = [False] * sz
            for i in norm:
                mask[i] = True
            return type(self)._from_frame(self._frame.filter_by_mask_list(mask))
        # Unsorted fallback: sort → mask → reorder (still cheaper than per-element Python)
        order = sorted(range(len(norm)), key=lambda k: norm[k])
        sorted_norm = [norm[k] for k in order]
        mask = [False] * sz
        for i in sorted_norm:
            mask[i] = True
        filtered = type(self)._from_frame(self._frame.filter_by_mask_list(mask))
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
        return type(self)(data)

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
                    return type(self)._from_frame(
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
                return type(self)._from_frame(
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
        return type(self)._from_frame(self._frame.select_columns(columns))

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
        return type(self)._from_frame(self._frame.filter_by_mask_list(mask))

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
