"""groupby/agg/aggregate/apply/transform and join/concat/merge."""

from __future__ import annotations

import math as _math
from typing import Callable, Optional

from .._groupby import _GroupBy
from .._helpers import _get_col, _is_nan


class _GroupByJoinMixin:

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
                return type(self)._from_frame(self._frame.reduce_all(func))
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
            return type(self)(results)
        if callable(func):
            results = {}
            for col in numeric_cols:
                raw = [float(v) for v in self[col]
                       if not (_is_nan(v) if isinstance(v, float) else False)]
                results[col] = [func(raw)]
            return type(self)(results)
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
                return type(self)(result)
            return type(self)({k: [v] for k, v in result.items()})
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
                return type(self)(out)
            return type(self)({"result": rows})

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
        return type(self)._from_frame(self._frame.join_by_index(other._frame, how))

    def concat(self, other: "DataFrame") -> "DataFrame":
        """Vertically concatenate two DataFrames (stack rows). Index resets to 0..N-1."""
        return type(self)._from_frame(self._frame.concat_frame(other._frame))

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

        return type(self)(out)
