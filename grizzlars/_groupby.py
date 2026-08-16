"""Returned by DataFrame.groupby(col)."""

from __future__ import annotations

import math as _math


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
                # DataFrame lives in ._frame, which never needs this module —
                # type(self._df) avoids importing it and matches self._df's
                # concrete (always DataFrame) type exactly.
                return type(self._df)._from_frame(
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
        return type(self._df)(result)

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
