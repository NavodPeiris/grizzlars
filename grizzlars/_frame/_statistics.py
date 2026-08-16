"""Scalar and whole-frame statistics: sum/mean/std/min/max/count/median/var/
sem/skew/kurt/prod/abs, idxmax/idxmin/rank/mode/describe/quantile/corr/cov,
nunique/unique/n_missing/value_counts.
"""

from __future__ import annotations

import math as _math
from typing import Optional

from .._helpers import _is_nan


class _StatisticsMixin:

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
        return type(self)._from_frame(self._frame.reduce_all("sum"))

    def mean(self, col: Optional[str] = None):
        """Mean of column values. No col → one-row DataFrame of all numeric columns."""
        if col is not None:
            if self._frame.col_type(col) == "bool":
                raw = list(self[col])
                n = len(raw)
                return sum(1 for v in raw if v) / n if n else float("nan")
            return self._frame.mean(col)
        return type(self)._from_frame(self._frame.reduce_all("mean"))

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
        return type(self)._from_frame(self._frame.reduce_all("std"))

    def min(self, col: Optional[str] = None):
        """Minimum value. No col → one-row DataFrame."""
        if col is not None:
            if self._frame.col_type(col) == "bool":
                raw = list(self[col])
                return 0.0 if any(not v for v in raw) else 1.0
            return self._frame.col_min(col)
        return type(self)._from_frame(self._frame.reduce_all("min"))

    def max(self, col: Optional[str] = None):
        """Maximum value. No col → one-row DataFrame."""
        if col is not None:
            if self._frame.col_type(col) == "bool":
                raw = list(self[col])
                return 1.0 if any(v for v in raw) else 0.0
            return self._frame.col_max(col)
        return type(self)._from_frame(self._frame.reduce_all("max"))

    def count(self, col: Optional[str] = None):
        """Count non-null values. No col → one-row DataFrame."""
        if col is not None:
            typ = self._frame.col_type(col)
            if typ == "double":
                return sum(1 for v in self[col] if not _is_nan(v))
            elif typ == "string":
                return sum(1 for v in self[col] if v != "")
            return int(self._frame.count(col))
        return type(self)._from_frame(self._frame.reduce_all("count"))

    def median(self, col: Optional[str] = None):
        """Median value. No col → one-row DataFrame."""
        if col is not None:
            raw = [v for v in self[col] if not (_is_nan(v) if isinstance(v, float) else False)]
            if not raw: return float("nan")
            sv = sorted(raw)
            n = len(sv)
            return (sv[n//2] + sv[(n-1)//2]) / 2.0
        return type(self)._from_frame(self._frame.reduce_all("median"))

    def var(self, col: Optional[str] = None, ddof: int = 1):
        """Variance. No col → one-row DataFrame."""
        if col is not None:
            raw = [float(v) for v in self[col] if not (_is_nan(v) if isinstance(v, float) else False)]
            n = len(raw)
            if n <= ddof: return float("nan")
            m = sum(raw) / n
            return sum((v - m)**2 for v in raw) / (n - ddof)
        return type(self)._from_frame(self._frame.reduce_all("var"))

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
        return type(self)({c: [_sem(c)] for c in self._numeric_cols()})

    def skew(self, col: Optional[str] = None):
        """Sample skewness — delegates to C++."""
        if col is not None:
            return self._frame.skew_col(col)
        return type(self)({c: [self._frame.skew_col(c)] for c in self._numeric_cols()})

    def kurt(self, col: Optional[str] = None):
        """Excess kurtosis — delegates to C++."""
        if col is not None:
            return self._frame.kurt_col(col)
        return type(self)({c: [self._frame.kurt_col(c)] for c in self._numeric_cols()})

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
        return type(self)({c: [_prod(c)] for c in self._numeric_cols()})

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
            return type(self)()
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
        return type(self)(data, index=list(range(len(stat_names))))

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
            return type(self)(matrix)

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
            return type(self)(matrix)

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
            return type(self)._from_frame(self._frame.value_counts_double(col))
        if t == "int64":
            return type(self)._from_frame(self._frame.value_counts_int64(col))
        return type(self)._from_frame(self._frame.value_counts_string(col))
