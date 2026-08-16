"""Introspection: keys/info/memory_usage, boolean reductions (any/all), and
row iteration (items/iterrows/itertuples/equals).
"""

from __future__ import annotations

import re

from .._helpers import _get_col, _is_nan


class _MetadataMixin:

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

    # ── boolean reductions ────────────────────────────────────────────────────

    def any(self, col: str) -> bool:
        """Return True if any value in *col* is truthy."""
        raw = self[col]
        return any(bool(v) and not _is_nan(v) for v in raw)

    def all(self, col: str) -> bool:
        """Return True if all values in *col* are truthy."""
        raw = self[col]
        return all(bool(v) and not _is_nan(v) for v in raw)

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
