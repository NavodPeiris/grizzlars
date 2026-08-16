"""Reshaping: transpose, set_index/reset_index, melt, pivot_table."""

from __future__ import annotations

from typing import Optional

from .._helpers import _get_col


class _ReshapingMixin:

    def transpose(self) -> "DataFrame":
        """Return the transpose — delegates to C++."""
        return type(self)._from_frame(self._frame.transpose_frame())

    def set_index(self, col: str, drop: bool = True) -> "DataFrame":
        """Set *col* as the index — delegates to C++."""
        return type(self)._from_frame(self._frame.set_index_col(col, drop))

    def reset_index(self, drop: bool = False) -> "DataFrame":
        """Reset the index to 0..N-1 — delegates to C++."""
        return type(self)._from_frame(self._frame.reset_index_frame(drop))

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
        return type(self)._from_frame(
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
        return type(self)(data)
