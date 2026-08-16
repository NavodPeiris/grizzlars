"""Lazy column comparisons returned by Series.__gt__ / __lt__ etc."""

from __future__ import annotations

from ._helpers import _OPS, _get_col, _is_nan


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
