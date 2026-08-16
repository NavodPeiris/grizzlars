"""Rolling window, cumulative, and shift/pct_change/diff functions."""

from __future__ import annotations


class _WindowMixin:

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
