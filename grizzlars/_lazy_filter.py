"""Lazy filter result — stores the boolean mask without materialising the data."""

from __future__ import annotations

from ._comparison import _ColComparison, _CombinedComparison
from ._helpers import _mask_to_list
from ._series import Series


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
            # Local import: DataFrame lives in ._frame, which itself never
            # needs to import this module, but importing it at module level
            # here would still make `import grizzlars` order-dependent.
            from ._frame import DataFrame
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
