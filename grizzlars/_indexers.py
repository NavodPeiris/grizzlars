"""df.iloc / df.loc / df.at / df.iat accessors."""

from __future__ import annotations

import sys as _sys

from ._helpers import _mask_to_list
from ._series import Series


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
