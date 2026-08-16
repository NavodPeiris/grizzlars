"""Constructor, deep_copy, and column access (__getitem__/__setitem__/dunders
and shape/index/dtypes-style properties) — the mixin every other DataFrame
mixin builds on.
"""

from __future__ import annotations

import sys as _sys
from typing import Optional

from .._comparison import _ColComparison, _CombinedComparison
from .._grizzlars import GrizzlarFrame as _GrizzlarFrame
from .._helpers import _get_col, _load_col, _mask_to_list
from .._indexers import _AtIndexer, _IAtIndexer, _ILocIndexer, _LocIndexer
from .._series import Series


class _CoreMixin:

    def __init__(
        self,
        data: Optional[dict] = None,
        index=None,
    ) -> None:
        self._frame = _GrizzlarFrame()
        if data:
            n = max(len(v) for v in data.values())
            idx = list(range(n)) if index is None else list(index)
            self._frame.load_index(idx)
            for name, values in data.items():
                _np = _sys.modules.get("numpy")
                if _np is not None and isinstance(values, _np.ndarray):
                    _load_col(self._frame, name, values)
                elif isinstance(values, list):
                    _load_col(self._frame, name, values)
                else:
                    try:
                        v = list(values)
                    except TypeError:
                        v = [values]
                    _load_col(self._frame, name, v)

    @classmethod
    def _from_frame(cls, frame: _GrizzlarFrame) -> "DataFrame":
        obj = cls.__new__(cls)
        obj._frame = frame
        return obj

    def _copy(self) -> "DataFrame":
        return type(self)._from_frame(self._frame.deep_copy())

    def copy(self) -> "DataFrame":
        return type(self)._from_frame(self._frame.deep_copy())

    # ── column access ─────────────────────────────────────────────────────────

    def __getitem__(self, key):
        # _ColComparison / _CombinedComparison → fast C++ filter
        if isinstance(key, _ColComparison):
            try:
                return type(self)._from_frame(
                    self._frame.filter_col_scalar_double(key._col, key._op, float(key._scalar))
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass
            return type(self)._from_frame(self._frame.filter_by_mask_list(key.to_mask()))
        if isinstance(key, _CombinedComparison):
            return type(self)._from_frame(self._frame.filter_by_mask_list(key.to_mask()))
        # Boolean Series → row filter (no numpy needed)
        if isinstance(key, Series) and key.dtype == "bool":
            return type(self)._from_frame(self._frame.filter_by_mask_list(key.to_list()))
        # Plain bool list → row filter
        if isinstance(key, list) and key and isinstance(key[0], bool):
            return type(self)._from_frame(self._frame.filter_by_mask_list(key))
        # numpy array support (optional) — use sys.modules to avoid slow cold import
        _np = _sys.modules.get("numpy")
        if _np is not None and isinstance(key, _np.ndarray) and (key.dtype == bool or key.dtype == _np.bool_):
            return type(self)._from_frame(self._frame.filter_by_mask_list(_mask_to_list(key)))
        # pandas Series support (optional) — use sys.modules to avoid slow cold import
        _pd = _sys.modules.get("pandas")
        if _pd is not None and isinstance(key, _pd.Series) and (str(key.dtype) in ("bool", "boolean")):
            return type(self)._from_frame(self._frame.filter_by_mask_list(_mask_to_list(key.to_numpy())))
        # Slice → row range
        if isinstance(key, slice):
            total = len(self)
            start, stop, step = key.indices(total)
            if step == 1:
                return type(self)._from_frame(self._frame.iloc(start, stop))
            indices = range(start, stop, step)
            return self.take(indices)
        # List of column names → sub-DataFrame
        if isinstance(key, list) and key and isinstance(key[0], str):
            return type(self)._from_frame(self._frame.select_columns(key))
        # String column name → lazy Series (data loaded on first access)
        if isinstance(key, str):
            dtype = self._frame.col_type(key)
            return Series(None, name=key, dtype=dtype,
                          _src_frame=self, _src_col=key)
        raise TypeError(
            f"DataFrame key must be a column name (str) or boolean array/Series, "
            f"got {type(key).__name__}"
        )

    def __setitem__(self, col, values) -> None:
        # List of column names + 2D array (e.g. from sklearn transform)
        if isinstance(col, list):
            try:
                import numpy as _np
                arr = _np.asarray(values)
                for i, c in enumerate(col):
                    _load_col(self._frame, c, arr[:, i].tolist())
                return
            except ImportError:
                pass
            # fallback: values is a list-of-lists / iterable of rows
            rows = list(values)
            for i, c in enumerate(col):
                _load_col(self._frame, c, [row[i] for row in rows])
            return
        if isinstance(values, list):
            v = values
        else:
            try:
                v = list(values)
            except TypeError:
                v = [values]
        _load_col(self._frame, col, v)

    def __contains__(self, col: str) -> bool:
        return self._frame.has_column(col)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        try:
            frame = object.__getattribute__(self, "_frame")
            if frame.has_column(name):
                self[name] = value
                return
        except AttributeError:
            pass
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        # Dot-notation column access: df.Age, df.Gender, …
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            cols = self.columns
        except AttributeError:
            raise AttributeError(name)
        if name in cols:
            return self[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    def __iter__(self):
        """Iterate over column names (pandas-compatible)."""
        return iter(self.columns)

    def __array__(self, dtype=None):
        """Allow numpy to convert this DataFrame to a 2-D array (used by sklearn etc.)."""
        import numpy as _np
        arr = self.values
        if dtype is not None:
            # Give a useful error listing which columns still contain strings
            string_cols = [c for c in self.columns if self._frame.col_type(c) == "string"]
            if string_cols:
                raise ValueError(
                    f"Cannot convert DataFrame to {dtype}: the following columns are still "
                    f"string type and must be encoded first: {string_cols}"
                )
            return arr.astype(dtype)
        return arr

    def __bool__(self):
        raise ValueError(
            "The truth value of a DataFrame is ambiguous. "
            "Use df.empty, df.any() or df.all()."
        )

    @property
    def index(self):
        import numpy as _np
        return _np.asarray(self._frame.get_index(), dtype=_np.uint64)

    @property
    def columns(self) -> list:
        return self._frame.columns()

    @property
    def shape(self) -> tuple:
        return self._frame.shape()

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def dtypes(self) -> dict:
        """Return a mapping of column name -> type string."""
        return {c: self._frame.col_type(c) for c in self.columns}

    @property
    def iloc(self) -> _ILocIndexer:
        """Integer-location based indexing: df.iloc[0:100], df.iloc[-1]."""
        return _ILocIndexer(self)

    @property
    def loc(self) -> _LocIndexer:
        """Label-based indexing: df.loc[label], df.loc[start:stop], df.loc[mask]."""
        return _LocIndexer(self)

    @property
    def at(self) -> _AtIndexer:
        """Fast scalar label-based access: df.at[row_label, col_name]."""
        return _AtIndexer(self)

    @property
    def iat(self) -> _IAtIndexer:
        """Fast scalar integer-position access: df.iat[row_pos, col_pos]."""
        return _IAtIndexer(self)

    @property
    def T(self) -> "DataFrame":
        """Transpose the DataFrame."""
        return self.transpose()

    @property
    def axes(self) -> list:
        """Return [index, columns]."""
        return [self.index, self.columns]

    @property
    def empty(self) -> bool:
        """True if the DataFrame has no rows."""
        return self.shape[0] == 0

    @property
    def ndim(self) -> int:
        """Always 2 for a DataFrame."""
        return 2

    @property
    def size(self) -> int:
        """Total number of elements (rows × columns)."""
        r, c = self.shape
        return r * c

    @property
    def values(self):
        """Return the DataFrame as a 2-D numpy array (object dtype if mixed types)."""
        import numpy as np
        cols = self.columns
        if not cols:
            return np.empty((len(self), 0))
        # get_column returns a numpy array for numeric types — no list round-trip needed
        arrays = [np.asarray(_get_col(self._frame, c)) for c in cols]
        dtypes_set = {a.dtype.kind for a in arrays}
        if dtypes_set <= {"f", "i", "u"}:  # all numeric — np.column_stack upcasts to float64
            return np.column_stack(arrays)
        out = np.empty((len(self), len(cols)), dtype=object)
        for j, a in enumerate(arrays):
            out[:, j] = a
        return out
