"""Lightweight 1-D column returned by DataFrame.__getitem__."""

from __future__ import annotations

from ._comparison import _ColComparison, _CombinedComparison
from ._helpers import _OPS, _get_col, _is_nan


class Series:
    """Lightweight 1-D column returned by DataFrame.__getitem__. No numpy/pandas dependency."""

    def __init__(self, data, name=None, dtype=None, _src_frame=None, _src_col=None):
        # Lazy mode: data is None, loaded on first access via _get_data()
        if data is None:
            self._data    = None
            self._src_frame = _src_frame
            self._src_col   = _src_col
        elif isinstance(data, list):
            self._data = data
            self._src_frame = None
            self._src_col   = None
        else:
            try:
                self._data = list(data)
            except TypeError:
                self._data = [data]
            self._src_frame = None
            self._src_col   = None
        self.name = name
        self._dtype_str = dtype  # "double"/"int64"/"bool"/"string"

    def _get_data(self) -> list:
        if self._data is None:
            self._data = _get_col(self._src_frame._frame, self._src_col)
            if not isinstance(self._data, list):
                self._data = list(self._data)
        return self._data

    # Support len(), iteration, indexing
    def __len__(self):
        if self._data is None and self._src_frame is not None:
            return len(self._src_frame)
        return len(self._data)
    def __iter__(self): return iter(self._get_data())
    def __getitem__(self, i): return self._get_data()[i]

    # Element-wise comparisons — fast path when we have a source frame
    def _cmp(self, other, op: str):
        if self._src_frame is not None and self._src_col is not None and not isinstance(other, Series):
            return _ColComparison(self._src_frame, self._src_col, op, other)
        data = self._get_data()
        fn = _OPS[op]
        return Series([fn(v, other) if not (_is_nan(v) if isinstance(v, float) else False)
                       else False for v in data], dtype="bool")

    def __eq__(self, other):
        if self._src_frame is not None and self._src_col is not None and not isinstance(other, Series):
            return _ColComparison(self._src_frame, self._src_col, "==", other)
        data = self._get_data()
        return Series([v == other for v in data], dtype="bool")
    def __ne__(self, other):
        if self._src_frame is not None and self._src_col is not None and not isinstance(other, Series):
            return _ColComparison(self._src_frame, self._src_col, "!=", other)
        data = self._get_data()
        return Series([v != other for v in data], dtype="bool")
    def __lt__(self, other): return self._cmp(other, "<")
    def __le__(self, other): return self._cmp(other, "<=")
    def __gt__(self, other): return self._cmp(other, ">")
    def __ge__(self, other): return self._cmp(other, ">=")

    # Logical operators on bool Series
    def __and__(self, other):
        if isinstance(other, (_ColComparison, _CombinedComparison)):
            return _CombinedComparison(self, other, "and")
        return Series([a and b for a, b in zip(self._get_data(), other._get_data())], dtype="bool")
    def __or__(self, other):
        if isinstance(other, (_ColComparison, _CombinedComparison)):
            return _CombinedComparison(self, other, "or")
        return Series([a or b for a, b in zip(self._get_data(), other._get_data())], dtype="bool")
    def __invert__(self):
        return Series([not v for v in self._get_data()], dtype="bool")

    @property
    def dtype(self): return self._dtype_str

    def unique(self):
        seen = set(); result = []
        for v in self._get_data():
            if v not in seen:
                seen.add(v); result.append(v)
        return result

    def value_counts(self):
        from collections import Counter
        c = Counter(v for v in self._get_data() if not (_is_nan(v) if isinstance(v, float) else False))
        return dict(sorted(c.items(), key=lambda x: -x[1]))

    def to_list(self): return list(self._get_data())

    def to_numpy(self):
        import numpy as np
        return np.asarray(self._get_data())

    def __array__(self, dtype=None):
        import numpy as np
        return np.asarray(self._get_data(), dtype=dtype)

    def isnull(self):
        return Series([_is_nan(v) or v == "" for v in self._get_data()], dtype="bool")

    def notnull(self):
        return Series([not (_is_nan(v) or v == "") for v in self._get_data()], dtype="bool")

    def nunique(self):
        return len(set(v for v in self._get_data()
                       if not (_is_nan(v) if isinstance(v, float) else v == "")))

    def head(self, n: int = 5) -> "Series":
        return Series(self._get_data()[:n], name=self.name, dtype=self._dtype_str)

    def tail(self, n: int = 5) -> "Series":
        d = self._get_data()
        return Series(d[-n:] if n else [], name=self.name, dtype=self._dtype_str)

    def take(self, indices, axis=0) -> "Series":
        d = self._get_data()
        sz = len(d)
        norm = [int(i) if i >= 0 else sz + int(i) for i in indices]
        return Series([d[i] for i in norm], name=self.name, dtype=self._dtype_str)

    def __repr__(self):
        d = self._get_data()
        lines = [f"{i}    {v}" for i, v in enumerate(d[:10])]
        if len(d) > 10:
            lines.append(f"... {len(d) - 10} more")
        if self.name:
            lines.append(f"Name: {self.name}")
        return "\n".join(lines)
