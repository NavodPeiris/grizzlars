"""Data cleaning: drop_duplicates/drop_na/dropna, fillna/ffill/bfill,
replace, isna/notna/where/mask, clip/round/rename/drop/astype.
"""

from __future__ import annotations

from typing import Optional


class _CleaningMixin:

    def drop_duplicates(
        self,
        subset=None,
        keep: str = "first",
    ) -> "DataFrame":
        """Return a new DataFrame with duplicate rows removed."""
        if subset is None or isinstance(subset, str):
            col = subset or (self.columns[0] if self.columns else None)
            if col:
                return type(self)._from_frame(self._frame.drop_duplicates(col))
            return self._copy()
        mask = self._frame.duplicated_rows(
            [subset] if isinstance(subset, str) else list(subset), keep)
        not_mask = [not v for v in mask]
        return type(self)._from_frame(self._frame.filter_by_mask_list(not_mask))

    def drop_na(self, col: str) -> "DataFrame":
        """Return a new DataFrame with NaN / empty-string rows in *col* removed."""
        return type(self)._from_frame(self._frame.drop_na(col))

    def dropna(
        self,
        subset=None,
    ) -> "DataFrame":
        """
        Return a new DataFrame with rows containing NaN removed.
        subset: column name or list of column names to check (default: all columns).
        """
        cols = self.columns if subset is None else (
            [subset] if isinstance(subset, str) else list(subset))
        result = self
        for col in cols:
            result = type(self)._from_frame(result._frame.drop_na(col))
        return result

    def fillna(self, col: str, value) -> "DataFrame":
        """Fill NaN / empty values in *col* with *value* (in-place; returns self)."""
        t = self._frame.col_type(col)
        if t == "string":
            self._frame.fillna_string(col, str(value))
        else:
            self._frame.fillna_double(col, float(value))
        return self

    def ffill(self, col: str) -> "DataFrame":
        """Forward-fill NaN values in *col* (in-place; returns self) — delegates to C++."""
        self._frame.ffill_col(col)
        return self

    def bfill(self, col: str) -> "DataFrame":
        """Backward-fill NaN values in *col* (in-place; returns self) — delegates to C++."""
        self._frame.bfill_col(col)
        return self

    def _apply_replace_col(self, col_name: str, mapping: dict) -> None:
        col_type = self._frame.col_type(col_name)
        if not mapping:
            return
        val_sample = next(iter(mapping.values()))
        # Cross-type: string column → numeric values; replace then astype
        if col_type == "string" and isinstance(val_sample, (int, float)):
            str_mapping = {str(k): str(v) for k, v in mapping.items()}
            self._frame.replace_col_string(col_name, list(str_mapping.keys()), list(str_mapping.values()))
            target_type = "int64" if all(isinstance(v, int) for v in mapping.values()) else "double"
            self._frame.astype_col(col_name, target_type)
        elif col_type == "string":
            self._frame.replace_col_string(col_name, [str(k) for k in mapping.keys()], [str(v) for v in mapping.values()])
        else:
            self._frame.replace_col_double(
                col_name,
                [float(k) for k in mapping.keys()],
                [float(v) for v in mapping.values()])

    def replace(
        self,
        to_replace,
        value=None,
        col: Optional[str] = None,
        inplace: bool = False,
    ) -> "DataFrame":
        """
        Replace values.

        replace(to_replace, value)              — replace scalar/list in all columns
        replace({old: new, …})                  — replace via dict in all columns
        replace({"ColName": {old: new, …}})     — pandas-style per-column nested dict
        replace(to_replace, value, col="col")   — restrict to one column
        inplace=True modifies in-place and returns None.
        """
        target = self if inplace else self._copy()

        # pandas nested-dict form: {"ColName": {old: new, ...}}
        if (
            isinstance(to_replace, dict)
            and to_replace
            and isinstance(next(iter(to_replace.values())), dict)
        ):
            for col_name, mapping in to_replace.items():
                target._apply_replace_col(col_name, mapping)
        else:
            if isinstance(to_replace, dict):
                mapping = to_replace
            else:
                vals = to_replace if isinstance(to_replace, list) else [to_replace]
                mapping = {v: value for v in vals}
            if col:
                target._apply_replace_col(col, mapping)
            else:
                for c in target.columns:
                    try:
                        target._apply_replace_col(c, mapping)
                    except Exception:
                        pass

        return None if inplace else target

    def isna(self) -> "DataFrame":
        """Return a boolean DataFrame: True where values are NaN / empty — delegates to C++."""
        return type(self)._from_frame(self._frame.isna_frame())

    def isnull(self) -> "DataFrame":
        """Alias for isna()."""
        return self.isna()

    def notna(self) -> "DataFrame":
        """Return a boolean DataFrame: True where values are NOT NaN / empty — delegates to C++."""
        return type(self)._from_frame(self._frame.notna_frame())

    def notnull(self) -> "DataFrame":
        """Alias for notna()."""
        return self.notna()

    def where(self, cond, other=float("nan")) -> "DataFrame":
        """
        Return a copy where values NOT satisfying *cond* are replaced by *other*.
        cond can be a boolean DataFrame or a boolean array/list.
        """
        if isinstance(cond, type(self)) and isinstance(other, (int, float)):
            return type(self)._from_frame(self._frame.where_frame(cond._frame, float(other)))
        # Fallback: Python-level
        result = self._copy()
        cols = self.columns
        for j, col in enumerate(cols):
            raw = list(self[col])
            if isinstance(cond, type(self)):
                mask_col = list(cond[col])
                result[col] = [v if mask_col[i] else other for i, v in enumerate(raw)]
            elif isinstance(cond, list):
                result[col] = [v if cond[i] else other for i, v in enumerate(raw)]
            else:
                try:
                    import numpy as np
                    arr = np.asarray(cond)
                    if arr.ndim == 2:
                        result[col] = [v if arr[i, j] else other for i, v in enumerate(raw)]
                    else:
                        result[col] = [v if arr[i] else other for i, v in enumerate(raw)]
                except ImportError:
                    result[col] = [v if cond[i] else other for i, v in enumerate(raw)]
        return result

    def mask(self, cond, other=float("nan")) -> "DataFrame":
        """Inverse of where() — replace values WHERE *cond* is True."""
        result = self._copy()
        cols = self.columns
        for j, col in enumerate(cols):
            raw = list(self[col])
            if isinstance(cond, type(self)):
                mask_col = list(cond[col])
                result[col] = [other if mask_col[i] else v for i, v in enumerate(raw)]
            elif isinstance(cond, list):
                result[col] = [other if cond[i] else v for i, v in enumerate(raw)]
            else:
                try:
                    import numpy as np
                    arr = np.asarray(cond)
                    if arr.ndim == 2:
                        result[col] = [other if arr[i, j] else v for i, v in enumerate(raw)]
                    else:
                        result[col] = [other if arr[i] else v for i, v in enumerate(raw)]
                except ImportError:
                    result[col] = [other if cond[i] else v for i, v in enumerate(raw)]
        return result

    def clip(
        self,
        lower=None,
        upper=None,
        col: Optional[str] = None,
    ) -> "DataFrame":
        """Clip numeric values to [lower, upper] bounds — delegates to C++."""
        result = self._copy()
        lo = float(lower) if lower is not None else float("-inf")
        hi = float(upper) if upper is not None else float("inf")
        target_cols = [col] if col else [
            c for c in self.columns
            if self._frame.col_type(c) in ("double", "int64")
        ]
        for c in target_cols:
            result._frame.clip_col(c, lo, hi)
        return result

    def round(self, decimals: int = 0) -> "DataFrame":
        """Round all numeric columns to *decimals* decimal places — delegates to C++."""
        result = self._copy()
        for col in self.columns:
            if self._frame.col_type(col) in ("double",):
                result._frame.round_col(col, decimals)
        return result

    def rename(self, mapping: dict) -> "DataFrame":
        """Rename columns in-place and return self."""
        for old, new in mapping.items():
            self._frame.rename_col(old, new)
        return self

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        inplace: bool = False,
        errors: str = "raise",
    ) -> Optional["DataFrame"]:
        """
        Drop rows or columns.
        labels: str or list.  axis: 0/'index' to drop rows, 1/'columns' to drop columns.
        """
        if columns is not None:
            labels = columns
            axis = 1
        elif index is not None:
            labels = index
            axis = 0

        if labels is None:
            raise ValueError("must specify labels, columns, or index")

        drop_list = [labels] if isinstance(labels, str) else list(labels)
        is_col_axis = axis in (1, "columns")

        if inplace:
            target = self
        else:
            target = self._copy()

        if is_col_axis:
            for c in drop_list:
                if c not in target:
                    if errors == "raise":
                        raise KeyError(c)
                    continue
                target._frame.drop_column(c)
        else:
            idx = list(target.index)
            drop_set = set(drop_list)
            mask = [int(v) not in drop_set for v in idx]
            new_frame = target._frame.filter_by_mask_list(mask)
            target._frame = new_frame

        if inplace:
            return None
        return target

    def astype(self, dtype) -> "DataFrame":
        """
        Cast columns to specified types — delegates to C++ astype_col.
        dtype can be a single type or a dict {col: type}.
        """
        result = self._copy()

        def _map_type(t):
            if t in (float, "float", "float64"): return "double"
            if t in (int, "int", "int64"): return "int64"
            if t in (str, "str", "string", "object"): return "string"
            if t in (bool, "bool"): return "bool"
            return None

        if isinstance(dtype, dict):
            for col, t in dtype.items():
                mapped = _map_type(t)
                if mapped:
                    result._frame.astype_col(col, mapped)
        else:
            mapped = _map_type(dtype)
            if mapped:
                for col in self.columns:
                    try:
                        result._frame.astype_col(col, mapped)
                    except Exception:
                        pass
        return result
