"""Module-level I/O helpers: read_csv, get_dummies."""

from __future__ import annotations

import csv
import re
from typing import Optional

from ._frame import DataFrame
from ._grizzlars import GrizzlarFrame as _GrizzlarFrame
from ._helpers import _load_col

# Matches hmdf column-name annotations like  :12265:<double>  or  :12265:<unsigned long>
_HMDF_ANNOTATION = re.compile(r":\d+:<[^>]+>$")


def read_csv(
    path: str,
    index_col: Optional[str] = None,
    dtype: Optional[dict] = None,
) -> DataFrame:
    """
    Read a CSV file into a DataFrame.

    Uses a native C++ reader by default (dramatically faster than Python's
    csv.DictReader for large files).  Automatically strips hmdf column-name
    annotations (e.g. ``:12265:<double>``) so hmdf-written files load cleanly.

    Falls back to a pure-Python path only when *dtype* overrides are supplied.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    index_col : str, optional
        Column to use as the index (must contain unsigned integers).
    dtype : dict, optional
        Mapping of clean column name -> callable for explicit type conversion.
        When set, forces the slower Python fallback path.
    """
    # ── fast path: native C++ reader ──────────────────────────────────────────
    if dtype is None:
        frame = _GrizzlarFrame.read_csv_native(str(path), index_col or "")
        return DataFrame._from_frame(frame)

    # ── fallback: Python reader (supports custom dtype converters) ────────────
    raw: dict = {}
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for col, val in row.items():
                raw.setdefault(col, []).append(val)

    raw = {_HMDF_ANNOTATION.sub("", k): v for k, v in raw.items()}

    data: dict = {}
    for col, vals in raw.items():
        if dtype and col in dtype:
            data[col] = [dtype[col](v) for v in vals]
            continue
        try:
            data[col] = [int(v) for v in vals]
        except ValueError:
            try:
                data[col] = [float(v) for v in vals]
            except ValueError:
                data[col] = vals

    index = None
    if index_col and index_col in data:
        raw_idx = data.pop(index_col)
        index = [int(v) for v in raw_idx]

    return DataFrame(data, index=index)


def get_dummies(
    data: "DataFrame",
    columns=None,
    prefix=None,
    prefix_sep: str = "_",
    dummy_na: bool = False,
    drop_first: bool = False,
    dtype=None,
) -> "DataFrame":
    """
    One-hot encode categorical columns (equivalent to pandas.get_dummies).

    Parameters
    ----------
    data : DataFrame
    columns : list of str, optional
        Columns to encode. Defaults to all string columns.
    prefix : str or list, optional
        Prefix for new column names. Defaults to the original column name.
    prefix_sep : str
        Separator between prefix and category value. Default "_".
    dummy_na : bool
        If True, add a column for NaN/empty values. Default False.
    drop_first : bool
        Drop the first category level to avoid multicollinearity. Default False.
    dtype : ignored
        Accepted for API compatibility; new columns are always int64 (0/1).
    """
    _ = dtype  # accepted for API compatibility only
    result = data._copy()

    if columns is None:
        columns = [c for c in data.columns if data._frame.col_type(c) == "string"]

    prefixes = {}
    if prefix is None:
        prefixes = {c: c for c in columns}
    elif isinstance(prefix, str):
        prefixes = {c: prefix for c in columns}
    else:
        prefixes = {c: p for c, p in zip(columns, prefix)}

    for col in columns:
        col_type = data._frame.col_type(col)
        if col_type != "string":
            raise TypeError(
                f"get_dummies: column '{col}' has type '{col_type}', expected 'string'. "
                f"Only string/categorical columns can be one-hot encoded."
            )
        col_vals = list(data[col])  # list of raw str values
        unique_vals = []
        seen = set()
        for v in col_vals:
            is_empty = (v == "" or v is None)
            if is_empty:
                if dummy_na and "<NA>" not in seen:
                    seen.add("<NA>")
                    unique_vals.append("<NA>")
            else:
                if v not in seen:
                    seen.add(v)
                    unique_vals.append(v)

        if drop_first and unique_vals:
            unique_vals = unique_vals[1:]

        pfx = prefixes.get(col, col)
        for uv in unique_vals:
            new_col = f"{pfx}{prefix_sep}{uv}"
            if uv == "<NA>":
                _load_col(result._frame, new_col, [1 if (v == "" or v is None) else 0 for v in col_vals])
            else:
                _load_col(result._frame, new_col, [1 if v == uv else 0 for v in col_vals])

        result._frame.drop_column(col)

    return result
