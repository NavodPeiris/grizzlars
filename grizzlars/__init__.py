"""
grizzlars — Python wrapper for the ultrafast hmdf C++ DataFrame library.

Supported column types: float64 (double), int64, bool, str.
Index type: unsigned 64-bit integer (auto-assigned 0..N-1 if omitted).

numpy and pandas are optional — they are only imported inside functions
that explicitly need them (to_pandas, plotting, values property, etc.).
All core data operations go through the C++ backend.

The implementation is split across sibling modules by concern (see
_frame/ for the DataFrame class's mixins, _series.py, _comparison.py,
etc.) — this file only re-exports the public API.
"""

from __future__ import annotations

from ._grizzlars import (
    set_thread_level,
    set_optimum_thread_level,
    get_thread_level,
)
from ._frame import DataFrame
from ._series import Series
from ._io_functions import read_csv, get_dummies

__version__ = "0.1.0"
__all__ = ["DataFrame", "Series", "read_csv",
           "set_thread_level", "set_optimum_thread_level", "get_thread_level"]

# Enable multithreading automatically on import using all logical CPU cores.
set_optimum_thread_level()
