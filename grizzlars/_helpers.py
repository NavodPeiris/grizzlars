"""Small stateless helpers shared across grizzlars' Python layer: native
column dtype dispatch (load_column_*/get_column_* are typed on the C++ side
since litgen needs concrete, non-template signatures — the runtime type
dispatch that used to live in C++ lives here instead) and value-display
helpers used by Series/DataFrame repr.
"""

from __future__ import annotations

import operator as _op
import sys as _sys

_OPS = {
    ">":  _op.gt,
    ">=": _op.ge,
    "<":  _op.lt,
    "<=": _op.le,
    "==": _op.eq,
    "!=": _op.ne,
}


def _load_col(frame, name: str, values) -> None:
    """Dispatch to the correctly-typed native load_column_* for *values*."""
    _np = _sys.modules.get("numpy")
    if _np is not None and isinstance(values, _np.ndarray):
        kind = values.dtype.kind
        if kind == "f":
            frame.load_column_double(name, values.tolist())
        elif kind in ("i", "u"):
            frame.load_column_int64(name, values.tolist())
        elif kind == "b":
            frame.load_column_bool(name, values.astype("uint8").tolist())
        else:
            frame.load_column_string(name, [str(v) for v in values.tolist()])
        return

    vals = values if isinstance(values, list) else list(values)
    if not vals:
        frame.load_column_double(name, [])
        return
    first = vals[0]
    if isinstance(first, bool):
        frame.load_column_bool(name, [1 if v else 0 for v in vals])
    elif isinstance(first, float):
        frame.load_column_double(name, [float(v) for v in vals])
    elif isinstance(first, int):
        frame.load_column_int64(name, [int(v) for v in vals])
    else:
        frame.load_column_string(name, [str(v) for v in vals])


def _mask_to_list(mask) -> list:
    """Coerce a boolean mask (numpy array, pandas Series, or list-like) into
    a plain list of 0/1 ints for filter_by_mask_list (nanobind's vector<T>
    caster wants a Sequence, not an ndarray)."""
    _np = _sys.modules.get("numpy")
    if _np is not None and isinstance(mask, _np.ndarray):
        return mask.astype("uint8").tolist()
    return [1 if v else 0 for v in mask]


def _get_col(frame, name: str):
    """Dispatch to the correctly-typed native get_column_* for *name*."""
    t = frame.col_type(name)
    if t == "double":
        import numpy as _np
        return _np.asarray(frame.get_column_double(name))
    if t == "int64":
        import numpy as _np
        return _np.asarray(frame.get_column_int64(name))
    if t == "bool":
        return list(frame.get_column_bool(name))
    return list(frame.get_column_string(name))


def _is_nan(v) -> bool:
    return isinstance(v, float) and v != v


def _display(v) -> str:
    return "NaN" if _is_nan(v) or v == "" else str(v)
