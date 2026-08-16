"""Arithmetic operators (+ - * / // % ** neg abs) and comparison operators
(== != < <= > >=, returning a boolean DataFrame)."""

from __future__ import annotations

import operator as _op


class _OperatorsMixin:

    # ── arithmetic operators ──────────────────────────────────────────────────

    def _arith(self, other, op_str, op_fn) -> "DataFrame":
        if isinstance(other, type(self)):
            return type(self)._from_frame(self._frame.arith_frame_op(op_str, other._frame))
        try:
            scalar = float(other)
            return type(self)._from_frame(self._frame.arith_scalar(op_str, scalar))
        except (TypeError, ValueError):
            pass
        # Fallback for non-numeric scalars
        result = self._copy()
        for col in self.columns:
            if self._frame.col_type(col) not in ("double", "int64"):
                continue
            raw = list(self[col])
            result[col] = [op_fn(v, other) for v in raw]
        return result

    def __add__(self, other):  return self._arith(other, "+", _op.add)
    def __radd__(self, other): return self._arith(other, "+", lambda a, b: _op.add(b, a))
    def __sub__(self, other):  return self._arith(other, "-", _op.sub)
    def __rsub__(self, other): return self._arith(other, "-", lambda a, b: _op.sub(b, a))
    def __mul__(self, other):  return self._arith(other, "*", _op.mul)
    def __rmul__(self, other): return self._arith(other, "*", lambda a, b: _op.mul(b, a))
    def __truediv__(self, other):   return self._arith(other, "/", _op.truediv)
    def __rtruediv__(self, other):  return self._arith(other, "/", lambda a, b: _op.truediv(b, a))
    def __floordiv__(self, other):  return self._arith(other, "//", _op.floordiv)
    def __rfloordiv__(self, other): return self._arith(other, "//", lambda a, b: _op.floordiv(b, a))
    def __mod__(self, other):  return self._arith(other, "%", _op.mod)
    def __rmod__(self, other): return self._arith(other, "%", lambda a, b: _op.mod(b, a))
    def __pow__(self, other):  return self._arith(other, "**", _op.pow)
    def __rpow__(self, other): return self._arith(other, "**", lambda a, b: _op.pow(b, a))
    def __neg__(self):         return type(self)._from_frame(self._frame.arith_scalar("*", -1.0))
    def __abs__(self):         return self.abs()

    # ── comparison operators (return bool DataFrame) ──────────────────────────

    def _compare(self, other, op_str, op_fn) -> "DataFrame":
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return type(self)._from_frame(self._frame.compare_scalar(op_str, float(other)))
        # Fallback
        result_data = {}
        for col in self.columns:
            raw = list(self[col])
            if isinstance(other, type(self)):
                oth = list(other[col]) if col in other else [False] * len(raw)
                try:
                    result_data[col] = [op_fn(a, b) for a, b in zip(raw, oth)]
                except TypeError:
                    result_data[col] = [False] * len(raw)
            else:
                try:
                    result_data[col] = [op_fn(v, other) for v in raw]
                except TypeError:
                    result_data[col] = [False] * len(raw)
        return type(self)(result_data, index=list(self.index))

    def __eq__(self, other): return self._compare(other, "==", _op.eq)
    def __ne__(self, other): return self._compare(other, "!=", _op.ne)
    def __lt__(self, other): return self._compare(other, "<",  _op.lt)
    def __le__(self, other): return self._compare(other, "<=", _op.le)
    def __gt__(self, other): return self._compare(other, ">",  _op.gt)
    def __ge__(self, other): return self._compare(other, ">=", _op.ge)
