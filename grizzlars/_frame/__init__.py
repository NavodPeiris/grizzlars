"""The DataFrame class itself, assembled from single-concern mixins.

Each _*.py sibling module holds one section of DataFrame's original,
much larger implementation (column access, statistics, sorting/filtering,
groupby/join, window functions, cleaning, reshaping, operators, I/O,
plotting, repr) — split purely for readability, with no behavior change.
Mixin methods refer to `type(self)` rather than importing DataFrame
directly, so none of the sibling modules need to import this one back.
"""

from __future__ import annotations

from ._cleaning import _CleaningMixin
from ._core import _CoreMixin
from ._groupby_join import _GroupByJoinMixin
from ._io import _IOMixin
from ._metadata import _MetadataMixin
from ._operators import _OperatorsMixin
from ._plotting import _PlottingMixin
from ._repr import _ReprMixin
from ._reshaping import _ReshapingMixin
from ._sorting_filtering import _SortingFilteringMixin
from ._statistics import _StatisticsMixin
from ._window import _WindowMixin


class DataFrame(
    _CoreMixin,
    _MetadataMixin,
    _StatisticsMixin,
    _SortingFilteringMixin,
    _GroupByJoinMixin,
    _WindowMixin,
    _CleaningMixin,
    _ReshapingMixin,
    _OperatorsMixin,
    _IOMixin,
    _PlottingMixin,
    _ReprMixin,
):
    """
    High-performance DataFrame backed by the C++ hmdf library.

    Parameters
    ----------
    data : dict, optional
        Mapping of column name -> list or numpy array.
    index : list or array-like, optional
        Unsigned integer index.  Defaults to 0..N-1.

    Examples
    --------
    >>> df = grizzlars.DataFrame({"price": [100.5, 200.0], "volume": [1000, 2000]})
    >>> df.filter(df["price"] > 150).sort("price")
    """
