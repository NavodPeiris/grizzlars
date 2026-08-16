"""to_csv/to_dict/to_numpy/to_pandas/__dataframe__/to_json/to_string/to_html."""

from __future__ import annotations

import json as _json
from typing import Optional


class _IOMixin:

    def to_csv(self, path: str, index: bool = True) -> None:
        self._frame.to_csv(path, index)

    def to_dict(self, orient: str = "dict") -> dict:
        """Convert DataFrame to a dictionary."""
        cols = self.columns
        idx = list(self.index)
        if orient == "list":
            return {col: list(self[col]) for col in cols}
        if orient == "records":
            return [{col: self[col][i] for col in cols} for i in range(len(self))]
        if orient == "index":
            return {int(idx[i]): {col: self[col][i] for col in cols}
                    for i in range(len(self))}
        if orient == "series":
            return {col: self[col] for col in cols}
        return {col: {int(idx[i]): self[col][i] for i in range(len(self))}
                for col in cols}

    def to_numpy(self):
        """Return the DataFrame as a 2-D numpy array."""
        return self.values

    def to_pandas(self):
        """Convert to a pandas DataFrame (enables plotly, seaborn, sklearn, etc.)."""
        import pandas as pd
        data = {col: list(self[col]) for col in self.columns}
        return pd.DataFrame(data, index=list(self.index))

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):
        """Implement the Python DataFrame Interchange Protocol."""
        return self.to_pandas().__dataframe__(
            nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    def to_json(
        self,
        path_or_buf=None,
        orient: str = "records",
        indent: Optional[int] = None,
    ) -> Optional[str]:
        """Serialize to JSON string or file."""
        data = self.to_dict(orient=orient)
        text = _json.dumps(data, indent=indent, default=str)
        if path_or_buf is None:
            return text
        with open(path_or_buf, "w") as f:
            f.write(text)
        return None

    def to_string(
        self,
        max_rows: Optional[int] = None,
        max_cols: Optional[int] = None,
    ) -> str:
        """Return string representation."""
        return self.__repr__()

    def to_html(self, index: bool = True) -> str:
        """Return HTML table representation."""
        return self._repr_html_()
