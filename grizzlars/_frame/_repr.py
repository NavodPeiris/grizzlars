"""__repr__ / _repr_html_ / __str__."""

from __future__ import annotations

from .._helpers import _get_col


class _ReprMixin:

    def __repr__(self) -> str:
        rows, ncols = self.shape
        cols = self.columns
        col_w = 13
        idx_w = 8

        lines: list = []
        header = f"{'':>{idx_w}}" + "".join(f"  {c:>{col_w}}" for c in cols)
        sep = "─" * len(header)
        lines += [header, sep]

        show = min(rows, 10)
        idx = list(self.index)
        # Pre-load all column data once (avoid O(show×ncols) lazy loads)
        col_data = {col: _get_col(self._frame, col) for col in cols}
        for i in range(show):
            row = f"{idx[i]:>{idx_w}}"
            for col in cols:
                v = col_data[col][i]
                val = "NaN" if (isinstance(v, float) and v != v) or v == "" else str(v)
                row += f"  {val:>{col_w}}"
            lines.append(row)

        if rows > 10:
            lines.append(f"  ... ({rows - 10} more rows)")

        lines.append(f"\n[{rows} rows × {ncols} columns]")
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        rows, ncols = self.shape
        cols = self.columns
        idx = list(self.index)

        max_rows = 10
        show = min(rows, max_rows)
        truncated = rows > max_rows

        th_style = "padding:4px 10px;border:1px solid #ddd;background:#c3c3c3;text-align:right;white-space:nowrap;color:black;"
        td_style = "padding:4px 10px;border:1px solid #ddd;text-align:right;white-space:nowrap;color:black;"

        html = [
            '<div style="max-width:100%;overflow-x:auto;font-family:monospace;font-size:13px;">',
            '<table style="border-collapse:collapse;border:1px solid #ddd;">',
            "<thead>",
            '<tr style="background:#dcdcdc;">',
            f'<th style="{th_style}"></th>',
        ]
        for col in cols:
            html.append(f'<th style="{th_style}">{col}</th>')
        html += ["</tr>", "</thead>", "<tbody>"]

        # Pre-load all column data once
        col_data = {col: _get_col(self._frame, col) for col in cols}
        for i in range(show):
            row_bg = "#fff" if i % 2 == 0 else "#fafafa"
            html.append(f'<tr style="background:{row_bg};">')
            html.append(f'<th style="{th_style}">{idx[i]}</th>')
            for col in cols:
                v = col_data[col][i]
                val = "NaN" if (isinstance(v, float) and v != v) or v == "" else v
                html.append(f'<td style="{td_style}">{val}</td>')
            html.append("</tr>")

        if truncated:
            html.append(
                f'<tr><td colspan="{ncols + 1}" style="text-align:center;padding:4px 10px;'
                f'color:#999;border:1px solid #ddd;">... {rows - max_rows} more rows</td></tr>'
            )

        html += [
            "</tbody>",
            "</table>",
            f'<p style="font-size:11px;color:#888;margin:4px 0 0;">{rows} rows × {ncols} columns</p>',
            "</div>",
        ]
        return "".join(html)

    def __str__(self) -> str:
        return self.__repr__()
