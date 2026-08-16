"""Plotting accessors — all delegate to pandas (hist/plot/boxplot)."""

from __future__ import annotations


class _PlottingMixin:

    def hist(self, column=None, by=None, grid=True, xlabelsize=None, xrot=None,
             ylabelsize=None, yrot=None, ax=None, sharex=False, sharey=False,
             figsize=None, layout=None, bins=10, legend=False, **kwargs):
        """Plot histograms of numeric columns (delegates to pandas)."""
        return self.to_pandas().hist(
            column=column, by=by, grid=grid, xlabelsize=xlabelsize, xrot=xrot,
            ylabelsize=ylabelsize, yrot=yrot, ax=ax, sharex=sharex, sharey=sharey,
            figsize=figsize, layout=layout, bins=bins, legend=legend, **kwargs,
        )

    def plot(self, *args, **kwargs):
        """Access matplotlib plotting (delegates to pandas)."""
        return self.to_pandas().plot(*args, **kwargs)

    def boxplot(self, column=None, by=None, ax=None, fontsize=None, rot=0,
                grid=True, figsize=None, layout=None, **kwargs):
        """Draw a boxplot (delegates to pandas)."""
        return self.to_pandas().boxplot(
            column=column, by=by, ax=ax, fontsize=fontsize, rot=rot,
            grid=grid, figsize=figsize, layout=layout, **kwargs,
        )
