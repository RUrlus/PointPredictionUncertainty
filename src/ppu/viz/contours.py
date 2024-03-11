"""Plotting functions.

Copyright (c) 2024 ING Analytics Wholesale Banking
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ppu.viz.color import _get_color_hexes
from ppu.viz.legend import LegendHandler

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["plot_pdf_contours"]


def plot_pdf_contours(
    x_grid: NDArray,
    y_grid: NDArray,
    grid_pdf: NDArray,
    ref_pdf: NDArray,
    levels: None | int | float | NDArray = None,
    filled: bool = True,
    ax: plt.Axes | None = None,
    alpha: float = 0.8,
    cmap: str | None = None,
    legend_loc: str | None = None,
    zorder: int = 10,
    **kwargs,
):
    """Plot contours at `levels` given reference PDF values.

    Args:
        x_grid:
        y_grid:
        grid_pdf:
        ref_pdf:
        levels: if int(s) levels is treated as the number of standard deviations for the confidence interval.
            If float(s) it is taken to be the density to be contained in the confidence interval
            By default we plot 1, 2 and 3 std deviations
        filled: create filled ellipses if true and line plot otherwhise. Ignore when number of levels > 4
        ax: Optional pre-existing axes for the plot
        cmap: matplotlib cmap name to use for CIs, default='Blues'
        legend_loc: location of the legend, default is `lower left`
        alpha: defualt=0.8 opacity value of the contours
        zorder: the back/foreground position order
        kwargs: arguments passed to the plotting functions

    Returns:
        ax: the axis with the ellipse added to it
    """
    if cmap is None:
        cmap = "Blues"
    if isinstance(levels, (list, tuple)):
        levels = np.asarray(levels)
    if levels is None:
        perc = np.array((10.0, 5.0, 1.0))
        labels = [f"{p}% CI" for p in (90, 95, 99)]
    elif isinstance(levels, float):
        labels = [f"{round(levels * 100, 3)}% CI"]
        perc = (1 - np.array((levels,))) * 100
    elif isinstance(levels, np.ndarray) and np.issubdtype(levels.dtype, np.floating):
        levels = np.sort(np.unique(levels))
        labels = [f"{round(level * 100, 3)}% CI" for level in levels]
        perc = (1 - levels) * 100
    else:
        msg = "`levels` must be a int, float, array-like or None"
        raise TypeError(msg)

    crit_vals = np.percentile(ref_pdf, perc)
    plot_levels = [*crit_vals[::-1].tolist(), 1]
    colors, c_marker = _get_color_hexes(cmap, n_colors=len(plot_levels), return_marker=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    lh = LegendHandler(ax)
    ax.contourf(x_grid, y_grid, grid_pdf, colors=colors[::-1], levels=plot_levels)
    handles = [Patch(facecolor=c, edgecolor=c, label=l) for c, l in zip(colors, labels)]
    lh.add_legend(handles=handles, loc=legend_loc, fontsize=12)
    fig.tight_layout()
    return ax
