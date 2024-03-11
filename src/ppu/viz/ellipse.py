"""Copyright (c) 2024 ING Analytics Wholesale Banking."""
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sts
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch

from ppu.viz.color import _get_color_hexes, _get_line_color
from ppu.viz.legend import LegendHandler

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["plot_ellipse_from_cov"]


def _create_filled_legend(c_hexes, c_marker, labels, vals):
    label = r"$\bar{x}$" + f"={round(vals[0], 4)}, " + r"$\bar{y}$" + f"={round(vals[1], 4)}"
    line = [
        Line2D(
            [0],
            [0],
            linestyle="None",
            markeredgewidth=3,
            markersize=8,
            marker="x",
            color=c_marker,
            alpha=0.6,
            label=label,
        )
    ]
    patches = [Patch(facecolor=c, edgecolor=c, label=label) for c, label in zip(c_hexes, labels)]
    return line + patches


def _plot_filled_ellipse(
    pos: Sequence[int | float] | NDArray[int | float],
    theta: float,
    scales: NDArray,
    labels: list[str],
    ax: plt.Axes | None = None,
    alpha: float = 0.8,
    cmap: str | None = None,
    zorder: int = 10,
    **kwargs,
):
    colors, c_marker = _get_color_hexes(cmap, n_colors=len(labels), return_marker=True, keep_alpha=False)

    scales = scales[::-1]
    colors = colors[::-1]
    for i in range(scales.shape[0]):
        width, height = scales[i, :]
        ellipse = Ellipse(
            xy=pos, width=width, height=height, angle=theta, alpha=alpha, color=colors[i], zorder=zorder, **kwargs
        )
        ax.add_patch(ellipse)

    ax.scatter(*pos, marker="x", s=6, color=c_marker, zorder=zorder, **kwargs)
    # create custom legend with the correct colours and labels
    handles = _create_filled_legend(colors, c_marker, labels, pos)
    return ax, handles


def _create_line_legend(styles, color, labels, vals):
    label = r"$\bar{x}$" + f"={round(vals[0], 4)}, " + r"$\bar{y}$" + f"={round(vals[1], 4)}"
    line = [
        Line2D(
            [0], [0], linestyle="None", markeredgewidth=3, markersize=8, marker="x", color=color, alpha=0.6, label=label
        )
    ]

    line_seg = [
        Line2D([0], [0], linestyle=ls, linewidth=2, color=color, label=label) for ls, label in zip(styles[::-1], labels)
    ]
    return line + line_seg


def _plot_ellipse(
    pos: Sequence[int | float] | NDArray[int | float],
    theta: float,
    scales: NDArray,
    labels: list[str],
    ax: plt.Axes | None = None,
    alpha: float = 0.8,
    cmap: str | None = None,
    zorder: int = 10,
    **kwargs,
):
    color = _get_line_color(cmap, False)
    line_styles = (":", "--", "-") if len(labels) <= 3 else (":", "-.", "--", "-")

    scales = scales[::-1]
    for i in range(scales.shape[0]):
        width, height = scales[i, :]
        ellipse = Ellipse(
            xy=pos,
            width=width,
            height=height,
            angle=theta,
            color=color,
            fill=False,
            ls=line_styles[i],
            lw=2,
            alpha=alpha,
            zorder=zorder,
            **kwargs,
        )
        ax.add_patch(ellipse)
    ax.scatter(*pos, marker="x", s=6, color=color, zorder=zorder, **kwargs)

    # create custom legend with the correct colours and labels
    handles = _create_line_legend(line_styles, color, labels, pos)
    return ax, handles


def plot_ellipse_from_cov(
    pos: Sequence[int | float] | NDArray[int | float],
    cov_mat: NDArray[int | float],
    levels: None | int | float | NDArray = None,
    filled: bool = True,
    ax: plt.Axes | None = None,
    alpha: float = 0.8,
    cmap: str | None = None,
    legend_loc: str | None = None,
    zorder: int = 10,
    **kwargs,
):
    """Plot ellipse for a covariance matrix.


    Args:
        pos: The location of the center of the ellipse. Expects a 2-element sequence of [x0, y0].
        cov_mat: The 2x2 covariance matrix to base the ellipse
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

    # quick catch for list and tuples
    if isinstance(levels, (list, tuple)):
        levels = np.asarray(levels)

    vals, vecs = np.linalg.eigh(cov_mat)
    order = vals.argsort()[::-1]
    vals = vals[None, order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # transform levels into scaling factors for the ellipse
    # Width and height are "full" widths, not radius
    if levels is None:
        # std: 1, 2, 3
        scales = np.array((2, 4, 6))[:, None] * np.sqrt(vals)
        labels = [r"$1\sigma$ CI", r"$2\sigma$ CI", r"$3\sigma$ CI"]
    elif isinstance(levels, int):
        labels = [f"{levels}" + r"$\sigma$ CI"]
        scales = 2 * np.array((levels,))[:, None] * np.sqrt(vals)
    elif isinstance(levels, np.ndarray) and np.issubdtype(levels.dtype, np.integer):
        scales = 2 * np.sort(np.unique(levels))[:, None] * np.sqrt(vals)
        labels = [f"{level}" + r"$\sigma$ CI" for level in levels]
    elif isinstance(levels, float):
        labels = [f"{round(levels * 100, 3)}% CI"]
        scales = 2 * np.sqrt(sts.chi2(2).ppf(np.array((levels,)))[:, None] * vals)
    elif isinstance(levels, np.ndarray) and np.issubdtype(levels.dtype, np.floating):
        levels = np.sort(np.unique(levels))
        labels = [f"{round(level * 100, 3)}% CI" for level in levels]
        scales = 2 * np.sqrt(sts.chi2(2).ppf(levels)[:, None] * vals)
    else:
        msg = "`levels` must be a int, float, array-like or None"
        raise TypeError(msg)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.get_figure()

    lh = LegendHandler(ax)
    if filled or len(labels) > 4:
        ax, handles = _plot_filled_ellipse(pos, theta, scales, labels, ax, alpha, cmap, zorder, **kwargs)
    else:
        ax, handles = _plot_ellipse(pos, theta, scales, labels, ax, alpha, cmap, zorder, **kwargs)
    lh.add_legend(handles=handles, loc=legend_loc, fontsize=12)
    fig.tight_layout()
    return ax
