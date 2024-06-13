"""Copyright (c) 2024 ING Analytics Wholesale Banking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from ppu.viz.legend import LegendHandler

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["plot_dense_binary_scatter", "plot_dense_scatter"]


def plot_dense_scatter(
    X: NDArray, ax: plt.Axes | None = None, color: str | None = None, alpha: float = 0.8, zorder: int = 1, **kwargs
):
    ax = ax or plt.gca()
    s = kwargs.get("s", 1)
    ax.scatter(X[:, 0], X[:, 1], s=s, marker="x", c=color, alpha=alpha, zorder=zorder, **kwargs)
    return ax


def plot_dense_binary_scatter(
    X: NDArray, y: NDArray, ax: plt.Axes | None = None, alpha: float = 0.8, zorder: int = 1, **kwargs
):
    s = kwargs.get("s", (10, 15))
    if isinstance(s, (int, float)):
        s0 = s1 = s
    elif isinstance(s, (list, tuple)):
        s0, s1 = s
    else:
        msg = "`s` should be scalar or tuple of scalars."
        raise TypeError(msg)
    ax = ax or plt.gca()
    lh = LegendHandler(ax)
    mask = y.astype(bool)
    ax.scatter(X[~mask, 0], X[~mask, 1], marker=".", s=s0, c="C0", label="$y=0$", alpha=alpha, zorder=zorder, **kwargs)
    ax.scatter(X[mask, 0], X[mask, 1], marker="+", s=s1, c="C1", label="$y=1$", alpha=alpha, zorder=zorder, **kwargs)
    lh.add_legend(fontsize=12)
    return ax
