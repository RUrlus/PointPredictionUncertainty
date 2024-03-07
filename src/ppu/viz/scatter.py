"""Copyright (c) 2024 ING Analytics Wholesale Banking."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["plot_dense_scatter"]


def plot_dense_scatter(
    X: NDArray, ax: plt.Axes | None = None, color: str | None = None, alpha: float = 0.8, zorder: int = 1, **kwargs
):
    ax = ax or plt.gca()
    ax.scatter(X[:, 0], X[:, 1], s=1, marker="x", c=color, alpha=alpha, zorder=zorder, **kwargs)
    return ax
