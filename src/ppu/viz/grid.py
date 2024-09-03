from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ppu.viz.scatter import plot_dense_binary_scatter, plot_dense_scatter

if TYPE_CHECKING:
    from numpy.typing import NDArray


class GridPlot:
    def __init__(self, X: NDArray, y: NDArray | None = None, n_ticks: int = 100, eps: float = 1.0) -> None:
        self.X = X
        self.y = y
        self.n_ticks = n_ticks
        self.grid_shape = (self.n_ticks, self.n_ticks)
        self.eps = eps

        self.X_grid = None
        self.x0 = None
        self.x1 = None
        self.x0_min, self.x1_min = self.X.min(0) - self.eps
        self.x0_max, self.x1_max = self.X.max(0) + self.eps
        self.x0 = np.linspace(self.x0_min, self.x0_max, self.n_ticks)
        self.x1 = np.linspace(self.x1_min, self.x1_max, self.n_ticks)
        x0_, x1_ = np.meshgrid(self.x0, self.x1)
        self.X_grid = np.c_[x0_.ravel(), x1_.ravel()]

    def plot(
        self,
        values: NDArray,
        overlay: bool = False,
        ax=None,
        vmax: float | None = None,
        cmap: str | None = None,
        **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 8))
        else:
            fig = ax.get_figure()

        if values.shape != self.grid_shape:
            values = values.reshape(*self.grid_shape)

        vmax = vmax or values.max()
        cmap = cmap or ("Blues_r" if not overlay else "Greys_r")
        c = ax.pcolormesh(self.x0, self.x1, values, cmap=cmap, vmin=0, vmax=vmax, alpha=0.95, **kwargs)
        # set the limits of the plot to the limits of the data
        fig.colorbar(c, ax=ax)

        if overlay:
            if self.y is None:
                ax = plot_dense_scatter(X=self.X, ax=ax, alpha=0.3)
            else:
                ax = plot_dense_binary_scatter(X=self.X, y=self.y, ax=ax, alpha=0.3)
        fig.tight_layout()
        return ax
