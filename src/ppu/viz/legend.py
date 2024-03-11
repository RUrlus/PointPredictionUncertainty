"""Copyright (c) 2024 ING Analytics Wholesale Banking."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


class LegendHandler:
    def __init__(self, ax: plt.Axes) -> LegendHandler:
        self.ax = ax
        self.og_legends = [c for c in ax.get_children() if isinstance(c, mpl.legend.Legend)]
        self.n_og_legends = len(self.og_legends)

    def add_legend(self, **kwargs):
        legend_loc = kwargs.pop("loc", None)
        if legend_loc is None:
            legend_loc = self.n_og_legends + 1
        if self.og_legends:
            self.ax.add_artist(self.og_legends[-1])
        self.ax.legend(loc=legend_loc, **kwargs)
