"""Copyright (c) 2024 ING Analytics Wholesale Banking."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex


def set_plot_style(get_colors=False) -> list[str] | None:
    plt.style.use("ggplot")
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    plt.rcParams["text.color"] = "black"
    plt.rcParams["figure.max_open_warning"] = 0
    if get_colors:
        return [i["color"] for i in plt.rcParams["axes.prop_cycle"]]  # type: ignore
    return None


def _get_color_hexes(cmap, n_colors=3, n_offset=3, return_marker=False, keep_alpha=True):
    if not isinstance(cmap, str):
        msg = "`cmap` must be a str"
        raise TypeError(msg)

    t_colors = n_colors + 2 * n_offset + 1
    cmap = plt.colormaps.get_cmap(cmap).resampled(t_colors)
    hexes = [rgb2hex(cmap(i), keep_alpha=keep_alpha) for i in range(cmap.N)]
    if return_marker:
        return hexes[n_offset : n_offset + n_colors][::-1], hexes[-3]
    return hexes[n_offset : n_offset + n_colors][::-1]


def _get_line_color(cmap, keep_alpha=True):
    if not isinstance(cmap, str):
        msg = "`cmap` must be a str"
        raise TypeError(msg)

    cmap = plt.colormaps.get_cmap(cmap)
    return rgb2hex(cmap.get_over(), keep_alpha=keep_alpha)
