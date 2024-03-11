# SPDX-FileCopyrightText: 2024-present Ralph Urlus <rurlus.dev@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

from ppu.viz.ellipse import plot_ellipse_from_cov

if TYPE_CHECKING:
    from numpy.typing import NDArray

class _2dBaseGenerator:
    def _set_loc_scale(self, loc, scale, def_loc=(0.0, 0.0), def_scale=(1.0, 1.0)):
        self.loc = np.asarray(loc or def_loc, dtype=np.float64)
        self.scale = np.asarray(scale or def_scale, dtype=np.float64)
        self.var = np.power(self.scale, 2)
        if self.loc.size != 2 or self.scale.size != 2:
            msg = "`loc` and `scale` must have length 2."
            raise ValueError(msg)

    def _set_rng(self, rng):
        if isinstance(rng, int) or rng is None:
            self.rng = np.random.Generator(np.random.PCG64DXSM(rng))
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            msg = "`rng` must be an int or a np.random.Generator"
            raise TypeError(msg)


class BivariateGaussian(_2dBaseGenerator):
    def __init__(
        self,
        loc: Sequence[int | float] | None = None,
        scale: Sequence[int | float] | None = None,
        cov: float | NDArray = None,
        rng: int | np.random.Generator | None = None,
    ) -> BivariateGaussian:
        self._set_loc_scale(loc, scale)
        if isinstance(cov, (int, float)):
            if not (-1.0 <= cov <= 1.0):
                msg = "`cov` must be in [-1., 1.] if a scalar."
                raise ValueError(msg)
            self.rho = cov
            self.cov = np.zeros((2, 2))
            self.cov[0, 0] = self.var[0]
            self.cov[1, 1] = self.var[1]
            self.cov[(0, 1), (1, 0)] = self.rho * np.prod(self.scale)
        elif isinstance(cov, np.ndarray) and cov.shape == (2, 2):
            self.scale = np.asarray(np.diag(cov), dtype=np.float64)
            self.var = np.power(self.scale, 2)
            self.cov = cov
            self.rho = self.cov[0, 1] / np.prod(self.scale)
        else:
            msg = "`cov` must be an scalar (correlation) or 2x2 NDArray"
            raise TypeError(msg)

        self._set_rng(rng)
        self.dist = multivariate_normal(mean=self.loc, cov=self.cov, allow_singular=False, seed=self.rng)

    def rvs(self, size: int) -> NDArray:
        """Generate samples from Bivariate Gaussian.

        Args:
            size: the sample size to generate

        Returns:
            X: 2D array containing the samples
        """
        return self.dist.rvs(size=size, random_state=self.rng)

    def plot(
        self, levels: Sequence[int | float] | None = None, filled: bool = False, ax: Axes | None = None, **kwargs
    ) -> Axes:
        """Plot DGP.

        Args:
            levels: if int(s) levels is treated as the number of standard deviations for the confidence interval.
                If float(s) it is taken to be the density to be contained in the confidence interval
                By default we plot 1, 2 and 3 std deviations
            filled: use filled contours or line plot
            ax: the Axes to plot on
            **kwargs: keyword arguments passed on to `ppu.viz.plot_ellipse_from_cov`
        """
        return plot_ellipse_from_cov(self.loc, self.cov, levels, filled=filled, ax=ax, **kwargs)

    def plot_rvs(self, sample: int | NDArray, ax: Axes | None = None, **kwargs) -> Axes:
        """Plot sample or generate and plot sample.

        Args:
            sample: the sample size if integer and otherwise a previously generated sample using `BivariateGaussian.rvs`
            ax: the Axes to plot on
            **kwargs: keyword arguments passed on to `ppu.viz.plot_dense_scatter`
        """
        if isinstance(sample, int):
            sample = self.rvs(sample)
        elif not (isinstance(sample, np.ndarray) and sample.ndim == 2):
            msg = "`sample` should be a integer or an 2D ndarray."
            raise TypeError(msg)
        return plot_dense_scatter(sample, ax=ax**kwargs)


