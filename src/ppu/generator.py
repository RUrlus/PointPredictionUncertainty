# SPDX-FileCopyrightText: 2024-present Ralph Urlus <rurlus.dev@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal

from ppu.viz.ellipse import plot_ellipse_from_cov


class BivariateGaussian:
    def __init__(
        self,
        loc: Sequence[int | float] | None = None,
        scale: Sequence[int | float] | None = None,
        cov: float | NDArray = None,
        rng: int | np.random.Generator | None = None,
    ) -> BivariateGaussian:
        self.loc = np.asarray(loc or (0.0, 0.0), dtype=np.float64)
        self.scale = np.asarray(scale or (1.0, 1.0), dtype=np.float64)
        self.var = np.power(self.scale, 2)
        if self.loc.size != 2 or self.scale.size != 2:
            msg = "`loc` and `scale` must have length 2."
            raise ValueError(msg)

        if isinstance(cov, (int, float)):
            if not (-1.0 <= cov <= 1.0):
                msg = "`cov` must be in [-1., 1.] if a scalar."
                raise ValueError(msg)
            self.rho = cov
            self.cov = np.zeros((2, 2))
            self.cov[0, 0] = self.var[0]
            self.cov[1, 1] = self.var[1]
            self.cov[(0, 1), (1, 0)] = self.rho * np.prod(self.scale)
        elif isinstance(cov, NDArray) and cov.shape == (2, 2):
            self.scale = np.asarray(np.diag(cov), dtype=np.float64)
            self.var = np.power(self.scale, 2)
            self.cov = cov
            self.rho = self.cov[0, 1] / np.prod(self.scale)
        else:
            msg = "`cov` must be an scalar (correlation) or 2x2 NDArray"
            raise TypeError(msg)
        if isinstance(rng, int):
            self.rng = np.random.Generator(np.random.PCG64DXSM(rng))
        elif isinstance(rng, np.random.Generator):
            self.rng = rng
        else:
            msg = "`rng` must be an int or a np.random.Generator"
            raise TypeError(msg)

        self.dist = multivariate_normal(mean=self.loc, cov=self.cov, allow_singular=False, seed=self.rng)

    def rvs(self, size):
        return self.dist.rvs(size=size, random_state=self.rng)

    def plot(self, levels: Sequence[int | float] | None = None, filled: bool = False, **kwargs):
        return plot_ellipse_from_cov(self.loc, self.cov, levels, filled=filled, **kwargs)
