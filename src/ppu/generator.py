# SPDX-FileCopyrightText: 2024-present Ralph Urlus <rurlus.dev@gmail.com>
#
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import multivariate_normal

from ppu.viz import plot_dense_binary_scatter, plot_dense_scatter
from ppu.viz.ellipse import plot_ellipse_from_cov

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes
    from numpy.typing import NDArray

__all__ = ["BivariateGaussian", "Circular", "GaussianBlobs", "Moons"]


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

    def plot_rvs(
        self, size: int | None = None, X: NDArray | None = None, y: NDArray | None = None, ax: Axes | None = None
    ):
        """Plot sample or generate and plot sample.

        Args:
            size: the sample size to generate, may be None when `X` and `y` are note None
            X: the feature samples, may be None when size is not None
            y: the corresponding labels to X, may be None when size is not None and X is None
            ax: the Axes to plot on
            **kwargs: keyword arguments passed on to `ppu.viz.plot_dense_scatter`
        """
        if isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
            size = y.size
        elif isinstance(size, int):
            X, y = self.rvs(size)
        else:
            msg = "`size` must not be None when `X` and `y` are None"
            raise TypeError(msg)
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        return plot_dense_binary_scatter(X, y, ax)


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
        return plot_dense_scatter(sample, ax=ax, **kwargs)


class Circular(_2dBaseGenerator):
    def __init__(
        self,
        loc: Sequence[int | float] | None = None,
        scale: Sequence[int | float] | None = None,
        class_sep: float = 1.0,
        class_balance: float = 0.5,
        rng: int | np.random.Generator | None = None,
    ) -> Circular:
        """Instantiate the class.

        Args:
            loc: the means of the noise added to the features, defaults to (0., 0.)
            scale: the standard deviation of the noise added to the features, defaults to (0.25, 0.25)
            class_sep: the distance between circles, defaults to 1.
            class_balance: the ratio of samples to generate for the two respective classes
            rng: seed or random state used to make generation reproducible
        """
        self._set_loc_scale(loc, scale, def_scale=(0.25, 0.25))
        self._set_rng(rng)
        self.class_sep = class_sep
        self.class_balance = class_balance

    def rvs(self, size: int) -> tuple[NDArray, NDArray]:
        """Generate binary classification samples from two Gaussians.

        Args:
            size: the sample size to generate

        Returns:
            X: 2D array containing the feature samples
            y: 1D array containing the labels
        """
        # sample polar coordinates
        angles = self.rng.uniform(low=0, high=2 * np.pi, size=size)
        y = radii = self.rng.binomial(n=1, p=self.class_balance, size=size)
        # transform to cartesian coordinates and add noise
        X = np.empty((size, 2), order="F")
        X[:, 0] = np.sin(angles) * (radii * self.class_sep) + self.rng.normal(
            loc=self.loc[0], scale=self.scale[0], size=size
        )
        X[:, 1] = np.cos(angles) * (radii * self.class_sep) + self.rng.normal(
            loc=self.loc[1], scale=self.scale[1], size=size
        )
        return X, y

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
        y0_loc = (self.loc[0] - self.class_sep, self.loc[1] + self.class_sep)
        ax = plot_ellipse_from_cov(y0_loc, self.cov, levels, filled=filled, cmap="Blues", ax=ax, **kwargs)
        y1_loc = (self.loc[0] + self.class_sep, self.loc[1] - self.class_sep)
        return plot_ellipse_from_cov(y1_loc, self.cov, levels, filled=filled, cmap="Reds", ax=ax, **kwargs)


class Moons(_2dBaseGenerator):
    # Adapted from scikit-learn.
    # sklearn.datasets._samples_generator.make_moons
    def __init__(
        self,
        loc: Sequence[int | float] | None = None,
        scale: Sequence[int | float] | None = None,
        class_sep: float = 0.0,
        class_balance: float = 0.5,
        rng: int | np.random.Generator | None = None,
    ) -> Moons:
        """Instantiate the class.

        Args:
            loc: the means of the noise added to the features, defaults to (0., 0.)
            scale: the standard deviation of the noise added to the features, defaults to (0.25, 0.25)
            class_sep: vertical spacing between the moons, defaults to 0.0.
            class_balance: the ratio of samples to generate for the two respective classes
            rng: seed or random state used to make generation reproducible
        """
        self._set_loc_scale(loc, scale, def_scale=(0.25, 0.25))
        self._set_rng(rng)
        self.class_sep = class_sep
        self.class_balance = class_balance

    def rvs(self, size: int) -> tuple[NDArray, NDArray]:
        """Generate samples from Moons DGP.

        Args:
            size: the sample size to generate

        Returns:
            X: 2D array containing the feature samples
            y: 1D array containing the labels
        """
        if isinstance(size, int):
            size_outer = int(np.floor(size * self.class_balance))
            size_inner = size - size_outer
        else:
            msg = "`size` should be an integer."
            raise TypeError(msg)

        X = np.empty(shape=(size, 2), order="F")
        X[:size_outer, 0] = np.cos(np.linspace(0, np.pi, size_outer))
        X[size_outer:, 0] = 1 - np.cos(np.linspace(0, np.pi, size_inner))

        X[:size_outer, 1] = np.sin(np.linspace(0, np.pi, size_outer))
        X[size_outer:, 1] = 1 - np.sin(np.linspace(0, np.pi, size_inner)) - 0.5

        X += self.rng.normal(self.loc, self.scale, size=(size, 2))
        X[:size_outer, 1] += self.class_sep

        y = np.zeros(size, dtype=int)
        y[size_outer:] += 1

        sidx = self.rng.choice(np.arange(y.size), replace=False, size=y.size)
        X = X[sidx, :].copy()
        y = y[sidx].copy()

        return X, y


class GaussianBlobs(_2dBaseGenerator):
    def __init__(
        self,
        loc: Sequence[int | float] | None = None,
        scale: Sequence[int | float] | None = None,
        class_sep: float = 0.5,
        class_balance: float = 0.5,
        rng: int | np.random.Generator | None = None,
    ) -> GaussianBlobs:
        """Instantiate the class.

        Args:
            loc: the means of the features, defaults to (0., 0.)
            scale: the standard deviation of the features, defaults to (0.5, 0.5)
            class_sep: the distance between the means, additive to `loc`
            class_balance: the ratio of samples to generate for the two respective classes
            rng: seed or random state used to make generation reproducible
        """
        self._set_rng(rng)
        self.class_sep = class_sep
        self._set_loc_scale(loc, scale, def_scale=(0.5, 0.5))
        self.cov = np.zeros((2, 2))
        self.cov[0, 0] = self.var[0]
        self.cov[1, 1] = self.var[1]
        self.class_balance = class_balance

    def rvs(self, size: int) -> tuple[NDArray, NDArray]:
        """Generate binary classification samples from two Gaussians.

        Args:
            size: the sample size to generate

        Returns:
            X: 2D array containing the feature samples
            y: 1D array containing the labels
        """
        y = self.rng.binomial(n=1, p=self.class_balance, size=size)
        X = np.empty((size, 2), order="F")
        X[:, 0] = self.rng.normal(loc=(self.loc[0] + self.class_balance) - y, scale=self.scale[0], size=size)
        X[:, 1] = self.rng.normal(loc=(self.loc[1] - self.class_balance) + y, scale=self.scale[1], size=size)
        return X, y

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
        y0_loc = (self.loc[0] - self.class_sep, self.loc[1] + self.class_sep)
        ax = plot_ellipse_from_cov(y0_loc, self.cov, levels, filled=filled, cmap="Blues", ax=ax, **kwargs)
        y1_loc = (self.loc[0] + self.class_sep, self.loc[1] - self.class_sep)
        return plot_ellipse_from_cov(y1_loc, self.cov, levels, filled=filled, cmap="Reds", ax=ax, **kwargs)


class RingBlobs(_2dBaseGenerator):
    def __init__(
        self,
        n_blobs: int = 6,
        scale: Sequence[int | float] | None = None,
        class_sep: float = 5.6,
        class_balance: float = 0.5,
        rng: int | np.random.Generator | None = None,
    ) -> GaussianBlobs:
        """Instantiate the class.

        The defaults are scaled s.t. that a blob's 2 sigma interval touches but doesn't
        overlap with its neighbours.

        Args:
            n_blobs: the number of Gaussian blobs to generate
            scale: the standard deviation of the features, defaults to (1.4, 1.4)
            class_sep: multiplcative radial scaling of the ring
            class_balance: the ratio of samples to generate for the two respective classes
            rng: seed or random state used to make generation reproducible
        """
        self._set_rng(rng)
        if n_blobs < 2:
            msg = "`n_blobs` must be >= 2"
            raise ValueError(msg)
        self.n_blobs = n_blobs
        self.class_sep = class_sep
        self.class_balance = class_balance
        self._set_loc_scale(None, scale, def_scale=(1.4, 1.4))
        self.cov = np.zeros((2, 2))
        self.cov[0, 0] = self.var[0]
        self.cov[1, 1] = self.var[1]

        self.centers = np.linspace(0, np.pi * 2, self.n_blobs + 1)[:-1]
        self.centroids = np.empty((self.centers.size, 2), order="F")
        self.centroids[:, 0] = self.class_sep * np.sin(self.centers)
        self.centroids[:, 1] = self.class_sep * np.cos(self.centers)
        order = (0, 1) if self.class_balance < 0.5 else (1, 0)
        self.labels = np.tile(order, int(np.ceil(self.n_blobs / 2)))[: self.n_blobs]

    def rvs(self, size: int) -> tuple[NDArray, NDArray]:
        """Generate binary classification samples from two Gaussians.

        Args:
            size: the sample size to generate

        Returns:
            X: 2D array containing the feature samples
            y: 1D array containing the labels
        """
        total_size = size * self.n_blobs
        blob_size = total_size // self.n_blobs

        lb = 0
        ub = blob_size
        X = np.empty((total_size, 2))
        y = np.empty(total_size, dtype=int)
        for i in range(self.centers.size):
            X[lb:ub, 0] = self.rng.normal(self.centroids[i, 0], scale=self.scale[0], size=blob_size)
            X[lb:ub, 1] = self.rng.normal(self.centroids[i, 1], scale=self.scale[1], size=blob_size)
            y[lb:ub] = self.labels[i]
            lb = ub
            ub += blob_size
        return X, y

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
        all_handles = []
        for i in range(self.centers.size):
            loc = self.centroids[i, :]
            cmap = "Blues" if self.labels[i] == 1 else "Reds"
            ax, handles = plot_ellipse_from_cov(
                loc, self.cov, levels, filled=filled, cmap=cmap, ax=ax, **kwargs, return_handles=True
            )
            all_handles.append(handles)

        lines = [
            Line2D([0], [0], linestyle="None", markeredgewidth=3, markersize=2, marker="x", color="C0", label="$y=0$"),
            Line2D([0], [0], linestyle="None", markeredgewidth=3, markersize=2, marker="x", color="C1", label="$y=1$"),
        ]
        ax.legend(handles=lines + all_handles[0][1:] + all_handles[1][1:])
        return ax
