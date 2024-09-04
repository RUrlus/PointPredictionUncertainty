from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

from ppu.methods.utils import view_as_blocks
from ppu.viz.scatter import plot_dense_binary_scatter

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ppu.methods.mlp import MLP


class TracIn:
    def __init__(
        self,
        mlp: MLP,
        X_train: NDArray,
        y_train: NDArray,
        grid_border: float = 1.0,
        n_points: int = 30,
        n_ticks: int = 1000,
        rng: np.random.Generator | None = None,
        reset_optimizer: bool = False,
    ) -> None:
        self.rng = rng or np.random.Generator(np.random.PCG64DXSM())
        self.reset_optimizer = reset_optimizer
        self.mlp = copy.deepcopy(mlp)
        self.device = self.mlp.device
        self.model = self.mlp.model
        self.og_model_state = copy.deepcopy(self.model.state_dict())
        self.og_opt_state = copy.deepcopy(self.mlp.optimizer.state_dict())
        self.X_train = X_train
        self.y_train = y_train
        self.X_idx = np.arange(X_train.shape[0])
        self._create_grids(self.X_train, n_points, n_ticks, grid_border)

    def plot(
        self,
        values: NDArray,
        overlay: bool = False,
        ax=None,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str | None = None,
        **kwargs,
    ):
        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 8))
        else:
            fig = ax.get_figure()

        if values.shape != (self.n_ticks, self.n_ticks):
            values = values.reshape(self.n_ticks, self.n_ticks)

        vmin = vmin if vmin is not None else values.min()
        vmax = vmax if vmax is not None else values.max()
        cmap = cmap or ("Blues_r" if not overlay else "Greys_r")
        c = ax.pcolormesh(self.x_axis, self.y_axis, values, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.95, **kwargs)
        # set the limits of the plot to the limits of the data
        fig.colorbar(c, ax=ax)

        if overlay:
            ax = plot_dense_binary_scatter(X=self.X_train, y=self.y_train, ax=ax, alpha=0.3)
        fig.tight_layout()
        return ax

    def _restore_og_model_state(self):
        self.model.load_state_dict(self.og_model_state)
        self.mlp.optimizer.load_state_dict(self.og_opt_state)

    def _create_grids(self, X: NDArray, n_points: int, n_ticks: int, border: float):
        self.grid_border = border
        x_min, y_min = X.min(0) - border
        x_max, y_max = X.max(0) + border

        # tick per point
        self.tpp = n_ticks // n_points
        self.n_ticks = self.tpp * n_points

        self.n_grid_points = n_points**2
        self.n_grid_ticks = self.n_ticks**2
        self.n_block_ticks = self.tpp**2

        self.x_axis = np.linspace(x_min, x_max, self.n_ticks)
        self.y_axis = np.linspace(y_min, y_max, self.n_ticks)

        self.x_mesh, self.y_mesh = np.meshgrid(self.x_axis, self.y_axis)
        midx = np.arange(self.x_mesh.size).reshape(self.x_mesh.shape)
        block_idx = view_as_blocks(midx, (self.tpp, self.tpp))
        self.block_idx_rvl = block_idx.ravel()

        x_mesh_blocked = self.x_mesh.ravel()[self.block_idx_rvl]
        y_mesh_blocked = self.y_mesh.ravel()[self.block_idx_rvl]

        self.block_X_grid = np.c_[x_mesh_blocked, y_mesh_blocked]
        self.mesh_X_grid = np.c_[self.x_mesh.ravel(), self.y_mesh.ravel()]

        self.point_X_grid = np.c_[
            np.median(x_mesh_blocked.reshape(self.n_grid_points, self.n_block_ticks), 1),
            np.median(y_mesh_blocked.reshape(self.n_grid_points, self.n_block_ticks), 1),
        ]

    def _grid_loss(self, X_t: torch.Tensor, y_t: torch.Tensor) -> NDArray:
        return self.mlp.criterion(self.model(X_t).squeeze(-1), y_t, reduction="none").detach().cpu().numpy()

    def _max_grid_loss(self, X_t: torch.Tensor, y0_t: torch.Tensor, y1_t: torch.Tensor) -> NDArray:
        outp = self.model(X_t).squeeze(-1)
        return (
            torch.maximum(
                self.mlp.criterion(outp, y0_t, reduction="none"), self.mlp.criterion(outp, y1_t, reduction="none")
            )
            .detach()
            .cpu()
            .numpy()
        )

    def loss_over_grid(self, X: NDArray, y: int | NDArray):
        # move to torch and device
        if isinstance(y, (int, float)):
            y = torch.full(size=(self.X.shape[0], 1), fill_value=y, dtype=torch.float32, device=self.device)
        else:
            y = torch.from_numpy(y).to(dtype=torch.float32, device=self.device)

        return (
            self.mlp.criterion(
                self.model(torch.from_numpy(X).to(dtype=torch.float32, device=self.device)), y, reduction="none"
            )
            .detach()
            .cpu()
            .numpy()
        )

    def _fit_point_early_stopping(self, label: float):
        p_iter = 0
        iter_cnt = 0
        min_loss = 1e6
        self.y_batch_t[-1] = label
        self.model.load_state_dict(self.model_state)
        self.mlp.optimizer.load_state_dict(self.opt_state)
        while iter_cnt < self.n_iters_:
            self.mlp.train_epoch(self.X_batch_t, self.y_batch_t)
            batch_loss = (
                self.mlp.criterion(self.model(self.X_batch_t).squeeze(-1), self.y_batch_t).detach().cpu().numpy()
            )
            if batch_loss < min_loss:
                min_loss = batch_loss
                self.checkpoint = copy.deepcopy(self.model.state_dict())
            self.loss_store[p_iter] = batch_loss
            p_iter += 1
            if iter_cnt % self.patience_ == 0:
                if self.loss_store.min() > min_loss:
                    break
                p_iter = 0
            iter_cnt += 1
        self.model.load_state_dict(self.checkpoint)
        return iter_cnt

    def _create_batch(self, batch_size):
        ridx = self.rng.choice(self.X_idx, size=batch_size, replace=False)
        # fine-tuning tensors
        self.X_batch_t = torch.empty((batch_size + 1, 2), device=self.device, dtype=torch.float32)
        self.X_batch_t[:batch_size] = torch.from_numpy(self.X_train[ridx]).to(dtype=torch.float32)
        self.y_batch_t = torch.empty(batch_size + 1, device=self.device, dtype=torch.float32)
        self.y_batch_t[:batch_size] = torch.from_numpy(self.y_train[ridx]).to(dtype=torch.float32)

    def _reset_store_state(self):
        # # reset optimizer
        if self.reset_optimizer:
            self.mlp.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.mlp._opt_lr, weight_decay=self.mlp._opt_weight_decay
            )
        self.model.to(self.device)
        self.model_state = copy.deepcopy(self.model.state_dict())
        self.opt_state = copy.deepcopy(self.mlp.optimizer.state_dict())

    def trace_points(self, batch_size: int = 500, n_iters: int = 40, patience: int = 5):
        self._restore_og_model_state()
        self.n_iters_ = n_iters
        self.patience_ = patience
        self.loss_store = np.empty(patience, dtype=np.float32)

        self._create_batch(batch_size)

        point_X_grid_t = torch.from_numpy(self.point_X_grid).to(self.device, dtype=torch.float32)
        block_X_grid_t = torch.from_numpy(self.block_X_grid).to(self.device, dtype=torch.float32)
        loss_y0_t = torch.zeros(size=(self.n_block_ticks,), dtype=torch.float32, device=self.device)
        loss_y1_t = torch.ones(size=(self.n_block_ticks,), dtype=torch.float32, device=self.device)

        # -- Initial losses --
        mesh_X_grid_t = torch.from_numpy(self.mesh_X_grid).to(self.device, dtype=torch.float32)
        initial_loss_0 = self._grid_loss(
            mesh_X_grid_t, torch.zeros(size=(self.n_grid_ticks,), dtype=torch.float32, device=self.device)
        ).reshape(self.n_ticks, self.n_ticks)
        initial_loss_1 = self._grid_loss(
            mesh_X_grid_t, torch.ones(size=(self.n_grid_ticks,), dtype=torch.float32, device=self.device)
        ).reshape(self.n_ticks, self.n_ticks)
        del mesh_X_grid_t

        self._reset_store_state()

        grid_l0 = np.empty(self.n_grid_ticks)
        grid_l1 = np.empty_like(grid_l0)

        for i in tqdm(range(self.n_grid_points), total=self.n_grid_points):
            self.X_batch_t[-1, :] = point_X_grid_t[i, :]
            lidx = i * self.n_block_ticks
            uidx = lidx + self.n_block_ticks
            grid_X_block_T = block_X_grid_t[lidx:uidx]
            order_index_block = self.block_idx_rvl[lidx:uidx]
            self._fit_point_early_stopping(label=0.0)
            grid_l1[order_index_block] = self.mlp.element_loss(grid_X_block_T, loss_y0_t).cpu().numpy()
            self._fit_point_early_stopping(label=1.0)
            grid_l1[order_index_block] = self.mlp.element_loss(grid_X_block_T, loss_y1_t).cpu().numpy()

        return (
            grid_l0.reshape(self.n_ticks, self.n_ticks),
            grid_l1.reshape(self.n_ticks, self.n_ticks),
            initial_loss_0,
            initial_loss_1,
        )

    def trace_point(self, X, batch_size: int = 500, n_iters: int = 40, weight: float = 1.0):
        self._restore_og_model_state()
        self.n_iters_ = n_iters
        iters_p1 = n_iters + 1

        self._create_batch(batch_size)
        self.X_batch_t[-1, :] = torch.from_numpy(np.asarray(X)).to(dtype=torch.float32, device=self.device)

        weight_t = torch.ones(batch_size + 1, device=self.device, dtype=torch.float32)
        weight_t[-1] = weight
        weight_t *= weight_t.nelement() / weight_t.sum()

        mesh_X_grid_t = torch.from_numpy(self.mesh_X_grid).to(self.device, dtype=torch.float32)
        loss_y0_t = torch.zeros(size=(self.n_grid_ticks,), dtype=torch.float32, device=self.device)
        loss_y1_t = torch.ones(size=(self.n_grid_ticks,), dtype=torch.float32, device=self.device)

        l0_loss_store = np.zeros((iters_p1, batch_size + 1), dtype=np.float32)
        l1_loss_store = np.zeros_like(l0_loss_store)
        grid_loss_l0 = np.zeros((iters_p1, self.n_grid_ticks), dtype=np.float32)
        grid_loss_l1 = np.zeros_like(grid_loss_l0)

        # -- Initial losses --
        grid_loss_l0[0, :] = self.mlp.element_loss(mesh_X_grid_t, loss_y0_t).cpu().numpy()
        grid_loss_l1[0, :] = self.mlp.element_loss(mesh_X_grid_t, loss_y1_t).cpu().numpy()
        # loss over batch
        self.y_batch_t[-1] = 0.0
        l0_loss_store[0, :] = self.mlp.element_loss(self.X_batch_t, self.y_batch_t).cpu().numpy()
        # loss over batch
        self.y_batch_t[-1] = 1.0
        l1_loss_store[0, :] = self.mlp.element_loss(self.X_batch_t, self.y_batch_t).cpu().numpy()

        self._reset_store_state()

        # iterate with label 0
        self.y_batch_t[-1] = 0.0
        for i in tqdm(range(1, n_iters + 1), total=n_iters):
            self.mlp.weighted_train_epoch(self.X_batch_t, self.y_batch_t, weight_t)
            l0_loss_store[i, :] = self.mlp.element_loss(self.X_batch_t, self.y_batch_t).cpu().numpy()
            grid_loss_l0[i, :] = self.mlp.element_loss(mesh_X_grid_t, loss_y0_t).cpu().numpy()

        # iterate with label 1
        self.y_batch_t[-1] = 1.0
        self.model.load_state_dict(self.model_state)
        self.mlp.optimizer.load_state_dict(self.opt_state)
        for i in tqdm(range(1, n_iters + 1), total=n_iters):
            self.mlp.weighted_train_epoch(self.X_batch_t, self.y_batch_t, weight_t)
            l1_loss_store[i, :] = self.mlp.element_loss(self.X_batch_t, self.y_batch_t).cpu().numpy()
            grid_loss_l1[i, :] = self.mlp.element_loss(mesh_X_grid_t, loss_y1_t).cpu().numpy()
        return grid_loss_l0, grid_loss_l1, l0_loss_store, l1_loss_store
