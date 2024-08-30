from __future__ import annotations

import copy
from math import floor
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from torch import Tensor

    from ppu.methods.mlp import MLP


def get_loss_over_grid(X: NDArray | Tensor, y: int | NDArray[int] | Tensor[int], mlp: MLP) -> NDArray[float]:
    """Get loss of `mlp` over every point in `X_test`.

    Args:
        X: the points to test
        y: the label(s) to evaluate for `X` with
        mlp: trained classifier

    Returns:
        loss: array of shape (X.shape[0])

    """
    # move to torch and device
    if isinstance(y, (int, float)):
        y = torch.full(size=(X.shape[0], 1), fill_value=y, dtype=torch.float32, device=mlp.device)
    else:
        y = torch.from_numpy(y).to(dtype=torch.float32, device=mlp.device)

    return (
        mlp.criterion(mlp.model(torch.from_numpy(X).to(dtype=torch.float32, device=mlp.device)), y, reduction="none")
        .detach()
        .cpu()
        .numpy()
    )


def get_proba_over_grid(X: NDArray | Tensor, mlp: MLP) -> NDArray[float]:
    """Get proba of `mlp` over every point in `X_test`.

    Args:
        X: the points to evaluate
        mlp: trained classifier

    Returns:
        loss: array of shape (X_test.shape[0])

    """
    return (
        torch.sigmoid(mlp.model(torch.from_numpy(X).to(dtype=torch.float32, device=mlp.device))).detach().cpu().numpy()
    )
def get_random_resampled_tracin(
    X_test: NDArray | Tensor,
    y_test: int | NDArray[int] | Tensor[int],
    X_train: NDArray | Tensor,
    y_train: NDArray | Tensor,
    mlp: MLP,
    n_iter: int = 1,
    batch_size: int | float = 10,
    rng: np.random.Generator | None = None,
) -> NDArray[float]:
    """Compute tracin score for `X_test` point(s) and label(s) `y_test` using a randomly sampled batch for each point.

    Args:
        X_test: the points to test
        y_test: the label(s) to evaluate for `X_test` with
        X_train: the training data to sample the batches from
        y_train: the training labels to sample the batches from
        mlp: trained classifier
        n_iter: number of iterations/epochs the model is trained with the new batch
        batch_size: number or fraction of training data to include in the batch
        rng: random state used for sampling the batch.

    Returns:
        tracin: array of shape (X_test.shape[0], n_iter)

    """
    rng = rng or np.random.Generator(np.random.PCG64DXSM())
    n_points, ncols = X_test.shape
    batch_size = batch_size if isinstance(batch_size, int) else floor(n_points * batch_size)
    x_rows, x_cols = X_train.shape

    mlp.model.to("cpu")
    mlp = copy.deepcopy(mlp)

    # reset optimizer
    mlp.optimizer = torch.optim.Adam(mlp.model.parameters(), lr=mlp._opt_lr, weight_decay=mlp._opt_weight_decay)
    mlp.model.to(mlp.device)

    # make checkpoints
    model_state = copy.deepcopy(mlp.model.state_dict())
    opt_state = copy.deepcopy(mlp.optimizer.state_dict())

    # move to torch and device
    if isinstance(y_test, (int, float)):
        y_test = torch.full(size=(n_points, 1), fill_value=y_test, dtype=torch.float32, device=mlp.device)
    else:
        y_test = torch.from_numpy(y_test).to(dtype=torch.float32, device=mlp.device)

    X_test = torch.from_numpy(X_test).to(dtype=torch.float32, device=mlp.device)
    y_train = torch.from_numpy(y_train).to(dtype=torch.float32, device=mlp.device)
    X_train = torch.from_numpy(X_train).to(dtype=torch.float32, device=mlp.device)

    x_idx = np.arange(x_rows)
    x_batch = torch.empty((batch_size + 1, x_cols), device=mlp.device, dtype=torch.float32)
    y_batch = torch.empty(batch_size + 1, device=mlp.device, dtype=torch.float32)

    tracin = np.empty((n_points, n_iter), dtype=float)
    for i in range(n_points):
        test_point = X_test[i, :]
        test_label = y_test[i]

        # restore state
        mlp.optimizer.load_state_dict(opt_state)
        mlp.model.load_state_dict(model_state)
        loss_before = mlp.criterion(mlp.model(test_point).squeeze(-1), test_label.squeeze(-1)).detach().cpu().numpy()

        # sample the training data
        ridx = rng.choice(x_idx, size=batch_size, replace=False)
        x_batch[:batch_size] = X_train[ridx]
        x_batch[-1] = test_point
        y_batch[:batch_size] = y_train[ridx]
        y_batch[-1] = test_label

        for j in range(n_iter):
            mlp.train_epoch(x_batch, y_batch)
            ll = mlp.criterion(mlp.model(test_point).squeeze(-1), test_label.squeeze(-1)).detach().cpu().numpy()
            tracin[i, j] = loss_before - ll

    return tracin


def get_random_tracin(
    X_test: NDArray | Tensor,
    y_test: int | NDArray[int] | Tensor[int],
    X_train: NDArray | Tensor,
    y_train: NDArray | Tensor,
    mlp: MLP,
    n_iter: int = 1,
    batch_size: int | float = 10,
    rng: np.random.Generator | None = None,
) -> NDArray[float]:
    """Compute tracin score for `X_test` point(s) and label(s) `y_test` using a randomly sampled batch for each point.

    Args:
        X_test: the points to test
        y_test: the label(s) to evaluate for `X_test` with
        X_train: the training data to sample the batches from
        y_train: the training labels to sample the batches from
        mlp: trained classifier
        n_iter: number of iterations/epochs the model is trained with the new batch
        batch_size: number or fraction of training data to include in the batch
        rng: random state used for sampling the batch.

    Returns:
        tracin: array of shape (X_test.shape[0], n_iter)

    """
    rng = rng or np.random.Generator(np.random.PCG64DXSM())
    n_points, ncols = X_test.shape
    x_rows, x_cols = X_train.shape

    mlp = copy.deepcopy(mlp)
    mlp.model.to("cpu")

    # reset optimizer
    mlp.optimizer = torch.optim.Adam(mlp.model.parameters(), lr=mlp._opt_lr, weight_decay=mlp._opt_weight_decay)
    mlp.model.to(mlp.device)

    # make checkpoints
    model_state = copy.deepcopy(mlp.model.state_dict())
    opt_state = copy.deepcopy(mlp.optimizer.state_dict())

    # move to torch and device
    if isinstance(y_test, (int, float)):
        y_test = torch.full(size=(n_points, 1), fill_value=y_test, dtype=torch.float32, device=mlp.device)
    else:
        y_test = torch.from_numpy(y_test).to(dtype=torch.float32, device=mlp.device)

    X_test = torch.from_numpy(X_test).to(dtype=torch.float32, device=mlp.device)
    y_train = torch.from_numpy(y_train).to(dtype=torch.float32, device=mlp.device)
    X_train = torch.from_numpy(X_train).to(dtype=torch.float32, device=mlp.device)

    x_idx = np.arange(x_rows)
    ridx = rng.choice(x_idx, size=batch_size, replace=False)
    x_batch = torch.empty((batch_size + 1, x_cols), device=mlp.device, dtype=torch.float32)
    x_batch[:batch_size] = X_train[ridx]
    y_batch = torch.empty(batch_size + 1, device=mlp.device, dtype=torch.float32)
    y_batch[:batch_size] = y_train[ridx]

    tracin = np.empty((n_points, n_iter), dtype=float)
    for i in range(n_points):
        test_point = X_test[i, :]
        test_label = y_test[i]

        # restore state
        mlp.optimizer.load_state_dict(opt_state)
        mlp.model.load_state_dict(model_state)
        loss_before = mlp.criterion(mlp.model(test_point).squeeze(-1), test_label.squeeze(-1)).detach().cpu().numpy()

        # sample the training data
        x_batch[-1] = test_point
        y_batch[-1] = test_label

        for j in range(n_iter):
            mlp.train_epoch(x_batch, y_batch)
            ll = mlp.criterion(mlp.model(test_point).squeeze(-1), test_label.squeeze(-1)).detach().cpu().numpy()
            tracin[i, j] = loss_before - ll

    return tracin
