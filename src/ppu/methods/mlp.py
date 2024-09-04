from __future__ import annotations

from copy import deepcopy
from math import floor
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from ppu.methods.bregman import BI_LSE

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MLP:
    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: list[int] = [100, 1],
        device: str = "cpu",
        lr: float = 1e-2,
        weight_decay: float = 0,
        criterion: Callable = F.binary_cross_entropy_with_logits,
        patience: int = 5,
        frequency: int = 1,
        checkpoint_patience: int = 1,
        test_size: float = 0.3,
    ) -> None:
        """Multi Layer perceptron.

        Args:
            in_channels: dimension of the inputs
            hidden_channels: size of layers
            device: the device to run model on, e.g. {"cpu", "cuda"}
            lr: learning rate
            weight_decay: weight decay/L2 penalty
            criterion: cost/loss function
            patience: number of iterations without improvement before terminating
            frequency: number of iterations before evaluating on validation set
            checkpoint_patience: number of iterations the model needs to improve before creating checkpoint.
            test_size: percentage of input data to use as holdout validation set, only used in for early stopping.
        """
        self.device = device
        self.model = torchvision.ops.MLP(in_channels=in_channels, hidden_channels=hidden_channels)
        self.model.to(self.device)
        self.criterion = criterion
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self._opt_lr = lr
        self._opt_weight_decay = weight_decay
        self.patience = patience
        self.fitted_ = False
        self.test_size = test_size
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.frequency = frequency

        self.checkpoint_patience = checkpoint_patience

        self._pred_from_output = self._pred_from_single if self.hidden_channels[-1] == 1 else self._pred_from_mutli

    def _arr_to_device(self, x: NDArray | torch.Tensor, dtype=None) -> torch.Tensor:
        if dtype:
            return (x if isinstance(x, torch.Tensor) else torch.from_numpy(x)).to(dtype=dtype, device=self.device)
        return (x if isinstance(x, torch.Tensor) else torch.from_numpy(x)).to(device=self.device)

    def _pred_from_single(self, outputs) -> torch.Tensor:
        return torch.gt(outputs.data, 0).squeeze(-1)

    def _pred_from_mutli(self, outputs) -> torch.Tensor:
        return outputs.data.argmax(-1)

    def fit(self, X: NDArray | torch.Tensor, y: NDArray | torch.Tensor, n_iter: int | None = None):
        X = X if isinstance(X, torch.Tensor) else torch.from_numpy(X)
        y = y if isinstance(y, torch.Tensor) else torch.from_numpy(y)

        if n_iter is not None:
            X_train = X.to(dtype=torch.float32, device=self.device)
            y_train = y.to(dtype=torch.float32, device=self.device)
            for _ in range(n_iter):
                self.train_epoch(X, y)
            self.fitted_ = True
            return

        n_samples = y.size(0)
        sample_idx = torch.ones(n_samples).multinomial(n_samples - 1, replacement=False)
        n_test_samples = floor(n_samples * self.test_size)

        X_train = X[sample_idx[:-n_test_samples]].to(dtype=torch.float32, device=self.device)
        y_train = y[sample_idx[:-n_test_samples]].to(dtype=torch.float32, device=self.device)

        X_val = X[sample_idx[-n_test_samples:]].to(dtype=torch.float32, device=self.device)
        y_val = y[sample_idx[-n_test_samples:]].to(dtype=torch.float32, device=self.device)

        acc_best = 0
        patience_akk = 0
        check_patience_akk = 0
        while True:
            iters_cnt = 0
            # run for 'frequency' number of times before validating
            while iters_cnt < self.frequency:
                iters_cnt += 1
                self.train_epoch(X_train, y_train)

            acc = self.test(X_val, y_val).detach().cpu().numpy()

            # if our accuracy goes down, increment the patience accumulator
            if acc_best > acc or np.isclose(acc_best, acc):
                patience_akk += 1
            # or reset the accumulator and the best accuracy
            else:
                acc_best = acc
                patience_akk = 0
                check_patience_akk += 1
                if check_patience_akk >= self.checkpoint_patience:
                    model_checkpoint = deepcopy(self.model.state_dict())
                    check_patience_akk = 0

            if patience_akk >= self.patience:
                self.model.load_state_dict(model_checkpoint)
                return

    def train_epoch(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        # criterion -> loss; loss.backward
        self.criterion(self.model(X).squeeze(-1), y).backward()
        self.optimizer.step()

    def weighted_train_epoch(self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> None:
        self.model.train()
        self.optimizer.zero_grad()
        # criterion -> loss; loss.backward
        (self.criterion(self.model(X).squeeze(-1), y, reduction="none") * w).mean().backward()
        self.optimizer.step()

    def element_loss(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.criterion(self.model(X).squeeze(-1), y, reduction="none").detach()

    def weighted_element_loss(self, X: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return (self.criterion(self.model(X).squeeze(-1), y, reduction="none") * w).detach()

    def test(self, X: torch.Tensor, y: torch.Tensor, threshold: float = 0.5):
        self.model.eval()
        with torch.no_grad():
            pred = self._pred_from_output(self.model(X))
        return torch.sum(torch.eq(pred, y)) / y.shape[0]

    def predict_proba(self, X: NDArray | torch.Tensor) -> NDArray:
        X = self._arr_to_device(X, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():  # Ensures that no gradients are computed
            # Apply sigmoid to convert logits to probabilities
            return torch.sigmoid(self.model(X).squeeze(-1)).detach().cpu().numpy()

    def score(self, X: NDArray | torch.Tensor, y: NDArray | torch.Tensor, threshold: float = 0.5) -> NDArray:
        X = self._arr_to_device(X, dtype=torch.float32)
        y = self._arr_to_device(y, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():  # Ensures that no gradients are computed
            # Apply sigmoid to convert logits to probabilities
            proba = torch.sigmoid(self.model(X).squeeze(-1)).detach()
        return torch.eq(torch.ge(proba, threshold), y).cpu().numpy().astype(int)

    def estimate_dropout_BI(self, X: NDArray | torch.Tensor, dropout=0.5, n_ens=10):
        X = self._arr_to_device(X)
        self.model.train()
        checkpoint = deepcopy(self.model.state_dict())
        self.model = torchvision.ops.MLP(
            in_channels=self.in_channels, hidden_channels=self.hidden_channels, dropout=dropout
        )
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint)

        with self.no_grad():
            preds = [self.model(X).squeeze(-1).detach().cpu().numpy() for _ in range(n_ens)]
        preds = np.array(preds).T
        return np.array([BI_LSE(zs, bound="lower") for zs in preds])
