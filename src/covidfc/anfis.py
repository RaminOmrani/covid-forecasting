"""Adaptive Neuro-Fuzzy Inference System (Takagi-Sugeno type) in PyTorch.

Structure (Jang, 1993):
  layer 1  Gaussian membership value of each input for each rule
  layer 2  rule firing strength = product over inputs
  layer 3  normalised firing strength
  layer 4  rule consequent, zero-order (constant) or first-order (linear in x)
  layer 5  weighted sum

Each rule owns its own membership function for every input (no grid
partition), so the number of rules does not grow exponentially with the
number of inputs. All parameters are trained jointly by gradient descent
(Adam) with early stopping on a validation block.

This is an independent implementation written for this repository.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class ANFIS(nn.Module):
    def __init__(self, n_inputs: int, n_rules: int, order: int = 1, init_x: np.ndarray | None = None, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        if init_x is not None and len(init_x) >= n_rules:
            # Initialise centres on randomly chosen training points.
            idx = torch.randperm(len(init_x), generator=g)[:n_rules]
            mu = torch.as_tensor(np.asarray(init_x)[idx.numpy()], dtype=torch.float32)
        else:
            mu = torch.rand(n_rules, n_inputs, generator=g)
        self.mu = nn.Parameter(mu)
        self.log_sigma = nn.Parameter(torch.full((n_rules, n_inputs), float(np.log(0.3))))
        self.order = order
        if order == 1:
            self.coef = nn.Parameter(0.01 * torch.randn(n_rules, n_inputs, generator=g))
        elif order != 0:
            raise ValueError("order must be 0 or 1")
        self.bias = nn.Parameter(0.01 * torch.randn(n_rules, generator=g))

    def firing(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma)
        # log of the product of Gaussians = sum of log-memberships (stable).
        log_w = -0.5 * (((x[:, None, :] - self.mu[None]) / sigma[None]) ** 2).sum(-1)
        return torch.softmax(log_w, dim=1)  # normalised firing strengths

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.firing(x)
        f = self.bias[None, :]
        if self.order == 1:
            f = f + x @ self.coef.T
        return (w * f).sum(1)


@dataclass
class FitResult:
    model: ANFIS
    train_loss: list
    val_loss: list
    best_epoch: int


def fit_anfis(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_rules: int = 500,
    lr: float = 0.015,
    epochs: int = 500,
    order: int = 1,
    patience: int = 50,
    seed: int = 0,
) -> FitResult:
    """Full-batch training with Adam; keeps the weights of the epoch with the
    lowest validation MSE. Inputs and targets are expected to be scaled."""
    torch.manual_seed(seed)
    n_rules = min(n_rules, len(x_tr))
    model = ANFIS(x_tr.shape[1], n_rules, order=order, init_x=x_tr, seed=seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xt, yt = torch.as_tensor(x_tr, dtype=torch.float32), torch.as_tensor(y_tr, dtype=torch.float32)
    xv, yv = torch.as_tensor(x_val, dtype=torch.float32), torch.as_tensor(y_val, dtype=torch.float32)
    best, best_state, best_epoch, since = float("inf"), None, 0, 0
    tr_hist, val_hist = [], []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        loss = torch.mean((model(xt) - yt) ** 2)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vl = torch.mean((model(xv) - yv) ** 2).item()
        tr_hist.append(loss.item())
        val_hist.append(vl)
        if vl < best - 1e-9:
            best, best_state, best_epoch, since = vl, copy.deepcopy(model.state_dict()), ep, 0
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    return FitResult(model, tr_hist, val_hist, best_epoch)


def predict(model: ANFIS, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).numpy()
