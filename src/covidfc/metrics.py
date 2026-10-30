"""Point-forecast metrics used in the papers.

All functions take 1-D arrays of observed (y) and predicted (p) values in the
original units of the target.
"""
from __future__ import annotations

import numpy as np


def _arr(y, p):
    y = np.asarray(y, dtype=float).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {p.shape}")
    return y, p


def mae(y, p):
    y, p = _arr(y, p)
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    y, p = _arr(y, p)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mape(y, p, eps: float = 1e-8):
    """Mean absolute percentage error (%). Days with y == 0 are skipped,
    because the percentage error is undefined there."""
    y, p = _arr(y, p)
    mask = np.abs(y) > eps
    if not mask.any():
        return float("nan")
    return float(100 * np.mean(np.abs((y[mask] - p[mask]) / y[mask])))


def smape(y, p, eps: float = 1e-8):
    y, p = _arr(y, p)
    denom = (np.abs(y) + np.abs(p)) / 2
    mask = denom > eps
    return float(100 * np.mean(np.abs(y[mask] - p[mask]) / denom[mask]))


def r2(y, p):
    y, p = _arr(y, p)
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def nse(y, p):
    """Nash-Sutcliffe efficiency. Numerically equal to R^2 computed against
    the mean of the observations; reported separately for comparability with
    the hydrology-style tables in the papers."""
    return r2(y, p)


def willmott_index(y, p):
    """Willmott's index of agreement (d), in [0, 1]."""
    y, p = _arr(y, p)
    ybar = y.mean()
    num = np.sum((y - p) ** 2)
    den = np.sum((np.abs(p - ybar) + np.abs(y - ybar)) ** 2)
    return float(1 - num / den) if den > 0 else float("nan")


def pbias(y, p):
    """Percent bias (%). Positive = model under-predicts on average."""
    y, p = _arr(y, p)
    s = np.sum(y)
    return float(100 * np.sum(y - p) / s) if s != 0 else float("nan")


ALL = {
    "MAE": mae,
    "RMSE": rmse,
    "MAPE": mape,
    "sMAPE": smape,
    "R2": r2,
    "NSE": nse,
    "WI": willmott_index,
    "PBIAS": pbias,
}


def evaluate(y, p, which=None) -> dict:
    which = which or list(ALL)
    return {k: ALL[k](y, p) for k in which}
