"""ANFIS experiments: driver-based model, time-series model, and
one-factor-at-a-time (OFAT) sensitivity analysis.

Every experiment
  * splits the series chronologically into train / validation / test,
  * fits the scalers on the training block only,
  * early-stops on the validation block,
  * evaluates once on the test block, in original units,
  * repeats over several seeds and reports mean and standard deviation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics
from .anfis import fit_anfis, predict
from .data import DRIVERS
from .split import MinMax, time_split


def _xy_driver(df: pd.DataFrame, target: str, inputs: list[str], horizon: int):
    x = df[inputs].to_numpy(float)
    y = df[target].to_numpy(float)
    if horizon > 0:  # drivers on day t -> target on day t + horizon
        x, y = x[:-horizon], y[horizon:]
    return x, y


def _xy_window(df: pd.DataFrame, target: str, window: int):
    s = df[target].to_numpy(float)
    x = np.stack([s[i - window : i] for i in range(window, len(s))])
    y = s[window:]
    return x, y


def _run(x, y, cfg: dict, seed: int):
    sp = time_split(len(y), cfg["test_frac"], cfg["val_frac"])
    sx, sy = MinMax().fit(x[sp.train]), MinMax().fit(y[sp.train][:, None])
    z = lambda a: sx.transform(a)
    zy = lambda a: sy.transform(a[:, None]).ravel()
    res = fit_anfis(
        z(x[sp.train]), zy(y[sp.train]), z(x[sp.val]), zy(y[sp.val]),
        n_rules=cfg["n_rules"], lr=cfg["lr"], epochs=cfg["epochs"],
        order=cfg.get("order", 1), patience=cfg.get("patience", 50), seed=seed,
    )
    out = {}
    for part in ("train", "test"):
        sl = getattr(sp, part)
        p = sy.inverse(predict(res.model, z(x[sl]))[:, None]).ravel()
        out[part] = metrics.evaluate(y[sl], p)
    out["best_epoch"] = res.best_epoch
    return out


def _summarise(runs: list[dict], part: str = "test") -> dict:
    keys = runs[0][part].keys()
    return {k: (float(np.mean([r[part][k] for r in runs])), float(np.std([r[part][k] for r in runs]))) for k in keys}


def driver_model(df: pd.DataFrame, target: str, cfg: dict, inputs: list[str] | None = None) -> dict:
    """HR from the eight indicators; DR from the eight indicators plus HR."""
    if inputs is None:
        inputs = list(DRIVERS) + (["HR"] if target == "DR" else [])
    x, y = _xy_driver(df, target, inputs, cfg.get("horizon", 0))
    runs = [_run(x, y, cfg, s) for s in cfg["seeds"]]
    return {"inputs": inputs, "train": _summarise(runs, "train"), "test": _summarise(runs, "test"), "runs": runs}


def timeseries_model(df: pd.DataFrame, target: str, cfg: dict) -> dict:
    """One-step-ahead ANFIS on the past `window` days of the target only."""
    x, y = _xy_window(df, target, cfg.get("window", 30))
    runs = [_run(x, y, cfg, s) for s in cfg["seeds"]]
    return {"window": cfg.get("window", 30), "train": _summarise(runs, "train"), "test": _summarise(runs, "test"), "runs": runs}


def ofat_sensitivity(df: pd.DataFrame, target: str, cfg: dict) -> pd.DataFrame:
    """Drop one input at a time, refit, and compare test error with the full
    model (same seeds). A larger increase in RMSE means the fitted model leans
    more on that input. This ranks inputs by their importance to the fitted
    model; it is not an estimate of the causal effect of an intervention."""
    full = driver_model(df, target, cfg)
    base_rmse = np.array([r["test"]["RMSE"] for r in full["runs"]])
    rows = []
    for drop in full["inputs"]:
        inputs = [c for c in full["inputs"] if c != drop]
        red = driver_model(df, target, cfg, inputs=inputs)
        rmse = np.array([r["test"]["RMSE"] for r in red["runs"]])
        delta = rmse - base_rmse
        rows.append({
            "dropped_input": drop,
            "test_RMSE_mean": rmse.mean(),
            "delta_RMSE_mean": delta.mean(),
            "delta_RMSE_sd": delta.std(),
            "delta_RMSE_pct": 100 * delta.mean() / base_rmse.mean(),
        })
    return pd.DataFrame(rows).sort_values("delta_RMSE_mean", ascending=False).reset_index(drop=True)
