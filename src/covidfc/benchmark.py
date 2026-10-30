"""Benchmark of nine forecasters for one-step-ahead hospitalisation rate.

Families
  classical   SLR, ARIMA, SARIMA
  ensembles   Random Forest, XGBoost, LightGBM
  deep        LSTM, RNN, 1D-CNN

Protocol (identical for every model)
  * chronological train / validation / test split;
  * target on day t is predicted with information up to day t-1 only;
  * tabular models use lag features (lags 1..L, rolling mean and standard
    deviation of the previous 7 days, day of week);
  * ARIMA/SARIMA are refitted on train+val and produce rolling one-step
    forecasts over the test block (observations are appended, the
    parameters are not re-estimated);
  * deep models see a window of the previous W days, scaled with a scaler
    fitted on the training block, and early-stop on the validation block;
  * all errors are reported in the original units of the target.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from . import metrics
from .split import MinMax, time_split


# --------------------------------------------------------------- features
def lag_features(s: pd.Series, dates: pd.Series, n_lags: int = 14) -> pd.DataFrame:
    f = pd.DataFrame({f"lag_{k}": s.shift(k) for k in range(1, n_lags + 1)})
    prev = s.shift(1)
    f["roll_mean_7"] = prev.rolling(7).mean()
    f["roll_std_7"] = prev.rolling(7).std()
    f["dow"] = pd.to_datetime(dates).dt.dayofweek
    return f


# --------------------------------------------------------------- tabular
def _tabular_models(seed: int):
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from xgboost import XGBRegressor

    return {
        "SLR": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=500, min_samples_leaf=2, random_state=seed, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.9,
                                colsample_bytree=0.9, random_state=seed, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=15, subsample=0.9,
                                  subsample_freq=1, colsample_bytree=0.9, random_state=seed, verbose=-1),
    }


# --------------------------------------------------------------- deep
def _deep_model(kind: str, window: int, hidden: int = 32):
    import torch
    from torch import nn

    class Rec(nn.Module):
        def __init__(self, cell):
            super().__init__()
            self.rnn = cell(1, hidden, batch_first=True)
            self.out = nn.Linear(hidden, 1)

        def forward(self, x):
            h, _ = self.rnn(x)
            return self.out(h[:, -1]).squeeze(-1)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, hidden, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x.transpose(1, 2)).squeeze(-1)

    return {"LSTM": lambda: Rec(nn.LSTM), "RNN": lambda: Rec(nn.RNN), "1D-CNN": CNN}[kind]()


def _fit_deep(kind, x_tr, y_tr, x_val, y_val, window, seed, epochs=300, lr=1e-3, patience=30):
    import copy

    import torch

    torch.manual_seed(seed)
    model = _deep_model(kind, window)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    xt, yt, xv, yv = t(x_tr)[..., None], t(y_tr), t(x_val)[..., None], t(y_val)
    best, best_state, since = float("inf"), None, 0
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(xt), generator=g).split(32):
            opt.zero_grad()
            loss = torch.mean((model(xt[idx]) - yt[idx]) ** 2)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = torch.mean((model(xv) - yv) ** 2).item()
        if vl < best - 1e-9:
            best, best_state, since = vl, copy.deepcopy(model.state_dict()), 0
        else:
            since += 1
            if since >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return lambda x: model(t(x)[..., None]).detach().numpy()


# --------------------------------------------------------------- runner
def run(df: pd.DataFrame, target: str = "HR", cfg: dict | None = None) -> pd.DataFrame:
    cfg = {"test_frac": 0.2, "val_frac": 0.1, "n_lags": 14, "window": 30, "seed": 0,
           "arima_order": (1, 1, 1), "sarima_order": (1, 1, 1), "seasonal_order": (1, 0, 1, 7),
           **(cfg or {})}
    s = df[target].astype(float).reset_index(drop=True)
    n = len(s)
    sp = time_split(n, cfg["test_frac"], cfg["val_frac"])
    y_true = s.iloc[sp.test].to_numpy()
    rows = []

    def record(name, family, pred, t0):
        r = {"model": name, "family": family, **metrics.evaluate(y_true, pred, ["MAE", "RMSE", "sMAPE", "R2", "WI"])}
        r["fit_seconds"] = round(time.time() - t0, 2)
        rows.append(r)

    # tabular
    feats = lag_features(s, df["date"], cfg["n_lags"])
    valid = feats.notna().all(axis=1).to_numpy()
    idx = np.arange(n)
    tr = idx[sp.train][valid[sp.train]]
    va = idx[sp.val]
    te = idx[sp.test]
    for name, model in _tabular_models(cfg["seed"]).items():
        t0 = time.time()
        fit_idx = np.concatenate([tr, va])  # no early stopping here: train on train+val
        model.fit(feats.iloc[fit_idx], s.iloc[fit_idx])
        record(name, "classical" if name == "SLR" else "ensemble", model.predict(feats.iloc[te]), t0)

    # ARIMA / SARIMA
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    hist = s.iloc[: sp.val.stop].to_numpy()
    test = s.iloc[sp.test].to_numpy()
    for name, make in [
        ("ARIMA", lambda h: ARIMA(h, order=cfg["arima_order"])),
        ("SARIMA", lambda h: SARIMAX(h, order=cfg["sarima_order"], seasonal_order=cfg["seasonal_order"])),
    ]:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = make(hist).fit()
            preds = []
            for obs in test:  # rolling one-step forecasts
                preds.append(float(res.forecast(1)[0]))
                res = res.append([obs])
        record(name, "classical", np.array(preds), t0)

    # deep models
    W = cfg["window"]
    sc = MinMax().fit(s.iloc[sp.train].to_numpy()[:, None])
    z = sc.transform(s.to_numpy()[:, None]).ravel()
    X = np.stack([z[i - W : i] for i in range(W, n)])
    Y = z[W:]
    pos = np.arange(W, n)  # position of each target in the original series

    def block(sl):
        m = (pos >= sl.start) & (pos < sl.stop)
        return X[m], Y[m]

    (xtr, ytr), (xva, yva), (xte, _) = block(sp.train), block(sp.val), block(sp.test)
    for kind in ["LSTM", "RNN", "1D-CNN"]:
        t0 = time.time()
        f = _fit_deep(kind, xtr, ytr, xva, yva, W, cfg["seed"])
        record(kind, "deep", sc.inverse(f(xte)[:, None]).ravel(), t0)

    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
