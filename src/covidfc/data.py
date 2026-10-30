"""Data loading and a synthetic stand-in dataset.

The original study used a 1,010-day national series held by a ministry of
health. Those data are not public and are NOT included here. To make the code
runnable, `make_synthetic` produces a series with the same column layout:
eight daily indicators, a hospitalisation rate (HR) and a death rate (DR).
The synthetic series only imitates the broad structure (waves, seasonality,
vaccination roll-out, lags); numbers produced on it say nothing about the
real epidemic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DRIVERS = [
    "temperature",
    "wind",
    "precipitation",
    "domestic_travel",
    "international_travel",
    "vaccination",
    "infection",
    "protocol_compliance",
]
TARGETS = ["HR", "DR"]


def make_synthetic(n_days: int = 1010, seed: int = 0, start: str = "2020-02-20") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    date = pd.date_range(start, periods=n_days, freq="D")

    season = np.cos(2 * np.pi * (t - 200) / 365.25)
    temperature = 17 + 10 * season + rng.normal(0, 2, n_days)
    wind = np.clip(3 + 0.8 * rng.standard_normal(n_days) + 0.5 * np.sin(2 * np.pi * t / 365.25), 0, None)
    precipitation = np.clip(rng.gamma(0.6, 2.0, n_days) * (1.2 - 0.5 * season), 0, None)

    # Vaccination: cumulative coverage (%), logistic roll-out from ~day 250.
    vaccination = 85 / (1 + np.exp(-(t - 400) / 45))

    # Behaviour: compliance rises with each wave and fades slowly.
    waves = sum(
        a * np.exp(-0.5 * ((t - c) / w) ** 2)
        for a, c, w in [(1.0, 60, 25), (1.6, 250, 35), (2.2, 430, 30), (2.8, 560, 28), (1.8, 700, 40), (1.2, 900, 45)]
    )
    protocol_compliance = np.clip(40 + 25 * np.tanh(np.convolve(waves, np.ones(14) / 14, mode="same")) + rng.normal(0, 3, n_days), 0, 100)
    domestic_travel = np.clip(100 - 30 * np.tanh(waves) + 8 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 5, n_days), 5, None)
    international_travel = np.clip(20 + 0.04 * t - 10 * np.tanh(waves) + rng.normal(0, 2, n_days), 0, None)

    # Infection (new cases per 100k): waves damped by vaccination and compliance.
    infection = np.clip(
        25 * waves * (1 - 0.6 * vaccination / 100) * (1.15 - 0.3 * protocol_compliance / 100)
        + 2 * (1 - season)
        + rng.normal(0, 1.5, n_days),
        0,
        None,
    )

    def lag(x, k):
        return np.concatenate([np.full(k, x[0]), x[:-k]])

    # Hospitalisation rate depends on infection ~3 days earlier and on vaccination.
    HR = np.clip(
        0.9 * lag(infection, 3) * (1 - 0.45 * vaccination / 100) + 0.03 * (25 - temperature) + rng.normal(0, 0.8, n_days),
        0,
        None,
    )
    # Death rate depends on hospitalisation ~4 days earlier.
    DR = np.clip(0.12 * lag(HR, 4) * (1 - 0.3 * vaccination / 100) + rng.normal(0, 0.08, n_days), 0, None)

    df = pd.DataFrame(
        {
            "date": date,
            "temperature": temperature,
            "wind": wind,
            "precipitation": precipitation,
            "domestic_travel": domestic_travel,
            "international_travel": international_travel,
            "vaccination": vaccination,
            "infection": infection,
            "protocol_compliance": protocol_compliance,
            "HR": HR,
            "DR": DR,
        }
    )
    return df


def load(path: str | Path | None, seed: int = 0) -> pd.DataFrame:
    """Load a CSV/Excel file with the columns above, or fall back to the
    synthetic series when no path is given."""
    if path is None or str(path).lower() == "synthetic":
        return make_synthetic(seed=seed)
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    missing = [c for c in ["date", *DRIVERS, *TARGETS] if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)
