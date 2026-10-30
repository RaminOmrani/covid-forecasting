import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from covidfc import metrics  # noqa: E402
from covidfc.data import DRIVERS, make_synthetic  # noqa: E402
from covidfc.split import MinMax, time_split  # noqa: E402


def test_metrics_perfect_forecast():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    m = metrics.evaluate(y, y)
    assert m["MAE"] == 0 and m["RMSE"] == 0 and m["R2"] == 1 and m["WI"] == 1 and m["PBIAS"] == 0


def test_time_split_is_chronological_and_disjoint():
    sp = time_split(100, 0.2, 0.1)
    assert sp.train.stop == sp.val.start and sp.val.stop == sp.test.start and sp.test.stop == 100
    assert sp.test.stop - sp.test.start == 20


def test_scaler_uses_training_block_only():
    x = np.arange(10.0)[:, None]
    sc = MinMax().fit(x[:5])
    assert sc.transform(x)[-1, 0] > 1  # test values may fall outside [0, 1]
    assert np.allclose(sc.inverse(sc.transform(x)), x)


def test_synthetic_layout():
    df = make_synthetic(n_days=120)
    assert list(df.columns) == ["date", *DRIVERS, "HR", "DR"]
    assert (df[["HR", "DR"]] >= 0).all().all()


def test_anfis_learns_a_smooth_function():
    from covidfc.anfis import fit_anfis, predict

    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (300, 2))
    y = np.sin(3 * x[:, 0]) + 0.5 * x[:, 1]
    r = fit_anfis(x[:200], y[:200], x[200:250], y[200:250], n_rules=20, lr=0.03, epochs=300, seed=0)
    assert metrics.r2(y[250:], predict(r.model, x[250:])) > 0.9
