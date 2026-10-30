"""Driver-based and time-series ANFIS for HR and DR."""
import json

from _common import anfis_cfg, fmt, setup
from covidfc.anfis_experiments import driver_model, timeseries_model

cfg, df, out = setup(__doc__)
acfg = anfis_cfg(cfg)
report = {}
for target in ["HR", "DR"]:
    d = driver_model(df, target, acfg)
    t = timeseries_model(df, target, acfg)
    print(f"[{target}] driver-based ANFIS, inputs={d['inputs']}")
    print("   test:", fmt(d["test"]))
    print(f"[{target}] time-series ANFIS, window={t['window']}")
    print("   test:", fmt(t["test"]))
    report[target] = {"driver_based": {k: d[k] for k in ("inputs", "train", "test")},
                      "time_series": {k: t[k] for k in ("window", "train", "test")}}
(out / "anfis_metrics.json").write_text(json.dumps(report, indent=2))
print(f"saved {out / 'anfis_metrics.json'}")
