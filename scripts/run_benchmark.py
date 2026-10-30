"""Nine-model benchmark for one-step-ahead forecasting."""
from _common import setup
from covidfc import benchmark

cfg, df, out = setup(__doc__)
b = cfg["benchmark"]
tab = benchmark.run(df, target=b["target"], cfg={**cfg["split"], **b})
print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
tab.to_csv(out / f"benchmark_{b['target']}.csv", index=False)
