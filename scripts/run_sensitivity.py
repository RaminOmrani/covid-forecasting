"""One-factor-at-a-time sensitivity of the driver-based ANFIS."""
from _common import anfis_cfg, setup
from covidfc.anfis_experiments import ofat_sensitivity

cfg, df, out = setup(__doc__)
for target in ["HR", "DR"]:
    tab = ofat_sensitivity(df, target, anfis_cfg(cfg))
    print(f"\n[{target}] increase in test RMSE when an input is removed")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    tab.to_csv(out / f"sensitivity_{target}.csv", index=False)
