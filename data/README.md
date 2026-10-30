# Data

The study data (a 1,010-day national series of hospitalisation and death rates
with daily indicators) are confidential and are not distributed.

`python scripts/make_synthetic_data.py` writes `synthetic_covid.csv`, a stand-in
with the same columns. To run the code on your own data, provide a CSV or Excel
file with these columns and set `data:` in `configs/default.yaml`:

| column | meaning |
|---|---|
| date | calendar day |
| temperature, wind, precipitation | daily weather averages |
| domestic_travel, international_travel | mobility indicators |
| vaccination | cumulative vaccination coverage (%) |
| infection | new infections (rate) |
| protocol_compliance | compliance with health protocols (%) |
| HR | hospitalisation rate (target) |
| DR | death rate (target) |
