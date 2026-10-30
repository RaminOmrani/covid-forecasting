# Forecasting COVID-19 hospitalisation and death rates

Code for two studies on forecasting hospital demand during the COVID-19 pandemic:

1. **Driver-based neuro-fuzzy forecasting.** An adaptive neuro-fuzzy inference system (ANFIS) predicts the hospitalisation rate (HR) and the death rate (DR) from eight daily indicators: weather (temperature, wind, precipitation), mobility (domestic and international travel), vaccination coverage, infection, and compliance with health protocols. A one-factor-at-a-time sensitivity analysis then ranks the indicators by how much the fitted model relies on each. A time-series ANFIS that sees only the past values of the target is included for comparison.
2. **Benchmark of forecasters.** Nine models from three families predict the next day's hospitalisation rate under one shared protocol: classical (SLR, ARIMA, SARIMA), tree ensembles (Random Forest, XGBoost, LightGBM) and deep sequence models (LSTM, RNN, 1D-CNN).

| Study | Publication | Code |
|---|---|---|
| 1. Driver-based ANFIS, time-series ANFIS, sensitivity analysis | R. Omrani, A. Nemati. *A neuro-fuzzy causal approach for pandemic severity forecasting: COVID-19 case study.* International Journal of Data Science and Analytics 20, 5979–5997 (2025). https://doi.org/10.1007/s41060-025-00813-z | `src/covidfc/anfis.py`, `src/covidfc/anfis_experiments.py`, `scripts/run_anfis.py`, `scripts/run_sensitivity.py` |
| 2. Benchmark of nine forecasters | R. Omrani, A. Nemati, F. Yousefi-Moghadam, M. Mehravaran. *Healthcare demand forecasting: ensemble vs. non-ensemble AI algorithms. COVID-19 hospitalization rate case study.* Under review at Scientific Reports. | `src/covidfc/benchmark.py`, `scripts/run_benchmark.py` |

Shared modules: `data.py` (loading and the synthetic series), `split.py` (chronological split and scaling), `metrics.py`.

## About the data and the numbers

The studies used a 1,010-day national series held by a ministry of health. Those data are confidential and are **not** in this repository. The code runs on a synthetic stand-in series with the same columns (`src/covidfc/data.py`), or on any CSV/Excel file with that layout (see `data/README.md`).

This repository is a clean, documented implementation of the methods. It is written so that it can be read and rerun. It does not reproduce the numbers reported in the papers, which come from the confidential data. Results on the synthetic series only show that the pipeline works.

## Design choices

- **No look-ahead.** Every split is chronological: train, then validation, then test. Scalers are fitted on the training block only. The validation block is used for early stopping; the test block is used once, for the final scores.
- **One-step-ahead benchmark.** Every model predicts day *t* with information up to day *t − 1*. Tabular models use 14 lags, a 7-day rolling mean and standard deviation, and the day of the week. ARIMA and SARIMA produce rolling one-step forecasts. The deep models see a 30-day window.
- **Original units.** All errors are reported in the units of the target, so the models can be compared directly.
- **Repeated runs.** ANFIS results are averaged over several seeds and reported as mean ± standard deviation.
- **Metrics.** MAE, RMSE, MAPE, sMAPE, R², Nash–Sutcliffe efficiency, Willmott's index of agreement and percent bias (`src/covidfc/metrics.py`).

## What the sensitivity analysis does and does not show

Removing one input and refitting shows how much the *fitted model* depends on that input. It does not identify the causal effect of changing that input in the real world. That would need an explicit causal model that deals with confounding, with feedback between policy and outcomes, and with reporting delays. The analysis is a way to rank drivers for further study. It is not evidence about interventions.

## ANFIS implementation

`src/covidfc/anfis.py` is a first-order Takagi–Sugeno ANFIS written in PyTorch for this repository. Each rule has its own Gaussian membership function for every input, so the number of rules does not grow exponentially with the number of inputs. Rule firing strengths are normalised with a log-sum-exp (softmax) for numerical stability. All parameters are trained jointly with Adam. The default settings (500 rules, learning rate 0.015, up to 500 epochs) are the values selected with a Taguchi design in the thesis.

## Layout

```
configs/      default.yaml (paper settings), quick.yaml (fast smoke run)
data/         README.md only; real data are never committed
scripts/      run_anfis.py, run_sensitivity.py, run_benchmark.py, make_synthetic_data.py
src/covidfc/  data.py, split.py, metrics.py, anfis.py, anfis_experiments.py, benchmark.py
tests/        unit tests (pytest)
```

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                            # unit tests
python scripts/run_anfis.py --config configs/quick.yaml
python scripts/run_sensitivity.py --config configs/quick.yaml
python scripts/run_benchmark.py                      # nine-model benchmark
```

With `configs/default.yaml` the ANFIS scripts use the full settings and take longer. Outputs (CSV/JSON) are written to `results/`.

## Licence

MIT. See `LICENSE`.
