# Cross-Asset Regime Detection Pipeline

This repository reproduces an end-to-end quantitative research workflow:
market data ingestion and alignment, feature engineering, walk-forward regime modeling (HMM + K-means baseline), strategy backtesting, diagnostics, robustness checks, figure generation, and final report assembly.

---

## 1) Pipeline Overview

Run the following scripts in order:

1. Download and align market data: `scripts/01_download_data.py`
2. Build modeling features: `scripts/02_build_features.py`
3. Fit walk-forward HMM: `scripts/06_fit_walkforward_hmm.py`
4. Fit walk-forward K-means baseline: `scripts/04_fit_clustering_baseline.py`
5. Run HMM backtest: `scripts/07_run_walkforward_backtest.py`
6. Run K-means backtest: `scripts/08_run_kmeans_backtest.py`
7. Build backtest diagnostics: `scripts/11_build_backtest_diagnostics.py`
8. Run HMM robustness checks: `scripts/12_run_hmm_robustness.py`
9. Run subperiod analysis: `scripts/13_run_subperiod_analysis.py`
10. Build final figures: `scripts/09_build_final_figures.py`


---

## 2) Project Results (Current Run)

The following metrics come directly from:

- `results/backtests/kmeans_vs_walkforward_strategy_metrics.csv`
- `results/models/hmm_robustness_summary.csv`
- `results/backtests/shared_window_subperiod_metrics.csv`

### Shared-window strategy comparison (3321 observations)

| Portfolio | Cumulative Return | Annualized Return | Annualized Volatility | Sharpe | Max Drawdown |
|---|---:|---:|---:|---:|---:|
| walkforward_hmm_regime_strategy | 2.2411 | 0.0933 | 0.0981 | 0.9592 | -0.2510 |
| kmeans_regime_strategy | 0.8704 | 0.0487 | 0.0874 | 0.5873 | -0.3237 |
| equal_weight_4_asset | 1.3364 | 0.0665 | 0.0814 | 0.8315 | -0.2163 |
| fixed_60_40_stock_bond | 2.3137 | 0.0952 | 0.1023 | 0.9401 | -0.2409 |

Key takeaways:

- The walk-forward HMM materially outperforms the walk-forward K-means baseline on return, Sharpe, and drawdown.
- Versus static benchmarks, HMM is competitive with fixed 60/40 on cumulative return and Sharpe, but does not strictly dominate drawdown.

### Subperiod behavior (HMM vs K-means)

From `results/backtests/shared_window_subperiod_metrics.csv`:

- `2013-2016`: HMM beats K-means (higher return and Sharpe)
- `2017-2019`: K-means has slightly higher Sharpe
- `2020-2022`: HMM strongly outperforms K-means (K-means return is negative)
- `2023-2026`: K-means is slightly stronger than HMM

This indicates HMM’s edge is real on the full shared window, but not uniform across all market regimes.

### HMM robustness highlights

From `results/models/hmm_robustness_summary.csv`:

- Baseline spec (`3 states`, monthly refit, diag covariance) has state persistence `0.9187`
- Alternative specs keep high persistence (roughly `0.91` to `0.97`)
- The tied-covariance variant reaches the highest persistence (`0.9696`)

Overall, regime assignments remain stable across reasonable specification changes.

---

## 3) Environment Setup From Scratch

### Requirements

- Python `3.10+` (recommended: `3.10` or `3.11`)
- Internet access to Yahoo Finance and FRED API
- macOS / Linux / Windows

### Clone repository

```bash
git clone <your-repo-url>
cd project_code
```

### Create and activate virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### Install dependencies

```bash
pip install -r requirements.txt
```

Manual equivalent:

```bash
pip install numpy pandas scikit-learn hmmlearn yfinance fredapi pyarrow pillow
```

---

## 4) API Key Setup (Required)

The data ingestion step uses FRED via `fredapi`.

1. Request a FRED API key at [FRED](https://fred.stlouisfed.org/)
2. Create `.env` at repository root
3. Add:

```env
FRED_API_KEY=your_fred_api_key
```

`scripts/01_download_data.py` automatically loads `.env`.

---

## 5) Data and Configuration

- Source config: `config/market_data_sources.json`
- Default start date: `2010-01-01`
- Default end date: `null` (up to current date)
- Main processed output: `data/processed/aligned_market_data.parquet`
- Raw per-series outputs: `data/raw/`

---

## 6) Step-by-Step Reproduction

Run all commands from repository root.

### Step 1: Download and align data

```bash
python scripts/01_download_data.py
```

Outputs:

- `data/processed/aligned_market_data.parquet`
- `data/raw/*.csv`

### Step 2: Build features

```bash
python scripts/02_build_features.py
```

Outputs:

- `data/processed/model_features.parquet`

### Step 3: Fit walk-forward HMM

```bash
python scripts/06_fit_walkforward_hmm.py
```

Outputs:

- `results/regimes/hmm_walkforward_state_labels.parquet`
- `results/regimes/hmm_walkforward_state_probabilities.parquet`
- `results/models/hmm_walkforward_metadata.json`
- `docs/hmm_walkforward_notes.md`

### Step 4: Fit walk-forward K-means baseline

```bash
python scripts/04_fit_clustering_baseline.py
```

Outputs:

- `results/regimes/cluster_labels.parquet`
- `results/models/clustering_baseline_metadata.json`
- `docs/clustering_baseline_notes.md`

### Step 5: Run HMM backtest

```bash
python scripts/07_run_walkforward_backtest.py
```

Outputs:

- `results/backtests/walkforward_strategy_returns.parquet`
- `results/backtests/walkforward_strategy_variant_returns.parquet`
- `results/backtests/walkforward_benchmark_returns.parquet`
- `results/backtests/walkforward_strategy_metrics.csv`
- `results/backtests/walkforward_allocation_sweep_metrics.csv`
- `results/backtests/walkforward_allocation_sweep_metadata.json`
- `docs/backtest_walkforward_notes.md`

### Step 6: Run K-means backtest

```bash
python scripts/08_run_kmeans_backtest.py
```

Outputs:

- `results/backtests/kmeans_strategy_returns.parquet`
- `results/backtests/kmeans_benchmark_returns.parquet`
- `results/backtests/kmeans_strategy_metrics.csv`
- `results/backtests/kmeans_vs_walkforward_strategy_metrics.csv`
- `docs/backtest_kmeans_notes.md`

### Step 7: Build diagnostics (turnover and cost sensitivity)

```bash
python scripts/11_build_backtest_diagnostics.py
```

Outputs:

- `results/backtests/strategy_turnover_summary.csv`
- `results/backtests/strategy_cost_sensitivity.csv`
- `docs/backtest_diagnostics.md`

### Step 8: Run HMM robustness checks

```bash
python scripts/12_run_hmm_robustness.py
```

Outputs:

- `results/models/hmm_robustness_summary.csv`
- `docs/hmm_robustness_notes.md`

### Step 9: Run subperiod analysis

```bash
python scripts/13_run_subperiod_analysis.py
```

Outputs:

- `results/backtests/shared_window_subperiod_metrics.csv`
- `docs/subperiod_analysis.md`

### Step 10: Build final figures

```bash
python scripts/09_build_final_figures.py
```

Outputs:

- `results/figures/figure_01_regime_persistence_comparison.png`
- `results/figures/figure_02_shared_window_nav_comparison.png`
- `results/figures/figure_03_walkforward_state_probabilities.png`
- `results/figures/figure_04_shared_window_strategy_metrics.png`


## 7) One-Command Full Run

```bash
python scripts/01_download_data.py && \
python scripts/02_build_features.py && \
python scripts/06_fit_walkforward_hmm.py && \
python scripts/04_fit_clustering_baseline.py && \
python scripts/07_run_walkforward_backtest.py && \
python scripts/08_run_kmeans_backtest.py && \
python scripts/11_build_backtest_diagnostics.py && \
python scripts/12_run_hmm_robustness.py && \
python scripts/13_run_subperiod_analysis.py && \
python scripts/09_build_final_figures.py && \
```

---

## 8) Reproducibility Checklist

After running the pipeline, verify these files exist and are non-empty:

- `data/processed/aligned_market_data.parquet`
- `data/processed/model_features.parquet`
- `results/regimes/hmm_walkforward_state_labels.parquet`
- `results/regimes/cluster_labels.parquet`
- `results/backtests/walkforward_strategy_metrics.csv`
- `results/backtests/kmeans_vs_walkforward_strategy_metrics.csv`
- `results/models/hmm_robustness_summary.csv`
- `results/figures/figure_04_shared_window_strategy_metrics.png`
- `deliverables/final_report/ELEN4904_final_report.md`

---

## 9) Troubleshooting

- `Missing FRED_API_KEY environment variable`  
  Add `FRED_API_KEY=...` to root `.env`.

- `Missing dependency 'yfinance'` or `Missing dependency 'fredapi'`  
  Reinstall dependencies with `pip install -r requirements.txt`.

- Parquet import/export error (`pyarrow` / `fastparquet`)  
  Install `pyarrow`.

- Network/data download failures  
  Retry, check network connectivity, and ensure Yahoo/FRED endpoints are reachable.

- Windows virtual environment activation issues  
  Run PowerShell as admin, then:
  `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

---

## 10) Collaboration Notes

- Keep Python minor version consistent across collaborators (for example `3.10.x`)
- Keep `.env` naming and output paths consistent
- Run scripts in the documented order
- Share these artifacts for easier validation:
  - `results/models/*.json`
  - `docs/*notes*.md`
  - key tables in `results/backtests/*.csv`
