# Cross-Asset Regime Detection Pipeline

本项目用于复现一个完整的量化研究流程：  
从市场数据下载与清洗开始，构建特征，训练 walk-forward HMM 与 walk-forward K-means 基线，执行策略回测，输出诊断、子区间分析、图表和最终报告。

---

## 1. 项目目标与流程概览

完整复现链路如下（按顺序执行）：

1. 下载并对齐数据：`scripts/01_download_data.py`
2. 构建特征：`scripts/02_build_features.py`
3. 训练 walk-forward HMM：`scripts/06_fit_walkforward_hmm.py`
4. 训练 walk-forward K-means 基线：`scripts/04_fit_clustering_baseline.py`
5. 运行 HMM 回测：`scripts/07_run_walkforward_backtest.py`
6. 运行 K-means 回测：`scripts/08_run_kmeans_backtest.py`
7. 回测诊断：`scripts/11_build_backtest_diagnostics.py`
8. HMM 稳健性检验：`scripts/12_run_hmm_robustness.py`
9. 子区间分析：`scripts/13_run_subperiod_analysis.py`
10. 生成最终图表：`scripts/09_build_final_figures.py`
11. 组装最终报告：`scripts/10_assemble_final_report.py`

---

## 2. 从 0 开始环境配置

### 2.1 系统要求

- macOS / Linux / Windows 均可（已在 macOS 下开发）
- Python `3.10+`（推荐 `3.10` 或 `3.11`）
- 可联网访问 Yahoo Finance 与 FRED API

### 2.2 克隆仓库

```bash
git clone <your-repo-url>
cd project_code
```

### 2.3 创建并激活虚拟环境

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2.4 安装依赖

优先使用仓库内的依赖文件安装：

```bash
pip install -r requirements.txt
```

如需手动安装，核心依赖如下：

```bash
pip install numpy pandas scikit-learn hmmlearn yfinance fredapi pyarrow pillow
```

> 说明：  
> - `pyarrow` 用于 parquet 读写；  
> - `pillow` 用于生成最终 PNG 图；  
> - `fredapi` 需要配合 FRED API Key 使用。

---

## 3. API Key 配置（必须）

数据下载步骤依赖 FRED。请先申请并配置 FRED API Key：

1. 在 [FRED 官网](https://fred.stlouisfed.org/)申请 API Key
2. 在项目根目录创建 `.env` 文件（与 `scripts/` 同级）
3. 写入：

```env
FRED_API_KEY=你的_fred_key
```

`scripts/01_download_data.py` 会自动读取根目录 `.env` 并注入环境变量。

---

## 4. 数据与配置说明

- 数据源配置文件：`config/market_data_sources.json`
- 默认抓取区间：
  - `start_date`: `2010-01-01`
  - `end_date`: `null`（表示跑到当天）
- 默认输出：
  - 原始单序列：`data/raw/`
  - 对齐后主数据：`data/processed/aligned_market_data.parquet`

---

## 5. 逐步复现（推荐首次）

以下命令均在项目根目录执行。

### Step 1) 下载并对齐市场数据

```bash
python scripts/01_download_data.py
```

预期产出：

- `data/processed/aligned_market_data.parquet`
- `data/raw/*.csv`

---

### Step 2) 构建建模特征表

```bash
python scripts/02_build_features.py
```

预期产出：

- `data/processed/model_features.parquet`

---

### Step 3) 训练 walk-forward HMM

```bash
python scripts/06_fit_walkforward_hmm.py
```

预期产出：

- `results/regimes/hmm_walkforward_state_labels.parquet`
- `results/regimes/hmm_walkforward_state_probabilities.parquet`
- `results/models/hmm_walkforward_metadata.json`
- `docs/hmm_walkforward_notes.md`

---

### Step 4) 训练 walk-forward K-means 基线

```bash
python scripts/04_fit_clustering_baseline.py
```

预期产出：

- `results/regimes/cluster_labels.parquet`
- `results/models/clustering_baseline_metadata.json`
- `docs/clustering_baseline_notes.md`

---

### Step 5) HMM 策略回测

```bash
python scripts/07_run_walkforward_backtest.py
```

预期产出：

- `results/backtests/walkforward_strategy_returns.parquet`
- `results/backtests/walkforward_strategy_variant_returns.parquet`
- `results/backtests/walkforward_benchmark_returns.parquet`
- `results/backtests/walkforward_strategy_metrics.csv`
- `results/backtests/walkforward_allocation_sweep_metrics.csv`
- `results/backtests/walkforward_allocation_sweep_metadata.json`
- `docs/backtest_walkforward_notes.md`

---

### Step 6) K-means 策略回测

```bash
python scripts/08_run_kmeans_backtest.py
```

预期产出：

- `results/backtests/kmeans_strategy_returns.parquet`
- `results/backtests/kmeans_benchmark_returns.parquet`
- `results/backtests/kmeans_strategy_metrics.csv`
- `results/backtests/kmeans_vs_walkforward_strategy_metrics.csv`
- `docs/backtest_kmeans_notes.md`

---

### Step 7) 回测诊断（换手率、交易成本敏感性）

```bash
python scripts/11_build_backtest_diagnostics.py
```

预期产出：

- `results/backtests/strategy_turnover_summary.csv`
- `results/backtests/strategy_cost_sensitivity.csv`
- `docs/backtest_diagnostics.md`

---

### Step 8) HMM 稳健性检查

```bash
python scripts/12_run_hmm_robustness.py
```

预期产出：

- `results/models/hmm_robustness_summary.csv`
- `docs/hmm_robustness_notes.md`

---

### Step 9) 子区间比较分析

```bash
python scripts/13_run_subperiod_analysis.py
```

预期产出：

- `results/backtests/shared_window_subperiod_metrics.csv`
- `docs/subperiod_analysis.md`

---

### Step 10) 生成最终图表

```bash
python scripts/09_build_final_figures.py
```

预期产出：

- `results/figures/figure_01_regime_persistence_comparison.png`
- `results/figures/figure_02_shared_window_nav_comparison.png`
- `results/figures/figure_03_walkforward_state_probabilities.png`
- `results/figures/figure_04_shared_window_strategy_metrics.png`

---

### Step 11) 组装最终报告

```bash
python scripts/10_assemble_final_report.py
```

预期产出：

- `deliverables/final_report/ELEN4904_final_report.md`
- `deliverables/final_report/assets/*.png`

---

## 6. 一键复现命令（已配好环境后）

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
python scripts/10_assemble_final_report.py
```

---

## 7. 结果自检清单

复现完成后，至少检查以下文件是否存在且非空：

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

## 8. 常见问题排查

- `Missing FRED_API_KEY environment variable`  
  没有配置 `.env` 或变量名不正确。确认根目录 `.env` 包含 `FRED_API_KEY=...`。

- `Missing dependency 'yfinance'` 或 `Missing dependency 'fredapi'`  
  说明当前 Python 环境未安装依赖，重新执行 `pip install ...`。

- parquet 读写报错（`pyarrow` / `fastparquet`）  
  执行 `pip install pyarrow`。

- 下载阶段网络报错或空数据  
  重试命令；确认网络能访问 Yahoo 与 FRED；必要时更换网络环境。

- Windows 激活虚拟环境失败  
  使用管理员 PowerShell，或执行：`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` 后重试。

---

## 9. 可复现性建议（给协作者）

- 每次实验固定同一套 Python 小版本（如 3.10.x）
- 将 `.env` 与关键输出路径保持一致
- 复现时按本文档顺序执行，不跳步
- 提交结果时建议附带：
  - `results/models/*.json` 元数据
  - `docs/*notes*.md` 简要说明
  - 关键 `results/backtests/*.csv`
