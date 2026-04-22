"""Shared portfolio backtest helpers for regime models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.metrics import build_nav_series, calculate_performance_metrics


TRADED_ASSETS = {
    "SPY": "spy_adj_close",
    "TLT": "tlt_adj_close",
    "IEF": "ief_adj_close",
    "GLD": "gld_adj_close",
}
RETURN_COLUMN_BY_ASSET = {asset: f"{asset.lower()}_return" for asset in TRADED_ASSETS}
WEIGHT_COLUMN_BY_ASSET = {asset: f"weight_{asset.lower()}" for asset in TRADED_ASSETS}

STATE_LABELS = {
    0: "risk_off",
    1: "transition",
    2: "risk_on",
}

REGIME_ALLOCATIONS = {
    0: {"SPY": 0.10, "TLT": 0.35, "IEF": 0.35, "GLD": 0.20},
    1: {"SPY": 0.40, "TLT": 0.20, "IEF": 0.25, "GLD": 0.15},
    2: {"SPY": 0.70, "TLT": 0.05, "IEF": 0.15, "GLD": 0.10},
}

BENCHMARK_ALLOCATIONS = {
    "equal_weight_4_asset": {"SPY": 0.25, "TLT": 0.25, "IEF": 0.25, "GLD": 0.25},
    "fixed_60_40_stock_bond": {"SPY": 0.60, "TLT": 0.20, "IEF": 0.20, "GLD": 0.00},
}


def validate_allocation_map(
    allocation_map: dict[object, dict[str, float]],
    expected_keys: list[object],
    name: str,
) -> None:
    for key in expected_keys:
        if key not in allocation_map:
            raise ValueError(f"{name} is missing allocation for key: {key}")

        allocation = allocation_map[key]
        missing_assets = [asset for asset in TRADED_ASSETS if asset not in allocation]
        extra_assets = [asset for asset in allocation if asset not in TRADED_ASSETS]
        if missing_assets:
            raise ValueError(
                f"{name} allocation {key} is missing assets: " + ", ".join(missing_assets)
            )
        if extra_assets:
            raise ValueError(
                f"{name} allocation {key} includes unsupported assets: "
                + ", ".join(extra_assets)
            )

        total_weight = float(sum(float(weight) for weight in allocation.values()))
        if not np.isclose(total_weight, 1.0):
            raise ValueError(
                f"{name} allocation {key} sums to {total_weight:.6f}, not 1.0."
            )

        negative_assets = [
            asset for asset, weight in allocation.items() if float(weight) < 0.0
        ]
        if negative_assets:
            raise ValueError(
                f"{name} allocation {key} has negative weights for: "
                + ", ".join(negative_assets)
            )


def load_market_data(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Aligned market data not found: {path}")

    frame = pd.read_parquet(path)
    required_columns = ["date", *TRADED_ASSETS.values()]
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "Aligned market data is missing required columns: "
            + ", ".join(missing_columns)
        )

    frame = frame[required_columns].copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    if frame["date"].isna().any():
        raise ValueError("Aligned market data contains invalid dates.")

    frame = frame.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    frame = frame.reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("Aligned market data contains duplicate dates after sorting.")

    if frame[list(TRADED_ASSETS.values())].isna().any().any():
        raise ValueError("Aligned market data contains missing prices for traded assets.")

    for column in TRADED_ASSETS.values():
        if (frame[column] <= 0.0).any():
            raise ValueError(f"Aligned market data contains non-positive prices in {column}.")

    return frame


def require_existing_file(path_like: str | Path, description: str) -> Path:
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def build_asset_return_frame(market_data: pd.DataFrame) -> pd.DataFrame:
    return_frame = pd.DataFrame({"date": market_data["date"]})
    for asset, price_column in TRADED_ASSETS.items():
        return_frame[RETURN_COLUMN_BY_ASSET[asset]] = market_data[price_column].pct_change()
    return return_frame


def build_allocation_lookup(
    allocation_map: dict[object, dict[str, float]],
    key_name: str,
) -> pd.DataFrame:
    lookup = pd.DataFrame.from_dict(allocation_map, orient="index")
    lookup = lookup[list(TRADED_ASSETS.keys())]
    lookup.index.name = key_name
    lookup = lookup.reset_index()
    return lookup.rename(
        columns={asset: WEIGHT_COLUMN_BY_ASSET[asset] for asset in TRADED_ASSETS}
    )


def run_strategy_backtest(
    signal_frame: pd.DataFrame,
    regime_allocations: dict[int, dict[str, float]],
) -> pd.DataFrame:
    allocation_lookup = build_allocation_lookup(regime_allocations, key_name="signal_state")
    strategy_frame = signal_frame.merge(
        allocation_lookup,
        on="signal_state",
        how="left",
        validate="many_to_one",
    )

    weight_columns = list(WEIGHT_COLUMN_BY_ASSET.values())
    if strategy_frame[weight_columns].isna().any().any():
        raise ValueError("Strategy frame contains missing regime allocation weights.")

    strategy_returns = pd.Series(0.0, index=strategy_frame.index, dtype=float)
    for asset in TRADED_ASSETS:
        strategy_returns = strategy_returns + (
            strategy_frame[RETURN_COLUMN_BY_ASSET[asset]]
            * strategy_frame[WEIGHT_COLUMN_BY_ASSET[asset]]
        )
    strategy_frame["strategy_return"] = strategy_returns
    strategy_frame["strategy_nav"] = build_nav_series(strategy_frame["strategy_return"]).to_numpy()
    return strategy_frame


def run_benchmark_backtests(
    signal_frame: pd.DataFrame,
    benchmark_allocations: dict[str, dict[str, float]],
) -> pd.DataFrame:
    benchmark_frame = pd.DataFrame({"date": signal_frame["date"]})
    for benchmark_name, allocation in benchmark_allocations.items():
        benchmark_returns = pd.Series(0.0, index=signal_frame.index, dtype=float)
        for asset in TRADED_ASSETS:
            benchmark_returns = benchmark_returns + (
                signal_frame[RETURN_COLUMN_BY_ASSET[asset]] * float(allocation[asset])
            )
        benchmark_frame[f"{benchmark_name}_return"] = benchmark_returns
        benchmark_frame[f"{benchmark_name}_nav"] = build_nav_series(benchmark_returns).to_numpy()

    return benchmark_frame


def build_metrics_table(
    strategy_frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    benchmark_names: list[str],
    strategy_name: str,
) -> pd.DataFrame:
    rows = [
        {
            "portfolio_name": strategy_name,
            **calculate_performance_metrics(strategy_frame["strategy_return"]),
        }
    ]
    for benchmark_name in benchmark_names:
        rows.append(
            {
                "portfolio_name": benchmark_name,
                **calculate_performance_metrics(benchmark_frame[f"{benchmark_name}_return"]),
            }
        )

    return pd.DataFrame(rows)


def ensure_parent_directory(path_like: str | Path) -> None:
    Path(path_like).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def format_percentage(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{100.0 * float(value):.2f}%"


def format_decimal(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    separator = ["---"] * len(headers)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        markdown_rows.append("| " + " | ".join(row) + " |")
    return "\n".join(markdown_rows)


def build_allocation_table() -> str:
    rows = []
    for state in sorted(REGIME_ALLOCATIONS):
        allocation = REGIME_ALLOCATIONS[state]
        rows.append(
            [
                str(state),
                STATE_LABELS[state],
                format_percentage(allocation["SPY"]),
                format_percentage(allocation["TLT"]),
                format_percentage(allocation["IEF"]),
                format_percentage(allocation["GLD"]),
            ]
        )
    return build_markdown_table(
        headers=["state", "label", "SPY", "TLT", "IEF", "GLD"],
        rows=rows,
    )


def build_benchmark_table() -> str:
    rows = []
    for name, allocation in BENCHMARK_ALLOCATIONS.items():
        rows.append(
            [
                name,
                format_percentage(allocation["SPY"]),
                format_percentage(allocation["TLT"]),
                format_percentage(allocation["IEF"]),
                format_percentage(allocation["GLD"]),
            ]
        )
    return build_markdown_table(
        headers=["benchmark", "SPY", "TLT", "IEF", "GLD"],
        rows=rows,
    )


def build_metrics_markdown(metrics_frame: pd.DataFrame) -> str:
    rows = []
    for _, row in metrics_frame.iterrows():
        rows.append(
            [
                str(row["portfolio_name"]),
                str(int(row["observations"])),
                format_percentage(row["cumulative_return"]),
                format_percentage(row["annualized_return"]),
                format_percentage(row["annualized_volatility"]),
                format_decimal(row["sharpe_ratio"]),
                format_percentage(row["max_drawdown"]),
            ]
        )
    return build_markdown_table(
        headers=[
            "portfolio",
            "obs",
            "cumulative_return",
            "annualized_return",
            "annualized_volatility",
            "sharpe_ratio",
            "max_drawdown",
        ],
        rows=rows,
    )
